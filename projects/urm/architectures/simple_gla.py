"""External model modules: Simple GLA (arch-018) and Lightning Attention (arch-016).

Both are U2.A additive linear attention delegating to the simple-GLA law
``S_t = exp(g_t) S_{t-1} + k_t v_t``, ``o_t = (q_t·scale) S_t`` (after-update
read) — verified against fla/ops/simple_gla + fla/layers @ 864a87f6.

- Simple GLA (018): per-HEAD, per-token data-dependent gate ``g =
  logsigmoid(gk_proj(x)) / gate_logit_normalizer`` (head scope).
- Lightning (016): STATIC per-head decay ``g_gamma[h] = -(8/H)·(1 −
  layer_idx/num_layers)·h`` (constant over time), supplied as an external
  operand; the sweep confirmed it is a fixed head-scalar gate, NOT
  data-dependent.

External stages: Q/K/V projections, the gate construction (projection or
static schedule), output projection. Short conv, cache/chunked paths are
residual external work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class SimpleGLALayer(K2LinearStateLayer):
    """arch-018: data-dependent head-scalar gate."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, gate_logit_normalizer: int = 8, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="head", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.gate_logit_normalizer = gate_logit_normalizer
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.gk_proj = torch.nn.Linear(hidden_size, num_heads, bias=True)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        g = F.logsigmoid(self.gk_proj(hidden_states)) / self.gate_logit_normalizer  # [B,T,H]
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


class LightningAttentionLayer(K2LinearStateLayer):
    """arch-016: static per-head decay schedule, supplied as an external operand."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, layer_idx: int = 0, num_layers: int = 1,
                 target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="head", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)
        # Static head-scalar log-decay: g_gamma[h] = -(8/H)(1 - layer_idx/num_layers)·h
        ratio = 1.0 - layer_idx / num_layers
        g_gamma = -(8.0 / num_heads) * ratio * torch.arange(num_heads, dtype=torch.float32)
        self.register_buffer("g_gamma", g_gamma)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        g = self.g_gamma.view(1, 1, H).expand(B, T, H)  # static, constant over time
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


__all__ = ["SimpleGLALayer", "LightningAttentionLayer"]
