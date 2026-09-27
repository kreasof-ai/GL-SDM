"""External model module: RetNet / multiscale retention (arch-017).

Verified composition row (U2.A static head decay): the state law is
``S_t = γ_h · S_{t-1} + k_t v_t``, ``o_t = (q_t/√d) S_t`` (after-update read)
with a STATIC per-head scalar decay ``γ_h = 1 − 2^(−5−h)`` — the sweep
disproved the data-dependent-gate hypothesis (A10) and confirmed the static
schedule, against fla/ops/retention + fla/layers/multiscale_retention.py @
864a87f6. External stages: Q/K/V projections, rotary embeddings, the swish
gating and RMSNorm output, the output projection. The static log-decay is
supplied as an external operand (log γ_h); rotary/cache/chunked paths are
residual external work.
"""

from __future__ import annotations

import torch

from architectures.k2_linear_state import K2LinearStateLayer


class RetNetLayer(K2LinearStateLayer):
    """One RetNet layer: external projections + typed U2.A static-decay mixer."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="head", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.g_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)
        # Static per-head decay γ_h = 1 − 2^(−5−h); the mixer takes log γ_h.
        gamma = 1.0 - torch.pow(torch.tensor(2.0), -5.0 - torch.arange(num_heads, dtype=torch.float32))
        self.register_buffer("log_gamma", torch.log(gamma))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        g = self.log_gamma.view(1, 1, H).expand(B, T, H)  # static, constant over time
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        o = out.transpose(1, 2).reshape(B, T, H * dv)
        o = o * torch.nn.functional.silu(self.g_proj(hidden_states))
        return self.o_proj(o)


__all__ = ["RetNetLayer"]
