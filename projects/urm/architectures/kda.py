"""External model module: KDA (arch-028).

Verified composition row (U2.D + key-channel decay): the mixer is the delta law
with a per-KEY-CHANNEL log-decay ``g_t ∈ R^K`` — ``S ← S·exp(g_t)``, ``S ← S +
(β_t k_t) ⊗ (v_t − k_tᵀ S_decayed)``, read after update with the explicit
1/√K read-scale — verified against fla/ops/kda/naive.py + fla/layers/kda.py @
864a87f6. External stages: Q/K/V projections (silu), the gate projection +
A_log/dt_bias schedule (g = −exp(A_log)·softplus(f_proj(x) + dt_bias), per
channel), beta projection, output projection. The in-kernel qk-l2norm / gate /
beta-sigmoid flags, GVA (num_v_heads ≠ num_heads) and cache/chunked paths are
residual external work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class KDALayer(K2LinearStateLayer):
    """One KDA layer: external projections + typed U2.D channel-gate mixer."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=True, gate_scope="channel", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.f_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)  # gate input
        self.b_proj = torch.nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = torch.nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = torch.nn.Parameter(torch.zeros(num_heads * head_k_dim))
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def _gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # g = −exp(A_log)·softplus(f_proj(x) + dt_bias), per channel — [B,T,H,K].
        g_in = self.f_proj(hidden_states)
        A_log = self.A_log.repeat_interleave(self.head_k_dim)
        return -torch.exp(A_log) * F.softplus(g_in + self.dt_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        beta = torch.sigmoid(self.b_proj(hidden_states))
        g = self._gate(hidden_states)  # [B,T,H*K]
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": beta.transpose(1, 2),
            "log_decay": g.view(B, T, H, dk).transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


__all__ = ["KDALayer"]
