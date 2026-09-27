"""External model module: Gated DeltaNet (arch-026).

Verified composition row (U2.D + head-scalar decay): the mixer is the canonical
delta law with a per-head scalar log-decay gate ``g_t`` — ``S' = exp(g_t)·S``,
``delta = β_t (v_t − k_tᵀ S')``, ``S_t = S' + k_t δᵀ``, read after update,
scale 1/√K — verified against fla/ops/gated_delta_rule + fla/layers/
gated_deltanet.py @ 864a87f6. The gate value ``g = −exp(A_log)·softplus(g_in +
dt_bias)`` is computed externally (fla/layers/gate.py). External stages: Q/K/V
projections, l2 q/k norm, the gate projection + A_log/dt_bias schedule, sigmoid
beta, output RMSNorm and output projection. Short conv, cache/chunked paths are
residual external work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class GatedDeltaNetLayer(K2LinearStateLayer):
    """One Gated DeltaNet layer: external projections + typed U2.D head-gate mixer."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=True, gate_scope="head", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.b_proj = torch.nn.Linear(hidden_size, num_heads, bias=False)
        self.g_proj = torch.nn.Linear(hidden_size, num_heads, bias=False)  # gate input g_in
        # Gate schedule parameters (fla/layers/gate.py): g = −exp(A_log)·softplus(g_in + dt_bias)
        self.A_log = torch.nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = torch.nn.Parameter(torch.zeros(num_heads))
        self.o_norm = torch.nn.RMSNorm(head_v_dim, eps=1e-5)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def _gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # g = −exp(A_log) · softplus(g_in + dt_bias), per head — [B, T, H].
        g_in = self.g_proj(hidden_states)
        return -torch.exp(self.A_log) * F.softplus(g_in + self.dt_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = F.normalize(self.q_proj(hidden_states).view(B, T, H, dk), p=2, dim=-1)
        k = F.normalize(self.k_proj(hidden_states).view(B, T, H, dk), p=2, dim=-1)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        beta = torch.sigmoid(self.b_proj(hidden_states))
        g = self._gate(hidden_states)
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": beta.transpose(1, 2),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        o = out.transpose(1, 2).reshape(B, T, H, dv)
        o = self.o_norm(o.float()).to(o.dtype)
        return self.o_proj(o.reshape(B, T, H * dv))


__all__ = ["GatedDeltaNetLayer"]
