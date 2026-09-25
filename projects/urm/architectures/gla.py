"""External model module: GLA (arch-019).

Verified composition row (U2.A channel-diagonal gate): the mixer is additive
linear attention with a per-key-channel, per-token data-dependent decay gate
``gk = logsigmoid(gk_proj(x)) / gate_logit_normalizer`` — the state law
``h_t = exp(gk_t) ⊗ h_{t-1} + k_t v_t``, after-update read — verified against
fla/ops/gla/naive.py + fla/layers/gla.py @ 864a87f6. External stages: Q/K/V
projections, the low-rank gate projection (gk_proj) and logsigmoid/normalizer,
the output gate and output projection. The short convolution, multi-kv-group
repetition, clamp_min variant and cache/chunked paths are residual external
work, not claimed here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class GLALayer(K2LinearStateLayer):
    """One GLA layer: external projections + typed U2.A channel-gate mixer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        *,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: int = 16,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__(
            hidden_size, num_heads, head_k_dim, head_v_dim,
            delta=False, gate_scope="channel", scale_rule="key_dim_rsqrt",
            target=target, intent=intent,
        )
        self.gate_logit_normalizer = gate_logit_normalizer
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.gk_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            torch.nn.Linear(gate_low_rank_dim, num_heads * head_k_dim, bias=True),
        )
        self.g_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        gk = self.gk_proj(hidden_states).view(B, T, H, dk)
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        out = self._run_mixer({
            "query": q.transpose(1, 2),
            "key": k.transpose(1, 2),
            "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": gk.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]  # [B, H, T, V]
        o = out.transpose(1, 2).reshape(B, T, H * dv)
        # External output gate (swish) + output projection.
        g = self.g_proj(hidden_states)
        o = o * F.silu(g)
        return self.o_proj(o)


__all__ = ["GLALayer"]
