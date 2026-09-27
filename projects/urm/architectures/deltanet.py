"""External model module: DeltaNet (arch-025).

Verified composition row (exact U2.D): the mixer is the canonical delta law
with c=1, β, G=I (no decay), read after update — verified against
fla/ops/delta_rule/naive.py + fla/layers/delta_net.py @ 864a87f6. External
stages (sweep): Q/K/V projections, l2 key/query normalization, the sigmoid
beta projection, the output RMSNorm and output projection. The short
convolution, cache/decode modes and chunked prefill path are residual external
work, not claimed here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class DeltaNetLayer(K2LinearStateLayer):
    """One DeltaNet layer: external projections + typed U2.D mixer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__(
            hidden_size, num_heads, head_k_dim, head_v_dim,
            delta=True, gate_scope="none", scale_rule="key_dim_rsqrt",
            target=target, intent=intent,
        )
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.b_proj = torch.nn.Linear(hidden_size, num_heads, bias=False)
        self.o_norm = torch.nn.RMSNorm(head_v_dim, eps=1e-5)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        # External: l2-norm q/k; sigmoid beta.
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        # Mixer (fla layout [B, H, T, *]); zero initial state.
        out = self._run_mixer({
            "query": q.transpose(1, 2),
            "key": k.transpose(1, 2),
            "value": v.transpose(1, 2),
            "beta": beta.transpose(1, 2),
            "log_decay": torch.zeros(B, H, T, device=q.device),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]  # [B, H, T, V]
        o = out.transpose(1, 2).reshape(B, T, H, dv)
        o = self.o_norm(o.float()).to(o.dtype)
        return self.o_proj(o.reshape(B, T, H * dv))


__all__ = ["DeltaNetLayer"]
