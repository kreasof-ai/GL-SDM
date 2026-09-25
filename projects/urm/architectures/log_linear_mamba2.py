"""External model module: LogLinearMamba2 (arch-046) / A4 hierarchical chunks.

A4 hierarchical client — shares the hierarchical dyadic level state with arch-009
Log-linear attention. Verified against fla/layers/log_linear_mamba2.py +
fla/ops/log_linear_attn/naive.py @ 864a87f6: the mixer is the SAME hierarchical
law as arch-009 (``o = (H ∘ qkᵀ)·v`` with the dyadic level-mask H), here driven
by the Mamba-2 frontend — the ``g`` log-decay comes from the Mamba-2
``-exp(A_log)·softplus(a_proj(x)+dt_bias)`` discretization and the per-level
scales from the level transform.

The hierarchical mixer (the banked dyadic state law) reuses
architectures/log_linear_attention.py — now routed through the public
``dyadic_banked_state`` op via :class:`BankedLogLinearMixer`. The Mamba-2 frontend
(A_log/dt_bias/a_proj, the level transform, conv short-circuit, norm/gate) is
external. The chunked *streaming* kernel (the pinned LogLinearAttentionState with
partial-chunk carry) is a native schedule — residual, not claimed.
"""

from __future__ import annotations

import torch

from architectures.log_linear_attention import BankedLogLinearMixer


class LogLinearMamba2Layer(torch.nn.Module):
    """LogLinearMamba2 mixer: Mamba-2 frontend (external) + the shared public banked law."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, num_levels: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_levels = num_levels
        # Mamba-2 frontend (external): the discretization producing g, and the level transform.
        self.A_log = torch.nn.Parameter(torch.zeros(num_heads))
        self.dt_bias = torch.nn.Parameter(torch.zeros(num_heads))
        self.a_proj = torch.nn.Linear(hidden_size, num_heads, bias=False)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.level_proj = torch.nn.Linear(hidden_size, num_heads * num_levels, bias=False)
        # The shared A4 banked mixer, executed through the public dyadic_banked_state op.
        self._mixer = BankedLogLinearMixer(num_heads, head_dim, num_levels,
                                           target=target, intent=intent)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, D, L = self.num_heads, self.head_dim, self.num_levels
        q = self.q_proj(hidden_states).view(B, T, H, D)
        k = self.k_proj(hidden_states).view(B, T, H, D)
        v = self.v_proj(hidden_states).view(B, T, H, D)
        # Mamba-2 discretization: g = -exp(A_log)·softplus(a_proj(x) + dt_bias), per head.
        g = -self.A_log.exp() * torch.nn.functional.softplus(
            self.a_proj(hidden_states) + self.dt_bias
        )  # [B,T,H]
        level_scales = self.level_proj(hidden_states).view(B, T, H, L)
        # Shared hierarchical law (arch-009), executed through the public banked op.
        return self._mixer(q, k, v, g, level_scales)


__all__ = ["LogLinearMamba2Layer"]
