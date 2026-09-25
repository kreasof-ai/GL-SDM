"""External model module: Log-linear attention (arch-009) / A4 hierarchical chunks.

A4 hierarchical client. Verified against fla/ops/log_linear_attn/naive.py @
864a87f6: the mixer is a hierarchical K2-family form — a hierarchy of dyadic
level states with per-level scales — evaluated in the full-matrix reference form

    H = construct_H_matrix(segsum(g), level_scales)   (dyadic level masks, Σ_level A ∘ mask_level)
    o = (H ∘ (q·kᵀ)) · v                                (elementwise-modulated contraction)

A = exp(segsum(g)) is the cumulative decay; mask_level selects dyadic sub-blocks
with per-level scales. H is NOT a softmax mask (it can be 0/negative) — this is a
linear-attention-like full-matrix form, not the canonical single-state K2 law
(sweep verdict). The hierarchical H construction (construct_H_matrix / segsum /
level masks) is the pinned external typed transform; the level_scales frontend
and the streaming/chunked partial-chunk state (LogLinearAttentionState with
q/k/v/g/level_scales carry) are residual, not claimed. Shared with arch-046
LogLinearMamba2 (which adds the Mamba-2 frontend).
"""

from __future__ import annotations

import math

import torch


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative segment sum: xs[i,j] = sum_{j<k<=i} x_k, masked lower-triangular."""
    T = x.size(-1)
    xc = torch.cumsum(x, dim=-1)
    xs = xc[..., :, None] - xc[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
    return xs.masked_fill(~mask, -torch.inf)


def construct_level_mask(level: int, L: torch.Tensor) -> torch.Tensor:
    """Dyadic level mask scaled by L[..., level, :]. L is [..., L_levels, T].

    Transcribed from the pinned fla/ops/log_linear_attn/naive.py.
    """
    T = L.size(-1)
    if level == 0:
        return torch.diag_embed(L[..., level, :])
    indices = torch.cartesian_prod(torch.arange(T), torch.arange(T)).to(L.device)
    mask = torch.where(
        torch.logical_and(
            torch.logical_and(
                indices[:, 0] % (1 << level) >= (1 << (level - 1)),
                indices[:, 1] + (1 << (level - 1)) >= indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
            ),
            indices[:, 1] < indices[:, 0] - (indices[:, 0] % (1 << (level - 1))),
        ),
        1.0, 0.0,
    ).view(T, T)
    # Pinned: scale the mask by L[..., level, i] (indexed by the row/query index i).
    scale = L[..., level, :].unsqueeze(-1).expand(*L.shape[:-2], T, T)
    return mask.to(L.dtype) * scale


def construct_H_matrix(a: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """H = Σ_level exp(segsum(a)) ∘ mask_level. a is [B,H,T]; L is [B,H,L_levels,T]."""
    T = a.size(-1)
    A = torch.exp(segsum(a))
    H = torch.zeros_like(A)
    for level in range(int(math.ceil(math.log2(T))) + 1):
        H = H + A * construct_level_mask(level, L)
    return H


class LogLinearAttentionLayer(torch.nn.Module):
    """Log-linear attention mixer (full-matrix reference form of the hierarchical law)."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, g, level_scales):
        """q/k/v [B,T,H,D]; g per-head log decay [B,T,H]; level_scales [B,T,H,L_levels]."""
        H = construct_H_matrix(g.permute(0, 2, 1), level_scales.permute(0, 2, 3, 1))  # [B,H,T,T]
        M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
        return torch.einsum("bhlc,bchp->blhp", M, v)


__all__ = ["LogLinearAttentionLayer", "construct_H_matrix", "segsum"]
