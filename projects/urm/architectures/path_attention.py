"""External model module: PaTH attention (arch-010) / UT Householder operand correction.

UT (typed causal triangular transform) client — shares the strict-causal
triangular transform sub-law with arch-013 DeltaFormer. Verified against
fla/ops/path_attn/naive.py @ 864a87f6: per chunk, a strictly-lower Householder
transform ``T_mat = I + inv(I + tril(w_β wᵀ, −1)) − I`` (forward substitution)
produces corrected operands — ``A_local = tril(qkᵀ) − tril(qwᵀ)@(T_mat@
tril(w_β kᵀ))``, ``q' = q − tril(qwᵀ)@(T_mat w_β)``, ``k' = k − (T_mat w_β kᵀ)ᵀ w``
— then cross-chunk scores with progressive q' correction, plus the FoX-style
cumulative-gate bias ``gc_i − gc_j``, softmax, @v.

The triangular transform is a data-dependent strictly-causal solve, NOT a
per-token projection (sweep verdict: 'no projection-only equivalence'). The
chunked Householder solve and cross-chunk progressive correction are the typed
UT transform here; the short-conv / l2-norm on w frontend and the FoX cumulative
gate are external. The chunk_size=full-sequence case reduces to a single-chunk
UT correction; the multi-chunk progressive path is residual.
"""

from __future__ import annotations

import math

import torch


def _householder_T(w_beta: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """T_mat = I + inv(I + tril(w_β wᵀ, −1)) − I via forward substitution (per chunk).

    w_beta/w [..., C, D]. Returns [..., C, C].
    """
    C = w.shape[-2]
    mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=w.device), diagonal=0)
    T_mat = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, C):
        T_mat[..., i, :i] = T_mat[..., i, :i] + (
            T_mat[..., i, :, None] * T_mat[..., :, :i]
        ).sum(-2)
    return T_mat + torch.eye(C, dtype=w.dtype, device=w.device)


class PaTHAttentionLayer(torch.nn.Module):
    """PaTH mixer: single-chunk Householder UT operand correction + cumulative-gate softmax."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, w, beta, g, scale):
        """Single-chunk (chunk_size >= T) path. q/k/v/w [B,T,H,D]; beta [B,T,H]; g [B,T,HQ]."""
        B, T, HQ, D = q.shape
        H = k.shape[2]
        G = HQ // H
        # expand shared KV/w/beta to the query heads (GQA)
        def exp(x, last=True):
            return x.unsqueeze(3).expand(B, T, H, G, x.shape[-1]).flatten(2, 3) if x.dim() == 4 else \
                x.unsqueeze(3).expand(B, T, H, G).flatten(2, 3)
        k, v, w = exp(k), exp(v), exp(w)
        beta = beta.unsqueeze(3).expand(B, T, H, G).flatten(2, 3)
        g_cumsum = g.cumsum(1)                                    # [B,T,HQ]

        # Householder UT correction over the whole sequence (single chunk).
        w_beta = w * beta.unsqueeze(-1)                           # [B,T,HQ,D]
        # work head-first [B,HQ,T,D]
        qf = q.permute(0, 2, 1, 3).float()
        kf = k.permute(0, 2, 1, 3).float()
        wf = w.permute(0, 2, 1, 3).float()
        wbf = w_beta.permute(0, 2, 1, 3).float()
        T_mat = _householder_T(wbf, wf)                           # [B,HQ,T,T]
        # Pinned mask: triu(diagonal=0) zeroed → keep the STRICT lower triangle.
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=0)
        Twbk = T_mat @ (wbf @ kf.transpose(-1, -2)).masked_fill(upper, 0)
        qw = (qf @ wf.transpose(-1, -2)).tril()
        Twb = T_mat @ wbf
        A_local = (qf @ kf.transpose(-1, -2)).tril() - qw @ Twbk
        A = A_local.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device)), float("-inf"))
        A = A + g_cumsum.permute(0, 2, 1).unsqueeze(-1) - g_cumsum.permute(0, 2, 1).unsqueeze(-2)
        o = (A * scale).softmax(-1) @ v.permute(0, 2, 1, 3).float()
        return o.permute(0, 2, 1, 3).to(q.dtype)                  # [B,T,HQ,D]


__all__ = ["PaTHAttentionLayer"]
