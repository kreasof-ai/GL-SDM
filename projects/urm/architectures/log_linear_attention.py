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

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


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


def dyadic_level_of(t: int, j: int, num_levels: int) -> int:
    """The disjoint dyadic level owning the causal pair (t, j).

    The pinned level masks partition the causal triangle so each (t, j≤t) pair belongs
    to exactly ONE level: j == t is level 0 (the diagonal); otherwise the LARGEST l
    whose aligned block ``[(t>>(l-1))<<(l-1) − 2^{l-1}, (t>>(l-1))<<(l-1))`` contains j.
    Verified to reconstruct ``construct_H_matrix`` exactly (0.0).
    """
    if j == t:
        return 0
    best = 0
    for l in range(1, num_levels):
        half = 1 << (l - 1)
        base = (t >> (l - 1)) << (l - 1)
        if base - half <= j < base:
            best = l
    return best


def log_linear_disjoint_reference(q, k, v, g, level_scales):
    """The disjoint dyadic-block reference form of the hierarchical law (exact oracle).

    Equivalent to ``construct_H_matrix`` + the contraction, but computed as a sum over
    the disjoint dyadic blocks — each key j contributes to query t at exactly one level
    with decay ``exp(gcum[t]−gcum[j])`` (gcum the per-head prefix cumsum of g) and the
    per-level scale ``level_scales[t, level]``. Verified against the pinned
    ``naive_log_linear_attn`` and the full-matrix form at 0.0. This is the honest
    reference oracle; the banked recurrent form (the pinned chunked kernel's
    ``LogLinearAttentionState`` with the carry-cascade promote/reset and within-chunk
    decay) is the residual schedule, not yet derived to parity.
    """
    B, T, H, D = q.shape
    num_levels = level_scales.shape[-1]
    gcum = g.permute(0, 2, 1).cumsum(-1)                      # [B,H,T]
    out = torch.zeros(B, T, H, D, dtype=torch.float32)
    for t in range(T):
        for j in range(t + 1):
            l = dyadic_level_of(t, j, num_levels)
            aij = torch.exp(gcum[..., t] - gcum[..., j])       # [B,H]
            score = (q[:, t].float() * k[:, j].float()).sum(-1)  # [B,H]
            out[:, t] += (level_scales[:, t, :, l].float() * aij * score).unsqueeze(-1) * v[:, j].float()
    return out.to(q.dtype)


class LogLinearAttentionLayer(torch.nn.Module):
    """Log-linear attention mixer (full-matrix reference form of the hierarchical law).

    Retained as the independent full-matrix comparator; the public banked path is
    :class:`BankedLogLinearMixer`.
    """

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, q, k, v, g, level_scales):
        """q/k/v [B,T,H,D]; g per-head log decay [B,T,H]; level_scales [B,T,H,L_levels]."""
        H = construct_H_matrix(g.permute(0, 2, 1), level_scales.permute(0, 2, 3, 1))  # [B,H,T,T]
        M = torch.einsum("bhlc,blhn,bchn->bhlc", H, q, k)
        return torch.einsum("bhlc,bchp->blhp", M, v)


class BankedLogLinearMixer(torch.nn.Module):
    """The shared A4 hierarchical mixer routed through the public ``dyadic_banked_state`` op.

    Both arch-009 (Log-linear attention) and arch-046 (LogLinearMamba2) use this:
    the banked dyadic state law is identical; only the frontend differs (arch-046
    adds the Mamba-2 discretization). ``log_linear_disjoint_reference`` and the
    full-matrix :class:`LogLinearAttentionLayer` are retained as comparators.
    """

    def __init__(self, num_heads: int, head_dim: int, num_levels: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_levels = num_levels
        H, D, L = num_heads, head_dim, num_levels
        document = {
            "schema_version": 2, "name": "log_linear_banked", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["B", "T", H]},
                    {"name": "level_scales", "dtype": "float32", "shape": ["B", "T", H, L]},
                ],
                "nodes": [
                    {
                        "id": "bank", "op": "dyadic_banked_state",
                        "inputs": ["query", "key", "value", "log_decay", "level_scales"],
                        "outputs": ["output"],
                        "params": {
                            "num_levels": L,
                            "roles": {
                                "query": "query", "key": "key", "value": "value",
                                "log_decay": "log_decay", "level_scales": "level_scales",
                            },
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        program = normalize_graph_document(load_graph_recipe_document(document).document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, q, k, v, g, level_scales):
        """q/k/v [B,T,H,D]; g per-head log decay [B,T,H]; level_scales [B,T,H,L_levels]."""
        return self._plan.execute(
            query=q.float(), key=k.float(), value=v.float(),
            log_decay=g.float(), level_scales=level_scales.float(),
        )["output"]


__all__ = [
    "LogLinearAttentionLayer",
    "BankedLogLinearMixer",
    "construct_H_matrix",
    "segsum",
    "dyadic_level_of",
    "log_linear_disjoint_reference",
]
