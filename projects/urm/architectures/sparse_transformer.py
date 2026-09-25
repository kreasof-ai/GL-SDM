"""External model module: Sparse Transformer (arch-072).

A2 indexed-K1 client. Verified against the pinned sparse-attention source:
plain causal softmax attention with an exact static sparsity pattern restricting
the visible key set per query. The admitted patterns (external route → typed
indexed-K1 gather-attend):

- ``all``: full causal band (the dense lower triangle).
- ``local``: causal band of bandwidth ``local_attn_ctx``.
- ``strided``: keys j ≤ i with (i − j) mod stride == 0.

The pattern→index route is external; the mixer is the typed indexed-K1 call
(architectures/indexed_attention.py). The 'fixed' per-head block layout and the
blockwise strided-transposed implementation detail are residual.
"""

from __future__ import annotations

import torch

from architectures.indexed_attention import IndexedAttentionBase


class SparseTransformerLayer(IndexedAttentionBase):
    """One Sparse Transformer mixer: external static pattern route + indexed-K1."""

    def __init__(self, num_heads: int, head_dim: int, pattern: str = "strided",
                 stride: int = 4, local_ctx: int = 4, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__(num_heads, head_dim, target=target, intent=intent)
        if pattern not in ("all", "local", "strided"):
            raise ValueError(f"unsupported pattern {pattern!r}")
        self.pattern = pattern
        self.stride = stride
        self.local_ctx = local_ctx

    def build_gather_indices(self, T: int, device=None) -> torch.Tensor:
        """Per-query visible source set under the static pattern; [1,1,T,W] broadcast."""
        sets = []
        for i in range(T):
            if self.pattern == "all":
                s = list(range(i + 1))
            elif self.pattern == "local":
                s = list(range(max(0, i - self.local_ctx + 1), i + 1))
            else:  # strided
                s = [j for j in range(i + 1) if (i - j) % self.stride == 0]
                if i not in s:
                    s.append(i)  # always include the query position
                s = sorted(set(s))
            sets.append(s)
        W = max(len(s) for s in sets)
        idx = torch.full((T, W), -1, dtype=torch.long)
        for i, s in enumerate(sets):
            idx[i, :len(s)] = torch.tensor(s)
        return idx.view(1, 1, T, W).to(device)

    def forward(self, query, key, value, gather_indices):
        B, _, H = query.shape[0], query.shape[1], self.num_heads
        if gather_indices.shape[0] == 1 and B > 1:
            gather_indices = gather_indices.expand(B, -1, -1, -1)
        if gather_indices.shape[1] == 1 and H > 1:
            gather_indices = gather_indices.expand(-1, H, -1, -1)
        return self._attend(query, key, value, gather_indices)


__all__ = ["SparseTransformerLayer"]
