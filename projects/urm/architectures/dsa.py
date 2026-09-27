"""External model module: DSA (DeepSeek Sparse Attention, arch-007).

A2 indexed-K1 client. Verified against fla/ops/dsa/naive.py @ 864a87f6. The
lightning indexer scores every causal key ``I[t,s] = Σ_j w_idx[t,j]·act(scale·
q_idx[t,j]·k_idx[s])`` (MQA: a single index key shared across heads), keeps the
top-k causal keys per query (invisible slots padded -1, selection shared across
all query heads), then the main attention is softmax(scale·q·k)·v over the
gathered selected tokens only.

The indexer route (q_idx/k_idx/w_idx projections, activation, top-k) is external;
the mixer is the typed indexed-K1 call (architectures/indexed_attention.py). The
indexer projections and GQA grouping are external.
"""

from __future__ import annotations

import torch

from architectures.indexed_attention import IndexedAttentionBase


class DSALayer(IndexedAttentionBase):
    """One DSA mixer: external lightning-indexer top-k route + typed indexed-K1."""

    def __init__(self, num_heads: int, head_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__(num_heads, head_dim, target=target, intent=intent)

    def forward(self, q, k, v, indices):
        """``indices`` [B,T,H,S] selected source positions (the external indexer output),
        -1 padded; shared across query heads. The mixer attends over the gathered set.
        """
        # The indexed path gathers per KV head; DSA shares the selection across heads.
        gather = indices.permute(0, 2, 1, 3)  # [B,H,T,W]
        return self._attend(q, k, v, gather)


__all__ = ["DSALayer"]
