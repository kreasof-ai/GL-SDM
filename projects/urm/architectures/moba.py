"""External model module: MoBA (Mixture of Block Attention, arch-006).

A2 indexed-K1 client. Verified against the pinned moba source: each sequence is
split into chunks (the last chunk per sample excluded from the target pool for
causality); the gate score for a candidate block is ``q · mean_pool(k_block)``;
top-(topk-1) blocks are selected per (head, token), with the current block always
served by chunk-local self-attention. The selected-block tokens are gathered and
attended; the chunk-local self-attention is always on.

The block-gating route (mean-pool scores, top-k, causality masking) is external;
the mixer is the typed indexed-K1 call (architectures/indexed_attention.py). The
chunk-local self-attention merge and the full MoBA layer are residual.
"""

from __future__ import annotations

import torch

from architectures.indexed_attention import IndexedAttentionBase


class MoBALayer(IndexedAttentionBase):
    """One MoBA mixer: external top-k block route + typed indexed-K1 attend."""

    def __init__(self, num_heads: int, head_dim: int, chunk_size: int, topk: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__(num_heads, head_dim, target=target, intent=intent)
        self.chunk_size = chunk_size
        self.topk = topk

    def route(self, q, k):
        """External top-k block route: score blocks by q·mean_pool(k_block), causal.

        q [B,T,H,K], k [B,S,H,K] (S = T). Returns token-position gather indices
        [B,H,T,W] (selected block tokens + the query's own chunk-local block,
        -1 padding for causally-invisible/surplus slots).
        """
        B, T, H, K = q.shape
        cs = self.chunk_size
        n_blocks = (T + cs - 1) // cs
        device = q.device
        # mean-pool each block's keys: [B, n_blocks, H, K]
        pad = n_blocks * cs - T
        kp = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad)) if pad else k
        blocks = kp.view(B, n_blocks, cs, H, K).mean(dim=2)         # [B,nb,H,K]
        # gate score: q_t · mean_pool(k_block)  -> [B,T,H,nb]
        scores = torch.einsum("bthk,bnhk->bthn", q, blocks)
        # causality: block n visible to query t iff its end (n+1)*cs-1 <= t
        q_pos = torch.arange(T, device=device).view(1, T, 1, 1)
        block_end = (torch.arange(n_blocks, device=device) + 1) * cs - 1
        visible = block_end.view(1, 1, 1, -1) <= q_pos               # [1,T,1,nb]
        scores = scores.masked_fill(~visible, float("-inf"))
        # top-k blocks per (head, token); the current block is always included
        cur_block = (torch.arange(T, device=device) // cs)           # [T]
        topk_idx = scores.topk(min(self.topk, n_blocks), dim=-1).indices  # [B,T,H,topk]
        # expand blocks to token positions
        arange_cs = torch.arange(cs, device=device)
        tok = topk_idx.unsqueeze(-1) * cs + arange_cs                # [B,T,H,topk,cs]
        tok = tok.flatten(-2)                                        # [B,T,H,W]
        # mask: block genuinely selected (score finite), in-range, causally visible
        sel_score = torch.gather(scores, -1, topk_idx)               # [B,T,H,topk]
        valid_score = sel_score.unsqueeze(-1).expand(B, T, H, topk_idx.shape[-1], cs).flatten(-2).isfinite()
        in_range = tok < T
        causal = tok <= q_pos                                        # [1,T,1,1] broadcast over W
        tok = torch.where(valid_score & in_range & causal, tok, torch.full_like(tok, -1))
        return tok.permute(0, 2, 1, 3)                               # [B,H,T,W]

    def forward(self, q, k, v):
        gather = self.route(q, k)
        return self._attend(q, k, v, gather)


__all__ = ["MoBALayer"]
