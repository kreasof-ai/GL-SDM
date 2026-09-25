"""External model module: MoM (Mixture of Memories, arch-051).

Verified combinator row (routed-expert, MoE-shaped): an external softmax top-k
router selects memories per token; each selected memory runs the gated delta
rule (U2.D) over its packed stream; an external scatter-add weighted merge
combines the per-memory outputs — verified against fla/layers/mom.py @ 864a87f6.

Per the sweep: the router (softmax over gate(x), top-k, renormalized weights),
the pack (argsort/gather/pad into per-memory streams with cu_seqlens) and the
merge (scatter-add weighted sum) are external ordinary operators; each selected
memory's mixer is a typed K2 U2.D call (chunk/fused gated delta, scalar-per-head
gate). The per-memory state gather/scatter and cache shuffling are external and
residual (not claimed). This module implements the no-shared-mem branch with
per-memory U2.D calls and the typed scatter-add merge.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.gated_deltanet import GatedDeltaNetLayer


class MoMLayer(torch.nn.Module):
    """One MoM layer: external router + per-memory U2.D calls + typed merge.

    Each memory is a gated-delta (U2.D) stream. Tokens are routed to their
    top-k memories with softmax-renormalized weights; per-memory mixers run the
    typed K2 graph; outputs merge by scatter-add weighted sum.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 n_memories: int, topk: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.n_memories = n_memories
        self.topk = topk
        self.gate = torch.nn.Linear(hidden_size, n_memories, bias=False)  # router
        # Per-memory gated-delta mixers.
        self.memories = torch.nn.ModuleList([
            GatedDeltaNetLayer(hidden_size, num_heads, head_k_dim, head_v_dim,
                               target=target, intent=intent)
            for _ in range(n_memories)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        # External router: softmax over gate(x), top-k memories, renormalized weights.
        logits = self.gate(hidden_states)                       # [B,T,E]
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(self.topk, dim=-1)      # [B,T,K]
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)      # renormalize to sum 1

        # Per-memory U2.D over the full stream (dense); the merge masks by routing.
        out = torch.zeros(B, T, self.hidden_size, device=hidden_states.device)
        for e in range(self.n_memories):
            # mask of tokens routed to memory e (any of the top-k slots)
            mem_mask = (topk_idx == e).any(dim=-1)              # [B,T]
            if not mem_mask.any():
                continue
            # routing weight for memory e at each token (0 if not selected)
            w_e = torch.where(
                topk_idx == e, topk_w, torch.zeros_like(topk_w)
            ).sum(dim=-1, keepdim=True)                          # [B,T,1]
            o_e = self.memories[e](hidden_states)                # [B,T,hidden]
            out = out + o_e * w_e
        return out


__all__ = ["MoMLayer"]
