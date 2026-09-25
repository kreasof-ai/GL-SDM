"""External model module: MoM (Mixture of Memories, arch-051).

Verified combinator row (routed-expert, MoE-shaped): an external softmax top-k
router selects memories per token; each selected memory runs the gated delta
rule (U2.D) over its packed stream; an external scatter-add weighted merge
combines the per-memory outputs — verified against fla/layers/mom.py @ 864a87f6.

Per the sweep: the router (softmax over gate(x), top-k, renormalized weights),
the pack (gather each memory's routed tokens into a time-ordered causal
subsequence) and the merge (scatter-add weighted sum back to the original token
positions) are external ordinary operators; each selected memory's mixer is a
typed K2 U2.D call (chunk/fused gated delta, scalar-per-head gate). Each memory
runs over ONLY its routed tokens — an unrouted token must not advance that
memory's recurrent state (the pinned fla/layers/mom.py packs by (batch, memory)
into cu_seqlens varlen streams for the same reason). The per-memory state
gather/scatter across calls and cache shuffling are external and residual (not
claimed). This module implements the no-shared-mem branch.
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
        device = hidden_states.device
        # External router: softmax over gate(x), top-k memories, renormalized weights.
        logits = self.gate(hidden_states)                       # [B,T,E]
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(self.topk, dim=-1)      # [B,T,K]
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)      # renormalize to sum 1

        # Per-memory weight of (token t → memory e): sum of the top-k weights
        # whose selected memory is e. [B,T,E]; zero where e is not selected.
        w_e_full = torch.zeros(B, T, self.n_memories, device=device)
        w_e_full.scatter_add_(2, topk_idx, topk_w)              # scatter topk_w into memory slots

        # Each memory runs its U2.D mixer over ONLY the tokens routed to it, in
        # original time order (a causal subsequence). An unrouted token must not
        # advance the memory's recurrent state — otherwise the state a routed
        # token later reads is contaminated by tokens never assigned to it. The
        # pinned fla/layers/mom.py packs tokens by (batch, memory) into
        # cu_seqlens varlen streams for the same reason. Outputs scatter back to
        # the routed token positions and merge by the routing weight.
        out = torch.zeros(B, T, self.hidden_size, device=device)
        for e in range(self.n_memories):
            routed = w_e_full[:, :, e] > 0                      # [B,T] tokens routed to e
            if not routed.any():
                continue
            # Pack each batch element's routed tokens (time-ordered, since t ascends).
            lengths = routed.sum(dim=1)                          # [B]
            max_len = int(lengths.max().item())
            packed = hidden_states.new_zeros(B, max_len, self.hidden_size)
            backpos = torch.zeros(B, max_len, dtype=torch.long, device=device)
            for b in range(B):
                pos = routed[b].nonzero(as_tuple=False).squeeze(-1)  # time-ordered positions
                packed[b, : pos.numel()] = hidden_states[b, pos]
                backpos[b, : pos.numel()] = pos
            o_packed = self.memories[e](packed)                  # [B, max_len, hidden]
            # Scatter each packed output back to its original token position,
            # weighted by the routing weight for that (token, memory) pair.
            for b in range(B):
                n = int(lengths[b].item())
                if n == 0:
                    continue
                pos = backpos[b, :n]                             # original token positions
                w_b = w_e_full[b, pos, e].unsqueeze(-1)          # [n,1]
                out[b, pos] = out[b, pos] + o_packed[b, :n] * w_b
        return out


__all__ = ["MoMLayer"]
