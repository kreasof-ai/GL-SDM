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
        self._dynamic_routing = True
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
        w_e_full = topk_w.new_zeros(B, T, self.n_memories)
        w_e_full.scatter_add_(2, topk_idx, topk_w)              # scatter topk_w into memory slots

        # Each memory runs its U2.D mixer over ONLY the tokens routed to it, in
        # original time order (a causal subsequence). An unrouted token must not
        # advance the memory's recurrent state — otherwise the state a routed
        # token later reads is contaminated by tokens never assigned to it. The
        # pinned fla/layers/mom.py packs tokens by (batch, memory) into
        # cu_seqlens varlen streams for the same reason. Outputs scatter back to
        # the routed token positions and merge by the routing weight.
        out = torch.zeros(B, T, self.hidden_size, device=device)
        routed_all = w_e_full > 0
        # One small host transfer per layer chooses a common padded length.
        # Padding follows every real causal stream and cannot affect its outputs.
        # Keep empty experts skipped so their optimizer state remains untouched.
        max_lengths = routed_all.sum(dim=1).amax(dim=0).tolist()
        max_len = max(max_lengths)
        positions = torch.arange(T, device=device).expand(B, T)
        for e, length in enumerate(max_lengths):
            if not length:
                continue
            routed = routed_all[:, :, e]
            # One batched pack/merge instead of a nonzero and .item() sync for
            # every sequence. Sentinel positions sort after the causal stream.
            backpos = positions.masked_fill(~routed, T).sort(dim=1).values[:, :max_len]
            valid = backpos < T
            backpos = backpos.clamp_max(T - 1)
            gather = backpos[..., None].expand(B, max_len, self.hidden_size)
            packed = hidden_states.gather(1, gather) * valid[..., None]
            o_packed = self.memories[e](packed)                  # [B, max_len, hidden]
            # Scatter each packed output back to its original token position,
            # weighted by the routing weight for that (token, memory) pair.
            packed_weights = w_e_full[:, :, e].gather(1, backpos) * valid
            out = out.scatter_add(1, gather, (o_packed * packed_weights[..., None]).to(out.dtype))
        return out


__all__ = ["MoMLayer"]
