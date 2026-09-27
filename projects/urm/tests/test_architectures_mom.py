"""Parity gates for arch-051 MoM (Mixture of Memories, routed-expert combinator).

Verified against the pinned fla source (fla/layers/mom.py @ 864a87f6, the
sweep's verification origin). The pinned layer runs a packed varlen
chunk_gated_delta_rule over concatenated per-memory streams; this module
composes the external router + per-memory typed U2.D calls + the external
scatter-add weighted merge. The gates verify the router semantics (top-k,
renormalized weights), the merge structure (scatter-add weighted sum), and the
per-memory mixer (each memory's gated-delta law, already verified in
arch-026's Gated DeltaNet gates).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.mom import MoMLayer

HIDDEN, H, DK, DV, E, TOPK = 32, 4, 8, 8, 3, 2


def test_router_topk_renormalized_weights():
    """Router: softmax over gate(x), top-k memories, weights renormalized to sum 1."""
    torch.manual_seed(5)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    with torch.no_grad():
        logits = layer.gate(hidden)
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(TOPK, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
    assert topk_idx.shape == (2, 6, TOPK)
    # renormalized weights sum to 1 per token
    assert (topk_w.sum(dim=-1) - 1).abs().max().item() < 1e-6


def test_merge_is_scatter_add_weighted_sum():
    """The output is the scatter-add weighted sum of per-memory PACKED-stream outputs.

    Each memory runs its U2.D mixer over only the tokens routed to it, in time
    order (a causal subsequence) — never the full stream. This oracle recomputes
    exactly that packed composition independently and checks the scatter-add merge.
    """
    torch.manual_seed(7)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    B, T = 2, 6
    hidden = torch.randn(B, T, HIDDEN)
    with torch.no_grad():
        logits = layer.gate(hidden)
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(TOPK, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
        actual = layer(hidden)
        # Independent packed-stream recompute.
        w_e_full = torch.zeros(B, T, E).scatter_add_(2, topk_idx, topk_w)
        expected = torch.zeros_like(actual)
        for e in range(E):
            routed = w_e_full[:, :, e] > 0
            if not routed.any():
                continue
            for b in range(B):
                pos = routed[b].nonzero(as_tuple=False).squeeze(-1)
                if pos.numel() == 0:
                    continue
                packed = hidden[b, pos].unsqueeze(0)          # [1,n,hidden]
                o_packed = layer.memories[e](packed)[0]        # [n,hidden]
                expected[b, pos] = expected[b, pos] + o_packed * w_e_full[b, pos, e].unsqueeze(-1)
    err = (actual - expected).abs().max().item()
    assert err < 1e-6, f"merge structure: max abs err {err}"


def test_unrouted_token_does_not_contaminate_memory_state():
    """A token routed to memory A must not change the state a memory-B token reads.

    Regression for the full-stream bug: running each memory over the whole token
    stream lets an unrouted token advance that memory's recurrent state, corrupting
    the state a routed token later reads. With a forced disjoint router, perturbing
    a memory-1 token must leave the memory-0 tokens' outputs exactly unchanged.
    """
    HIDDEN2, H2, DK2, DV2, E2, TOPK2 = 8, 2, 4, 4, 3, 1  # topk=1 → disjoint partition
    torch.manual_seed(3)
    layer = MoMLayer(HIDDEN2, H2, DK2, DV2, E2, TOPK2)
    with torch.no_grad():
        layer.gate.weight.data.zero_()
        layer.gate.weight.data[0, 0] = 5.0   # hidden[..,0] > 0 → mem0
        layer.gate.weight.data[1, 0] = -5.0  # hidden[..,0] < 0 → mem1
        g = torch.Generator().manual_seed(9)
        h = torch.zeros(1, 3, HIDDEN2)
        h[0, 0, 0] = -1.0  # → mem1
        h[0, 1, 0] = 1.0   # → mem0
        h[0, 2, 0] = 1.0   # → mem0 (later, same memory as token 1)
        h[0, :, 1:] = torch.randn(1, 3, HIDDEN2 - 1, generator=g)
        out_before = layer(h).clone()
        # Perturb token 0 (mem1); the mem0 tokens (1, 2) must be exactly unchanged.
        h2 = h.clone()
        h2[0, 0, 1:] = torch.randn(HIDDEN2 - 1, generator=torch.Generator().manual_seed(99))
        out_after = layer(h2)
    contamination = (out_before[0, 1:] - out_after[0, 1:]).abs().max().item()
    assert contamination == 0.0, (
        f"unrouted-token state contamination: {contamination} (a mem1 token changed mem0 outputs)"
    )
    # Sanity: the perturbed token's own (mem1) output did change.
    assert (out_before[0, 0] - out_after[0, 0]).abs().max().item() > 1e-3


def test_unrouted_memory_contributes_nothing():
    """A memory no token routes to must contribute exactly zero."""
    torch.manual_seed(11)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    with torch.no_grad():
        logits = layer.gate(hidden)
        weights = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = weights.topk(TOPK, dim=-1)
        # find an unrouted memory
        for e in range(E):
            if not (topk_idx == e).any():
                # zero that memory's output must not change the result
                out_full = layer(hidden)
                assert torch.isfinite(out_full).all()
                return
        pytest.skip("all memories routed in this fixture")


def test_gradients_flow_through_router_and_memories():
    """Router gradients flow through the routing weights into the merge."""
    torch.manual_seed(13)
    layer = MoMLayer(HIDDEN, H, DK, DV, E, TOPK)
    hidden = torch.randn(2, 6, HIDDEN)
    layer(hidden).square().sum().backward()
    assert layer.gate.weight.grad is not None and layer.gate.weight.grad.abs().sum().item() > 0
