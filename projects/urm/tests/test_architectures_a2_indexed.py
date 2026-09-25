"""Parity gates for the A2 indexed-K1 clients: MoBA (006), DSA (007), Sparse
Transformer (072).

Each is an external route (top-k blocks / lightning indexer top-k / static
pattern) + the shared typed indexed-K1 gather-attend mixer. Verified against the
pinned equation transcriptions (MoBA block-gating, DSA top-k selection, Sparse
Transformer static patterns) — the pinned DSA/MoBA reference ops are fla Triton
kernels, so the oracle is the equation transcription of the gather-attend over
the externally-routed source set.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.dsa import DSALayer
from architectures.moba import MoBALayer
from architectures.sparse_transformer import SparseTransformerLayer

H, D, T = 2, 8, 8


def _manual_gather_attend(q, k, v, gather):
    """Oracle: gather the source set per query and softmax-attend over it."""
    B, T, H, D = q.shape
    qq, kk, vv = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    outs = torch.zeros(B, T, H, D)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                sel = gather[b, h, t]
                sel = sel[sel >= 0]
                if len(sel) == 0:
                    continue
                ks, vs = kk[b, h, sel], vv[b, h, sel]
                sc = (qq[b, h, t] @ ks.T) * D ** -0.5
                outs[b, t, h] = torch.softmax(sc, -1) @ vs
    return outs


# --- arch-006 MoBA ---

def test_moba_gather_attend_matches_oracle():
    torch.manual_seed(5)
    cs, topk = 2, 2
    q = torch.randn(1, T, H, D); k = torch.randn(1, T, H, D); v = torch.randn(1, T, H, D)
    layer = MoBALayer(H, D, cs, topk)
    with torch.no_grad():
        actual = layer(q, k, v)
        gather = layer.route(q, k)
    expected = _manual_gather_attend(q, k, v, gather)
    assert (actual - expected).abs().max().item() < 1e-5


def test_moba_route_respects_causality():
    """A block whose end is after the query is never selected."""
    torch.manual_seed(7)
    cs, topk = 2, 3
    q = torch.randn(1, T, H, D); k = torch.randn(1, T, H, D)
    layer = MoBALayer(H, D, cs, topk)
    gather = layer.route(q, k)  # [1,H,T,W]
    for t in range(T):
        sel = gather[0, :, t]
        sel = sel[sel >= 0]
        assert (sel <= t).all(), f"query {t} selected a future token"


# --- arch-007 DSA ---

def test_dsa_gather_attend_matches_oracle():
    torch.manual_seed(9)
    q = torch.randn(1, T, H, D); k = torch.randn(1, T, H, D); v = torch.randn(1, T, H, D)
    layer = DSALayer(H, D)
    # external indexer output: top-3 causal keys per query, -1 padded to width 3
    indices = torch.full((T, 3), -1, dtype=torch.long)
    for t in range(T):
        sel = list(range(max(0, t - 2), t + 1))
        indices[t, :len(sel)] = torch.tensor(sel)
    indices = indices.view(1, T, 1, 3).expand(1, T, H, 3).contiguous()
    with torch.no_grad():
        actual = layer(q, k, v, indices)
    gather = indices.permute(0, 2, 1, 3)
    expected = _manual_gather_attend(q, k, v, gather)
    assert (actual - expected).abs().max().item() < 1e-5


def test_dsa_padding_masked():
    torch.manual_seed(11)
    q = torch.randn(1, T, H, D); k = torch.randn(1, T, H, D); v = torch.randn(1, T, H, D)
    layer = DSALayer(H, D)
    indices = torch.zeros(1, T, H, 2, dtype=torch.long)
    indices[..., 1] = -1  # second slot padded
    with torch.no_grad():
        out = layer(q, k, v, indices)
    assert torch.isfinite(out).all()


# --- arch-072 Sparse Transformer ---

@pytest.mark.parametrize("pattern,kw", [("all", {}), ("local", {"local_ctx": 3}), ("strided", {"stride": 3})])
def test_sparse_transformer_patterns_match_oracle(pattern, kw):
    torch.manual_seed(5)
    q = torch.randn(1, T, H, D); k = torch.randn(1, T, H, D); v = torch.randn(1, T, H, D)
    layer = SparseTransformerLayer(H, D, pattern=pattern, **kw)
    gather = layer.build_gather_indices(T)
    with torch.no_grad():
        actual = layer(q, k, v, gather.expand(1, H, T, gather.shape[-1]))
    expected = _manual_gather_attend(q, k, v, gather.expand(1, H, T, gather.shape[-1]))
    assert (actual - expected).abs().max().item() < 1e-5


def test_sparse_transformer_strided_pattern():
    """The strided pattern selects j ≤ i with (i−j) mod stride == 0."""
    layer = SparseTransformerLayer(H, D, pattern="strided", stride=3)
    gather = layer.build_gather_indices(T)
    # query 6 (stride 3): keys j with (6-j)%3==0 → {6,3,0}
    sel = gather[0, 0, 6]
    sel = sorted(sel[sel >= 0].tolist())
    assert sel == [0, 3, 6], f"strided pattern wrong: {sel}"
