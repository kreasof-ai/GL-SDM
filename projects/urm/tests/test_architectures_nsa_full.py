"""Parity gate for the full three-branch NSA (arch-005 completed).

The remodeled NSAFullLayer (compressed + selected + sliding, gate-merged) against
the pinned fla branch oracles (fla/ops/nsa/naive.py @ 864a87f6): the compressed
branch vs ``naive_nsa_compression``, the selected branch vs ``naive_nsa_selection``
on shared block indices, the sliding branch vs a direct windowed softmax
reference, and the uniform-gate merge of the three.
"""

from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, "/tmp/urm-comparator-pins/fla")

from architectures.nsa import NSAFullLayer

B, T, H, D = 1, 24, 2, 16
BS, TOPK, WIN = 8, 2, 6

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="indexed-K1 plan runs on CUDA")


def _inputs(seed: int):
    torch.manual_seed(seed)
    return torch.randn(B, T, H * D, device="cuda")


def _fla_oracles(q, k, v, block_indices):
    from fla.ops.nsa.naive import naive_nsa_compression, naive_nsa_selection
    from fla.ops.utils.pooling import mean_pooling

    qh = q.view(B, T, H, D)
    kh = k.view(B, T, H, D)
    vh = v.view(B, T, H, D)
    k_cmp = mean_pooling(kh, BS)
    v_cmp = mean_pooling(vh, BS)
    scale = D ** -0.5
    o_cmp, _ = naive_nsa_compression(qh, k_cmp, v_cmp, BS, scale)
    o_slc = naive_nsa_selection(qh, kh, vh, block_indices, BS, scale)

    # Sliding window: direct causal windowed softmax reference.
    qw = qh.permute(0, 2, 1, 3).float()  # [B,H,T,D]
    kw = kh.permute(0, 2, 1, 3).float()
    vw = vh.permute(0, 2, 1, 3).float()
    attn = qw @ kw.transpose(-1, -2) * scale
    i = torch.arange(T, device=qw.device).view(T, 1)
    j = torch.arange(T, device=qw.device).view(1, T)
    allow = (j <= i) & (j > i - WIN)
    attn = attn.masked_fill(~allow, float("-inf"))
    o_swa = (torch.softmax(attn, dim=-1) @ vw).permute(0, 2, 1, 3)
    return o_cmp, o_slc, o_swa


def test_full_nsa_matches_pinned_branches():
    hidden = _inputs(seed=7)
    layer = NSAFullLayer(H * D, H, D, block_size=BS, topk=TOPK, window_size=WIN,
                         target="reference").eval().cuda()
    # Uniform gates (sigmoid(0) = 1/2 each) and no projection mixing, so the
    # output is exactly 0.5 * (o_cmp + o_slc + o_swa) per head.
    with torch.no_grad():
        for proj in (layer.q_proj, layer.k_proj, layer.v_proj):
            proj.weight.copy_(torch.eye(H * D))
        layer.o_proj.weight.copy_(torch.eye(H * D))
        layer.gate_proj.weight.zero_()

    n_blocks = (T + BS - 1) // BS
    t_idx = torch.arange(T, device="cuda")
    # Real NSA forces the current (and previous) block into the selection — the
    # all-(-1) route never occurs (fla's selection oracle NaNs there by design).
    current = t_idx // BS
    previous = torch.where(current > 0, current - 1, torch.full_like(current, -1))
    block_indices = torch.stack([current, previous], dim=-1)  # [T, 2]
    block_indices = block_indices.view(1, T, 1, TOPK).expand(B, T, H, TOPK).contiguous()

    with torch.no_grad():
        out = layer(hidden, block_indices=block_indices)

    o_cmp, o_slc, o_swa = _fla_oracles(hidden, hidden, hidden, block_indices)
    expected = 0.5 * (o_cmp + o_slc + o_swa)
    err = (out.view(B, T, H, D) - expected).abs().max().item()
    assert err < 2e-4, f"full-NSA vs pinned branches: max abs err {err:.2e}"


def test_full_nsa_training_step_finite():
    torch.manual_seed(0)
    layer = NSAFullLayer(H * D, H, D, block_size=BS, topk=TOPK, window_size=WIN,
                         target="reference").train().cuda()
    out = layer(_inputs(seed=3))
    loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
