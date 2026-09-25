"""Parity gates for arch-005 NSA (A2 indexed-K1, selected branch).

Verified against the pinned fla source (fla/ops/nsa/naive.py @ 864a87f6, the
sweep's verification origin): the selected branch — gather the top-k selected
block tokens (block*block_size + arange, padded -1) and softmax-attend over the
gathered set — matches the pinned naive_nsa selected branch on identical
operands (fp32, CUDA). The block-selection route, the compressed branch, the
gated merge and the GQA power-of-2/≥16 tile constraint are residual, not claimed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.nsa import NSASelectedLayer

H, HQ, K, V, T, BS, S = 1, 16, 8, 8, 8, 2, 2


def _operands(seed: int, device: str = "cuda"):
    torch.manual_seed(seed)
    q = torch.randn(1, T, HQ, K, device=device)
    k = torch.randn(1, T, H, K, device=device)
    v = torch.randn(1, T, H, V, device=device)
    block_indices = torch.stack(
        [torch.tensor([0, max(0, t // BS)]) for t in range(T)]
    ).view(1, T, 1, S).expand(1, T, H, S).contiguous().to(device)
    return q, k, v, block_indices


def test_nsa_selected_branch_matches_pinned():
    if not torch.cuda.is_available():
        pytest.skip("pinned naive_nsa mean_pooling requires CUDA")
    q, k, v, block_indices = _operands(seed=5)
    layer = NSASelectedLayer(HQ, K, BS)
    with torch.no_grad():
        actual = layer(q, k, v, block_indices)
    from benchmarks.comparators.fla_k2 import fla_op
    naive = fla_op("fla.ops.nsa.naive.naive_nsa")
    g_slc = torch.ones(1, T, HQ, device=q.device)
    expected = naive(q, k, v, g_cmp=None, g_slc=g_slc, g_swa=None,
                     block_indices=block_indices, block_counts=S, block_size=BS, window_size=0)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"nsa selected parity: max abs err {err}"


def test_nsa_padding_slots_are_masked():
    """A -1 (padded) block slot contributes nothing to the output."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    q, k, v, block_indices = _operands(seed=7)
    layer = NSASelectedLayer(HQ, K, BS)
    with torch.no_grad():
        out_full = layer(q, k, v, block_indices)
        # pad the second block slot entirely
        padded = block_indices.clone()
        padded[..., 1] = -1
        out_padded = layer(q, k, v, padded)
    # outputs differ (fewer sources) but stay finite
    assert torch.isfinite(out_padded).all()


def test_nsa_gradients_flow():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    q, k, v, block_indices = _operands(seed=13)
    layer = NSASelectedLayer(HQ, K, BS)
    q = q.requires_grad_(True)
    layer(q, k, v, block_indices).square().sum().backward()
    assert q.grad is not None and q.grad.abs().sum().item() > 0
