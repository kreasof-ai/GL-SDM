"""Parity gates for arch-054 AttnRes depth-domain residual aggregation.

Verified against the pinned fla naive_attnres (fla/ops/attnres/naive.py @
864a87f6, the sweep's verification origin): depth-domain softmax over the
residual sources, RMSNorm-ed keys, unnormalized values, optional fused output
RMSNorm, and gradients to the residual sources.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.attnres import AttnResLayer
from benchmarks.comparators.attnres import attnres_adapter

D, L = 16, 3


def _inputs(seed: int, T: int = 5):
    torch.manual_seed(seed)
    query = torch.randn(D)
    residuals = [torch.randn(T, D) for _ in range(L)]
    rms_weight = torch.randn(D)
    return query, residuals, rms_weight


def test_layer_matches_pinned_attnres():
    query, residuals, rms_weight = _inputs(seed=5)
    expected, _id = attnres_adapter(query, residuals, rms_weight)
    urm = AttnResLayer(L)
    with torch.no_grad():
        actual = urm(query, residuals, rms_weight)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"layer parity: max abs err {err}"


def test_output_rms_fusion_matches_pinned():
    query, residuals, rms_weight = _inputs(seed=7)
    out_rms = torch.randn(D)
    expected, _id = attnres_adapter(
        query, residuals, rms_weight, output_rms_weight=out_rms
    )
    urm = AttnResLayer(L)
    with torch.no_grad():
        actual = urm(query, residuals, rms_weight, output_rms_weight=out_rms)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"fused-output parity: max abs err {err}"


def test_non_unit_scale_matches_pinned():
    query, residuals, rms_weight = _inputs(seed=11)
    expected, _id = attnres_adapter(query, residuals, rms_weight, scale=0.5)
    urm = AttnResLayer(L)
    with torch.no_grad():
        actual = urm(query, residuals, rms_weight, scale=0.5)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"scaled parity: max abs err {err}"


def test_gradients_flow_to_residual_sources():
    query, residuals, rms_weight = _inputs(seed=17)
    residuals = [r.requires_grad_(True) for r in residuals]
    urm = AttnResLayer(L)
    urm(query, residuals, rms_weight).square().sum().backward()
    for i, r in enumerate(residuals):
        assert r.grad is not None and r.grad.abs().sum().item() > 0, f"residual {i} has no gradient"
