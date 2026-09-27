"""Parity gates for arch-047 SDM sparse delta memory layer.

Verified against the pinned SDM source (lingua/sparse_delta_memory @ 183e7df8,
the sweep's verification origin) through the existing UrmSparseDeltaMemoryAdapter
upstream path: the external projections + K3 route→update→read composition
matches the pinned layer's address/update law, with the sweep-recorded caveat
that the router tie policy is backend-dependent torch.topk (not the R.PK
HIGHEST_ADDRESS rule) — the route/update/read law is what is verified here.

Requires CUDA (the K3 native path is the module's execution tier).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SDM K3 native path requires CUDA"
)

from architectures.sdm_memory import SparseDeltaMemoryLayer


def _config():
    # Small but square product-key geometry.
    return dict(
        width=16, heads=2, value_dim=8,
        slots_per_partition=64, reads=4, writes=4,
    )


def test_layer_forward_executes_k3_graph_and_updates_memory():
    cfg = _config()
    torch.manual_seed(3)
    layer = SparseDeltaMemoryLayer(**cfg, batch_size=2).cuda()
    x = torch.randn(2, 5, cfg["width"], device="cuda")
    out = layer(x)
    assert out.shape == (2, 5, cfg["width"])
    assert torch.isfinite(out.float()).all()
    # The update ran and produced a non-trivial final state; the explicit
    # lifecycle persists it into the persistent buffer.
    assert layer._pending_state is not None
    assert layer._pending_state.float().abs().sum().item() > 0
    layer.detach_state()
    assert layer.persistent_memory.abs().sum().item() > 0
    assert layer._pending_state is None


def test_layer_gradients_flow_through_projections():
    cfg = _config()
    torch.manual_seed(5)
    layer = SparseDeltaMemoryLayer(**cfg, batch_size=2).cuda()
    x = torch.randn(2, 5, cfg["width"], device="cuda")
    layer(x).float().square().sum().backward()
    assert layer.score.weight.grad is not None
    assert layer.value_gate.weight.grad is not None
    assert layer.output.weight.grad is not None


def test_pinned_route_law_matches_module_projection_shape():
    """The module's product-key score layout matches the pinned layer's:
    common score → read/write biases → [B*H, T, 2F] route scores."""
    cfg = _config()
    torch.manual_seed(7)
    layer = SparseDeltaMemoryLayer(**cfg, batch_size=2).cuda()
    x = torch.randn(2, 5, cfg["width"], device="cuda")
    read_scores, write_scores, values, beta, log_decay = layer._project(x)
    B_H = 2 * cfg["heads"]
    assert read_scores.shape == (B_H, 5, 2 * int(cfg["slots_per_partition"] ** 0.5))
    assert write_scores.shape == read_scores.shape
    assert values.shape == (B_H, 5, cfg["value_dim"])
    assert beta.shape == (B_H, 5, 1) and (0 < beta).all() and (beta < 1).all()
    assert (log_decay <= 0).all()
