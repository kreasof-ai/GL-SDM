"""GPU differential gates for URM-native sparse route selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.adapters.sparse_delta_memory import (
    MODE_INFERENCE,
    UrmSparseDeltaMemoryAdapter,
    probe_sdm_support,
)
from urm.adapters.sparse_delta_memory_reference import torch_product_key
from urm.backends.sparse_route import (
    CertifiedSparseRouteScores,
    TritonSparseRouteBackend,
)
from urm.compiler.semantic import DType, SparseRouteSelectionSpec

TOLERANCES = {
    torch.float32: {"atol": 2e-5, "rtol": 2e-5},
    torch.bfloat16: {"atol": 4e-3, "rtol": 4e-3},
}


def _input(dtype, slots, width, *, sequence=3, requires_grad=False):
    half = round(slots**0.5)
    generator = torch.Generator(device="cuda").manual_seed(709 + width + slots)
    spec = SparseRouteSelectionSpec(
        1,
        sequence,
        slots,
        width,
        DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16,
    )
    row_spec = SparseRouteSelectionSpec(1, 1, slots, width, spec.dtype)
    rows = []
    attempts = 0
    while len(rows) < sequence and attempts < 10_000:
        candidate = torch.randn(
            (1, 1, 2 * half),
            device="cuda",
            dtype=dtype,
            generator=generator,
        ).contiguous()
        attempts += 1
        try:
            CertifiedSparseRouteScores.certify(row_spec, candidate)
        except ValueError:
            continue
        rows.append(candidate)
    if len(rows) != sequence:
        raise AssertionError(
            "could not generate deterministic boundary-tie-free scores"
        )
    scores = torch.cat(rows, dim=1).contiguous().requires_grad_(requires_grad)
    return spec, scores


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("slots,width", [(64, 1), (64, 7), (4096, 64), (65536, 4)])
def test_native_route_matches_transparent_reference(dtype, slots, width) -> None:
    spec, scores = _input(dtype, slots, width)
    output = TritonSparseRouteBackend(spec).generate_certified(
        CertifiedSparseRouteScores.certify(spec, scores)
    )
    values, addresses = torch_product_key(scores, width, spec.factor_extent)
    weights = torch.softmax(values, dim=-1)
    assert torch.equal(output.addresses.to(torch.int64), addresses)
    torch.testing.assert_close(
        output.weights.float(), weights.float(), **TOLERANCES[dtype]
    )
    assert bool((output.addresses[..., 1:] > output.addresses[..., :-1]).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("slots,width", [(64, 7), (4096, 64)])
def test_native_route_score_gradients_match_reference(dtype, slots, width) -> None:
    spec, base = _input(dtype, slots, width)
    native_scores = base.detach().clone().requires_grad_(True)
    reference_scores = base.detach().clone().requires_grad_(True)
    native = TritonSparseRouteBackend(spec).generate_certified(
        CertifiedSparseRouteScores.certify(spec, native_scores)
    )
    values, addresses = torch_product_key(reference_scores, width, spec.factor_extent)
    reference_weights = torch.softmax(values, dim=-1)
    cotangent = torch.randn_like(reference_weights)
    (native.weights.float() * cotangent.float()).sum().backward()
    (reference_weights.float() * cotangent.float()).sum().backward()
    assert torch.equal(native.addresses.to(torch.int64), addresses)
    assert torch.isfinite(native_scores.grad).all()
    torch.testing.assert_close(
        native_scores.grad.float(),
        reference_scores.grad.float(),
        **TOLERANCES[dtype],
    )


@pytest.mark.skipif(
    not probe_sdm_support().supported, reason="pinned upstream SDM is unavailable"
)
def test_native_route_addresses_match_exact_pinned_upstream_callable() -> None:
    spec, scores = _input(torch.float32, 256, 4, sequence=5)
    adapter = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=256,
        value_dim=37,
        num_writes=4,
        num_reads=4,
        chunk_size=16,
        mode=MODE_INFERENCE,
        device="cuda",
        dtype=torch.float32,
    )
    upstream_values, upstream_addresses = adapter.direct_calls["address"](scores, 4, 16)
    native = TritonSparseRouteBackend(spec).generate_certified(
        CertifiedSparseRouteScores.certify(spec, scores)
    )
    assert torch.equal(native.addresses.to(torch.int64), upstream_addresses)
    torch.testing.assert_close(
        native.weights,
        torch.softmax(upstream_values, dim=-1),
        atol=2e-5,
        rtol=2e-5,
    )
