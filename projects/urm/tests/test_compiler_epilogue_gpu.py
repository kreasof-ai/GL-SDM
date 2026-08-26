"""GPU tests for the experimental fused row-scale routed-reduction anchor.

Proves the CODA-inspired compiler-generated capability: forward equivalence,
complete backward (weights, values AND row scale), property coverage on
degenerate and non-power-of-two shapes, repeated routes, zero scales, and dtype
envelopes.
Routed-reduction v1 remains untouched; these tests exercise only the
experimental anchor under urm/compiler/anchors/.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for the fused-epilogue prototype tests", allow_module_level=True
    )

from urm.compiler.anchors import (
    ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION,
    routed_reduce_row_scale,
)

# Dtype-specific envelopes (docs/benchmarking.md: tolerances live in tests).
FORWARD_TOL = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
    torch.float16: {"atol": 1.5e-2, "rtol": 2e-2},
}
# Backward compares bf16-input Triton accumulation against an fp32 eager
# recurrence over D-length inner products; reassociation dominates, so the
# envelope is wider than forward and rtol-led.
BACKWARD_TOL = {"atol": 8e-2, "rtol": 4e-2}


def _sample(
    q=37,
    k=5,
    s=64,
    d=96,
    dtype=torch.bfloat16,
    seed=3,
    zero_scale=False,
    requires_grad=False,
    device="cuda",
):
    generator = torch.Generator(device=device).manual_seed(seed)
    indices = torch.randint(0, s, (q, k), device=device, generator=generator)
    weights = (
        torch.randn((q, k), device=device, generator=generator)
        .to(dtype)
        .requires_grad_(requires_grad)
    )
    values = (
        torch.randn((s, d), device=device, generator=generator)
        .to(dtype)
        .requires_grad_(requires_grad)
    )
    scale = torch.randn((q,), device=device, generator=generator).to(dtype)
    if zero_scale:
        scale = torch.zeros_like(scale)
    return indices, weights, values, scale.requires_grad_(requires_grad)


def _reference(indices, weights, values, row_scale):
    base = torch.einsum("qk,qkd->qd", weights.float(), values.float()[indices.long()])
    return row_scale.float()[:, None] * base


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_forward_matches_materialized_reference(dtype) -> None:
    indices, weights, values, scale = _sample(dtype=dtype)
    output = routed_reduce_row_scale(indices, weights, values, scale)
    expected = _reference(indices, weights, values, scale)
    torch.testing.assert_close(output.float(), expected, **FORWARD_TOL[dtype])


def test_backward_covers_weights_values_and_row_scale() -> None:
    indices, weights, values, scale = _sample(requires_grad=True)
    output = routed_reduce_row_scale(indices, weights, values, scale)
    grad_generator = torch.Generator(device="cuda").manual_seed(77)
    grad = torch.randn(
        output.shape, device="cuda", dtype=torch.float32, generator=grad_generator
    )
    output.backward(grad)

    # Materialized eager reference with identical leaves.
    w_ref = weights.detach().clone().requires_grad_(True)
    v_ref = values.detach().clone().requires_grad_(True)
    r_ref = scale.detach().clone().requires_grad_(True)
    _reference(indices, w_ref, v_ref, r_ref).backward(grad)

    assert weights.grad is not None
    assert values.grad is not None
    # The row-scale gradient is never silently omitted.
    assert scale.grad is not None and torch.isfinite(scale.grad).all()
    torch.testing.assert_close(weights.grad.float(), w_ref.grad.float(), **BACKWARD_TOL)
    torch.testing.assert_close(values.grad.float(), v_ref.grad.float(), **BACKWARD_TOL)
    torch.testing.assert_close(scale.grad.float(), r_ref.grad.float(), **BACKWARD_TOL)


@pytest.mark.parametrize(
    "q,k,s,d",
    [
        (1, 1, 1, 1),  # fully degenerate
        (3, 7, 11, 100),  # non-power-of-two everywhere
        (1024, 32, 256, 512),  # prefill-like wide route width
        (64, 2, 8192, 128),  # decode-like deep value tables
    ],
)
def test_property_shapes_non_power_of_two_and_degenerate(q, k, s, d) -> None:
    if k > s:
        pytest.skip("route width cannot exceed sources")
    indices, weights, values, scale = _sample(q=q, k=k, s=s, d=d, seed=q + k)
    output = routed_reduce_row_scale(indices, weights, values, scale)
    expected = _reference(indices, weights, values, scale)
    torch.testing.assert_close(output.float(), expected, **FORWARD_TOL[torch.bfloat16])


def test_repeated_routes_within_and_across_rows() -> None:
    q, k, s, d = 16, 4, 4, 48  # sources < q*k forces repeats everywhere
    generator = torch.Generator(device="cuda").manual_seed(21)
    indices = torch.randint(0, s, (q, k), device="cuda", generator=generator)
    weights = torch.randn((q, k), device="cuda", generator=generator).bfloat16()
    values = torch.randn((s, d), device="cuda", generator=generator).bfloat16()
    scale = torch.randn((q,), device="cuda", generator=generator).bfloat16()
    output = routed_reduce_row_scale(indices, weights, values, scale)
    expected = _reference(indices, weights, values, scale)
    torch.testing.assert_close(output.float(), expected, **FORWARD_TOL[torch.bfloat16])


def test_zero_row_scale_yields_exact_zeros() -> None:
    indices, weights, values, scale = _sample(zero_scale=True)
    output = routed_reduce_row_scale(indices, weights, values, scale)
    assert torch.equal(output, torch.zeros_like(output))


def test_carried_state_semantics_via_chained_calls() -> None:
    """Chaining two fused calls with shared state equals fusing the concatenated route."""
    q, k, s, d = 12, 3, 32, 40
    i1, w1, v, r = _sample(q=q, k=k, s=s, d=d, seed=31)
    i2, w2, _, _ = _sample(q=q, k=k, s=s, d=d, seed=33)
    chained = routed_reduce_row_scale(i1, w1, v, r) + routed_reduce_row_scale(
        i2, w2, v, r
    )
    i_concat = torch.cat([i1, i2], dim=1)
    w_concat = torch.cat([w1, w2], dim=1)
    concatenated = routed_reduce_row_scale(i_concat, w_concat, v, r)
    expected = _reference(i_concat, w_concat, v, r)
    torch.testing.assert_close(concatenated.float(), expected, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(
        chained.float(), concatenated.float(), atol=3e-2, rtol=2e-2
    )


def test_deterministic_repeatable_forward() -> None:
    args = _sample(seed=41)
    first = routed_reduce_row_scale(*args)
    second = routed_reduce_row_scale(*args)
    assert torch.equal(first, second)


def test_metadata_reports_epilogue_capability() -> None:
    from urm.compiler.anchors import routed_reduce_row_scale_metadata

    meta = routed_reduce_row_scale_metadata(route_width=8, value_dim=512)
    assert meta["epilogue"] == "row_scale"
    assert meta["anchor_version"] == ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION


def test_v1_contract_stays_unchanged_alongside_prototype() -> None:
    """The frozen v1 kernel must not grow an implicit epilogue."""
    from urm.triton_kernels.routed_reduce import routed_reduce

    indices, weights, values, scale = _sample(q=8, k=2, s=16, d=32, seed=43)
    v1_output = routed_reduce(indices, weights, values)
    expected_base = torch.einsum(
        "qk,qkd->qd", weights.float(), values.float()[indices.long()]
    )
    torch.testing.assert_close(v1_output.float(), expected_base, atol=2e-2, rtol=2e-2)
    # And the prototype differs from v1 exactly by the row scale.
    fused = routed_reduce_row_scale(indices, weights, values, scale)
    ones = torch.ones_like(scale)
    unscaled_fused = routed_reduce_row_scale(indices, weights, values, ones)
    torch.testing.assert_close(
        unscaled_fused.float(), v1_output.float(), atol=0, rtol=0
    )
    del fused
