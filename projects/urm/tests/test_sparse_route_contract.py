"""Dependency-light contract and compiler gates for native route selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.execution import (
    NATIVE_SPARSE_ROUTE_ANCHOR_NAME,
    TRUSTED_ANCHORS,
    AnchorRegistry,
    make_sparse_route_selector,
)
from urm.compiler.planner import ScheduleParams, UrmCompiler
from urm.compiler.semantic import (
    DType,
    SparseAddressCanonicalization,
    SparseRouteGeneration,
    SparseRouteSelectionSpec,
    SparseRouteTiePolicy,
    SparseScoreComposition,
    sparse_delta_memory_program,
    sparse_route_selection_program,
)


def _compiler(*, supported=True, code="supported", reason=None):
    anchor = next(
        item for item in TRUSTED_ANCHORS if item.name == NATIVE_SPARSE_ROUTE_ANCHOR_NAME
    )
    registry = AnchorRegistry()
    status = SimpleNamespace(supported=supported, code=code, reason=reason)
    registry.register(make_sparse_route_selector(anchor, lambda _spec: status))
    return UrmCompiler(anchors=registry)


def test_route_semantics_are_typed_and_schedule_independent() -> None:
    program = sparse_route_selection_program(
        parallel=2,
        sequence=7,
        source_extent=4096,
        route_width=7,
        dtype=DType.FLOAT32,
    )
    op = program.ops[0]
    assert isinstance(op, SparseRouteGeneration)
    assert op.spec.composition is SparseScoreComposition.PAIRWISE_ADDITIVE_FACTORS
    assert op.spec.canonicalization is SparseAddressCanonicalization.ASCENDING
    assert op.spec.tie_policy is SparseRouteTiePolicy.HIGHEST_ADDRESS
    assert op.spec.score_width == 128
    serialized = repr(program).lower()
    assert "facebook" not in serialized
    assert "triton" not in serialized
    assert "tile" not in serialized


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"source_extent": 65}, "square"),
        ({"route_width": 0}, "route width"),
        ({"route_width": 65, "source_extent": 64}, "route width"),
        ({"route_width": 9, "source_extent": 64}, "factor extent"),
        ({"dtype": DType.FLOAT16}, "float32 and bfloat16"),
    ],
)
def test_route_semantic_drift_fails_closed(kwargs, match) -> None:
    values = {
        "parallel": 1,
        "sequence": 1,
        "source_extent": 64,
        "route_width": 4,
        "dtype": DType.FLOAT32,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        SparseRouteSelectionSpec(**values)


def test_composite_product_key_width_must_not_exceed_factor_extent() -> None:
    with pytest.raises(ValueError, match="factor extent"):
        sparse_delta_memory_program(
            slots_per_partition=64,
            writes=9,
            reads=4,
        )


def test_compiler_selects_native_route_and_records_exact_schedule() -> None:
    result = _compiler().compile(
        sparse_route_selection_program(source_extent=4096, route_width=7)
    )
    assert result.trace.anchors == (NATIVE_SPARSE_ROUTE_ANCHOR_NAME,)
    launch = result.plan.steps[0].launch_config
    assert launch == {
        "schedule_family": "row_owned_factor_topk_canonical_softmax",
        "block_half": 64,
        "block_route": 8,
        "block_pair": 64,
        "num_warps": 4,
        "num_stages": 2,
    }


def test_route_decline_and_override_fail_closed() -> None:
    program = sparse_route_selection_program()
    with pytest.raises(CompilerError) as caught:
        _compiler(
            supported=False, code="unsupported_shape", reason="width exceeds v0"
        ).compile(program)
    assert caught.value.diagnostics[0].code is DiagnosticCode.ANCHOR_DECLINED
    assert caught.value.diagnostics[0].subject == "sparse_route_generation"

    params = ScheduleParams(
        anchor_overrides={"sparse_route_generation": "routed_reduction_v1"}
    )
    with pytest.raises(CompilerError) as caught:
        _compiler().compile(program, schedule_params=params)
    assert caught.value.diagnostics[0].code is DiagnosticCode.ANCHOR_DECLINED
