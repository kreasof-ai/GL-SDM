"""Registered verified rewrite rules and their contracts.

Every rule declares its source/replacement patterns, semantic and
shape/dtype/layout preconditions, locality requirements, effect-preservation
obligations, equivalence classification (exact vs floating point) with a
numerical envelope, forward and backward mappings (or an explicit forward-only
restriction), saved-state/recomputation requirements, communication-volume
change, and estimated compute/traffic/launch effects. The proof vocabulary lives
in :mod:`urm.compiler.rewrite.proof`; the deterministic engine that applies these
rules lives in :mod:`urm.compiler.rewrite.engine`.

Rules move computation only through these contracts; there is no path for
arbitrary tensor callbacks to enter the IR.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from urm.compiler.common.diagnostics import DiagnosticCode
from urm.compiler.rewrite.proof import (
    BackwardContract,
    BackwardStrategy,
    EquivalenceClass,
    ForwardOnlyRestriction,
    SavedStatePolicy,
)
from urm.ir.effects import BARRIERS, EffectClass
from urm.ir.program import (
    DType,
    EpilogueSpec,
    Matmul,
    SemanticNode,
    SemanticProgram,
    Transform,
    TransformKind,
    WeightedReduce,
)


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    ok: bool
    reason_code: DiagnosticCode | None = None
    message: str | None = None

    @classmethod
    def pass_(cls) -> CheckOutcome:
        return cls(ok=True)

    @classmethod
    def fail(cls, code: DiagnosticCode, message: str) -> CheckOutcome:
        return cls(ok=False, reason_code=code, message=message)


@dataclass(frozen=True, slots=True)
class RewriteMatch:
    """One pattern occurrence: the subject op and its producing neighbor."""

    subject: SemanticNode
    producer: SemanticNode | None
    consumed_tensor: str


@dataclass(frozen=True, slots=True)
class Precondition:
    """A named predicate over IR metadata (never over live tensors)."""

    name: str
    check: Callable[[SemanticProgram, RewriteMatch], CheckOutcome]


def _is_barrier_free_between(
    program: SemanticProgram, match: RewriteMatch
) -> CheckOutcome:
    """No barrier-effect op may sit between producer and subject."""
    if match.producer is None:
        return CheckOutcome.pass_()
    start = program.op_names.index(match.producer.name)
    stop = program.op_names.index(match.subject.name)
    for op in program.ops[start + 1 : stop]:
        crossed = set(op.effect.all_classes) & BARRIERS
        if crossed:
            names = ", ".join(sorted(c.value for c in crossed))
            return CheckOutcome.fail(
                DiagnosticCode.REWRITE_EFFECT_UNSAFE,
                f"{op.name} ({names}) sits between "
                f"{match.producer.name} and {match.subject.name}",
            )
    return CheckOutcome.pass_()


BARRIER_FREE = Precondition(
    "no_effect_barrier_between_producer_and_subject",
    _is_barrier_free_between,
)


def _single_consumer(program: SemanticProgram, match: RewriteMatch) -> CheckOutcome:
    consumers = program.consumers_of(match.consumed_tensor)
    if len(consumers) != 1:
        return CheckOutcome.fail(
            DiagnosticCode.REWRITE_PRECONDITION_FAILED,
            f"intermediate {match.consumed_tensor!r} has {len(consumers)} "
            "consumers; fusing would duplicate or drop work",
        )
    return CheckOutcome.pass_()


SINGLE_CONSUMER = Precondition("intermediate_has_single_consumer", _single_consumer)


def _subject_is_row_scale_transform(
    program: SemanticProgram, match: RewriteMatch
) -> CheckOutcome:
    del program
    if isinstance(match.producer, Transform) and match.producer.kind is (
        TransformKind.ROW_SCALE
    ):
        return CheckOutcome.pass_()
    kind = getattr(match.producer, "kind", type(match.producer).__name__)
    return CheckOutcome.fail(
        DiagnosticCode.REWRITE_PRECONDITION_FAILED,
        f"intervening op ({kind}) is not a row-wise scale; movement through a "
        "linear map changes semantics",
    )


SCALE_IS_ROWWISE_LINEAR = Precondition(
    "intervening_transform_is_rowwise_linear", _subject_is_row_scale_transform
)


@dataclass(frozen=True, slots=True)
class RewriteRule:
    """Full verified-rewrite contract (see module docstring)."""

    name: str
    description: str
    subject_kind: type[SemanticNode]
    producer_kind: type[SemanticNode] | None
    matcher: Callable[[SemanticProgram, RewriteMatch], bool]
    preconditions: tuple[Precondition, ...]
    equivalence: EquivalenceClass
    tolerance_envelope: dict[str, float] | None
    forward_mapping: Callable[[SemanticProgram, RewriteMatch], tuple[SemanticNode, ...]]
    backward_contract: BackwardContract | None
    backward_mapping: (
        Callable[[SemanticProgram, RewriteMatch], tuple[SemanticNode, ...]] | None
    ) = None
    forward_only_restriction: ForwardOnlyRestriction = (
        ForwardOnlyRestriction.NOT_FORWARD_ONLY
    )
    saved_state_policy: SavedStatePolicy = SavedStatePolicy.NONE
    preserved_effects: frozenset[EffectClass] = frozenset()
    locality_floor: str | None = None
    communication_volume_delta_bytes: int = 0
    traffic_bytes_delta: int = 0
    launch_count_delta: int = 0

    @property
    def forward_only(self) -> bool:
        return self.backward_mapping is None and self.backward_contract is None

    def backward_covers(self, dtype: DType) -> bool:
        return self.backward_contract is not None and self.backward_contract.covers(
            dtype
        )


def _match_row_scale_after_reduce(
    program: SemanticProgram, match: RewriteMatch
) -> bool:
    del program
    return (
        isinstance(match.subject, Transform)
        and match.subject.kind is TransformKind.ROW_SCALE
        and isinstance(match.producer, WeightedReduce)
        and match.producer.epilogue is None
    )


def _fold_row_scale_forward(
    program: SemanticProgram, match: RewriteMatch
) -> tuple[SemanticNode, ...]:
    reduce_op = match.producer
    assert isinstance(reduce_op, WeightedReduce)
    scale_tensor = match.subject.inputs[1]
    fused = WeightedReduce(
        name=reduce_op.name,
        inputs=reduce_op.inputs,
        outputs=match.subject.outputs,
        spec=reduce_op.spec,
        epilogue=EpilogueSpec(kind=TransformKind.ROW_SCALE, scale=scale_tensor),
        shape_hint=reduce_op.shape_hint,
    )
    return (fused,)


FOLD_ROW_SCALE_EPILOGUE = RewriteRule(
    name="fold_row_scale_into_routed_reduction_epilogue",
    description=(
        "base[q,d]=sum_k w[q,k]*V[idx,d]; out[q,d]=r[q]*base[q,d] becomes one "
        "routed reduction whose typed epilogue applies r before store; base is "
        "no longer an externally visible tensor."
    ),
    subject_kind=Transform,
    producer_kind=WeightedReduce,
    matcher=_match_row_scale_after_reduce,
    preconditions=(BARRIER_FREE, SINGLE_CONSUMER),
    equivalence=EquivalenceClass.FLOATING_POINT,
    tolerance_envelope={
        "float32_atol": 1e-5,
        "float16_atol": 1.5e-2,
        "bfloat16_atol": 2e-2,
    },
    forward_mapping=_fold_row_scale_forward,
    backward_contract=BackwardContract(
        strategy=BackwardStrategy.TILE_RECOMPUTE,
        verified_dtypes=(DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16),
        tolerance_envelope={"atol": 8e-2, "rtol": 4e-2},
        evidence=(
            "tests/test_compiler_epilogue_gpu.py::"
            "test_backward_covers_weights_values_and_row_scale"
        ),
    ),
    saved_state_policy=SavedStatePolicy.RECOMPUTE,
    communication_volume_delta_bytes=0,
    traffic_bytes_delta=-2,
    launch_count_delta=-1,
)


def _match_row_scale_before_matmul(
    program: SemanticProgram, match: RewriteMatch
) -> bool:
    del program
    return (
        isinstance(match.subject, Matmul)
        and isinstance(match.producer, Transform)
        and match.producer.kind is TransformKind.ROW_SCALE
        and len(match.subject.inputs) == 2
        and len(match.producer.inputs) == 2
    )


def _delay_row_scale_through_gemm(
    program: SemanticProgram, match: RewriteMatch
) -> tuple[SemanticNode, ...]:
    producer = match.producer
    assert isinstance(producer, Transform)
    matmul = match.subject
    x, r = producer.inputs
    w = matmul.inputs[1]
    unscaled_name = f"{matmul.outputs[0]}__unscaled"
    gemm = Matmul(
        name=matmul.name,
        inputs=(x, w),
        outputs=(unscaled_name,),
        transpose_rhs=matmul.transpose_rhs,
    )
    rescale = Transform(
        name=f"{producer.name}__delayed",
        inputs=(unscaled_name, r),
        outputs=matmul.outputs,
        kind=TransformKind.ROW_SCALE,
    )
    return (gemm, rescale)


DELAY_ROW_SCALE_THROUGH_GEMM = RewriteRule(
    name="delay_row_scale_through_linear_matmul",
    description=(
        "Linear(RowScale(x, r), W) <-> RowScale(Linear(x, W), r): move a "
        "per-row scale through an intervening linear map so it executes in "
        "the GEMM epilogue lifetime instead of materializing an intermediate."
    ),
    subject_kind=Matmul,
    producer_kind=Transform,
    matcher=_match_row_scale_before_matmul,
    preconditions=(
        BARRIER_FREE,
        SCALE_IS_ROWWISE_LINEAR,
    ),
    equivalence=EquivalenceClass.FLOATING_POINT,
    tolerance_envelope={
        "float32_atol": 1e-5,
        "float16_atol": 4e-2,
        "bfloat16_atol": 9e-2,
    },
    forward_mapping=_delay_row_scale_through_gemm,
    backward_contract=BackwardContract(
        strategy=BackwardStrategy.LINEARITY,
        verified_dtypes=(DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16),
        tolerance_envelope={
            "float32_atol": 1e-5,
            "float16_atol": 4e-2,
            "bfloat16_atol": 9e-2,
        },
        evidence="tests/test_compiler_delayed_scaling.py",
    ),
    backward_mapping=_delay_row_scale_through_gemm,
    saved_state_policy=SavedStatePolicy.NONE,
    communication_volume_delta_bytes=0,
    traffic_bytes_delta=-2,
    launch_count_delta=0,
)


DEFAULT_RULES: tuple[RewriteRule, ...] = (
    DELAY_ROW_SCALE_THROUGH_GEMM,
    FOLD_ROW_SCALE_EPILOGUE,
)


__all__ = [
    "BARRIER_FREE",
    "CheckOutcome",
    "DEFAULT_RULES",
    "DELAY_ROW_SCALE_THROUGH_GEMM",
    "FOLD_ROW_SCALE_EPILOGUE",
    "Precondition",
    "RewriteMatch",
    "RewriteRule",
    "SCALE_IS_ROWWISE_LINEAR",
    "SINGLE_CONSUMER",
]
