"""Verified reparameterization: registered rewrite rules and deterministic traces.

A rewrite rule is a *verified contract*, not a code snippet. Every rule
declares its source/replacement patterns, semantic and shape/dtype/layout
preconditions, locality requirements, effect-preservation obligations,
equivalence classification (exact vs floating point) with a numerical
envelope, forward and backward mappings (or an explicit forward-only
restriction), saved-state/recomputation requirements, communication-volume
change, and estimated compute/traffic/launch effects. The engine records a
deterministic trace: rules considered, accepted, rejected, why, plus the
semantic obligations the compiled plan must still honor.

Rules move computation only through these contracts; there is no path for
arbitrary tensor callbacks to enter the IR.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from urm.compiler.diagnostics import DiagnosticCode
from urm.compiler.effects import BARRIERS, EffectClass
from urm.compiler.semantic import (
    DType,
    EpilogueSpec,
    Matmul,
    SemanticNode,
    SemanticProgram,
    Transform,
    TransformKind,
    WeightedReduce,
)


class EquivalenceClass(StrEnum):
    """What kind of equivalence a verified rewrite promises.

    ``EXACT`` is reserved for rewrites whose supported execution model
    promises exact or bitwise-equivalent results on every supported dtype.
    Reassociations that change floating-point operation order are
    ``FLOATING_POINT`` and must carry a dtype-specific numerical envelope;
    they are algebraically justified over real arithmetic only.
    """

    EXACT = "exact"
    FLOATING_POINT = "floating_point"


class SavedStatePolicy(StrEnum):
    NONE = "none"
    SAVE_TENSORS = "save_tensors"
    RECOMPUTE = "recompute"


class ForwardOnlyRestriction(StrEnum):
    """Why a rule may be forward-only."""

    NOT_FORWARD_ONLY = "not_forward_only"
    BACKWARD_UNVERIFIED = "backward_unverified_this_prototype"


class BackwardStrategy(StrEnum):
    """How the verified backward is obtained."""

    LINEARITY = "linearity"  # adjoint follows from declared linearity
    TILE_RECOMPUTE = "tile_recompute"  # un-scaled tiles are recomputed
    MATERIALIZED_AUTOGRAD = "materialized_autograd"  # framework autograd


@dataclass(frozen=True, slots=True)
class BackwardContract:
    """A certified backward for a rewrite, per supported dtype.

    A rule with ``backward_contract=None`` is forward-only and is rejected by
    training compilations. Certification is evidence-linked: the dtypes listed
    here are exactly those exercised by committed differential tests.
    """

    strategy: BackwardStrategy
    verified_dtypes: tuple[DType, ...]
    tolerance_envelope: dict[str, float]  # per-dtype atol/rtol keys
    evidence: str  # pointer to the differential tests / benchmark artifact

    def covers(self, dtype: DType) -> bool:
        return dtype in self.verified_dtypes


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
    # ``None`` means forward-only; a certified contract enables training use.
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


# -- Rule 1: fold a row-scale into the routed-reduction epilogue --------------


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
    # Forward envelopes match the GPU differential gates in
    # tests/test_compiler_epilogue_gpu.py (FORWARD_TOL).
    tolerance_envelope={
        "float32_atol": 1e-5,
        "float16_atol": 1.5e-2,
        "bfloat16_atol": 2e-2,
    },
    forward_mapping=_fold_row_scale_forward,
    # Certified backward: the experimental anchor computes gradients for
    # weights, values AND the row scale by recomputing un-scaled reduction
    # tiles; differential gates pass on every supported dtype
    # (tests/test_compiler_epilogue_gpu.py::BACKWARD_TOL). No forward-only
    # restriction remains, so no contradictory obligation pair can be emitted.
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
    traffic_bytes_delta=-2,  # avoids one full read+write of [Q, D]
    launch_count_delta=-1,
)


# -- Rule 2: delayed row scaling through a linear map (CODA-style identity) ---


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
    # Algebraically exact over real arithmetic only: the rewrite changes the
    # floating-point operation order (scale-after-GEMM vs scale-before-GEMM),
    # so it is classified FLOATING_POINT and carries dtype-specific envelopes
    # validated by tests/test_compiler_delayed_scaling.py. `exact` is reserved
    # for execution models promising bitwise-equivalent results.
    equivalence=EquivalenceClass.FLOATING_POINT,
    tolerance_envelope={
        "float32_atol": 1e-5,
        "float16_atol": 4e-2,
        "bfloat16_atol": 9e-2,
    },
    forward_mapping=_delay_row_scale_through_gemm,
    # Self-inverse direction: swapping back is the same construction with the
    # ops transposed; gradients follow from linearity of the intervening map.
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


# -- Engine and trace ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleAttempt:
    rule: str
    subject_op: str
    outcome: str  # "considered" | "accepted" | "rejected"
    reason_code: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Obligation:
    """A semantic duty the executing plan must still honor."""

    kind: str  # "recompute_backward" | "forward_only" | ...
    subject_op: str
    detail: str


@dataclass(frozen=True, slots=True)
class RewriteTrace:
    attempts: tuple[RuleAttempt, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    anchors: tuple[str, ...] = ()  # filled by the planner
    estimated_costs: dict[str, int | float] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": [
                {
                    "rule": attempt.rule,
                    "subject_op": attempt.subject_op,
                    "outcome": attempt.outcome,
                    "reason_code": attempt.reason_code,
                    "detail": attempt.detail,
                }
                for attempt in self.attempts
            ],
            "obligations": [
                {
                    "kind": obligation.kind,
                    "subject_op": obligation.subject_op,
                    "detail": obligation.detail,
                }
                for obligation in self.obligations
            ],
            "chosen_anchors": list(self.anchors),
            "estimated_costs": self.estimated_costs,
        }


@dataclass(frozen=True, slots=True)
class RewriteResult:
    program: SemanticProgram
    trace: RewriteTrace
    changed: bool


class RewriteEngine:
    """Deterministic application of registered rules."""

    def __init__(self, rules: Sequence[RewriteRule] = DEFAULT_RULES) -> None:
        self._rules: tuple[RewriteRule, ...] = tuple(
            sorted(rules, key=lambda rule: rule.name)
        )

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self._rules)

    def candidates_for(
        self, program: SemanticProgram
    ) -> tuple[tuple[RewriteRule, RewriteMatch], ...]:
        found: list[tuple[RewriteRule, RewriteMatch]] = []
        for op in program.ops:
            for rule in self._rules:
                if not isinstance(op, rule.subject_kind):
                    continue
                match = self._build_match(program, op, rule)
                if match is None or not rule.matcher(program, match):
                    continue
                found.append((rule, match))
        return tuple(found)

    @staticmethod
    def candidate_id(rule: RewriteRule, match: RewriteMatch) -> str:
        """Stable candidate identifier for one rewrite occurrence."""
        return f"rewrite:{rule.name}@{match.subject.name}"

    def apply_candidate(
        self,
        program: SemanticProgram,
        rule: RewriteRule,
        match: RewriteMatch,
    ) -> RewriteResult:
        """Apply exactly one enumerated candidate; ``program`` is not mutated.

        Raises :class:`CompilerError` when the candidate's preconditions no
        longer hold; callers must re-enumerate against the current program.
        """
        from urm.compiler.diagnostics import CompilerError, Diagnostic

        verdict = self._evaluate(program, rule, match)
        if not verdict.ok:
            raise CompilerError(
                (
                    Diagnostic(
                        code=verdict.reason_code
                        or DiagnosticCode.REWRITE_PRECONDITION_FAILED,
                        message=f"{self.candidate_id(rule, match)}: {verdict.message}",
                        subject=match.subject.name,
                    ),
                )
            )
        replacement = rule.forward_mapping(program, match)
        ops = self._splice(list(program.ops), match, replacement)
        rewritten = program.replaced(tuple(ops))
        rewritten.validate()
        attempt = RuleAttempt(
            rule=rule.name, subject_op=match.subject.name, outcome="accepted"
        )
        return RewriteResult(
            program=rewritten,
            trace=RewriteTrace(
                attempts=(attempt,),
                obligations=self._obligations_for(rule, match),
            ),
            changed=True,
        )

    def apply(self, program: SemanticProgram) -> RewriteResult:
        attempts: list[RuleAttempt] = []
        obligations: list[Obligation] = []
        ops = list(program.ops)
        changed = False
        for op in tuple(program.ops):
            applicable = [
                rule for rule in self._rules if isinstance(op, rule.subject_kind)
            ]
            for rule in applicable:
                attempts.append(
                    RuleAttempt(
                        rule=rule.name, subject_op=op.name, outcome="considered"
                    )
                )
                match = self._build_match(program, op, rule)
                reject = (
                    self._evaluate(program, rule, match)
                    if match is not None
                    else CheckOutcome.fail(
                        DiagnosticCode.REWRITE_PRECONDITION_FAILED,
                        "pattern does not match",
                    )
                )
                if not reject.ok:
                    attempts[-1] = RuleAttempt(
                        rule=rule.name,
                        subject_op=op.name,
                        outcome="rejected",
                        reason_code=reject.reason_code.value
                        if reject.reason_code
                        else None,
                        detail=reject.message,
                    )
                    continue
                assert match is not None
                replacement = rule.forward_mapping(program, match)
                ops = self._splice(ops, match, replacement)
                changed = True
                attempts[-1] = RuleAttempt(
                    rule=rule.name, subject_op=op.name, outcome="accepted"
                )
                obligations.extend(self._obligations_for(rule, match))
                break  # one accepted rule per site per deterministic pass

        rewritten = program.replaced(tuple(ops)) if changed else program
        if changed:
            rewritten.validate()
        trace = RewriteTrace(attempts=tuple(attempts), obligations=tuple(obligations))
        return RewriteResult(program=rewritten, trace=trace, changed=changed)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _evaluate(
        program: SemanticProgram, rule: RewriteRule, match: RewriteMatch
    ) -> CheckOutcome:
        if not rule.matcher(program, match):
            return CheckOutcome.fail(
                DiagnosticCode.REWRITE_PRECONDITION_FAILED,
                "pattern does not match",
            )
        for precondition in rule.preconditions:
            verdict = precondition.check(program, match)
            if not verdict.ok:
                return verdict
        return CheckOutcome.pass_()

    @staticmethod
    def _build_match(
        program: SemanticProgram, op: SemanticNode, rule: RewriteRule
    ) -> RewriteMatch | None:
        # Identify the latest producing neighbor feeding the subject that has
        # the rule's expected producer kind.
        producer: SemanticNode | None = None
        consumed = ""
        best = -1
        for tensor in op.inputs:
            source = program.producer_of(tensor)
            if source is None:
                continue
            if rule.producer_kind is not None and not isinstance(
                source, rule.producer_kind
            ):
                continue
            index = program.op_names.index(source.name)
            if index > best:
                best, producer, consumed = index, source, tensor
        if rule.producer_kind is not None and producer is None:
            return None
        return RewriteMatch(subject=op, producer=producer, consumed_tensor=consumed)

    @staticmethod
    def _splice(
        ops: list[SemanticNode],
        match: RewriteMatch,
        replacement: tuple[SemanticNode, ...],
    ) -> list[SemanticNode]:
        start = (
            ops.index(match.producer)
            if match.producer is not None and match.producer in ops
            else ops.index(match.subject)
        )
        end = ops.index(match.subject) + 1
        return ops[:start] + list(replacement) + ops[end:]

    @staticmethod
    def _obligations_for(
        rule: RewriteRule, match: RewriteMatch
    ) -> tuple[Obligation, ...]:
        items: list[Obligation] = []
        if rule.saved_state_policy is SavedStatePolicy.RECOMPUTE:
            items.append(
                Obligation(
                    kind="recompute_backward",
                    subject_op=match.subject.name,
                    detail=(
                        f"{rule.name}: backward recomputes un-scaled reduction "
                        "tiles so the row-scale gradient exists; it is never "
                        "silently omitted"
                    ),
                )
            )
        if rule.forward_only:
            items.append(
                Obligation(
                    kind="forward_only",
                    subject_op=match.subject.name,
                    detail=f"{rule.name}: {rule.forward_only_restriction.value}",
                )
            )
        return tuple(items)
