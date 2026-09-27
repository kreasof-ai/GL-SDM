"""Deterministic rewrite engine: application of registered rules with traces.

The engine records a deterministic trace: rules considered, accepted, rejected,
why, plus the semantic obligations the compiled plan must still honor. The proof
vocabulary lives in :mod:`urm.compiler.rewrite.proof`; the registered rule
contracts live in :mod:`urm.compiler.rewrite.rules`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from urm.compiler.common.diagnostics import DiagnosticCode
from urm.compiler.rewrite.proof import (
    BackwardContract,
    BackwardStrategy,
    EquivalenceClass,
    ForwardOnlyRestriction,
    Obligation,
    SavedStatePolicy,
)
from urm.compiler.rewrite.rules import (
    BARRIER_FREE,
    CheckOutcome,
    DEFAULT_RULES,
    DELAY_ROW_SCALE_THROUGH_GEMM,
    FOLD_ROW_SCALE_EPILOGUE,
    Precondition,
    RewriteMatch,
    RewriteRule,
    SCALE_IS_ROWWISE_LINEAR,
    SINGLE_CONSUMER,
)
from urm.ir.program import SemanticNode, SemanticProgram

# Re-export the rule and proof vocabulary so existing importers of
# ``urm.compiler.rewrite.engine`` keep working while the modules are split.
__all__ = [
    "BARRIER_FREE",
    "BackwardContract",
    "BackwardStrategy",
    "CheckOutcome",
    "DEFAULT_RULES",
    "DELAY_ROW_SCALE_THROUGH_GEMM",
    "EquivalenceClass",
    "FOLD_ROW_SCALE_EPILOGUE",
    "ForwardOnlyRestriction",
    "Obligation",
    "Precondition",
    "RewriteEngine",
    "RewriteMatch",
    "RewriteResult",
    "RewriteRule",
    "RewriteTrace",
    "RuleAttempt",
    "SCALE_IS_ROWWISE_LINEAR",
    "SINGLE_CONSUMER",
    "SavedStatePolicy",
]


@dataclass(frozen=True, slots=True)
class RuleAttempt:
    rule: str
    subject_op: str
    outcome: str
    reason_code: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RewriteTrace:
    attempts: tuple[RuleAttempt, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    anchors: tuple[str, ...] = ()
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
        from urm.compiler.common.diagnostics import CompilerError, Diagnostic

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
                break

        rewritten = program.replaced(tuple(ops)) if changed else program
        if changed:
            rewritten.validate()
        trace = RewriteTrace(attempts=tuple(attempts), obligations=tuple(obligations))
        return RewriteResult(program=rewritten, trace=trace, changed=changed)


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
