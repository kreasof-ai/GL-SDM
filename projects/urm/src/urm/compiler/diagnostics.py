"""Structured compiler diagnostics.

Every rejection in the URM compiler is explicit and coded. The compiler never
falls back silently: an unsupported program produces a structured diagnostic,
and a backend declines with a reason instead of changing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class DiagnosticCode(StrEnum):
    """Stable codes for compile-time rejections and warnings."""

    UNBOUND_INPUT = "unbound_input"
    DUPLICATE_DEFINITION = "duplicate_definition"
    UNKNOWN_TENSOR = "unknown_tensor"
    UNUSED_OUTPUT = "unused_output"
    UNSUPPORTED_ROUTING = "unsupported_routing"
    MISSING_COLLISION_POLICY = "missing_collision_policy"
    MULTIPLE_COMMITS = "multiple_commits"
    RECURSION_NOT_LOWERED = "recurrence_not_lowered"
    REWRITE_PRECONDITION_FAILED = "rewrite_precondition_failed"
    REWRITE_EFFECT_UNSAFE = "rewrite_effect_unsafe"
    REWRITE_LOCALITY_UNSAFE = "rewrite_locality_unsafe"
    ANCHOR_DECLINED = "anchor_declined"
    NO_ANCHOR_AVAILABLE = "no_anchor_available"
    PLACEMENT_INCOMPLETE = "placement_incomplete"
    CAPACITY_DROP_REQUIRED = "capacity_drop_required"
    SCHEDULE_HINT_INVALID = "schedule_hint_invalid"
    INTENT_CONFLICT = "intent_conflict"
    CANDIDATE_NOT_FOUND = "candidate_not_found"
    CANDIDATE_ILLEGAL = "candidate_illegal"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    SOLVER_UNKNOWN = "solver_unknown"
    UNSAT_CONSTRAINTS = "unsat_constraints"
    MODEL_VERIFICATION_FAILED = "model_verification_failed"
    NOGOOD_RETRY_LIMIT = "nogood_retry_limit"
    PROTOCOL_VIOLATION = "protocol_violation"
    DEPENDENCY_MISSING = "dependency_missing"
    UPSTREAM_REVISION_MISMATCH = "upstream_revision_mismatch"
    UNSUPPORTED_HARDWARE = "unsupported_hardware"
    UNSUPPORTED_SEMANTICS = "unsupported_semantics"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: DiagnosticCode
    message: str
    severity: Severity = Severity.ERROR
    subject: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.subject is not None:
            payload["subject"] = self.subject
        return payload


# NOTE: exceptions must be neither slotted nor frozen - the interpreter
# assigns ``__traceback__`` on every raise, which requires a writable
# instance dict and breaks under both ``slots=True`` (layout mismatch) and
# ``frozen=True`` (blocked assignment).
@dataclass
class CompilerError(Exception):
    """Raised when validation fails; carries every accumulated diagnostic."""

    diagnostics: tuple[Diagnostic, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.diagnostics:
            raise ValueError("CompilerError requires at least one diagnostic")

    def __str__(self) -> str:
        first = self.diagnostics[0]
        rest = len(self.diagnostics) - 1
        suffix = f" (+{rest} more)" if rest else ""
        return f"[{first.code.value}] {first.message}{suffix}"

    def to_list(self) -> list[dict[str, str]]:
        return [diagnostic.to_dict() for diagnostic in self.diagnostics]


class DiagnosticsCollector:
    """Deterministic accumulator used by validation passes."""

    def __init__(self) -> None:
        self._items: list[Diagnostic] = []

    def add(self, diagnostic: Diagnostic) -> None:
        self._items.append(diagnostic)

    def error(
        self, code: DiagnosticCode, message: str, subject: str | None = None
    ) -> None:
        self.add(
            Diagnostic(
                code=code, message=message, severity=Severity.ERROR, subject=subject
            )
        )

    def warning(
        self, code: DiagnosticCode, message: str, subject: str | None = None
    ) -> None:
        self.add(
            Diagnostic(
                code=code, message=message, severity=Severity.WARNING, subject=subject
            )
        )

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self._items if d.severity is Severity.ERROR)

    def raise_if_errors(self) -> None:
        errors = self.errors
        if errors:
            raise CompilerError(errors)

    def __iter__(self) -> Iterator[Diagnostic]:
        return iter(tuple(self._items))

    def __len__(self) -> int:
        return len(self._items)


def first_error(diagnostics: Sequence[Diagnostic]) -> Diagnostic:
    return diagnostics[0]
