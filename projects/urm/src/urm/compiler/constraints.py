"""Backend-independent constraint vocabulary for the URM compiler.

This module is the *only* place schedule/placement decision problems are
described. Z3 (or any future solver) is a translation target of this IR and
never leaks into it: semantic IR, execution IR, serialized artifacts, and
public adapter APIs contain no solver expressions.

Every assertion in a :class:`ConstraintModel` carries:

- a stable name;
- a category (semantic / rewrite / anchor / schedule / resource / ...);
- a human-readable explanation;
- provenance (originating object kind + id);
- a severity;
- the variables involved.

The vocabulary covers Boolean choices, bounded integers, enumerated choices,
(ine)quality over small linear expressions, divisibility, implication,
at-most-one/exactly-one, capacity/resource bounds, and lexicographic
objective terms.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Union

from urm.compiler.diagnostics import Severity

# -- Variables -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoolVar:
    """A named Boolean choice."""

    name: str


@dataclass(frozen=True, slots=True)
class IntVar:
    """A named integer with an inclusive domain."""

    name: str
    lower: int
    upper: int

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"{self.name}: inverted integer domain")


@dataclass(frozen=True, slots=True)
class EnumVar:
    """A named choice from an ordered set of symbols.

    The value is represented by its index into ``values``; index 0 is the
    deterministic default when one is needed.
    """

    name: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError(f"{self.name}: enumerated choice needs values")
        if len(set(self.values)) != len(self.values):
            raise ValueError(f"{self.name}: duplicate enumeration values")

    def index_of(self, value: str) -> int:
        return self.values.index(value)


Variable = BoolVar | IntVar | EnumVar

Assignment = dict[str, bool | int | str]


def variable_domain(variable: Variable) -> str:
    """Human-readable domain of one variable."""
    if isinstance(variable, BoolVar):
        return "bool"
    if isinstance(variable, IntVar):
        return f"[{variable.lower}, {variable.upper}]"
    return "{" + ", ".join(variable.values) + "}"


# -- Linear expressions ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LinearExpr:
    """A linear expression over variable names with an integer constant."""

    terms: tuple[tuple[str, int], ...] = ()  # (variable, coefficient), sorted by name
    constant: int = 0

    @staticmethod
    def const(value: int | bool) -> LinearExpr:
        return LinearExpr(constant=int(value))

    @staticmethod
    def var(name: str, coefficient: int = 1) -> LinearExpr:
        return LinearExpr(terms=((name, coefficient),))

    @staticmethod
    def _canonical(terms: dict[str, int], constant: int) -> tuple[tuple[str, int], ...]:
        merged: dict[str, int] = {}
        for name, coefficient in terms.items():
            if coefficient:
                merged[name] = merged.get(name, 0) + coefficient
        return tuple(sorted((n, c) for n, c in merged.items() if c))

    def __add__(self, other: object) -> LinearExpr:
        if isinstance(other, int | bool):
            other = LinearExpr.const(int(other))
        if not isinstance(other, LinearExpr):
            return NotImplemented
        terms: dict[str, int] = dict(self.terms)
        for name, coefficient in other.terms:
            terms[name] = terms.get(name, 0) + coefficient
        return LinearExpr(
            terms=LinearExpr._canonical(terms, 0),
            constant=self.constant + other.constant,
        )

    __radd__ = __add__

    def __sub__(self, other: object) -> LinearExpr:
        if isinstance(other, int | bool):
            other = LinearExpr.const(int(other))
        if not isinstance(other, LinearExpr):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: object) -> LinearExpr:
        if isinstance(other, int | bool):
            other = LinearExpr.const(int(other))
        return other + (-self)

    def __neg__(self) -> LinearExpr:
        return LinearExpr(
            terms=tuple((name, -c) for name, c in self.terms),
            constant=-self.constant,
        )

    def __mul__(self, scalar: int) -> LinearExpr:
        if not isinstance(scalar, int):
            return NotImplemented
        return LinearExpr(
            terms=tuple((name, c * scalar) for name, c in self.terms),
            constant=self.constant * scalar,
        )

    __rmul__ = __mul__

    def evaluate(self, assignment: Assignment) -> int:
        total = self.constant
        for name, coefficient in self.terms:
            value = assignment[name]
            numeric = int(value) if isinstance(value, bool) else value
            if isinstance(numeric, str):
                raise TypeError(f"cannot evaluate string-valued {name!r} numerically")
            total += coefficient * numeric
        return total

    def describe(self) -> str:
        parts = [
            f"{coefficient}*{name}" if coefficient != 1 else name
            for name, coefficient in self.terms
        ]
        body = " + ".join(parts) if parts else "0"
        if self.constant:
            body += f" {'+' if self.constant > 0 else '-'} {abs(self.constant)}"
        return body


ValueOrExpr = "int | bool | LinearExpr"  # runtime union alias for annotations


def _as_expr(value: ValueOrExpr) -> LinearExpr:
    if isinstance(value, LinearExpr):
        return value
    return LinearExpr.const(int(value))


# -- Provenance ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Origin:
    """Where a constraint came from."""

    kind: str  # e.g. "semantic_op" | "rewrite_rule" | "anchor" | "schedule_param"
    id: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


SEMANTIC_ORIGIN_KINDS = (
    "semantic_op",
    "rewrite_rule",
    "anchor",
    "schedule_param",
    "intent",
    "device",
    "placement",
    "route_protocol",
    "nogood",
    "solver_internal",
)


# -- Constraints -----------------------------------------------------------------


class ConstraintCategory(StrEnum):
    SEMANTIC = "semantic"
    REWRITE = "rewrite"
    ANCHOR_CAPABILITY = "anchor_capability"
    SCHEDULE = "schedule"
    RESOURCE = "resource"
    TRAINING = "training"
    DETERMINISM = "determinism"
    PLACEMENT = "placement"
    COMMUNICATION = "communication"
    SEARCH = "search"


@dataclass(frozen=True, slots=True)
class ConstraintHeader:
    """Metadata shared by every named assertion."""

    name: str
    category: ConstraintCategory
    explanation: str
    origin: Origin
    severity: Severity = Severity.ERROR

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "category": self.category.value,
            "explanation": self.explanation,
            "origin": self.origin.to_dict(),
            "severity": self.severity.value,
        }


@dataclass(frozen=True, slots=True)
class Equality(ConstraintHeader):
    """lhs == rhs over linear expressions."""

    lhs: LinearExpr = LinearExpr()
    rhs: LinearExpr = LinearExpr()

    @property
    def variables(self) -> tuple[str, ...]:
        names = {name for name, _ in self.lhs.terms} | {
            name for name, _ in self.rhs.terms
        }
        return tuple(sorted(names))

    def holds(self, assignment: Assignment) -> bool:
        return self.lhs.evaluate(assignment) == self.rhs.evaluate(assignment)

    def describe(self) -> str:
        return f"{self.lhs.describe()} == {self.rhs.describe()}"


@dataclass(frozen=True, slots=True)
class LessEqual(ConstraintHeader):
    """lhs <= rhs over linear expressions."""

    lhs: LinearExpr = LinearExpr()
    rhs: LinearExpr = LinearExpr()

    @property
    def variables(self) -> tuple[str, ...]:
        names = {name for name, _ in self.lhs.terms} | {
            name for name, _ in self.rhs.terms
        }
        return tuple(sorted(names))

    def holds(self, assignment: Assignment) -> bool:
        return self.lhs.evaluate(assignment) <= self.rhs.evaluate(assignment)

    def describe(self) -> str:
        return f"{self.lhs.describe()} <= {self.rhs.describe()}"


@dataclass(frozen=True, slots=True)
class Divisibility(ConstraintHeader):
    """A variable must be divisible by ``divisor`` (remainder 0)."""

    variable: str = ""
    divisor: int = 1

    def __post_init__(self) -> None:
        if self.divisor <= 0:
            raise ValueError("divisor must be positive")

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.variable,)

    def holds(self, assignment: Assignment) -> bool:
        value = assignment[self.variable]
        numeric = int(value) if isinstance(value, bool) else value
        if isinstance(numeric, str):
            raise TypeError(f"{self.variable!r} is not integer-valued")
        return numeric % self.divisor == 0

    def describe(self) -> str:
        return f"{self.variable} % {self.divisor} == 0"


ImplicationConsequent = Union[Equality, LessEqual, Divisibility, "Implication"]


@dataclass(frozen=True, slots=True)
class Implication(ConstraintHeader):
    """If the guard holds, every consequent must hold.

    The guard is either a Boolean variable equal to ``expected``, or any
    single equality used as a predicate. Consequents may include nested
    implications, so conjunctions of guards decompose naturally.
    """

    guard_variable: str | None = None
    guard_expected: bool = True
    guard_equality: Equality | None = None
    consequents: tuple[ImplicationConsequent, ...] = ()

    @property
    def variables(self) -> tuple[str, ...]:
        names: set[str] = set()
        if self.guard_variable is not None:
            names.add(self.guard_variable)
        if self.guard_equality is not None:
            names.update(self.guard_equality.variables)
        for consequent in self.consequents:
            names.update(consequent.variables)
        return tuple(sorted(names))

    def guard_holds(self, assignment: Assignment) -> bool:
        if self.guard_equality is not None:
            return self.guard_equality.holds(assignment)
        assert self.guard_variable is not None
        value = assignment[self.guard_variable]
        return bool(value) is self.guard_expected

    def holds(self, assignment: Assignment) -> bool:
        if not self.guard_holds(assignment):
            return True
        return all(consequent.holds(assignment) for consequent in self.consequents)

    def describe(self) -> str:
        if self.guard_equality is not None:
            guard = self.guard_equality.describe()
        else:
            guard = self.guard_variable or "?"
            if not self.guard_expected:
                guard = f"not {guard}"
        consequents = ", ".join(c.describe() for c in self.consequents) or "true"
        return f"{guard} => ({consequents})"


@dataclass(frozen=True, slots=True)
class AtMostOne(ConstraintHeader):
    """At most one listed Boolean variable may be true."""

    choices: tuple[str, ...] = ()

    @property
    def variables(self) -> tuple[str, ...]:
        return self.choices

    def holds(self, assignment: Assignment) -> bool:
        return sum(bool(assignment[name]) for name in self.choices) <= 1

    def describe(self) -> str:
        return f"sum({', '.join(self.choices)}) <= 1"


@dataclass(frozen=True, slots=True)
class ExactlyOne(ConstraintHeader):
    """Exactly one listed Boolean variable must be true."""

    choices: tuple[str, ...] = ()

    @property
    def variables(self) -> tuple[str, ...]:
        return self.choices

    def holds(self, assignment: Assignment) -> bool:
        return sum(bool(assignment[name]) for name in self.choices) == 1

    def describe(self) -> str:
        return f"sum({', '.join(self.choices)}) == 1"


@dataclass(frozen=True, slots=True)
class CapacityBound(ConstraintHeader):
    """Resource bound on one unit: expression <= limit."""

    unit: str = ""
    expression: LinearExpr = LinearExpr()
    limit: int = 0

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.expression.terms)

    def holds(self, assignment: Assignment) -> bool:
        return self.expression.evaluate(assignment) <= self.limit

    def describe(self) -> str:
        return f"{self.unit}: {self.expression.describe()} <= {self.limit}"


@dataclass(frozen=True, slots=True)
class AllowedSet(ConstraintHeader):
    """An integer variable must take one of the listed values."""

    variable: str = ""
    allowed: tuple[int, ...] = ()

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.variable,)

    def holds(self, assignment: Assignment) -> bool:
        value = assignment[self.variable]
        numeric = int(value) if isinstance(value, bool) else value
        if isinstance(numeric, str):
            raise TypeError(f"{self.variable!r} is not integer-valued")
        return numeric in self.allowed

    def describe(self) -> str:
        return f"{self.variable} in {{{', '.join(map(str, self.allowed))}}}"


@dataclass(frozen=True, slots=True)
class Nogood(ConstraintHeader):
    """A learned exclusion: this exact (partial) assignment may not recur.

    Produced by compile feedback (a failed compilation) or by independent
    model verification rejecting a solver model. The forbidden pairs must all
    match for the nogood to fire.
    """

    forbidden: tuple[tuple[str, bool | int | str], ...] = ()

    @property
    def variables(self) -> tuple[str, ...]:
        seen: list[str] = []
        for name, _ in self.forbidden:
            if name not in seen:
                seen.append(name)
        return tuple(seen)

    def fires(self, assignment: Assignment) -> bool:
        return all(assignment[name] == value for name, value in self.forbidden)

    def holds(self, assignment: Assignment) -> bool:
        return not self.fires(assignment)

    def describe(self) -> str:
        body = " and ".join(f"{n} == {v!r}" for n, v in self.forbidden)
        return f"not({body})"


Constraint = (
    Equality
    | LessEqual
    | Divisibility
    | AllowedSet
    | Implication
    | AtMostOne
    | ExactlyOne
    | CapacityBound
    | Nogood
)


# -- Objectives -------------------------------------------------------------------


class ObjectiveSense(StrEnum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass(frozen=True, slots=True)
class ObjectiveTerm:
    """One lexicographic objective; list order defines priority."""

    name: str
    expression: LinearExpr
    sense: ObjectiveSense = ObjectiveSense.MINIMIZE

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "expression": self.expression.describe(),
            "sense": self.sense.value,
        }


# -- The model ---------------------------------------------------------------------


class ModelValidationError(ValueError):
    """Raised when a constraint model references unknown variables."""


@dataclass(slots=True)
class ConstraintModel:
    """A bounded, fully-named decision problem.

    Objectives are lexicographic: earlier entries dominate. A final
    deterministic tie-break term should order equally-good solutions by a
    stable problem-defined ordering so solving is reproducible.
    """

    name: str
    variables: list[Variable] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    objectives: list[ObjectiveTerm] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    # -- construction -----------------------------------------------------

    def add_variable(self, variable: Variable) -> Variable:
        existing = self.variable_named(variable.name)
        if existing is not None:
            raise ModelValidationError(
                f"duplicate variable {variable.name!r} in model {self.name}"
            )
        self.variables.append(variable)
        return variable

    def add_constraint(self, constraint: Constraint) -> Constraint:
        unknown = set(constraint.variables) - {v.name for v in self.variables}
        if unknown:
            raise ModelValidationError(
                f"constraint {constraint.name!r} uses unregistered variables: "
                f"{sorted(unknown)}"
            )
        self.constraints.append(constraint)
        return constraint

    def add_objective(self, term: ObjectiveTerm) -> ObjectiveTerm:
        unknown = {name for name, _ in term.expression.terms} - {
            v.name for v in self.variables
        }
        if unknown:
            raise ModelValidationError(
                f"objective {term.name!r} uses unregistered variables: {sorted(unknown)}"
            )
        self.objectives.append(term)
        return term

    # -- accessors ----------------------------------------------------------

    def variable_named(self, name: str) -> Variable | None:
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def constraints_in_category(
        self, category: ConstraintCategory
    ) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.category is category)

    # -- validation & serialization -----------------------------------------

    def validate(self) -> None:
        seen: set[str] = set()
        for variable in self.variables:
            if variable.name in seen:
                raise ModelValidationError(f"duplicate variable {variable.name!r}")
            seen.add(variable.name)
        for constraint in self.constraints:
            unknown = set(constraint.variables) - seen
            if unknown:
                raise ModelValidationError(
                    f"constraint {constraint.name!r} uses unregistered "
                    f"variables: {sorted(unknown)}"
                )
        for objective in self.objectives:
            unknown = {n for n, _ in objective.expression.terms} - seen
            if unknown:
                raise ModelValidationError(
                    f"objective {objective.name!r} uses unregistered variables: "
                    f"{sorted(unknown)}"
                )

    def to_summary(self) -> dict[str, object]:
        """Compact serializable summary (never raw solver expressions)."""
        return {
            "model": self.name,
            "metadata": dict(sorted(self.metadata.items())),
            "variables": [
                {"name": v.name, "domain": variable_domain(v)} for v in self.variables
            ],
            "constraints": [
                {
                    "name": c.name,
                    "category": c.category.value,
                    "explanation": c.explanation,
                    "origin": c.origin.to_dict(),
                    "severity": c.severity.value,
                    "formula": c.describe(),
                    "variables": list(c.variables),
                }
                for c in self.constraints
            ],
            "objectives": [o.to_dict() for o in self.objectives],
        }

    def summary_hash(self) -> str:
        canonical = json.dumps(self.to_summary(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -- Convenience builders ----------------------------------------------------


def equality(header: ConstraintHeader, lhs: ValueOrExpr, rhs: ValueOrExpr) -> Equality:
    return Equality(
        name=header.name,
        category=header.category,
        explanation=header.explanation,
        origin=header.origin,
        severity=header.severity,
        lhs=_as_expr(lhs),
        rhs=_as_expr(rhs),
    )


def less_equal(
    header: ConstraintHeader, lhs: ValueOrExpr, rhs: ValueOrExpr
) -> LessEqual:
    return LessEqual(
        name=header.name,
        category=header.category,
        explanation=header.explanation,
        origin=header.origin,
        severity=header.severity,
        lhs=_as_expr(lhs),
        rhs=_as_expr(rhs),
    )


def implies_true(header: ConstraintHeader, guard_var: str) -> _ImplicationBuilder:
    """Start an implication whose guard is ``guard_var == True``."""
    return _ImplicationBuilder(header, guard_variable=guard_var, guard_expected=True)


def implies_equal(
    header: ConstraintHeader, expr: ValueOrExpr, value: int
) -> _ImplicationBuilder:
    guard = equality(
        ConstraintHeader(
            name=f"{header.name}::guard",
            category=header.category,
            explanation=f"guard of {header.name}",
            origin=header.origin,
            severity=header.severity,
        ),
        expr,
        value,
    )
    return _ImplicationBuilder(header, guard_equality=guard)


class _ImplicationBuilder:
    def __init__(
        self,
        header: ConstraintHeader,
        *,
        guard_variable: str | None = None,
        guard_expected: bool = True,
        guard_equality: Equality | None = None,
    ) -> None:
        self._header = header
        self._guard_variable = guard_variable
        self._guard_expected = guard_expected
        self._guard_equality = guard_equality
        self._consequents: list[ImplicationConsequent] = []

    def then_equal(self, lhs: ValueOrExpr, rhs: ValueOrExpr) -> _ImplicationBuilder:
        self._consequents.append(
            equality(
                self._sub_header("then_eq"),
                lhs,
                rhs,
            )
        )
        return self

    def then_less_equal(
        self, lhs: ValueOrExpr, rhs: ValueOrExpr
    ) -> _ImplicationBuilder:
        self._consequents.append(less_equal(self._sub_header("then_le"), lhs, rhs))
        return self

    def then_divisible(self, variable: str, divisor: int) -> _ImplicationBuilder:
        sub = self._sub_header("then_div")
        self._consequents.append(
            Divisibility(
                name=sub.name,
                category=sub.category,
                explanation=sub.explanation,
                origin=sub.origin,
                severity=sub.severity,
                variable=variable,
                divisor=divisor,
            )
        )
        return self

    def then_contradiction(self) -> _ImplicationBuilder:
        """Force unsatisfiability whenever the guard holds."""
        self._consequents.append(equality(self._sub_header("then_false"), 1, 0))
        return self

    def done(self) -> Implication:
        if not self._consequents:
            raise ValueError(f"implication {self._header.name!r} has no consequents")
        return Implication(
            name=self._header.name,
            category=self._header.category,
            explanation=self._header.explanation,
            origin=self._header.origin,
            severity=self._header.severity,
            guard_variable=self._guard_variable,
            guard_expected=self._guard_expected,
            guard_equality=self._guard_equality,
            consequents=tuple(self._consequents),
        )

    def _sub_header(self, suffix: str) -> ConstraintHeader:
        return ConstraintHeader(
            name=f"{self._header.name}::{suffix}",
            category=self._header.category,
            explanation=f"consequent of {self._header.name}",
            origin=self._header.origin,
            severity=self._header.severity,
        )


def make_exactly_one(
    name: str,
    category: ConstraintCategory,
    explanation: str,
    origin: Origin,
    choices: Sequence[str],
    severity: Severity = Severity.ERROR,
) -> ExactlyOne:
    return ExactlyOne(
        name=name,
        category=category,
        explanation=explanation,
        origin=origin,
        severity=severity,
        choices=tuple(choices),
    )


def make_at_most_one(
    name: str,
    category: ConstraintCategory,
    explanation: str,
    origin: Origin,
    choices: Sequence[str],
    severity: Severity = Severity.ERROR,
) -> AtMostOne:
    return AtMostOne(
        name=name,
        category=category,
        explanation=explanation,
        origin=origin,
        severity=severity,
        choices=tuple(choices),
    )


def capacity_bound(
    name: str,
    category: ConstraintCategory,
    explanation: str,
    origin: Origin,
    unit: str,
    expression: ValueOrExpr,
    limit: int,
    severity: Severity = Severity.ERROR,
) -> CapacityBound:
    return CapacityBound(
        name=name,
        category=category,
        explanation=explanation,
        origin=origin,
        severity=severity,
        unit=unit,
        expression=_as_expr(expression),
        limit=limit,
    )


def make_nogood(
    name: str,
    explanation: str,
    origin_kind: str,
    origin_id: str,
    forbidden: dict[str, bool | int | str],
    category: ConstraintCategory = ConstraintCategory.SEARCH,
) -> Nogood:
    """Build a learned exclusion from compile feedback or model rejection."""
    return Nogood(
        name=name,
        category=category,
        explanation=explanation,
        origin=Origin(kind=origin_kind, id=origin_id),
        severity=Severity.ERROR,
        forbidden=tuple(sorted(forbidden.items(), key=lambda kv: kv[0])),
    )
