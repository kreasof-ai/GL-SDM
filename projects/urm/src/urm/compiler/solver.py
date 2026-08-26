"""Optional Z3 integration: feasibility and optimization passes.

Z3 is an optional compiler dependency (``pip install urm-kernel-lab[solver]``).
Core URM functionality works without it; importing this module never fails
when Z3 is absent - calling a pass does, with a structured diagnostic.

Two strictly separate passes:

**A. Feasibility pass.** A standard solver with tracked, named assertions.
Returns SAT with a model, UNSAT with an unsat core mapped to structured
diagnostics, or UNKNOWN/TIMEOUT with the reason and resource statistics.
Diagnostics and optimization are never merged into one opaque call.

**B. Optimization pass.** Runs only after feasibility succeeds. Bounded
lexicographic objectives in a fixed order (obligations first, memory,
communication critical path, communication bytes, materialization bytes,
launch count, analytical runtime, then a deterministic stable-ordering
tie-break). Timeouts and candidate limits are explicit; serialization is by
model summary, never raw solver dumps.

Solver models are untrusted until :mod:`urm.compiler.verification` accepts
them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from urm.compiler.constraints import (
    AllowedSet,
    Assignment,
    AtMostOne,
    BoolVar,
    CapacityBound,
    ConstraintModel,
    Divisibility,
    EnumVar,
    Equality,
    ExactlyOne,
    Implication,
    IntVar,
    LessEqual,
    Nogood,
)
from urm.compiler.diagnostics import (
    CompilerError,
    Diagnostic,
    DiagnosticCode,
    Severity,
)

try:  # optional dependency: the solver extra pins this version
    import z3  # type: ignore[import-not-found]

    _Z3_IMPORT_ERROR: str | None = None
except ImportError as error:  # pragma: no cover - exercised via absence
    z3 = None  # type: ignore[assignment]
    _Z3_IMPORT_ERROR = str(error)

DEFAULT_TIMEOUT_MS = 10_000
DEFAULT_RLIMIT = 10_000_000


def z3_available() -> bool:
    return z3 is not None


def z3_version() -> str | None:
    if z3 is None:
        return None
    return z3.get_version_string()


def require_z3() -> object:
    if z3 is None:
        raise CompilerError(
            (
                Diagnostic(
                    code=DiagnosticCode.SOLVER_UNAVAILABLE,
                    message=(
                        "the Z3 solver extra is not installed "
                        f"({_Z3_IMPORT_ERROR}); install "
                        "'urm-kernel-lab[solver]' or use the deterministic "
                        "cost-heuristic selection policy"
                    ),
                ),
            )
        )
    return z3


class FeasibilityStatus(StrEnum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SolverLimits:
    """Explicit resource bounds for every search."""

    timeout_ms: int = DEFAULT_TIMEOUT_MS
    rlimit: int = DEFAULT_RLIMIT  # Z3 resource (propagation) budget
    max_nogoods: int = 64


@dataclass(slots=True)
class PassOutcome:
    """Shared result shape for both passes."""

    status: FeasibilityStatus
    assignment: Assignment | None = None
    objective_values: tuple[int, ...] = ()
    unsat_core_names: tuple[str, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    statistics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is FeasibilityStatus.SAT


class _Translator:
    """Constraint-model -> Z3 translation with tracked assertion names."""

    def __init__(self, model: ConstraintModel) -> None:
        self.model = model
        self.vars: dict[str, object] = {}
        self.bool_vars: set[str] = set()
        for variable in model.variables:
            if isinstance(variable, BoolVar):
                self.vars[variable.name] = z3.Bool(variable.name)  # type: ignore[union-attr]
                self.bool_vars.add(variable.name)
            elif isinstance(variable, (IntVar, EnumVar)):
                self.vars[variable.name] = z3.Int(variable.name)  # type: ignore[union-attr]

    def domain_assertions(self) -> list[tuple[str, object]]:
        """Per-variable range assertions, each individually tracked."""
        out: list[tuple[str, object]] = []
        for variable in self.model.variables:
            expr = self.vars[variable.name]
            if isinstance(variable, BoolVar):
                continue
            if isinstance(variable, IntVar):
                out.append(
                    (
                        f"{variable.name}:in_range",
                        z3.And(  # type: ignore[union-attr]
                            expr >= variable.lower, expr <= variable.upper
                        ),
                    )
                )
            else:  # EnumVar: index into values
                upper = len(variable.values) - 1
                out.append(
                    (
                        f"{variable.name}:in_enum",
                        z3.And(expr >= 0, expr <= upper),  # type: ignore[union-attr]
                    )
                )
        return out

    def literal(self, value: bool | int | str, var_type: str | None = None):
        if isinstance(value, bool):
            return z3.BoolVal(value)  # type: ignore[union-attr]
        if isinstance(value, str):
            assert var_type is not None, "string literals require an enum target"
            enum_var = next(
                v
                for v in self.model.variables
                if isinstance(v, EnumVar) and v.name == var_type
            )
            return enum_var.index_of(value)
        return value

    def expression(self, linear):
        total = z3.IntVal(linear.constant)  # type: ignore[union-attr]
        for name, coefficient in linear.terms:
            term = self.vars[name]
            if name in self.bool_vars:
                # Booleans participate in linear expressions as 0/1.
                term = z3.If(term, 1, 0)  # type: ignore[union-attr]
            total = total + coefficient * term
        return total

    def constraint_expr(self, constraint):
        match constraint:
            case Equality():
                return self.expression(constraint.lhs) == self.expression(
                    constraint.rhs
                )
            case LessEqual():
                return self.expression(constraint.lhs) <= self.expression(
                    constraint.rhs
                )
            case Divisibility():
                return self.vars[constraint.variable] % constraint.divisor == 0
            case AllowedSet():
                target = self.vars[constraint.variable]
                return z3.Or([target == value for value in constraint.allowed])  # type: ignore[union-attr]
            case AtMostOne():
                flags = [z3.Bool(name) for name in constraint.choices]  # type: ignore[union-attr]
                return z3.Sum([z3.If(f, 1, 0) for f in flags]) <= 1  # type: ignore[union-attr]
            case ExactlyOne():
                flags = [z3.Bool(name) for name in constraint.choices]  # type: ignore[union-attr]
                return z3.Sum([z3.If(f, 1, 0) for f in flags]) == 1  # type: ignore[union-attr]
            case CapacityBound():
                return self.expression(constraint.expression) <= constraint.limit
            case Nogood():
                clauses = []
                for name, value in constraint.forbidden:
                    variable = self.model.variable_named(name)
                    assert variable is not None
                    if isinstance(variable, BoolVar):
                        target = z3.Bool(name)  # type: ignore[union-attr]
                        clauses.append(target != z3.BoolVal(bool(value)))  # type: ignore[union-attr]
                    else:
                        literal = self.literal(value, var_type=name)
                        clauses.append(self.vars[name] != literal)
                return z3.Or(clauses)  # type: ignore[union-attr]
            case Implication():
                if constraint.guard_equality is not None:
                    guard = self.constraint_expr(constraint.guard_equality)
                else:
                    guard_var = z3.Bool(constraint.guard_variable)  # type: ignore[union-attr]
                    guard = (
                        guard_var if constraint.guard_expected else z3.Not(guard_var)  # type: ignore[union-attr]
                    )
                consequents = [
                    self.constraint_expr(consequent)
                    for consequent in constraint.consequents
                ]
                return z3.Implies(guard, z3.And(consequents))  # type: ignore[union-attr]
            case _:  # pragma: no cover - exhaustive by construction
                raise TypeError(f"untranslatable constraint {type(constraint)}")

    def tracked_assertions(self) -> list[tuple[str, object]]:
        out = self.domain_assertions()
        for constraint in self.model.constraints:
            out.append((constraint.name, self.constraint_expr(constraint)))
        return out


def _decode_assignment(translator: _Translator, z3_model) -> Assignment:
    decoded: Assignment = {}
    for variable in translator.model.variables:
        ast = translator.vars[variable.name]
        value = z3_model.eval(ast, model_completion=True)  # type: ignore[union-attr]
        if isinstance(variable, BoolVar):
            decoded[variable.name] = bool(value)
        elif isinstance(variable, IntVar):
            decoded[variable.name] = int(str(value))
        else:  # EnumVar
            index = int(str(value))
            if not 0 <= index < len(variable.values):
                raise ValueError(
                    f"solver returned out-of-range index {index} for {variable.name}"
                )
            decoded[variable.name] = variable.values[index]
    return decoded


def _core_diagnostics(
    core_names: tuple[str, ...], model: ConstraintModel
) -> tuple[Diagnostic, ...]:
    """Map an unsat core onto structured diagnostics with explanations."""
    by_name = {}
    for constraint in model.constraints:
        by_name[constraint.name] = constraint
    messages: list[str] = []
    involved_categories: set[str] = set()
    for name in core_names:
        constraint = by_name.get(name)
        if constraint is None:
            continue  # a variable-domain assertion; listed but unnamed above
        involved_categories.add(constraint.category.value)
        messages.append(f"{name}: {constraint.explanation}")
    primary = (
        f"no legal plan: {len(core_names)} named constraints conflict "
        f"(categories: {sorted(involved_categories) or ['variable_domain']})"
    )
    diagnostics = [
        Diagnostic(
            code=DiagnosticCode.UNSAT_CONSTRAINTS,
            severity=Severity.ERROR,
            message=primary,
        )
    ]
    diagnostics.extend(
        Diagnostic(
            code=DiagnosticCode.UNSAT_CONSTRAINTS,
            severity=Severity.ERROR,
            message=message,
        )
        for message in messages[:8]  # bounded, most relevant first
    )
    if len(core_names) > 8:
        diagnostics.append(
            Diagnostic(
                code=DiagnosticCode.UNSAT_CONSTRAINTS,
                severity=Severity.WARNING,
                message=f"... and {len(core_names) - 8} more core members",
            )
        )
    return tuple(diagnostics)


def _base_statistics(
    elapsed_ms: float, limits: SolverLimits, **extra: float | str
) -> dict[str, float | int | str]:
    stats: dict[str, float | int | str] = {
        "wall_ms": round(elapsed_ms, 3),
        "timeout_ms": limits.timeout_ms,
        "rlimit": limits.rlimit,
    }
    stats.update(extra)
    return stats


# -- Pass A: feasibility -------------------------------------------------------


class FeasibilityPass:
    """Standard solver pass with tracked, named assertions."""

    def run(
        self,
        model: ConstraintModel,
        limits: SolverLimits | None = None,
    ) -> PassOutcome:
        engine = require_z3()
        limits = limits or SolverLimits()
        started = time.perf_counter()
        translator = _Translator(model)
        solver = engine.Solver()
        solver.set("timeout", limits.timeout_ms)
        solver.set("rlimit", limits.rlimit)
        for name, assertion in translator.tracked_assertions():
            solver.assert_and_track(assertion, name)
        result = solver.check()
        elapsed = (time.perf_counter() - started) * 1000.0

        statistics = _base_statistics(elapsed, limits)
        if result == engine.sat:
            assignment = _decode_assignment(translator, solver.model())
            return PassOutcome(
                status=FeasibilityStatus.SAT,
                assignment=assignment,
                statistics=statistics,
            )
        if result == engine.unsat:
            core_names = tuple(
                str(name) for name in sorted(map(str, solver.unsat_core()))
            )
            return PassOutcome(
                status=FeasibilityStatus.UNSAT,
                unsat_core_names=core_names,
                diagnostics=_core_diagnostics(core_names, model),
                statistics=statistics,
            )
        reason = solver.reason_unknown()
        statistics["unknown_reason"] = reason or ""
        return PassOutcome(
            status=FeasibilityStatus.UNKNOWN,
            diagnostics=(
                Diagnostic(
                    code=DiagnosticCode.SOLVER_UNKNOWN,
                    severity=Severity.ERROR,
                    message=(
                        f"feasibility could not be decided ({reason}); no plan "
                        "is selected on an undecided problem"
                    ),
                ),
            ),
            statistics=statistics,
        )


# -- Pass B: bounded lexicographic optimization ---------------------------------

DEFAULT_OBJECTIVE_ORDER = (
    "unresolved_obligations",
    "peak_temporary_bytes",
    "comm_critical_path",
    "communication_bytes",
    "materialization_bytes",
    "launch_count",
    "analytical_runtime_us",
)


class OptimizationPass:
    """Bounded lexicographic optimization; runs only after feasibility.

    The model's own objective list defines priority order; callers should end
    it with the deterministic stable-ordering tie-break so equal-cost plans
    are resolved reproducibly.
    """

    def __init__(self, limits: SolverLimits | None = None) -> None:
        self.limits = limits or SolverLimits()

    def run(self, model: ConstraintModel) -> PassOutcome:
        engine = require_z3()
        feasibility = FeasibilityPass().run(model, self.limits)
        if feasibility.status is not FeasibilityStatus.SAT:
            return feasibility  # UNSAT/UNKNOWN diagnostics flow through

        started = time.perf_counter()
        translator = _Translator(model)
        optimizer = engine.Optimize()
        optimizer.set("timeout", self.limits.timeout_ms)
        optimizer.set("rlimit", self.limits.rlimit)
        for _name, assertion in translator.tracked_assertions():
            optimizer.add(assertion)
        handles = []
        for objective in model.objectives:
            expression = translator.expression(objective.expression)
            handle = (
                optimizer.minimize(expression)
                if objective.sense.value == "minimize"
                else optimizer.maximize(expression)
            )
            handles.append(handle)
        result = optimizer.check()
        elapsed = (time.perf_counter() - started) * 1000.0
        statistics = _base_statistics(elapsed, self.limits)
        statistics["objectives"] = len(handles)

        if result == engine.sat:
            z3_model = optimizer.model()
            assignment = _decode_assignment(translator, z3_model)
            objective_values = []
            for objective, handle in zip(model.objectives, handles, strict=True):
                del objective
                objective_values.append(int(str(z3_model.eval(handle.value()))))
            return PassOutcome(
                status=FeasibilityStatus.SAT,
                assignment=assignment,
                objective_values=tuple(objective_values),
                statistics=statistics,
            )
        if result == engine.unsat:  # cannot happen after feasible Phase A
            return PassOutcome(
                status=FeasibilityStatus.UNKNOWN,
                diagnostics=(
                    Diagnostic(
                        code=DiagnosticCode.SOLVER_UNKNOWN,
                        severity=Severity.ERROR,
                        message="optimizer contradicted the feasibility pass",
                    ),
                ),
                statistics=statistics,
            )
        reason = optimizer.reason_unknown()
        statistics["unknown_reason"] = reason or ""
        return PassOutcome(
            status=FeasibilityStatus.UNKNOWN,
            diagnostics=(
                Diagnostic(
                    code=DiagnosticCode.SOLVER_UNKNOWN,
                    severity=Severity.ERROR,
                    message=(
                        f"optimization exceeded its bounds ({reason}); the "
                        "feasible plan is available but optimality is unproven"
                    ),
                ),
            ),
            statistics=statistics,
        )


# Historical result aliases kept for API readability at call sites.
FeasibilityResult = PassOutcome
OptimizationResult = PassOutcome

__all__ = [
    "DEFAULT_OBJECTIVE_ORDER",
    "DEFAULT_RLIMIT",
    "DEFAULT_TIMEOUT_MS",
    "FeasibilityPass",
    "FeasibilityResult",
    "FeasibilityStatus",
    "OptimizationPass",
    "OptimizationResult",
    "PassOutcome",
    "SolverLimits",
    "z3_available",
    "z3_version",
]
