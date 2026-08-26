"""Solver passes: SAT/UNSAT/UNKNOWN handling, cores, determinism, timeouts.

These tests exercise the Z3-backed passes; they skip cleanly when the
optional ``solver`` extra is absent (test_solver_absent_raises_structured_
diagnostic also runs without Z3 via monkeypatching).
"""

from __future__ import annotations

import pytest

from urm.compiler.constraints import (
    BoolVar,
    ConstraintCategory,
    ConstraintModel,
    Equality,
    ExactlyOne,
    Implication,
    IntVar,
    LinearExpr,
    ObjectiveSense,
    ObjectiveTerm,
    Origin,
    make_nogood,
)
from urm.compiler.diagnostics import CompilerError, DiagnosticCode
from urm.compiler.schedule_space import exhaustive_optimum
from urm.compiler.solver import (
    FeasibilityPass,
    FeasibilityStatus,
    OptimizationPass,
    SolverLimits,
    z3_available,
    z3_version,
)

pytestmark = pytest.mark.skipif(
    not z3_available(), reason="z3-solver optional extra not installed"
)


def _tiny_model() -> ConstraintModel:
    model = ConstraintModel(name="tiny")
    model.add_variable(BoolVar("use_fused"))
    model.add_variable(IntVar("block_d", 32, 256))
    model.add_constraint(
        Implication(
            name="fused_requires_big_tiles",
            category=ConstraintCategory.SCHEDULE,
            explanation="the fused plan requires BLOCK_D >= 128",
            origin=Origin(kind="schedule_param", id="t"),
            guard_variable="use_fused",
            consequents=(
                Equality(
                    name="fused_requires_big_tiles::le",
                    category=ConstraintCategory.SCHEDULE,
                    explanation="lower bound on BLOCK_D under fused",
                    origin=Origin(kind="schedule_param", id="t"),
                    lhs=LinearExpr.const(128),
                    rhs=LinearExpr.var("block_d"),
                ),
            ),
        )
    )
    return model


def test_sat_returns_decoded_model() -> None:
    result = FeasibilityPass().run(_tiny_model())
    assert result.status is FeasibilityStatus.SAT
    assert isinstance(result.assignment["use_fused"], bool)
    assert 32 <= result.assignment["block_d"] <= 256


def test_unsat_maps_core_to_named_diagnostics() -> None:
    model = _tiny_model()
    model.add_constraint(
        Equality(
            name="hint_pins_small_tile",
            category=ConstraintCategory.SCHEDULE,
            explanation="caller pinned BLOCK_D=64 while requesting fused",
            origin=Origin(kind="schedule_param", id="BLOCK_D_hint"),
            lhs=LinearExpr.var("block_d"),
            rhs=LinearExpr.const(64),
        )
    )
    model.add_constraint(
        Equality(
            name="fused_required",
            category=ConstraintCategory.SCHEDULE,
            explanation="policy requires the fused plan",
            origin=Origin(kind="intent", id="policy"),
            lhs=LinearExpr.var("use_fused"),
            rhs=LinearExpr.const(1),
        )
    )
    result = FeasibilityPass().run(model)
    assert result.status is FeasibilityStatus.UNSAT
    names = set(result.unsat_core_names)
    assert "fused_required" in names  # core members are tracked by name
    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert DiagnosticCode.UNSAT_CONSTRAINTS in codes
    joined = " ".join(diagnostic.message for diagnostic in result.diagnostics)
    assert "no legal plan" in joined


def test_unknown_status_under_starved_resource_limit() -> None:
    # A propagation budget of 1 forces UNKNOWN even on tiny problems.
    result = FeasibilityPass().run(_tiny_model(), SolverLimits(rlimit=1))
    assert result.status is FeasibilityStatus.UNKNOWN
    assert any(
        diagnostic.code is DiagnosticCode.SOLVER_UNKNOWN
        for diagnostic in result.diagnostics
    )


def test_optimization_is_lexicographic_and_deterministic() -> None:
    model = ConstraintModel(name="lex")
    for index in range(4):
        model.add_variable(BoolVar(f"choose_{index}"))
    origin = Origin(kind="intent", id="auto")
    model.add_constraint(
        ExactlyOne(
            name="one",
            category=ConstraintCategory.SEARCH,
            explanation="pick one",
            origin=origin,
            choices=tuple(f"choose_{i}" for i in range(4)),
        )
    )
    traffic = LinearExpr()
    ordinal = LinearExpr()
    for index, cost in enumerate((8, 4, 4, 2)):
        traffic += LinearExpr.var(f"choose_{index}") * cost
        ordinal += LinearExpr.var(f"choose_{index}") * index
    model.add_objective(ObjectiveTerm("traffic", traffic, ObjectiveSense.MINIMIZE))
    model.add_objective(ObjectiveTerm("tie", ordinal, ObjectiveSense.MINIMIZE))
    first = OptimizationPass(SolverLimits(timeout_ms=10_000)).run(model)
    second = OptimizationPass(SolverLimits(timeout_ms=10_000)).run(model)
    assert first.status is FeasibilityStatus.SAT
    assert first.assignment == second.assignment  # deterministic tie-breaking
    assert first.objective_values[0] == 2
    assert first.assignment["choose_3"] in (True, 1)


def test_solver_agrees_with_exhaustive_sweep_on_tiny_problem() -> None:
    from urm.compiler.constraints import Divisibility

    model = ConstraintModel(name="agree")
    model.add_variable(IntVar("x", 0, 12))
    model.add_constraint(
        Divisibility(
            name="x_divisible_by_three",
            category=ConstraintCategory.SCHEDULE,
            explanation="x must be a multiple of 3",
            origin=Origin(kind="schedule_param", id="a"),
            variable="x",
            divisor=3,
        )
    )
    exhaustive = exhaustive_optimum(model)
    solved = FeasibilityPass().run(model)
    assert solved.status is FeasibilityStatus.SAT
    assert exhaustive.best_assignment is not None
    assert all(constraint.holds(solved.assignment) for constraint in model.constraints)


def test_nogood_excludes_rejected_assignment() -> None:
    model = ConstraintModel(name="nogood")
    model.add_variable(BoolVar("a"))
    model.add_variable(BoolVar("b"))
    model.add_constraint(
        make_nogood(
            name="nogood_1",
            explanation="this exact combo failed compilation",
            origin_kind="compile_feedback",
            origin_id="cand",
            forbidden={"a": True, "b": True},
        )
    )
    result = FeasibilityPass().run(model)
    assert result.status is FeasibilityStatus.SAT
    assert not (result.assignment["a"] and result.assignment["b"])


def test_solver_absent_raises_structured_diagnostic(monkeypatch) -> None:
    import urm.compiler.solver as solver_module

    monkeypatch.setattr(solver_module, "z3", None)
    with pytest.raises(CompilerError) as excinfo:
        solver_module.FeasibilityPass().run(_tiny_model())
    assert excinfo.value.diagnostics[0].code is DiagnosticCode.SOLVER_UNAVAILABLE


def test_z3_version_recorded() -> None:
    version = z3_version()
    assert version is not None and version.count(".") >= 2
