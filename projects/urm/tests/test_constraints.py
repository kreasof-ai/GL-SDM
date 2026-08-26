"""Backend-independent constraint IR: construction, evaluation, serialization."""

from __future__ import annotations

import json

import pytest

from urm.compiler.constraints import (
    AllowedSet,
    Assignment,
    AtMostOne,
    BoolVar,
    ConstraintCategory,
    ConstraintModel,
    Divisibility,
    EnumVar,
    Equality,
    ExactlyOne,
    IntVar,
    LessEqual,
    LinearExpr,
    ModelValidationError,
    Nogood,
    ObjectiveSense,
    ObjectiveTerm,
    Origin,
    implies_equal,
    implies_true,
    make_nogood,
)


def test_linear_expression_algebra_and_evaluation() -> None:
    expr = LinearExpr.var("a") * 3 + LinearExpr.const(2) - LinearExpr.var("b")
    assignment: Assignment = {"a": 4, "b": 1}
    assert expr.evaluate(assignment) == 13
    assert (expr * 2).evaluate(assignment) == 26
    negated = -expr
    assert negated.evaluate(assignment) == -13
    assert json.loads(json.dumps(expr.describe()))


def test_model_rejects_unknown_variable_references() -> None:
    model = ConstraintModel(name="m")
    model.add_variable(BoolVar("x"))
    with pytest.raises(ModelValidationError, match="unregistered"):
        model.add_constraint(
            Equality(
                name="bad",
                category=ConstraintCategory.SCHEDULE,
                explanation="references unknown variable",
                origin=Origin(kind="schedule_param", id="x"),
                lhs=LinearExpr.var("nope"),
            )
        )


def test_every_constraint_kind_evaluates_imperatively() -> None:
    model = ConstraintModel(name="kinds")
    for name in ("p", "q", "r"):
        model.add_variable(BoolVar(name))
    model.add_variable(IntVar("n", 0, 64))
    model.add_variable(EnumVar("mode", ("fast", "safe")))
    origin = Origin(kind="schedule_param", id="test")

    model.add_constraint(
        Equality(
            name="eq",
            category=ConstraintCategory.SCHEDULE,
            explanation="",
            origin=origin,
            lhs=LinearExpr.var("n"),
            rhs=LinearExpr.const(32),
        )
    )
    model.add_constraint(
        LessEqual(
            name="le",
            category=ConstraintCategory.RESOURCE,
            explanation="",
            origin=origin,
            lhs=LinearExpr.var("n"),
            rhs=LinearExpr.const(40),
        )
    )
    model.add_constraint(
        Divisibility(
            name="div",
            category=ConstraintCategory.SCHEDULE,
            explanation="",
            origin=origin,
            variable="n",
            divisor=32,
        )
    )
    model.add_constraint(
        AllowedSet(
            name="allowed",
            category=ConstraintCategory.SCHEDULE,
            explanation="",
            origin=origin,
            variable="n",
            allowed=(32, 64),
        )
    )
    model.add_constraint(
        implies_true(ConstraintHeader_stub("imp", "guard n==32 implies mode safe"), "p")
        .then_equal(LinearExpr.var("q"), 1)
        .done()
    )
    model.add_constraint(
        AtMostOne(
            name="amo",
            category=ConstraintCategory.SEARCH,
            explanation="",
            origin=origin,
            choices=("p", "q"),
        )
    )
    model.add_constraint(
        ExactlyOne(
            name="eo_mode",
            category=ConstraintCategory.SCHEDULE,
            explanation="",
            origin=origin,
            choices=("r",),
        )
    )
    good: Assignment = {"p": False, "q": False, "r": True, "n": 32, "mode": "safe"}
    bad_div: Assignment = {**good, "n": 48}
    bad_guard: Assignment = {**good, "p": True, "q": False}
    assert all(c.holds(good) for c in model.constraints)
    assert not all(c.holds(bad_div) for c in model.constraints)
    assert not all(c.holds(bad_guard) for c in model.constraints)
    model.validate()


def ConstraintHeader_stub(name: str, explanation: str):
    from urm.compiler.constraints import ConstraintHeader

    return ConstraintHeader(
        name=name,
        category=ConstraintCategory.TRAINING,
        explanation=explanation,
        origin=Origin(kind="intent", id="test"),
    )


def test_nogood_fires_only_on_exact_match() -> None:
    ng: Nogood = make_nogood(
        name="ng1",
        explanation="rejected schedule",
        origin_kind="compile_feedback",
        origin_id="cand-1",
        forbidden={"block_d": 128, "fused": True},
    )
    assert ng.fires({"block_d": 128, "fused": True})
    assert not ng.fires({"block_d": 64, "fused": True})
    assert ng.holds({"block_d": 64, "fused": False})


def test_implication_builder_supports_nested_and_contradiction() -> None:
    header = ConstraintHeader_stub("nested", "plan guard implies mode guard")
    implication = (
        implies_equal(header, LinearExpr.var("plan"), 0)
        .then_equal(LinearExpr.var("mode"), 1)
        .done()
    )
    assert implication.guard_holds({"plan": 0})
    assert not implication.guard_holds({"plan": 1})

    contradiction = (
        implies_true(ConstraintHeader_stub("contra", "forbidden combo"), "flag")
        .then_contradiction()
        .done()
    )
    assert not contradiction.holds({"flag": True})
    assert contradiction.holds({"flag": False})


def test_model_summary_is_serializable_and_stable() -> None:
    model = ConstraintModel(name="summary", metadata={"intent": "training"})
    model.add_variable(BoolVar("x"))
    model.add_constraint(
        ExactlyOne(
            name="one_x",
            category=ConstraintCategory.SEARCH,
            explanation="pick x",
            origin=Origin(kind="intent", id="t"),
            choices=("x",),
        )
    )
    model.add_objective(
        ObjectiveTerm("obj", LinearExpr.var("x"), ObjectiveSense.MINIMIZE)
    )
    summary = model.to_summary()
    payload = json.loads(json.dumps(summary))
    assert payload["metadata"]["intent"] == "training"
    assert payload["constraints"][0]["name"] == "one_x"
    assert model.summary_hash() == model.summary_hash()
    # Every constraint carries provenance and severity.
    constraint_record = payload["constraints"][0]
    assert {"name", "category", "explanation", "origin", "severity"} <= set(
        constraint_record
    )


def test_objective_terms_validate_variables() -> None:
    model = ConstraintModel(name="objs")
    model.add_variable(BoolVar("x"))
    with pytest.raises(ModelValidationError, match="objective"):
        model.add_objective(
            ObjectiveTerm("bad", LinearExpr.var("ghost"), ObjectiveSense.MINIMIZE)
        )
