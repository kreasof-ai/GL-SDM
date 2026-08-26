"""Independent model verification with intentionally invalid fake models.

Every rejection here is produced WITHOUT any solver: assignments are
hand-constructed to violate exactly one check, proving the verifier catches
broken models that a buggy translation or solver bug could emit.
"""

from __future__ import annotations

from urm.compiler.constraints import (
    BoolVar,
    ConstraintCategory,
    ConstraintModel,
    Divisibility,
    ExactlyOne,
    IntVar,
    Origin,
)
from urm.compiler.execution import TRUSTED_ANCHORS
from urm.compiler.kernel_plan import verify_schedule_assignment
from urm.compiler.planner import (
    BASE_CANDIDATE_ID,
    CompilationIntent,
    ScheduleParams,
    UrmCompiler,
)
from urm.compiler.semantic import row_scaled_routed_reduction_program
from urm.compiler.verification import (
    AnchorFacts,
    AssignmentFacts,
    ModelVerifier,
    ResourceFacts,
)


def _schedule_assignment():
    """Build a model and derive an assignment WITHOUT any solver."""
    from urm.compiler.kernel_plan import exhaustive_schedule_sweep

    compiler = UrmCompiler()
    program = row_scaled_routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=16
    )
    model = compiler.build_constraints(program, BASE_CANDIDATE_ID)
    legal, ranked, _total = exhaustive_schedule_sweep(model)
    assert legal, "expected at least one legal schedule"
    return model, ranked[0][0]


def test_verifier_accepts_valid_solver_assignment() -> None:
    model, assignment = _schedule_assignment()
    report = verify_schedule_assignment(model, assignment)
    assert report.ok
    assert "variables_in_range" in report.checks_run


def test_out_of_range_value_is_rejected() -> None:
    model, assignment = _schedule_assignment()
    broken = {**assignment, "block_d": 999}
    report = ModelVerifier().verify(model, broken)
    assert not report.ok
    assert any(f.check == "variables_in_range" for f in report.failures)


def test_missing_and_unknown_variables_are_rejected() -> None:
    model, assignment = _schedule_assignment()
    missing = dict(assignment)
    missing.pop("block_d")
    report = ModelVerifier().verify(model, missing)
    assert not report.ok

    extra = {**assignment, "ghost_var": True}
    report_extra = ModelVerifier().verify(model, extra)
    assert not report_extra.ok


def test_divisibility_violation_is_rejected() -> None:
    model = ConstraintModel(name="div")
    model.add_variable(IntVar("block", 32, 256))
    model.add_constraint(
        Divisibility(
            name="aligned",
            category=ConstraintCategory.SCHEDULE,
            explanation="multiple of 32",
            origin=Origin(kind="schedule_param", id="b"),
            variable="block",
            divisor=32,
        )
    )
    report = ModelVerifier().verify(model, {"block": 48})
    assert not report.ok
    assert any("aligned" in f.message for f in report.failures)


def test_forward_only_anchor_rejects_training_facts() -> None:
    model = ConstraintModel(name="facts")
    model.add_variable(BoolVar("x"))
    facts = AssignmentFacts(
        selected_anchor=AnchorFacts(
            name="forward_only_anchor",
            kind="routed_reduction",
            trusted=True,
            forward_only=True,
        ),
        intent_training=True,
        required_backward_dtypes=frozenset({"bfloat16"}),
    )
    report = ModelVerifier().verify(model, {"x": True}, facts)
    assert not report.ok
    checks = {failure.check for failure in report.failures}
    assert "training_backward_compatibility" in checks


def test_untrusted_anchor_is_rejected() -> None:
    model = ConstraintModel(name="untrusted")
    model.add_variable(BoolVar("x"))
    facts = AssignmentFacts(
        selected_anchor=AnchorFacts(
            name="experimental_unverified",
            kind="grouped_gemm",
            trusted=False,
        ),
    )
    report = ModelVerifier().verify(model, {"x": True}, facts)
    assert not report.ok
    assert any(f.check == "anchor_trusted" for f in report.failures)


def test_shared_memory_ceiling_is_enforced() -> None:
    model = ConstraintModel(name="smem")
    model.add_variable(BoolVar("x"))
    facts = AssignmentFacts(
        resource_limits=ResourceFacts(max_shared_mem_bytes_per_block=64 * 1024),
        estimated_shared_mem_bytes=96 * 1024,
    )
    report = ModelVerifier().verify(model, {"x": True}, facts)
    assert not report.ok
    assert any(f.check == "shared_memory_limits" for f in report.failures)


def test_communication_conservation_requires_exactly_one_return() -> None:
    from urm.compiler.verification import RouteEdgeFacts

    routes = (
        RouteEdgeFacts(query_id=0, peer_id=1, ordinal=0, requires_return=True),
        RouteEdgeFacts(query_id=1, peer_id=2, ordinal=0, requires_return=True),
    )
    model = ConstraintModel(name="comm")
    model.add_variable(BoolVar("x"))

    good = AssignmentFacts(
        routes=routes,
        dispatched_counts={(0, 1): 1, (1, 2): 1},
        returned_counts={0: 1, 1: 1},
    )
    bad = AssignmentFacts(
        routes=routes,
        dispatched_counts={(0, 1): 1, (1, 2): 1},
        returned_counts={0: 1},  # query 1's required return is lost
    )
    assert ModelVerifier().verify(model, {"x": True}, good).ok
    report_bad = ModelVerifier().verify(model, {"x": True}, bad)
    assert not report_bad.ok
    assert any(f.check == "communication_conservation" for f in report_bad.failures)


def test_placement_ownership_and_capacity_checked_without_solver() -> None:
    from urm.compiler.verification import PlacementItemFacts

    model = ConstraintModel(name="place")
    for name in ("x", "owner_e0", "owner_e1"):
        model.add_variable(IntVar(name, 0, 3))
    items = (
        PlacementItemFacts(name="e0", size_bytes=900, owner_variable="owner_e0"),
        PlacementItemFacts(name="e1", size_bytes=900, owner_variable="owner_e1"),
    )

    good = AssignmentFacts(
        devices=(0, 1),
        device_capacity_bytes={0: 1000, 1: 1000},
        items=items,
    )
    good_assignment = {"x": 0, "owner_e0": 0, "owner_e1": 1}
    assert ModelVerifier().verify(model, good_assignment, good).ok

    overloaded = AssignmentFacts(
        devices=(0,),
        device_capacity_bytes={0: 1000},
        items=items,
    )
    report = ModelVerifier().verify(
        model, {"x": 0, "owner_e0": 0, "owner_e1": 0}, overloaded
    )
    assert not report.ok
    checks = {f.check for f in report.failures}
    assert "capacity" in checks

    unknown_device = ModelVerifier().verify(
        model, {"x": 0, "owner_e0": 3, "owner_e1": 0}, overloaded
    )
    assert not unknown_device.ok
    assert any(f.check == "placement_ownership" for f in unknown_device.failures)


def test_nogood_added_for_verification_failures_and_respected() -> None:
    from urm.compiler.diagnostics import Severity
    from urm.compiler.kernel_plan import exhaustive_schedule_sweep
    from urm.compiler.verification import (
        VerificationFailure,
        add_nogood_for_failures,
    )

    model, assignment = _schedule_assignment()
    failures = (
        VerificationFailure(
            check="injected",
            message="fake rejection for testing",
            severity=Severity.ERROR,
        ),
    )
    assert add_nogood_for_failures(model, assignment, failures, origin_id="t")
    # The model now carries the exact nogood for the rejected assignment.
    nogood = model.constraints[-1]
    assert not nogood.holds(assignment)  # fires on exactly that assignment
    # Every other legal assignment survives the nogood.
    legal, _ranked, _total = exhaustive_schedule_sweep(model)
    assert any(candidate != assignment for candidate in legal)


def test_trusted_anchor_catalog_stays_backward_capable() -> None:
    epilogue = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.name == "routed_reduction_row_scale_epilogue_v0"
    )
    assert epilogue.honored_obligations >= {"recompute_backward"}
    v1 = next(
        anchor for anchor in TRUSTED_ANCHORS if anchor.name == "routed_reduction_v1"
    )
    assert not v1.forward_only


def test_exactly_one_violation_detected_imperatively() -> None:
    model = ConstraintModel(name="onehot")
    for name in ("a", "b"):
        model.add_variable(BoolVar(name))
    model.add_constraint(
        ExactlyOne(
            name="pick",
            category=ConstraintCategory.SEARCH,
            explanation="exactly one",
            origin=Origin(kind="intent", id="t"),
            choices=("a", "b"),
        )
    )
    assert ModelVerifier().verify(model, {"a": True, "b": False}).ok
    assert not ModelVerifier().verify(model, {"a": False, "b": False}).ok
    assert not ModelVerifier().verify(model, {"a": True, "b": True}).ok


def test_build_schedule_model_metadata_roundtrip() -> None:
    compiler = UrmCompiler()
    program = row_scaled_routed_reduction_program(
        queries=8, route_width=2, sources=8, value_dim=16
    )
    model = compiler.build_constraints(
        program,
        BASE_CANDIDATE_ID,
        CompilationIntent.TRAINING,
        schedule_params=ScheduleParams(deterministic=True),
    )
    assert model.metadata["intent"] == "training"
    assert model.metadata["deterministic"] == "1"
    summary_hash_a = model.summary_hash()
    summary_hash_b = model.summary_hash()
    assert summary_hash_a == summary_hash_b
