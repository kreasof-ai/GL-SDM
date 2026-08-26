"""Solver-guided simulated placement: capacity, baselines, exhaustive parity."""

from __future__ import annotations

import pytest

from urm.compiler.placement_solver import (
    PlacementEdge,
    PlacementItem,
    PlacementProblem,
    build_placement_model,
    decode_placement,
    exhaustive_placement_optimum,
    feasible_owners,
    greedy_placement,
    placement_metrics,
    round_robin_placement,
)
from urm.compiler.solver import OptimizationPass, z3_available
from urm.compiler.verification import (
    AssignmentFacts,
    ModelVerifier,
    PlacementItemFacts,
)

pytestmark = pytest.mark.skipif(
    not z3_available(), reason="z3-solver optional extra not installed"
)


def _problem(rows: int, cols: int, items: int, capacity: int) -> PlacementProblem:
    names = [f"e{i}" for i in range(items)]
    return PlacementProblem(
        items=tuple(
            PlacementItem(name=name, size_bytes=100 + 10 * index)
            for index, name in enumerate(names)
        ),
        device_count=rows * cols,
        device_capacity_bytes=capacity,
        mesh_rows=rows,
        mesh_cols=cols,
        edges=(
            PlacementEdge(source_item=names[i], query_item=names[(i + 1) % items])
            for i in range(items - 1)
        ),
    )


def test_capacity_is_respected_on_2x2() -> None:
    problem = _problem(2, 2, items=4, capacity=400)
    model = build_placement_model(problem)
    result = OptimizationPass().run(model)
    assert result.status.value == "sat"
    owners = decode_placement(result.assignment, problem)
    assert feasible_owners(owners, problem)


def test_infeasible_when_no_device_has_room() -> None:
    problem = _problem(2, 2, items=4, capacity=90)
    model = build_placement_model(problem)
    result = OptimizationPass().run(model)
    assert result.status.value == "unsat"
    # The core names the capacity constraints that cannot all hold.
    assert any("device_capacity" in name for name in result.unsat_core_names)


def test_z3_matches_exhaustive_optimum_on_tiny_instance() -> None:
    problem = _problem(2, 2, items=4, capacity=10_000)
    model = build_placement_model(problem)
    z3_result = OptimizationPass().run(model)
    owners_z3 = decode_placement(z3_result.assignment, problem)
    metrics_z3 = placement_metrics(owners_z3, problem)

    exhaustive = exhaustive_placement_optimum(problem)
    assert exhaustive is not None
    owners_ex, metrics_ex = exhaustive
    assert tuple(
        metrics_z3[key]
        for key in ("max_load", "critical_path_proxy", "total_bytes", "peer_pairs")
    ) == (
        metrics_ex["max_load"],
        metrics_ex["critical_path_proxy"],
        metrics_ex["total_bytes"],
        metrics_ex["peer_pairs"],
    )
    assert owners_z3 == owners_ex  # deterministic tie-break agrees


def test_matches_or_beats_baselines_on_skewed_sizes() -> None:
    problem = PlacementProblem(
        items=(
            PlacementItem("big0", size_bytes=900),
            PlacementItem("big1", size_bytes=900),
            PlacementItem("small", size_bytes=50),
        ),
        device_count=4,
        device_capacity_bytes=1000,
        mesh_rows=2,
        mesh_cols=2,
    )
    rr = round_robin_placement(problem)
    greedy = greedy_placement(problem)
    assert feasible_owners(greedy, problem)
    model = build_placement_model(problem)
    z3_owners = decode_placement(OptimizationPass().run(model).assignment, problem)
    # The solver minimizes max load: never worse than round-robin.
    assert (
        placement_metrics(z3_owners, problem)["max_load"]
        <= placement_metrics(rr, problem)["max_load"]
    )
    # With no edges, greedy achieves the same optimal max load here.
    assert (
        placement_metrics(greedy, problem)["max_load"]
        == placement_metrics(z3_owners, problem)["max_load"]
    )


def test_colocation_and_anti_affinity_honored() -> None:
    problem = PlacementProblem(
        items=tuple(PlacementItem(name, size_bytes=10) for name in ("a", "b", "c")),
        device_count=4,
        device_capacity_bytes=10_000,
        mesh_rows=2,
        mesh_cols=2,
        colocated_pairs=(("a", "b"),),
        anti_affine_pairs=(("a", "c"),),
    )
    model = build_placement_model(problem)
    result = OptimizationPass().run(model)
    owners = decode_placement(result.assignment, problem)
    assert owners["a"] == owners["b"]
    assert owners["a"] != owners["c"]


def test_replication_factor_two_places_each_item_twice() -> None:
    problem = PlacementProblem(
        items=(PlacementItem("p0", size_bytes=100),),
        device_count=4,
        device_capacity_bytes=10_000,
        mesh_rows=2,
        mesh_cols=2,
        replication_factor=2,
    )
    model = build_placement_model(problem)
    result = OptimizationPass().run(model)
    assert result.status.value == "sat"
    true_copies = [
        device
        for device in range(4)
        if result.assignment[f"assign_p0_d{device}"] in (True, 1)
    ]
    assert len(true_copies) == 2


def test_returned_plans_pass_independent_verification() -> None:
    problem = _problem(2, 2, items=4, capacity=500)
    model = build_placement_model(problem)
    result = OptimizationPass().run(model)
    assert result.status.value == "sat"
    facts = AssignmentFacts(
        devices=tuple(range(problem.device_count)),
        device_capacity_bytes={
            device: problem.device_capacity_bytes for device in problem.devices()
        },
        items=tuple(
            PlacementItemFacts(
                name=item.name,
                size_bytes=item.size_bytes,
                owner_variable="unused",
                one_hot_devices=tuple(range(problem.device_count)),
            )
            for item in problem.items
        ),
    )
    report = ModelVerifier().verify(model, result.assignment, facts)
    assert report.ok


def test_mesh_topology_distances_shape_critical_path() -> None:
    problem = _problem(2, 4, items=3, capacity=10_000)
    assert problem.distance(0, 1) == 1
    assert problem.distance(0, 3) == 3
    assert problem.distance(0, 7) == 1 + 3
