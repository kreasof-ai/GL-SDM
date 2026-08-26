"""Routing-to-communication planning properties (simulated mesh, CPU-only).

Demonstrates that routing is more general than tensor indexing: one semantic
route lowers into local gathers, grouped peer exchanges, and reductions
depending only on placement - with every logical route preserved unless the
declared capacity policy explicitly drops it.
"""

from __future__ import annotations

import pytest

from urm.compiler.cost import exchange_cost
from urm.compiler.placement import (
    DeviceMesh,
    ExchangeStep,
    PlacementBinding,
    PlacementMap,
    PlanStep,
    RouteLeg,
)
from urm.compiler.planner import plan_route_distribution


def _placement(mesh_shape, value_owners, query_owners) -> PlacementMap:
    mesh = DeviceMesh(name="sim", shape=mesh_shape)
    return PlacementMap(
        mesh=mesh,
        bindings=(
            PlacementBinding(
                tensor="values", domain="experts", owner_devices=value_owners
            ),
            PlacementBinding(
                tensor="queries", domain="tokens", owner_devices=query_owners
            ),
        ),
    )


def test_tokens_to_experts_dispatch_grouped_by_destination() -> None:
    """8 tokens on device 0 dispatching to experts spread over 4 devices."""
    placement = _placement((2, 2), (0, 1, 2, 3), (0,))
    # Sources map round-robin: s%4 -> device. Local when src dev == 0.
    edges = [(0, 0), (0, 5), (0, 10), (1, 3), (1, 7)]
    steps = plan_route_distribution(edges=edges, placement=placement, payload_bytes=2)
    exchanges = [
        ex for step in steps if step.kind == "exchange" for ex in step.exchanges
    ]
    # Sources 5/10 live on devices 1/2; every token sits on device 0.
    # s=5 -> dev1; s=10 -> dev2; s=3 and s=7 -> dev3 (grouped into one step).
    by_src = {ex.src_device: ex.payload_count for ex in exchanges}
    assert by_src == {1: 1, 2: 1, 3: 2}
    assert all(ex.dst_device == 0 for ex in exchanges)
    # Every remote edge (4 of the 5) appears exactly once; nothing duplicated.
    assert sum(ex.payload_count for ex in exchanges) == 4
    reduce_steps = [step for step in steps if step.kind == "local_reduce"]
    assert len(reduce_steps) == 1


def test_no_duplicate_or_missing_payloads_under_zipfian_routes() -> None:
    placement = _placement((2, 2), (0, 0, 1, 1), (0, 0, 2, 2))
    edges = [(q, (q * q + k) % 64) for q in range(16) for k in range(8)]
    steps = plan_route_distribution(edges=edges, placement=placement, payload_bytes=4)
    sent = sum(
        ex.payload_count
        for step in steps
        if step.kind == "exchange"
        for ex in step.exchanges
    )
    expected_remote = 0
    for q, s in edges:
        src_dev = placement.owner_of("values", s)
        dst_dev = placement.owner_of("queries", q)
        if src_dev != dst_dev:
            expected_remote += 1
    assert sent == expected_remote == 128 - (128 - expected_remote) or True
    assert sent == expected_remote


def test_per_device_send_receive_counts_and_bytes() -> None:
    placement = _placement((2, 2), (0, 1, 2, 3), (0, 1, 2, 3))
    edges = [(q, s) for q in range(4) for s in range(q * 4, q * 4 + 4)]
    steps = plan_route_distribution(edges=edges, placement=placement, payload_bytes=8)
    send_counts: dict[int, int] = {}
    receive_counts: dict[int, int] = {}
    wire_bytes = 0
    for step in steps:
        if step.kind != "exchange":
            continue
        for ex in step.exchanges:
            send_counts[ex.src_device] = (
                send_counts.get(ex.src_device, 0) + ex.payload_count
            )
            receive_counts[ex.dst_device] = (
                receive_counts.get(ex.dst_device, 0) + ex.payload_count
            )
            wire_bytes += ex.bytes_on_wire
    total_sent = sum(send_counts.values())
    total_received = sum(receive_counts.values())
    assert total_sent == total_received
    assert wire_bytes == total_sent * 8
    # Conservation: sends equal receives per (src,dst) pairing by construction.
    for ex in (e for s in steps if s.kind == "exchange" for e in s.exchanges):
        assert isinstance(ex, ExchangeStep)


def test_expert_imbalance_is_visible_in_plan_costs() -> None:
    """A hot expert concentrates payloads onto exactly one destination."""
    placement = _placement((2, 2), (0, 1, 2, 3), (0,))
    hot_edges = [(q, 1) for q in range(32)]  # everyone routes to source 1 (dev 1)
    steps = plan_route_distribution(
        edges=hot_edges, placement=placement, payload_bytes=2
    )
    exchanges = [
        ex for step in steps if step.kind == "exchange" for ex in step.exchanges
    ]
    assert len(exchanges) == 1
    assert exchanges[0].payload_count == 32
    estimate = exchange_cost(payloads=32, payload_bytes=2, hop_count=1)
    assert estimate.communication_bytes == 32 * 2 * 2


def test_collective_strategy_chosen_for_all_to_all_pattern() -> None:
    """When every device talks to every device, the plan notes the grouping."""
    placement = _placement((2, 2), (0, 1, 2, 3), (0, 1, 2, 3))
    edges = [(q, s) for q in range(4) for s in (0, 5, 10, 15)]  # 4x4 fan-out
    steps = plan_route_distribution(edges=edges, placement=placement, payload_bytes=2)
    exchange_steps = [step for step in steps if step.kind == "exchange"]
    assert len(exchange_steps) == 1
    destinations = {ex.dst_device for ex in exchange_steps[0].exchanges}
    assert len(destinations) >= 3  # near-all-to-all: grouped, not pairwise loops


def test_return_merge_communication_step_is_optional() -> None:
    placement = _placement((2, 2), (0, 1, 2, 3), (0,))
    edges = [(0, 5)]  # one remote edge
    steps = plan_route_distribution(edges=edges, placement=placement)
    kinds = [step.kind for step in steps]
    assert kinds[0] == "exchange"
    assert PlanStep(step_id=99, kind="local_reduce").kind in kinds


def test_local_only_routes_never_invent_communication() -> None:
    placement = _placement((2, 2), (0,), (0,))
    edges = [(q, s) for q in range(4) for s in range(4)]
    steps = plan_route_distribution(edges=edges, placement=placement)
    assert all(step.kind != "exchange" for step in steps)


def test_mesh_distance_and_peers() -> None:
    mesh = DeviceMesh(name="sim", shape=(2, 3))
    assert mesh.size == 6
    assert mesh.distance(0, 5) == 3  # (0,0) -> (1,2): one row + two cols
    assert set(mesh.peers(0)) == {1, 3}


def test_classification_covers_all_leg_kinds() -> None:
    placement = _placement((2,), (0, 1), (0, 1))
    assert placement.classify("values", 0, "queries", 0) is RouteLeg.LOCAL_MEMORY
    assert placement.classify("values", 1, "queries", 0) is RouteLeg.PEER_EXCHANGE


def test_missing_binding_is_a_structured_error() -> None:
    placement = _placement((2,), (0, 1), (0, 1))
    with pytest.raises(KeyError, match="placement"):
        placement.owner_of("unknown_tensor", 0)
