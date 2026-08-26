"""Typed route protocols: push dispatch and pull gather stay separated.

The v0 prototype emitted source-owner -> query-owner traffic while tests
described token -> expert dispatch. These tests pin the corrected semantics:
each protocol plans its own direction, preserves every non-dropped route,
weights, stable order, exactly-one return, and declared capacity drops.
"""

from __future__ import annotations

import pytest

from urm.compiler.placement import DeviceMesh, PlacementBinding, PlacementMap
from urm.compiler.route_protocols import (
    RouteEdge,
    RouteProtocol,
    plan_by_protocol,
    plan_pull_gather,
    plan_push_dispatch_return,
    verify_plan_conservation,
)


def _placement(value_owners, query_owners, shape=(2, 2)) -> PlacementMap:
    mesh = DeviceMesh(name="sim", shape=shape)
    return PlacementMap(
        mesh=mesh,
        bindings=(
            PlacementBinding(
                tensor="experts", domain="expert", owner_devices=value_owners
            ),
            PlacementBinding(
                tensor="tokens", domain="sequence", owner_devices=query_owners
            ),
        ),
    )


def _dispatch_edges() -> list[RouteEdge]:
    # 4 tokens on device 0 dispatching to experts spread across devices 0..3.
    return [
        RouteEdge(
            query_id=token,
            peer_id=expert,
            weight=0.5 + 0.1 * token,
            ordinal=k,
            payload_type="hidden",
            payload_bytes=2,
            response_bytes=2,
            requires_return=True,
        )
        for token, expert in enumerate((0, 1, 2, 3))
        for k in (0,)
    ]


def test_push_dispatch_sends_token_owner_to_expert_owner() -> None:
    placement = _placement((0, 1, 2, 3), (0,))
    edges = [e for e in _dispatch_edges() if e.peer_id != 0]
    steps = plan_push_dispatch_return(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    dispatch_steps = [
        step
        for step in steps
        if step.kind == "exchange" and step.note.startswith("push_dispatch")
    ]
    assert len(dispatch_steps) == 1
    directions = {(ex.src_device, ex.dst_device) for ex in dispatch_steps[0].exchanges}
    # Token owner is device 0; payloads must FLOW OUT to expert owners.
    assert all(src == 0 and dst != 0 for src, dst in directions)


def test_pull_gather_moves_sources_toward_query_owner() -> None:
    placement = _placement((0, 1, 2, 3), (0,))
    edges = [
        RouteEdge(query_id=0, peer_id=5, weight=1.0, ordinal=0, payload_bytes=2)
    ]  # source 5 lives on device 1
    steps = plan_pull_gather(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    exchange_steps = [s for s in steps if s.kind == "exchange"]
    assert len(exchange_steps) == 1
    ex = exchange_steps[0].exchanges[0]
    assert (ex.src_device, ex.dst_device) == (1, 0)  # source owner -> query owner


def test_directions_are_not_conflated() -> None:
    """The same edge set yields opposite exchange directions per protocol."""
    placement = _placement((0, 1, 2, 3), (0,))
    edges = [
        RouteEdge(
            query_id=0,
            peer_id=1,
            requires_return=True,
            payload_type="hidden",
        )
    ]

    def directions(steps):
        out = []
        for step in steps:
            if step.kind != "exchange":
                continue
            for ex in step.exchanges:
                out.append((step.note.split(":")[0], ex.src_device, ex.dst_device))
        return sorted(out)

    push = plan_push_dispatch_return(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    pull = plan_pull_gather(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    push_dirs = [(note, s, d) for note, s, d in directions(push)]
    pull_dirs = [(note, s, d) for note, s, d in directions(pull)]
    # Dispatch leaves the token owner; gather arrives at it.
    assert any(note == "push_dispatch" and s == 0 for note, s, d in push_dirs)
    assert any(note == "pull_gather" and d == 0 for note, s, d in pull_dirs)


def test_exactly_one_return_per_dispatched_route() -> None:
    placement = _placement((0, 1, 2, 3), (0,))
    edges = _dispatch_edges()
    steps = plan_push_dispatch_return(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    violations = verify_plan_conservation(
        edges,
        placement,
        steps,
        protocol=RouteProtocol.PUSH_DISPATCH_RETURN,
        source_tensor="experts",
        query_tensor="tokens",
    )
    assert violations == []
    return_steps = [
        step
        for step in steps
        if step.kind == "exchange" and step.note.startswith("push_return")
    ]
    returned = sum(ex.payload_count for s in return_steps for ex in s.exchanges)
    dispatched = sum(
        ex.payload_count
        for s in steps
        if s.kind == "exchange"
        for ex in s.exchanges
        if ex.grouped_key.startswith("dispatch")
    )
    assert dispatched == returned == len([e for e in edges if e.peer_id != 0])


def test_weights_and_stable_order_are_carried_on_edges() -> None:
    edges = _dispatch_edges()
    assert all(edge.weight > 0 for edge in edges)
    ordinals = [edge.ordinal for edge in edges]
    assert ordinals == sorted(ordinals)  # stable semantic order


def test_capacity_drops_remove_exactly_the_declared_edges() -> None:
    placement = _placement((0, 1, 2, 3), (0,))
    edges = _dispatch_edges()
    edges_with_drop = [
        (
            RouteEdge(
                query_id=edge.query_id,
                peer_id=edge.peer_id,
                weight=edge.weight,
                ordinal=edge.ordinal,
                payload_type=edge.payload_type,
                payload_bytes=edge.payload_bytes,
                response_bytes=edge.response_bytes,
                requires_return=edge.requires_return,
                dropped=(index == 1),
            )
        )
        for index, edge in enumerate(edges)
    ]
    steps = plan_push_dispatch_return(
        edges_with_drop, placement, source_tensor="experts", query_tensor="tokens"
    )
    dispatch_sent = sum(
        ex.payload_count
        for step in steps
        if step.kind == "exchange"
        for ex in step.exchanges
        if ex.grouped_key.startswith("dispatch")
    )
    return_sent = sum(
        ex.payload_count
        for step in steps
        if step.kind == "exchange"
        for ex in step.exchanges
        if ex.grouped_key.startswith("return")
    )
    live_remote = sum(1 for e in edges_with_drop if not e.dropped and e.peer_id != 0)
    assert dispatch_sent == return_sent == live_remote


def test_merge_policy_is_visible_in_plan_notes() -> None:
    placement = _placement((0,), (0,))
    edges = [
        RouteEdge(query_id=0, peer_id=0, merge_policy="ordered"),
    ]
    steps = plan_push_dispatch_return(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    notes = " ".join(step.note or "" for step in steps)
    assert "ordered" in notes or "collision policy" in notes


def test_reserved_protocols_decline_explicitly() -> None:
    placement = _placement((0,), (0,))
    with pytest.raises(Exception, match="no planner"):
        plan_by_protocol(
            RouteProtocol.PAGE_REQUEST_RESPONSE,
            [RouteEdge(query_id=0, peer_id=0)],
            placement,
        )


def test_local_only_routes_never_invent_communication() -> None:
    placement = _placement((0,), (0,))
    edges = [
        RouteEdge(query_id=q, peer_id=s, requires_return=(s == 0))
        for q in range(2)
        for s in range(1)
    ]
    steps = plan_push_dispatch_return(
        edges, placement, source_tensor="experts", query_tensor="tokens"
    )
    assert all(step.kind != "exchange" for step in steps)
