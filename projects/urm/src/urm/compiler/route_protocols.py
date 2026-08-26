"""Typed distributed route protocols.

The v0 communication prototype conflated two different traffic directions:
tests described token-to-expert dispatch while the planner emitted
source-owner -> query-owner gather steps. This module separates them.

Protocols:

- ``PULL_GATHER``: the query owner references remote source rows; source
  owners send payloads toward query owners; the local weighted reduction
  happens at the query owner.
- ``PUSH_DISPATCH_RETURN``: the token owner pushes token payloads to expert
  owners; expert computation is local; results return to the token owner;
  weighted merges happen in stable semantic order where required.
- ``COLLECTIVE_EXCHANGE``, ``PAGE_REQUEST_RESPONSE``, ``STATE_REPLICATION``,
  ``TRANSACTIONAL_UPDATE``: reserved typed intents for the corresponding
  semantic ops; planned as first-class communication steps, never disguised
  local tensor work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from urm.compiler.diagnostics import DiagnosticCode
from urm.compiler.placement import ExchangeStep, PlacementMap, PlanStep, RouteLeg


class RouteProtocol(StrEnum):
    PULL_GATHER = "pull_gather"
    PUSH_DISPATCH_RETURN = "push_dispatch_return"
    COLLECTIVE_EXCHANGE = "collective_exchange"
    PAGE_REQUEST_RESPONSE = "page_request_response"
    STATE_REPLICATION = "state_replication"
    TRANSACTIONAL_UPDATE = "transactional_update"


@dataclass(frozen=True, slots=True)
class RouteEdge:
    """One logical route edge with its full protocol contract."""

    query_id: int  # logical token/query index
    peer_id: int  # logical source/expert/page index
    weight: float = 1.0
    ordinal: int = 0  # stable position within the query's route
    payload_type: str = "hidden"  # e.g. hidden | expert_partial | page_value
    payload_bytes: int = 2
    response_bytes: int = 0  # >0 when the protocol requires a return
    requires_return: bool = False
    dropped: bool = False  # declared capacity-policy drop
    merge_policy: str = "sum"  # sum | mean | last_write | ordered | reject
    capacity_policy: str = "dropless"  # dropless | fixed_capacity | expert_quota


def classify_edge(
    edge: RouteEdge,
    placement: PlacementMap,
    *,
    source_tensor: str,
    query_tensor: str,
) -> RouteLeg:
    if edge.dropped:
        return RouteLeg.LOCAL_MEMORY  # dropped edges generate no traffic
    return placement.classify(source_tensor, edge.peer_id, query_tensor, edge.query_id)


# -- Pull gather -----------------------------------------------------------------


def plan_pull_gather(
    edges: list[RouteEdge],
    placement: PlacementMap,
    *,
    source_tensor: str = "values",
    query_tensor: str = "queries",
) -> tuple[PlanStep, ...]:
    """Query-owner-centric gather: sources move to queries; reduce locally."""
    live = [edge for edge in edges if not edge.dropped]
    grouped: dict[tuple[int, int], list[RouteEdge]] = {}
    local_count = 0
    for edge in live:
        leg = classify_edge(
            edge, placement, source_tensor=source_tensor, query_tensor=query_tensor
        )
        if leg is RouteLeg.PEER_EXCHANGE:
            src = placement.owner_of(source_tensor, edge.peer_id)
            dst = placement.owner_of(query_tensor, edge.query_id)
            grouped.setdefault((src, dst), []).append(edge)
        else:
            local_count += 1
    steps: list[PlanStep] = []
    step_id = 0
    if grouped:
        exchanges = []
        for (src, dst), members in sorted(grouped.items()):
            exchanges.append(
                ExchangeStep(
                    step_id=len(exchanges),
                    src_device=src,
                    dst_device=dst,
                    payload_count=len(members),
                    payload_bytes=max(m.payload_bytes for m in members),
                    grouped_key=f"gather:dst:{dst}",
                )
            )
        steps.append(
            PlanStep(
                step_id=step_id,
                kind="exchange",
                exchanges=tuple(exchanges),
                note="pull_gather: source rows move to query owners",
            )
        )
        step_id += 1
    steps.append(
        PlanStep(
            step_id=step_id,
            kind="local_reduce",
            note=(
                f"{local_count} co-resident gathers; {len(live)} routes "
                "reduced at query owners in stable ordinal order"
            ),
        )
    )
    return tuple(steps)


# -- Push dispatch + return --------------------------------------------------------


def plan_push_dispatch_return(
    edges: list[RouteEdge],
    placement: PlacementMap,
    *,
    source_tensor: str = "experts",
    query_tensor: str = "tokens",
) -> tuple[PlanStep, ...]:
    """Token-owner-centric MoE dispatch.

    1. token owners push token payloads to expert owners (grouped);
    2. expert computation is local to the expert owner;
    3. results return to token owners - exactly one required return per route;
    4. weighted merge in stable semantic order at the token owner.
    """
    live = [edge for edge in edges if not edge.dropped]
    dispatch: dict[tuple[int, int], list[RouteEdge]] = {}
    returns: dict[tuple[int, int], list[RouteEdge]] = {}
    local_routes = 0
    for edge in live:
        expert_dev = placement.owner_of(source_tensor, edge.peer_id)
        token_dev = placement.owner_of(query_tensor, edge.query_id)
        if expert_dev == token_dev:
            local_routes += 1
            continue
        dispatch.setdefault((token_dev, expert_dev), []).append(edge)
        returns.setdefault((expert_dev, token_dev), []).append(edge)

    steps: list[PlanStep] = []
    step_id = 0

    def _exchange_group(
        group: dict[tuple[int, int], list[RouteEdge]], note: str, key_prefix: str
    ) -> PlanStep:
        exchanges = []
        for (src, dst), members in sorted(group.items()):
            exchanges.append(
                ExchangeStep(
                    step_id=len(exchanges),
                    src_device=src,
                    dst_device=dst,
                    payload_count=len(members),
                    payload_bytes=max(m.payload_bytes for m in members),
                    grouped_key=f"{key_prefix}:dst:{dst}",
                )
            )
        return PlanStep(
            step_id=step_id, kind="exchange", exchanges=tuple(exchanges), note=note
        )

    if dispatch:
        steps.append(
            _exchange_group(
                dispatch,
                "push_dispatch: token payloads move to expert owners",
                "dispatch",
            )
        )
        step_id += 1
    steps.append(
        PlanStep(
            step_id=step_id,
            kind="anchor_dispatch",
            note=f"expert compute runs locally on expert owners "
            f"({local_routes} co-resident routes skip the network)",
        )
    )
    step_id += 1
    if returns:
        steps.append(
            _exchange_group(
                returns,
                "push_return: expert outputs return to token owners "
                "(exactly one return per dispatched route)",
                "return",
            )
        )
        step_id += 1
    merge_note = "weighted merge at token owners; stable ordinal order preserved"
    policies = sorted({edge.merge_policy for edge in live})
    if policies:
        merge_note += f"; collision policy {policies}"
    steps.append(PlanStep(step_id=step_id, kind="local_reduce", note=merge_note))
    return tuple(steps)


def plan_by_protocol(
    protocol: RouteProtocol, edges: list[RouteEdge], placement: PlacementMap, **kwargs
) -> tuple[PlanStep, ...]:
    if protocol is RouteProtocol.PUSH_DISPATCH_RETURN:
        return plan_push_dispatch_return(edges, placement, **kwargs)
    if protocol is RouteProtocol.PULL_GATHER:
        return plan_pull_gather(edges, placement, **kwargs)
    raise CompilerErrorForProtocol(protocol)


class CompilerErrorForProtocol(Exception):
    def __init__(self, protocol: RouteProtocol) -> None:
        self.protocol = protocol
        super().__init__(
            f"[{DiagnosticCode.PROTOCOL_VIOLATION.value}] protocol "
            f"{protocol.value} has no planner in this prototype; it must be "
            "planned explicitly rather than silently treated as local work"
        )


def verify_plan_conservation(
    edges: list[RouteEdge],
    placement: PlacementMap,
    steps: tuple[PlanStep, ...],
    *,
    protocol: RouteProtocol,
    source_tensor: str = "values",
    query_tensor: str = "queries",
) -> list[str]:
    """Structural conservation checks; returns human-readable violations."""
    violations: list[str] = []
    live = [edge for edge in edges if not edge.dropped]

    def _dispatched() -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for step in steps:
            if step.kind != "exchange":
                continue
            for exchange in step.exchanges:
                if exchange.grouped_key and exchange.grouped_key.startswith(
                    ("gather:", "dispatch:")
                ):
                    key = (exchange.src_device, exchange.dst_device)
                    counts[key] = counts.get(key, 0) + exchange.payload_count
        return counts

    expected_remote: dict[tuple[int, int], int] = {}
    expected_return: dict[tuple[int, int], int] = {}
    for edge in live:
        src_dev = placement.owner_of(source_tensor, edge.peer_id)
        dst_dev = placement.owner_of(query_tensor, edge.query_id)
        if src_dev == dst_dev:
            continue
        if protocol is RouteProtocol.PUSH_DISPATCH_RETURN:
            # Dispatch leaves the token owner; returns come back.
            dispatch_pair = (dst_dev, src_dev)
            return_pair = (src_dev, dst_dev)
        else:
            dispatch_pair = (src_dev, dst_dev)
            return_pair = None  # pull gather has no required return leg
        expected_remote[dispatch_pair] = expected_remote.get(dispatch_pair, 0) + 1
        if protocol is RouteProtocol.PUSH_DISPATCH_RETURN:
            assert return_pair is not None
            expected_return[return_pair] = expected_return.get(return_pair, 0) + 1
    got = _dispatched()
    if got != expected_remote:
        violations.append(
            f"remote traffic mismatch: plan {got} != expected {expected_remote}"
        )
    if protocol is RouteProtocol.PUSH_DISPATCH_RETURN:
        returned: dict[tuple[int, int], int] = {}
        for step in steps:
            if step.kind != "exchange":
                continue
            for exchange in step.exchanges:
                if exchange.grouped_key and exchange.grouped_key.startswith("return:"):
                    key = (exchange.src_device, exchange.dst_device)
                    returned[key] = returned.get(key, 0) + exchange.payload_count
        if returned != expected_return:
            violations.append(
                f"return traffic mismatch: {returned} != one-per-dispatch "
                f"{expected_return}"
            )
    if not any(step.kind == "local_reduce" for step in steps):
        violations.append("plan never performs the local reduction/merge step")
    return violations


__all__ = [
    "RouteEdge",
    "RouteProtocol",
    "classify_edge",
    "plan_by_protocol",
    "plan_pull_gather",
    "plan_push_dispatch_return",
    "verify_plan_conservation",
]
