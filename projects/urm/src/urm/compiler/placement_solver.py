"""Solver-guided simulated placement for experts/pages.

Given one :class:`PlacementProblem` (items with memory demands, a device mesh,
route traffic between items), builds a bounded one-hot assignment model:

- every item has exactly ``replication_factor`` owners;
- per-device memory capacity is respected;
- optional colocation / anti-affinity pairs are honored;
- communication is modeled through per-edge device-pair indicators so byte,
  critical-path and peer-pair objectives stay linear.

Objectives (lexicographic): maximum per-device load, communication critical-
path proxy, total inter-device bytes, number of communicating peer pairs, then
a deterministic placement tie-break. Baselines (round-robin, greedy) and an
independent exhaustive optimum are provided for agreement checks on tiny
instances; returned plans are re-verified without any solver.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from urm.compiler.constraints import (
    Assignment,
    BoolVar,
    ConstraintCategory,
    ConstraintHeader,
    ConstraintModel,
    Equality,
    IntVar,
    LinearExpr,
    ObjectiveSense,
    ObjectiveTerm,
    Origin,
    capacity_bound,
    implies_equal,
    less_equal,
    make_exactly_one,
)


@dataclass(frozen=True, slots=True)
class PlacementItem:
    name: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class PlacementEdge:
    source_item: str  # expert/page producing or holding data
    query_item: str  # token/query group consuming it
    payload_bytes: int = 2


@dataclass(frozen=True, slots=True)
class PlacementProblem:
    items: tuple[PlacementItem, ...]
    device_count: int
    device_capacity_bytes: int
    mesh_rows: int  # 2x2 -> rows=2 cols=2 ; 2x4 -> rows=2 cols=4
    mesh_cols: int
    edges: tuple[PlacementEdge, ...] = ()
    colocated_pairs: tuple[tuple[str, str], ...] = ()
    anti_affine_pairs: tuple[tuple[str, str], ...] = ()
    replication_factor: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "edges",
            tuple(self.edges) if not isinstance(self.edges, tuple) else self.edges,
        )
        if self.replication_factor < 1:
            raise ValueError("replication_factor must be >= 1")
        if self.replication_factor > self.device_count:
            raise ValueError(
                f"replication factor {self.replication_factor} exceeds the "
                f"{self.device_count}-device mesh"
            )

    def item(self, name: str) -> PlacementItem:
        return next(item for item in self.items if item.name == name)

    def distance(self, device_a: int, device_b: int) -> int:
        row_a, col_a = divmod(device_a, self.mesh_cols)
        row_b, col_b = divmod(device_b, self.mesh_cols)
        return abs(row_a - row_b) + abs(col_a - col_b)

    def devices(self) -> range:
        return range(self.device_count)


def _header(
    name: str,
    explanation: str,
    origin_kind: str,
    origin_id: str,
    category: ConstraintCategory = ConstraintCategory.PLACEMENT,
) -> ConstraintHeader:
    return ConstraintHeader(
        name=name,
        category=category,
        explanation=explanation,
        origin=Origin(kind=origin_kind, id=origin_id),
    )


# -- Model -----------------------------------------------------------------------


def build_placement_model(problem: PlacementProblem) -> ConstraintModel:
    model = ConstraintModel(name=f"placement::{problem.mesh_rows}x{problem.mesh_cols}")
    added_names: set[str] = set()

    def _add_unique(constraint):
        if constraint.name in added_names:
            return None  # mirrored encodings can imply identical facts
        added_names.add(constraint.name)
        return model.add_constraint(constraint)

    devices = problem.devices()

    # One-hot assignment variables.
    assign: dict[tuple[str, int], str] = {}
    for item in problem.items:
        names = []
        for device in devices:
            variable = f"assign_{item.name}_d{device}"
            model.add_variable(BoolVar(variable))
            assign[(item.name, device)] = variable
            names.append(variable)
        if problem.replication_factor == 1:
            _add_unique(
                make_exactly_one(
                    name=f"single_owner_{item.name}",
                    category=ConstraintCategory.PLACEMENT,
                    explanation=(f"item {item.name!r} has exactly one owner device"),
                    origin=Origin(kind="placement", id=item.name),
                    choices=tuple(names),
                )
            )
        else:
            _add_unique(
                Equality(
                    name=f"replication_count_{item.name}",
                    category=ConstraintCategory.PLACEMENT,
                    explanation=(
                        f"{item.name!r} is replicated on exactly "
                        f"{problem.replication_factor} devices"
                    ),
                    origin=Origin(kind="placement", id=item.name),
                    lhs=LinearExpr(terms=tuple((n, 1) for n in sorted(names))),
                    rhs=LinearExpr.const(problem.replication_factor),
                )
            )

    # Capacity per device.
    for device in devices:
        load_expr = LinearExpr(
            terms=tuple(
                sorted(
                    (assign[(item.name, device)], item.size_bytes)
                    for item in problem.items
                )
            )
        )
        _add_unique(
            capacity_bound(
                name=f"device_capacity_d{device}",
                category=ConstraintCategory.PLACEMENT,
                explanation=(
                    f"device {device} holds at most "
                    f"{problem.device_capacity_bytes} B of assigned items"
                ),
                origin=Origin(kind="device", id=f"d{device}"),
                unit=f"device:{device}",
                expression=load_expr,
                limit=problem.device_capacity_bytes,
            )
        )

    # Colocation / anti-affinity.
    for first, second in problem.colocated_pairs:
        for device in devices:
            _add_unique(
                implies_equal(
                    _header(
                        f"colocate_{first}_{second}_d{device}",
                        f"colocation places {first!r} and {second!r} together on "
                        f"device {device} when {first!r} is there",
                        "placement",
                        f"{first}+{second}",
                    ),
                    LinearExpr.var(assign[(first, device)]),
                    1,
                )
                .then_equal(LinearExpr.var(assign[(second, device)]), 1)
                .done()
            )
    for first, second in problem.anti_affine_pairs:
        for device in devices:
            _add_unique(
                implies_equal(
                    _header(
                        f"anti_affine_{first}_{second}_d{device}",
                        f"{first!r} and {second!r} must not share device {device}",
                        "placement",
                        f"{first}!={second}",
                    ),
                    LinearExpr.var(assign[(first, device)]),
                    1,
                )
                .then_less_equal(
                    LinearExpr.var(assign[(first, device)])
                    + LinearExpr.var(assign[(second, device)]),
                    1,
                )
                .done()
            )

    # Communication indicators: exactly one ordered device pair per edge.
    edge_pair_vars: list[dict[tuple[int, int], str]] = []
    for index, edge in enumerate(problem.edges):
        chosen: dict[tuple[int, int], str] = {}
        for src_device in devices:
            for dst_device in devices:
                variable = f"edge{index}_on_{src_device}_{dst_device}"
                model.add_variable(BoolVar(variable))
                chosen[(src_device, dst_device)] = variable
        edge_pair_vars.append(chosen)
        _add_unique(
            make_exactly_one(
                name=f"edge{index}_exactly_one_device_pair",
                category=ConstraintCategory.COMMUNICATION,
                explanation=(
                    f"route edge {index} ({edge.source_item} -> "
                    f"{edge.query_item}) uses exactly one ordered device pair"
                ),
                origin=Origin(kind="placement", id=f"edge{index}"),
                choices=tuple(sorted(chosen.values())),
            )
        )
        source_name = edge.source_item.split("#")[0]
        query_name = edge.query_item.split("#")[0]
        for (src_device, dst_device), variable in sorted(chosen.items()):
            # One conjunctive implication per ordered pair: the chosen pair
            # forces BOTH endpoint ownerships simultaneously.
            builder = implies_equal(
                _header(
                    f"edge{index}_links_{src_device}_{dst_device}",
                    "the chosen device pair must match item ownership",
                    "communication",
                    f"edge{index}",
                    ConstraintCategory.COMMUNICATION,
                ),
                LinearExpr.var(variable),
                1,
            )
            builder.then_equal(LinearExpr.var(assign[(source_name, src_device)]), 1)
            builder.then_equal(LinearExpr.var(assign[(query_name, dst_device)]), 1)
            model.add_constraint(builder.done())

    # Max-load indicator for the first objective.
    total_demand = sum(item.size_bytes for item in problem.items)
    model.add_variable(IntVar("max_load", 0, total_demand))
    for device in devices:
        load_expr = LinearExpr(
            terms=tuple(
                sorted(
                    (assign[(item.name, device)], item.size_bytes)
                    for item in problem.items
                )
            )
        )
        _add_unique(
            less_equal(
                _header(
                    f"load_d{device}_le_max_load",
                    "per-device load is bounded by the max_load indicator",
                    "placement",
                    f"max_load_d{device}",
                ),
                load_expr,
                LinearExpr.var("max_load"),
            )
        )

    # Peer-pair flags.
    peer_flags: list[str] = []
    for first_device in devices:
        for second_device in devices:
            if first_device >= second_device:
                continue
            flag = f"peers_{first_device}_{second_device}"
            model.add_variable(BoolVar(flag))
            peer_flags.append(flag)
    for index in range(len(problem.edges)):
        for first_device in devices:
            for second_device in devices:
                if first_device >= second_device:
                    continue
                flag = f"peers_{first_device}_{second_device}"
                for triggering in (
                    edge_pair_vars[index][(first_device, second_device)],
                    edge_pair_vars[index].get((second_device, first_device)),
                ):
                    if triggering is None:
                        continue
                    model.add_constraint(
                        implies_equal(
                            _header(
                                f"edge{index}_fires_peers_"
                                f"{first_device}_{second_device}"
                                f"_via_{triggering}",
                                "traffic between two devices marks the peer pair",
                                "communication",
                                f"edge{index}",
                                ConstraintCategory.COMMUNICATION,
                            ),
                            LinearExpr.var(triggering),
                            1,
                        )
                        .then_equal(LinearExpr.var(flag), 1)
                        .done()
                    )

    # Objectives (lexicographic order).
    model.add_objective(
        ObjectiveTerm(
            name="max_device_load_bytes",
            expression=LinearExpr.var("max_load"),
            sense=ObjectiveSense.MINIMIZE,
        )
    )
    path_terms: list[tuple[str, int]] = []
    byte_terms: list[tuple[str, int]] = []
    for index, edge in enumerate(problem.edges):
        for (src_device, dst_device), variable in sorted(edge_pair_vars[index].items()):
            hops = problem.distance(src_device, dst_device)
            if hops:
                path_terms.append((variable, edge.payload_bytes * hops))
                byte_terms.append((variable, edge.payload_bytes))
    model.add_objective(
        ObjectiveTerm(
            name="comm_critical_path_proxy",
            expression=LinearExpr(terms=tuple(sorted(path_terms))),
        )
    )
    model.add_objective(
        ObjectiveTerm(
            name="inter_device_bytes",
            expression=LinearExpr(terms=tuple(sorted(byte_terms))),
        )
    )
    model.add_objective(
        ObjectiveTerm(
            name="peer_pair_count",
            expression=LinearExpr(
                terms=tuple((flag, 1) for flag in sorted(peer_flags))
            ),
        )
    )
    # Injective positional encoding: the owner vector maps to one integer.
    radix = problem.device_count
    tie_terms: list[tuple[str, int]] = [
        (
            assign[(item.name, device)],
            (device + 1) * radix**position,
        )
        for position, item in enumerate(problem.items)
        for device in devices
    ]
    model.add_objective(
        ObjectiveTerm(
            name="stable_placement_tiebreak",
            expression=LinearExpr(terms=tuple(sorted(tie_terms))),
        )
    )
    model.validate()
    return model


def decode_placement(
    assignment: Assignment, problem: PlacementProblem
) -> dict[str, int]:
    owners: dict[str, int] = {}
    for item in problem.items:
        for device in problem.devices():
            if assignment.get(f"assign_{item.name}_d{device}") is True:
                owners[item.name] = device
                break
    missing = [item.name for item in problem.items if item.name not in owners]
    if missing:
        raise ValueError(f"placement assignment misses owners for {missing}")
    return owners


# -- Metrics and baselines ----------------------------------------------------------


def placement_metrics(
    owners: dict[str, int], problem: PlacementProblem
) -> dict[str, float | int]:
    loads = {device: 0 for device in problem.devices()}
    for item in problem.items:
        owner = owners.get(item.name)
        if owner is not None:
            loads[owner] += item.size_bytes // max(1, problem.replication_factor)
    comm_bytes = 0
    path_proxy = 0
    peers: set[tuple[int, int]] = set()
    for edge in problem.edges:
        source_name = edge.source_item.split("#")[0]
        query_name = edge.query_item.split("#")[0]
        src = owners[source_name]
        dst = owners[query_name]
        hops = problem.distance(src, dst)
        if hops > 0:
            comm_bytes += edge.payload_bytes
            path_proxy += edge.payload_bytes * hops
            peers.add(tuple(sorted((src, dst))))
    return {
        "max_load": max(loads.values()),
        "total_bytes": comm_bytes,
        "critical_path_proxy": path_proxy,
        "peer_pairs": len(peers),
    }


def feasible_owners(owners: dict[str, int], problem: PlacementProblem) -> bool:
    loads = {device: 0 for device in problem.devices()}
    for item in problem.items:
        owner = owners.get(item.name)
        if owner is None or not 0 <= owner < problem.device_count:
            return False
        loads[owner] += item.size_bytes
    if any(load > problem.device_capacity_bytes for load in loads.values()):
        return False
    for first, second in problem.colocated_pairs:
        if owners[first] != owners[second]:
            return False
    for first, second in problem.anti_affine_pairs:
        if owners[first] == owners[second]:
            return False
    return True


def round_robin_placement(problem: PlacementProblem) -> dict[str, int]:
    """Deterministic round-robin over devices (ignores sizes/edges)."""
    return {
        item.name: position % problem.device_count
        for position, item in enumerate(problem.items)
    }


def greedy_placement(problem: PlacementProblem) -> dict[str, int]:
    """Capacity-aware least-loaded greedy by descending size."""
    owners: dict[str, int] = {}
    loads = {device: 0 for device in problem.devices()}
    for item in sorted(problem.items, key=lambda i: (-i.size_bytes, i.name)):
        feasible_devices = [
            device
            for device in problem.devices()
            if loads[device] + item.size_bytes <= problem.device_capacity_bytes
        ]
        pool = feasible_devices or list(problem.devices())
        device = min(pool, key=lambda d: (loads[d], d))
        owners[item.name] = device
        loads[device] += item.size_bytes
    return owners


def exhaustive_placement_optimum(
    problem: PlacementProblem,
) -> tuple[dict[str, int], dict[str, float | int]] | None:
    """Brute-force lexicographic optimum over all owner assignments."""
    best_rank: tuple | None = None
    best_owners: dict[str, int] | None = None
    names = [item.name for item in problem.items]
    for combination in itertools.product(
        range(problem.device_count), repeat=len(names)
    ):
        owners = dict(zip(names, combination, strict=True))
        if not feasible_owners(owners, problem):
            continue
        metrics = placement_metrics(owners, problem)
        rank = (
            metrics["max_load"],
            metrics["critical_path_proxy"],
            metrics["total_bytes"],
            metrics["peer_pairs"],
            sum(
                (owner + 1) * len(names) ** position
                for position, owner in enumerate(combination)
            ),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_owners = owners
    if best_owners is None:
        return None
    return best_owners, placement_metrics(best_owners, problem)


__all__ = [
    "PlacementEdge",
    "PlacementItem",
    "PlacementProblem",
    "build_placement_model",
    "decode_placement",
    "exhaustive_placement_optimum",
    "feasible_owners",
    "greedy_placement",
    "placement_metrics",
    "round_robin_placement",
]
