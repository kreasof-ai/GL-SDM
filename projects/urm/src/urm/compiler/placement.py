"""Placement, sharding, and communication planning over a simulated device mesh.

Placement decides whether a logical route becomes local memory access, a
kernel dispatch, or communication. The mesh here is *simulated*: plans are
deterministic data that a real multi-device runtime could execute, and every
remote exchange is represented as a first-class communication step - never as
an ordinary local tensor operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from urm.compiler.diagnostics import DiagnosticCode

if TYPE_CHECKING:
    from collections.abc import Iterator


class RouteLeg(StrEnum):
    """How one logical edge is realized after placement."""

    LOCAL_MEMORY = "local_memory"  # source row co-resident with the query
    KERNEL_DISPATCH = "kernel_dispatch"  # same device; go through an anchor
    PEER_EXCHANGE = "peer_exchange"  # remote source; explicit communication


@dataclass(frozen=True, slots=True)
class DeviceMesh:
    """A simulated multi-device mesh with row-major device ids."""

    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.shape or any(n <= 0 for n in self.shape):
            raise ValueError("mesh shape must be positive and non-empty")
        if len(self.shape) > 2:
            raise ValueError("prototype meshes are 1-D or 2-D")

    @property
    def size(self) -> int:
        result = 1
        for n in self.shape:
            result *= n
        return result

    @property
    def devices(self) -> tuple[int, ...]:
        return tuple(range(self.size))

    def coords(self, device: int) -> tuple[int, ...]:
        if not 0 <= device < self.size:
            raise ValueError(f"unknown device {device}")
        coords: list[int] = []
        remainder = device
        for n in reversed(self.shape):
            coords.append(remainder % n)
            remainder //= n
        return tuple(reversed(coords))

    def distance(self, a: int, b: int) -> int:
        ca, cb = self.coords(a), self.coords(b)
        return sum(abs(x - y) for x, y in zip(ca, cb))

    def peers(self, device: int) -> tuple[int, ...]:
        """Devices exactly one hop away."""
        return tuple(
            other
            for other in self.devices
            if other != device and self.distance(device, other) == 1
        )


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    """Which device owns which slice of a logical domain instance."""

    tensor: str
    domain: str
    owner_devices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.owner_devices:
            raise ValueError("placement requires at least one owner device")


@dataclass(frozen=True, slots=True)
class PlacementMap:
    """Bindings of logical tensors/domains onto mesh devices."""

    mesh: DeviceMesh
    bindings: tuple[PlacementBinding, ...]

    def owner_of(self, tensor: str, index: int) -> int:
        """Owner device of logical ``index`` within ``tensor``'s domain."""
        binding = self._binding_for(tensor)
        return binding.owner_devices[index % len(binding.owner_devices)]

    def _binding_for(self, tensor: str) -> PlacementBinding:
        for binding in self.bindings:
            if binding.tensor == tensor:
                return binding
        raise KeyError(
            f"no placement binding for {tensor!r} "
            f"({DiagnosticCode.PLACEMENT_INCOMPLETE.value})"
        )

    def classify(
        self, source_tensor: str, source_index: int, query_tensor: str, query_index: int
    ) -> RouteLeg:
        src = self.owner_of(source_tensor, source_index)
        dst = self.owner_of(query_tensor, query_index)
        if src == dst:
            return RouteLeg.LOCAL_MEMORY
        return RouteLeg.PEER_EXCHANGE


@dataclass(frozen=True, slots=True)
class ExchangeStep:
    """One first-class communication step in an executable plan."""

    step_id: int
    src_device: int
    dst_device: int
    payload_count: int
    payload_bytes: int
    grouped_key: str | None = None  # destination bucket when grouped by peer

    @property
    def bytes_on_wire(self) -> int:
        return self.payload_count * self.payload_bytes


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One executable step: anchor dispatch, local access, or exchange."""

    step_id: int
    kind: str  # "anchor_dispatch" | "exchange" | "local_reduce" | "commit"
    anchor: str | None = None
    exchanges: tuple[ExchangeStep, ...] = ()
    note: str | None = None
    # Verified schedule/launch configuration for anchor-dispatch steps;
    # plain serializable data only (see urm.compiler.search.ScheduleDecision).
    launch_config: dict[str, str | int] | None = None


def group_exchanges_by_destination(
    edges: Iterator[tuple[int, int]], *, payload_bytes: int
) -> dict[int, list[ExchangeStep]]:
    """Group per-edge (src, dst) pairs into per-destination exchange steps.

    Deterministic: steps are ordered by (src_device, dst_device).
    """
    grouped: dict[tuple[int, int], int] = {}
    for edge_src, edge_dst in edges:
        grouped[(edge_src, edge_dst)] = grouped.get((edge_src, edge_dst), 0) + 1
    steps: dict[int, list[ExchangeStep]] = {}
    for step_id, ((pair_src, pair_dst), count) in enumerate(sorted(grouped)):
        steps.setdefault(pair_src, []).append(
            ExchangeStep(
                step_id=step_id,
                src_device=pair_src,
                dst_device=pair_dst,
                payload_count=count,
                payload_bytes=payload_bytes,
                grouped_key=f"dst:{pair_dst}",
            )
        )
    return steps
