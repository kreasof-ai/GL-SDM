"""Assignment-level facts and nogood machinery for independent verification.

These types describe everything the domain layer needs to judge one assignment
(anchor capability, resources, routes, placement, protocol policies) and the
bounded exact-nogood helper used to reject a failed assignment. The imperative
checker lives in :mod:`urm.compiler.verify.plan`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from urm.compiler.common.diagnostics import Severity
from urm.compiler.solve.constraints import (
    Assignment,
    ConstraintModel,
    ModelValidationError,
    Nogood,
    make_nogood,
)


@dataclass(frozen=True, slots=True)
class VerificationFailure:
    check: str
    message: str
    severity: Severity = Severity.ERROR

    def to_dict(self) -> dict[str, str]:
        return {
            "check": self.check,
            "message": self.message,
            "severity": self.severity.value,
        }


@dataclass(frozen=True, slots=True)
class AnchorFacts:
    """Capability contract of one concrete anchor."""

    name: str
    kind: str
    trusted: bool = True
    forward_only: bool = False
    backward_verified_dtypes: frozenset[str] = frozenset()
    deterministic_accumulation: bool = True
    honored_obligations: frozenset[str] = frozenset()
    supported_dtypes: frozenset[str] | None = None

    def backward_covers(self, dtype_name: str) -> bool:
        return dtype_name in self.backward_verified_dtypes


@dataclass(frozen=True, slots=True)
class ResourceFacts:
    """Device-level resource ceilings."""

    max_shared_mem_bytes_per_block: int | None = None
    max_registers_per_thread: int | None = None
    max_threads_per_block: int | None = None


@dataclass(frozen=True, slots=True)
class RouteEdgeFacts:
    """One logical route edge with its protocol requirements."""

    query_id: int
    peer_id: int
    ordinal: int
    requires_return: bool = False
    dropped: bool = False


@dataclass(frozen=True, slots=True)
class PlacementItemFacts:
    """One placeable item (expert/page) and its demands."""

    name: str
    size_bytes: int
    owner_variable: str
    one_hot_devices: tuple[int, ...] = ()
    replication_factor: int = 1


@dataclass(frozen=True, slots=True)
class AssignmentFacts:
    """Everything the domain layer needs to judge one assignment."""

    selected_anchor: AnchorFacts | None = None
    intent_training: bool = False
    required_backward_dtypes: frozenset[str] = frozenset()
    unresolved_forward_only_obligations: int = 0
    requires_commit_capable_lowering: bool = False
    merge_policy_ordered: bool = False
    stable_order_required: bool = False
    locality_floor_rank: int | None = None
    achieved_locality_rank: int | None = None
    barrier_classes_crossed: tuple[str, ...] = ()
    resource_limits: ResourceFacts | None = None
    estimated_shared_mem_bytes: int | None = None
    estimated_registers_per_thread: int | None = None
    estimated_threads_per_block: int | None = None
    routes: tuple[RouteEdgeFacts, ...] = ()
    dispatched_counts: Mapping[tuple[int, int], int] | None = None
    returned_counts: Mapping[int, int] | None = None
    devices: tuple[int, ...] = ()
    device_capacity_bytes: Mapping[int, int] | None = None
    items: tuple[PlacementItemFacts, ...] = ()


def assignment_dtype_values(assignment: Assignment) -> set[str]:
    return {
        str(value)
        for key, value in assignment.items()
        if key.startswith("dtype") or key.endswith("_dtype")
    }


def add_nogood_for_failures(
    model: ConstraintModel,
    assignment: Assignment,
    failures: Sequence[VerificationFailure],
    origin_id: str,
    max_nogoods: int = 64,
) -> bool:
    """Add a bounded exact nogood excluding this rejected assignment.

    Returns True when the nogood was added; False when the retry budget is
    exhausted (the caller must stop requesting schedules).
    """
    existing = sum(
        1 for constraint in model.constraints if isinstance(constraint, Nogood)
    )
    if existing >= max_nogoods:
        return False
    forbidden = {name: assignment[name] for name in sorted(assignment)}
    model.add_constraint(
        make_nogood(
            name=f"nogood_rejected_{existing + 1}",
            explanation=(
                "rejected by independent model verification: "
                + "; ".join(failure.message for failure in failures[:3])
            ),
            origin_kind="verification_failure",
            origin_id=origin_id,
            forbidden=forbidden,
        )
    )
    return True


class AssignmentFactsError(ModelValidationError):
    """Raised when domain facts are internally inconsistent."""


__all__ = [
    "AnchorFacts",
    "AssignmentFacts",
    "AssignmentFactsError",
    "PlacementItemFacts",
    "ResourceFacts",
    "RouteEdgeFacts",
    "VerificationFailure",
    "add_nogood_for_failures",
    "assignment_dtype_values",
]
