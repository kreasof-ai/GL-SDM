"""Independent, solver-free verification of decision-model assignments.

A Z3 model is *never* trusted: before any plan or kernel is generated from
it, this module re-checks the assignment imperatively - no solver in the
loop, so translator or solver bugs cannot launder an invalid model into a
compiled artifact.

Two layers:

1. **IR layer** - every named assertion is re-evaluated against the raw
   assignment (ranges, divisibility, implications, at-most-one/exactly-one,
   capacity bounds, nogoods). This catches translation bugs.
2. **Domain layer** - capability/resource/protocol facts that only make
   sense outside the pure IR: anchor capability compatibility, dtype/layout
   support, training/backward compatibility, locality and effect barriers,
   placement ownership, communication conservation, capacity policy,
   stable-order requirements, collision/merge policies, transaction commit
   obligations.

Verification failures reject the model outright; callers may then add a
bounded nogood and request another schedule, but must never generate or run
a kernel from the rejected assignment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from urm.compiler.constraints import (
    AllowedSet,
    Assignment,
    AtMostOne,
    BoolVar,
    CapacityBound,
    ConstraintModel,
    Divisibility,
    EnumVar,
    Equality,
    ExactlyOne,
    Implication,
    IntVar,
    LessEqual,
    ModelValidationError,
    Nogood,
    make_nogood,
)
from urm.compiler.diagnostics import Severity


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
class VerificationReport:
    ok: bool
    failures: tuple[VerificationFailure, ...] = ()
    checks_run: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks_run": list(self.checks_run),
            "failures": [failure.to_dict() for failure in self.failures],
        }


# -- Domain facts ---------------------------------------------------------------
# Plain data describing the world the assignment executes in. Every group is
# optional so tests can construct minimal contexts.


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
    supported_dtypes: frozenset[str] | None = None  # None = unrestricted

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
    """One placeable item (expert/page) and its demands.

    Either ``owner_variable`` holds the chosen device directly, or
    ``one_hot_devices`` names the Boolean indicator variables
    ``assign_<item>_d<device>`` from a one-hot encoding.
    """

    name: str
    size_bytes: int
    owner_variable: str
    one_hot_devices: tuple[int, ...] = ()
    replication_factor: int = 1


@dataclass(frozen=True, slots=True)
class AssignmentFacts:
    """Everything the domain layer needs to judge one assignment."""

    # Schedule-side facts
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
    # Resource ceilings and the plan's declared estimates
    resource_limits: ResourceFacts | None = None
    estimated_shared_mem_bytes: int | None = None
    estimated_registers_per_thread: int | None = None
    estimated_threads_per_block: int | None = None
    # Communication / placement facts
    routes: tuple[RouteEdgeFacts, ...] = ()
    dispatched_counts: Mapping[tuple[int, int], int] | None = None
    returned_counts: Mapping[int, int] | None = None
    devices: tuple[int, ...] = ()
    device_capacity_bytes: Mapping[int, int] | None = None
    items: tuple[PlacementItemFacts, ...] = ()


# -- The verifier -----------------------------------------------------------------


class ModelVerifier:
    """Imperative, deterministic, solver-independent model checker."""

    def verify(
        self,
        model: ConstraintModel,
        assignment: Assignment,
        facts: AssignmentFacts | None = None,
    ) -> VerificationReport:
        failures: list[VerificationFailure] = []
        checks: list[str] = ["variables_in_range"]
        failures.extend(self._check_ranges(model, assignment))

        ir_checks = (
            ("equalities", Equality),
            ("inequalities", LessEqual),
            ("divisibility", Divisibility),
            ("allowed_sets", AllowedSet),
            ("implications", Implication),
            ("at_most_one", AtMostOne),
            ("exactly_one", ExactlyOne),
            ("capacity_bounds", CapacityBound),
            ("nogoods", Nogood),
        )
        by_type: dict[type, list[object]] = {}
        for constraint in model.constraints:
            by_type.setdefault(type(constraint), []).append(constraint)
        for check_name, constraint_type in ir_checks:
            checks.append(check_name)
            for constraint in by_type.get(constraint_type, []):
                try:
                    holds = constraint.holds(assignment)  # type: ignore[union-attr]
                except (KeyError, TypeError) as error:
                    failures.append(
                        VerificationFailure(
                            check=check_name,
                            message=(
                                f"{constraint.name}: could not evaluate ({error})"
                            ),
                        )
                    )
                    continue
                if not holds:
                    failures.append(
                        VerificationFailure(
                            check=check_name,
                            message=(
                                f"{constraint.name} violated: {constraint.describe()}"
                            ),
                        )
                    )

        if facts is not None:
            failures.extend(self._check_domain(assignment, facts))
        return VerificationReport(
            ok=not failures, failures=tuple(failures), checks_run=tuple(checks)
        )

    # -- IR layer -------------------------------------------------------------

    def _check_ranges(
        self, model: ConstraintModel, assignment: Assignment
    ) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        for variable in model.variables:
            if variable.name not in assignment:
                failures.append(
                    VerificationFailure(
                        check="variables_in_range",
                        message=f"variable {variable.name!r} is unassigned",
                    )
                )
                continue
            value = assignment[variable.name]
            if isinstance(variable, BoolVar):
                if not isinstance(value, bool):
                    failures.append(
                        VerificationFailure(
                            check="variables_in_range",
                            message=f"{variable.name}: expected bool, got {value!r}",
                        )
                    )
            elif isinstance(variable, IntVar):
                numeric = int(value) if isinstance(value, bool) else value
                if isinstance(numeric, str) or not isinstance(numeric, int):
                    failures.append(
                        VerificationFailure(
                            check="variables_in_range",
                            message=f"{variable.name}: expected int, got {value!r}",
                        )
                    )
                elif not variable.lower <= numeric <= variable.upper:
                    failures.append(
                        VerificationFailure(
                            check="variables_in_range",
                            message=(
                                f"{variable.name}={numeric} outside "
                                f"[{variable.lower}, {variable.upper}]"
                            ),
                        )
                    )
            elif isinstance(variable, EnumVar) and (
                not isinstance(value, str) or value not in variable.values
            ):
                failures.append(
                    VerificationFailure(
                        check="variables_in_range",
                        message=(f"{variable.name}={value!r} not in enumeration"),
                    )
                )
        extra = set(assignment) - {v.name for v in model.variables}
        if extra:
            failures.append(
                VerificationFailure(
                    check="variables_in_range",
                    message=f"assignment contains unknown variables: {sorted(extra)}",
                )
            )
        return failures

    def _evaluate_linear(self, expression, assignment: Assignment) -> int:
        total = expression.constant
        for name, coefficient in expression.terms:
            value = assignment[name]
            total += coefficient * (int(value) if isinstance(value, bool) else value)
        return total

    # -- domain layer -----------------------------------------------------------

    def _check_domain(
        self, assignment: Assignment, facts: AssignmentFacts
    ) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        failures.extend(self._check_anchor_capability(assignment, facts))
        failures.extend(self._check_resources(facts))
        failures.extend(self._check_locality_and_effects(facts))
        failures.extend(self._check_communication_conservation(facts))
        failures.extend(self._check_placement(assignment, facts))
        failures.extend(self._check_protocol_policies(facts))
        return failures

    def _check_anchor_capability(
        self, assignment: Assignment, facts: AssignmentFacts
    ) -> list[VerificationFailure]:
        anchor = facts.selected_anchor
        failures: list[VerificationFailure] = []
        if anchor is None:
            return failures
        if not anchor.trusted:
            failures.append(
                VerificationFailure(
                    check="anchor_trusted",
                    message=f"anchor {anchor.name!r} is not trusted",
                )
            )
        if facts.intent_training:
            if anchor.forward_only:
                failures.append(
                    VerificationFailure(
                        check="training_backward_compatibility",
                        message=(
                            f"training requested but anchor {anchor.name!r} is "
                            "forward-only"
                        ),
                    )
                )
            missing = {
                dtype
                for dtype in facts.required_backward_dtypes
                if not anchor.backward_covers(dtype)
            }
            if missing:
                failures.append(
                    VerificationFailure(
                        check="training_backward_compatibility",
                        message=(
                            f"anchor {anchor.name!r} backward not verified for "
                            f"dtypes {sorted(missing)}"
                        ),
                    )
                )
            if facts.unresolved_forward_only_obligations:
                failures.append(
                    VerificationFailure(
                        check="obligations_resolved",
                        message=(
                            f"{facts.unresolved_forward_only_obligations} "
                            "unresolved forward-only obligations under a "
                            "training intent"
                        ),
                    )
                )
        if anchor.supported_dtypes is not None:
            unsupported = {
                dtype
                for dtype in _assignment_dtype_values(assignment)
                if dtype and dtype not in anchor.supported_dtypes
            }
            if unsupported:
                failures.append(
                    VerificationFailure(
                        check="dtype_layout_support",
                        message=(
                            f"anchor {anchor.name!r} does not support dtypes "
                            f"{sorted(unsupported)} selected by this assignment"
                        ),
                    )
                )
        return failures

    @staticmethod
    def _check_resources(facts: AssignmentFacts) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        limits = facts.resource_limits
        if limits is None:
            return failures
        if (
            facts.estimated_shared_mem_bytes is not None
            and limits.max_shared_mem_bytes_per_block is not None
            and facts.estimated_shared_mem_bytes > limits.max_shared_mem_bytes_per_block
        ):
            failures.append(
                VerificationFailure(
                    check="shared_memory_limits",
                    message=(
                        f"estimated shared memory "
                        f"{facts.estimated_shared_mem_bytes} B exceeds the "
                        f"device limit {limits.max_shared_mem_bytes_per_block} B"
                    ),
                )
            )
        if (
            facts.estimated_registers_per_thread is not None
            and limits.max_registers_per_thread is not None
            and facts.estimated_registers_per_thread > limits.max_registers_per_thread
        ):
            failures.append(
                VerificationFailure(
                    check="resource_limits",
                    message=(
                        f"estimated registers/thread "
                        f"{facts.estimated_registers_per_thread} exceeds the "
                        f"limit {limits.max_registers_per_thread}"
                    ),
                )
            )
        if (
            facts.estimated_threads_per_block is not None
            and limits.max_threads_per_block is not None
            and facts.estimated_threads_per_block > limits.max_threads_per_block
        ):
            failures.append(
                VerificationFailure(
                    check="resource_limits",
                    message=(
                        f"estimated {facts.estimated_threads_per_block} threads "
                        "per block exceeds the device limit "
                        f"{limits.max_threads_per_block}"
                    ),
                )
            )
        return failures

    @staticmethod
    def _check_locality_and_effects(
        facts: AssignmentFacts,
    ) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        if (
            facts.locality_floor_rank is not None
            and facts.achieved_locality_rank is not None
            and facts.achieved_locality_rank < facts.locality_floor_rank
        ):
            failures.append(
                VerificationFailure(
                    check="locality_requirements",
                    message=(
                        f"plan achieves locality rank {facts.achieved_locality_rank} "
                        f"below the required floor {facts.locality_floor_rank}"
                    ),
                )
            )
        if facts.barrier_classes_crossed:
            failures.append(
                VerificationFailure(
                    check="effect_barriers",
                    message=(
                        f"assignment moves work across effect barriers: "
                        f"{list(facts.barrier_classes_crossed)}"
                    ),
                )
            )
        return failures

    @staticmethod
    def _check_communication_conservation(
        facts: AssignmentFacts,
    ) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        if not facts.routes:
            return failures
        live = [route for route in facts.routes if not route.dropped]
        expected_dispatch: dict[tuple[int, int], int] = {}
        for route in live:
            key = (route.query_id, route.peer_id)
            expected_dispatch[key] = expected_dispatch.get(key, 0) + 1
        if (
            facts.dispatched_counts is not None
            and dict(facts.dispatched_counts) != expected_dispatch
        ):
            failures.append(
                VerificationFailure(
                    check="communication_conservation",
                    message=(
                        "dispatched traffic does not preserve every "
                        "non-dropped route exactly once"
                    ),
                )
            )
        expected_returns: dict[int, int] = {}
        for route in live:
            if route.requires_return:
                expected_returns[route.query_id] = (
                    expected_returns.get(route.query_id, 0) + 1
                )
        if (
            facts.returned_counts is not None
            and dict(facts.returned_counts) != expected_returns
        ):
            failures.append(
                VerificationFailure(
                    check="communication_conservation",
                    message=("return traffic violates exactly-one-required-return"),
                )
            )
        return failures

    def _check_placement(
        self, assignment: Assignment, facts: AssignmentFacts
    ) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        if not facts.items:
            return failures
        device_set = set(facts.devices)
        loads: dict[int, int] = {}
        for item in facts.items:
            if item.one_hot_devices:
                owner_values = [
                    device
                    for device in item.one_hot_devices
                    if assignment.get(f"assign_{item.name}_d{device}") in (True, 1)
                ]
            elif item.owner_variable in assignment:
                owner_values = [assignment[item.owner_variable]]
            else:
                failures.append(
                    VerificationFailure(
                        check="placement_ownership",
                        message=f"item {item.name!r} has no assigned owner",
                    )
                )
                continue
            if not owner_values:
                failures.append(
                    VerificationFailure(
                        check="placement_ownership",
                        message=f"item {item.name!r} has no assigned owner",
                    )
                )
                continue
            expected_copies = max(1, item.replication_factor)
            if len(owner_values) != expected_copies and not item.one_hot_devices:
                pass  # single-value encodings carry one owner by construction
            if len(owner_values) != expected_copies and item.one_hot_devices:
                failures.append(
                    VerificationFailure(
                        check="placement_ownership",
                        message=(
                            f"item {item.name!r} has {len(owner_values)} owners "
                            f"but replication factor {expected_copies}"
                        ),
                    )
                )
            for owner_value in owner_values:
                owner = (
                    int(owner_value) if isinstance(owner_value, bool) else owner_value
                )
                if isinstance(owner, str):
                    try:
                        owner = int(owner)
                    except ValueError:
                        failures.append(
                            VerificationFailure(
                                check="placement_ownership",
                                message=(
                                    f"item {item.name!r} owner {owner!r} is "
                                    "not a device id"
                                ),
                            )
                        )
                        continue
                if owner not in device_set:
                    failures.append(
                        VerificationFailure(
                            check="placement_ownership",
                            message=(
                                f"item {item.name!r} placed on unknown device {owner}"
                            ),
                        )
                    )
                    continue
                loads[owner] = loads.get(owner, 0) + item.size_bytes // max(
                    1, expected_copies
                )
        if facts.device_capacity_bytes is not None:
            capacities = dict(facts.device_capacity_bytes)
            for device, load in sorted(loads.items()):
                if device in capacities and load > capacities[device]:
                    failures.append(
                        VerificationFailure(
                            check="capacity",
                            message=(
                                f"device {device} load {load} B exceeds its "
                                f"capacity {capacities[device]} B"
                            ),
                        )
                    )
        return failures

    @staticmethod
    def _check_protocol_policies(
        facts: AssignmentFacts,
    ) -> list[VerificationFailure]:
        failures: list[VerificationFailure] = []
        if facts.merge_policy_ordered and facts.stable_order_required:
            ordinals = [route.ordinal for route in facts.routes]
            if ordinals != sorted(ordinals):
                failures.append(
                    VerificationFailure(
                        check="stable_order_requirements",
                        message=(
                            "ordered merge requires route processing in stable "
                            "ordinal order"
                        ),
                    )
                )
        if facts.requires_commit_capable_lowering:
            # A commit-capable lowering must be part of the selected plan;
            # the schedule side records this via the anchor's commit flag.
            anchor = facts.selected_anchor
            if anchor is not None and not anchor.honored_obligations:
                failures.append(
                    VerificationFailure(
                        check="transaction_commit_obligations",
                        message=(
                            "transactional update requires a commit-capable "
                            "lowering; the selected anchor honors no commit-"
                            "related obligations"
                        ),
                    )
                )
        return failures


def _assignment_dtype_values(assignment: Assignment) -> set[str]:
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
    "ModelVerifier",
    "PlacementItemFacts",
    "ResourceFacts",
    "RouteEdgeFacts",
    "VerificationFailure",
    "VerificationReport",
    "add_nogood_for_failures",
]
