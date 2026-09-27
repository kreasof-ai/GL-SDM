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

The assignment-level facts and the nogood machinery live in
:mod:`urm.compiler.verify.assignments`; this module owns the imperative checker
(:class:`ModelVerifier`) and the report type.

Verification failures reject the model outright; callers may then add a
bounded nogood and request another schedule, but must never generate or run
a kernel from the rejected assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from urm.compiler.solve.constraints import (
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
    Nogood,
)
from urm.compiler.verify.assignments import (
    AnchorFacts,
    AssignmentFacts,
    AssignmentFactsError,
    PlacementItemFacts,
    ResourceFacts,
    RouteEdgeFacts,
    VerificationFailure,
    add_nogood_for_failures,
    assignment_dtype_values,
)

# Back-compat re-exports: the public verification surface is reachable from
# this module while the assignment-level facts live in verify/assignments.py.
_assignment_dtype_values = assignment_dtype_values


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
                for dtype in assignment_dtype_values(assignment)
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
                pass
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
