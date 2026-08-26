"""Representative impossible problems and their unsat-core diagnostics.

Each case builds a deliberately unsatisfiable :class:`ConstraintModel` from
real compiler facts, runs the feasibility pass, and maps the unsat core onto
a concise human-readable message. Raw solver formulas are never the primary
diagnostic; named constraints and their explanations are.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from urm.compiler.constraints import (
    AllowedSet,
    BoolVar,
    ConstraintCategory,
    ConstraintModel,
    Equality,
    Implication,
    IntVar,
    LinearExpr,
    Origin,
    capacity_bound,
    make_exactly_one,
)
from urm.compiler.diagnostics import DiagnosticCode, Severity
from urm.compiler.schedule_space import SUPPORTED_BLOCKS


@dataclass(frozen=True, slots=True)
class UnsatCase:
    """One representative impossible problem with its expected diagnosis."""

    name: str
    category: str
    build: Callable[[], ConstraintModel]
    expected_core_substring: str  # a constraint name that must appear in core
    concise_message: str


def _training_with_forward_only_anchor() -> ConstraintModel:
    model = ConstraintModel(name="unsat::training_forward_only_anchor")
    model.add_variable(BoolVar("use_fused_epilogue"))
    model.add_variable(BoolVar("anchor_forward_only"))
    model.add_constraint(
        Equality(
            name="selected_anchor_is_forward_only",
            category=ConstraintCategory.ANCHOR_CAPABILITY,
            explanation=(
                "the only registered anchor for this request declares "
                "forward-only execution (backward unverified)"
            ),
            origin=Origin(kind="anchor", id="routed_reduction_forward_only_v0"),
            lhs=LinearExpr.var("anchor_forward_only"),
            rhs=LinearExpr.const(1),
        )
    )
    model.add_constraint(
        Equality(
            name="training_requires_verified_backward",
            category=ConstraintCategory.TRAINING,
            explanation=(
                "training compilation requires an anchor whose backward is "
                "verified; forward-only anchors are rejected"
            ),
            origin=Origin(kind="intent", id="training"),
            lhs=LinearExpr.var("anchor_forward_only"),
            rhs=LinearExpr.const(0),
        )
    )
    return model


def _tile_incompatible_with_vector_width() -> ConstraintModel:
    model = ConstraintModel(name="unsat::tile_vector_mismatch")
    model.add_variable(IntVar("block_d", 32, 256))
    model.add_constraint(
        Equality(
            name="hint_pinned_block_d",
            category=ConstraintCategory.SCHEDULE,
            explanation="caller pinned BLOCK_D=48 via schedule hints",
            origin=Origin(kind="schedule_param", id="BLOCK_D_hint"),
            lhs=LinearExpr.var("block_d"),
            rhs=LinearExpr.const(48),
        )
    )
    model.add_constraint(
        AllowedSet(
            name="block_lane_alignment",
            category=ConstraintCategory.SCHEDULE,
            explanation=(
                "BLOCK_D must be a multiple of 32 (whole 32-lane vector "
                "tiles); implemented values are 32/64/128/256"
            ),
            origin=Origin(kind="schedule_param", id="BLOCK_D"),
            variable="block_d",
            allowed=SUPPORTED_BLOCKS,
        )
    )
    return model


def _shared_memory_exceeded() -> ConstraintModel:
    model = ConstraintModel(name="unsat::shared_memory_budget")
    for index, block in enumerate(SUPPORTED_BLOCKS):
        model.add_variable(BoolVar(f"tile_{index}_b{block}"))
    model.add_constraint(
        make_exactly_one(
            name="exactly_one_tile",
            category=ConstraintCategory.SCHEDULE,
            explanation="one tile configuration is selected",
            origin=Origin(kind="schedule_param", id="tiles"),
            choices=tuple(f"tile_{i}_b{b}" for i, b in enumerate(SUPPORTED_BLOCKS)),
        )
    )
    smem = LinearExpr()
    for index, block in enumerate(SUPPORTED_BLOCKS):
        smem += LinearExpr.var(f"tile_{index}_b{block}") * (4 * block * 4)
    model.add_constraint(
        capacity_bound(
            name="shared_mem_budget_511",
            category=ConstraintCategory.RESOURCE,
            explanation=(
                "staged tiles must fit the 511 B per-block shared-memory "
                "budget (4 stages x BLOCK_D x 4 B fp32 staging; the smallest "
                "supported tile already needs 512 B)"
            ),
            origin=Origin(kind="device", id="smem_per_block"),
            unit="smem_bytes",
            expression=smem,
            limit=511,
        )
    )
    return model


def _unsupported_dtype() -> ConstraintModel:
    model = ConstraintModel(name="unsat::unsupported_dtype")
    model.add_variable(BoolVar("dtype_float8"))
    model.add_constraint(
        Equality(
            name="requested_dtype_float8",
            category=ConstraintCategory.SCHEDULE,
            explanation="caller requested the float8 compute dtype via hints",
            origin=Origin(kind="schedule_param", id="dtype_hint"),
            lhs=LinearExpr.var("dtype_float8"),
            rhs=LinearExpr.const(1),
        )
    )
    model.add_constraint(
        Equality(
            name="anchor_supported_dtypes",
            category=ConstraintCategory.ANCHOR_CAPABILITY,
            explanation=(
                "the selected anchor supports only float32/float16/bfloat16 "
                "for this visitor"
            ),
            origin=Origin(kind="anchor", id="routed_reduction_row_scale_epilogue_v0"),
            lhs=LinearExpr.var("dtype_float8"),
            rhs=LinearExpr.const(0),
        )
    )
    return model


def _no_device_fits_item() -> ConstraintModel:
    model = ConstraintModel(name="unsat::no_device_fits_expert")
    for device in range(2):
        model.add_variable(BoolVar(f"assign_expert_d{device}"))
    model.add_constraint(
        make_exactly_one(
            name="single_owner_expert",
            category=ConstraintCategory.PLACEMENT,
            explanation="every item has exactly one owner device",
            origin=Origin(kind="placement", id="expert"),
            choices=("assign_expert_d0", "assign_expert_d1"),
        )
    )
    for device in range(2):
        model.add_constraint(
            capacity_bound(
                name=f"device_capacity_d{device}",
                category=ConstraintCategory.PLACEMENT,
                explanation=f"device {device} holds at most 1024 B",
                origin=Origin(kind="device", id=f"d{device}"),
                unit=f"device:{device}",
                expression=LinearExpr.var(f"assign_expert_d{device}") * 4096,
                limit=1024,
            )
        )
    return model


def _replication_exceeds_mesh() -> ConstraintModel:
    model = ConstraintModel(name="unsat::replication_exceeds_mesh")
    for device in range(2):
        model.add_variable(BoolVar(f"assign_page_d{device}"))
    model.add_constraint(
        Equality(
            name="replication_factor_three",
            category=ConstraintCategory.PLACEMENT,
            explanation="declared replication factor is 3 copies per page",
            origin=Origin(kind="placement", id="page_replication"),
            lhs=LinearExpr(terms=(("assign_page_d0", 1), ("assign_page_d1", 1))),
            rhs=LinearExpr.const(3),
        )
    )
    return model


def _deterministic_with_atomic_anchor() -> ConstraintModel:
    model = ConstraintModel(name="unsat::deterministic_atomic_anchor")
    model.add_variable(BoolVar("decomp_per_route"))
    model.add_variable(BoolVar("atomic_anchor_available"))
    model.add_constraint(
        Equality(
            name="only_atomic_nondeterministic_anchor",
            category=ConstraintCategory.ANCHOR_CAPABILITY,
            explanation=(
                "the only grad-value lowering uses cross-program relaxed "
                "atomics (nondeterministic accumulation order)"
            ),
            origin=Origin(kind="anchor", id="grad_values_per_route"),
            lhs=LinearExpr.var("atomic_anchor_available"),
            rhs=LinearExpr.const(1),
        )
    )
    model.add_constraint(
        Equality(
            name="deterministic_mode_requested",
            category=ConstraintCategory.DETERMINISM,
            explanation=(
                "deterministic mode requires bitwise-stable accumulation "
                "ordering across runs"
            ),
            origin=Origin(kind="intent", id="deterministic"),
            lhs=LinearExpr.var("decomp_per_route"),
            rhs=LinearExpr.const(1),
        )
    )
    model.add_constraint(
        Implication(
            name="deterministic_forbids_cross_program_atomics",
            category=ConstraintCategory.DETERMINISM,
            explanation=(
                "per-route accumulation orders float adds nondeterministically; "
                "deterministic mode forbids it"
            ),
            origin=Origin(kind="intent", id="deterministic"),
            guard_variable="decomp_per_route",
            consequents=(
                Equality(
                    name="deterministic_forbids_cross_program_atomics::reject",
                    category=ConstraintCategory.DETERMINISM,
                    explanation="forbidden under deterministic mode",
                    origin=Origin(kind="intent", id="deterministic"),
                    lhs=LinearExpr.const(1),
                    rhs=LinearExpr.const(0),
                ),
            ),
        )
    )
    return model


def _transactional_without_commit_lowering() -> ConstraintModel:
    model = ConstraintModel(name="unsat::transactional_no_commit")
    model.add_variable(BoolVar("commit_boundary_required"))
    model.add_variable(BoolVar("commit_capable_lowering_registered"))
    model.add_constraint(
        Equality(
            name="program_requires_commit_boundary",
            category=ConstraintCategory.SEMANTIC,
            explanation=(
                "the semantic program declares a transactional version "
                "boundary that must publish exactly one merged write"
            ),
            origin=Origin(kind="semantic_op", id="state_update"),
            lhs=LinearExpr.var("commit_boundary_required"),
            rhs=LinearExpr.const(1),
        )
    )
    model.add_constraint(
        Equality(
            name="no_commit_capable_lowering",
            category=ConstraintCategory.ANCHOR_CAPABILITY,
            explanation=(
                "no registered lowering for this state family declares "
                "commit capability"
            ),
            origin=Origin(kind="anchor", id="page_gather_update_reserved"),
            lhs=LinearExpr.var("commit_capable_lowering_registered"),
            rhs=LinearExpr.const(0),
        )
    )
    model.add_constraint(
        Implication(
            name="commit_requires_capable_lowering",
            category=ConstraintCategory.TRAINING,
            explanation="a commit boundary requires a commit-capable lowering",
            origin=Origin(kind="semantic_op", id="state_update"),
            guard_variable="commit_boundary_required",
            consequents=(
                Equality(
                    name="commit_requires_capable_lowering::need",
                    category=ConstraintCategory.ANCHOR_CAPABILITY,
                    explanation="capability must be present when committing",
                    origin=Origin(kind="semantic_op", id="state_update"),
                    lhs=LinearExpr.var("commit_capable_lowering_registered"),
                    rhs=LinearExpr.const(1),
                ),
            ),
        )
    )
    return model


def _push_dispatch_missing_return() -> ConstraintModel:
    model = ConstraintModel(name="unsat::push_missing_return")
    model.add_variable(BoolVar("dispatch_sent"))
    model.add_variable(BoolVar("return_path_planned"))
    model.add_constraint(
        Equality(
            name="dispatch_moves_payload_off_token_owner",
            category=ConstraintCategory.COMMUNICATION,
            explanation="MoE dispatch sends the token payload to the expert owner",
            origin=Origin(kind="route_protocol", id="push_dispatch_return"),
            lhs=LinearExpr.var("dispatch_sent"),
            rhs=LinearExpr.const(1),
        )
    )
    model.add_constraint(
        Equality(
            name="no_return_capacity_declared",
            category=ConstraintCategory.COMMUNICATION,
            explanation=(
                "the plan declares no return path from expert owner back to "
                "the token owner"
            ),
            origin=Origin(kind="route_protocol", id="push_dispatch_return"),
            lhs=LinearExpr.var("return_path_planned"),
            rhs=LinearExpr.const(0),
        )
    )
    model.add_constraint(
        Implication(
            name="dispatch_requires_exactly_one_return",
            category=ConstraintCategory.COMMUNICATION,
            explanation="every dispatched route requires exactly one return",
            origin=Origin(kind="route_protocol", id="push_dispatch_return"),
            guard_variable="dispatch_sent",
            consequents=(
                Equality(
                    name="dispatch_requires_exactly_one_return::need",
                    category=ConstraintCategory.COMMUNICATION,
                    explanation="return path must exist for dispatched routes",
                    origin=Origin(kind="route_protocol", id="push_dispatch_return"),
                    lhs=LinearExpr.var("return_path_planned"),
                    rhs=LinearExpr.const(1),
                ),
            ),
        )
    )
    return model


REPRESENTATIVE_UNSAT_CASES: tuple[UnsatCase, ...] = (
    UnsatCase(
        name="training_with_forward_only_anchor",
        category="intent_conflict",
        build=_training_with_forward_only_anchor,
        expected_core_substring="forward_only",
        concise_message=(
            "No legal training plan: the selected anchor declares forward-only "
            "execution, but training requires verified backward."
        ),
    ),
    UnsatCase(
        name="tile_incompatible_with_vector_width",
        category="schedule_resource",
        build=_tile_incompatible_with_vector_width,
        expected_core_substring="lane_alignment",
        concise_message=(
            "Hinted BLOCK_D=48 cannot tile whole 32-lane vector groups; use a "
            "multiple of 32 such as 32/64/128/256."
        ),
    ),
    UnsatCase(
        name="shared_memory_budget_exceeded",
        category="resource",
        build=_shared_memory_exceeded,
        expected_core_substring="shared_mem_budget",
        concise_message=(
            "Every supported tile configuration exceeds the 16 KiB "
            "shared-memory budget at 4 stages of fp32 staging."
        ),
    ),
    UnsatCase(
        name="requested_dtype_unsupported",
        category="anchor_capability",
        build=_unsupported_dtype,
        expected_core_substring="supported_dtypes",
        concise_message=(
            "Requested dtype float8 is not supported by the selected anchor "
            "(supported: float32, float16, bfloat16)."
        ),
    ),
    UnsatCase(
        name="no_device_has_enough_memory",
        category="placement_capacity",
        build=_no_device_fits_item,
        expected_core_substring="device_capacity",
        concise_message=(
            "No device on the 2-device mesh can hold the expert (4096 B > "
            "1024 B capacity everywhere)."
        ),
    ),
    UnsatCase(
        name="replication_factor_exceeds_mesh",
        category="placement_capacity",
        build=_replication_exceeds_mesh,
        expected_core_substring="replication_factor",
        concise_message=(
            "Replication factor 3 exceeds the 2-device mesh; each copy needs "
            "its own device."
        ),
    ),
    UnsatCase(
        name="deterministic_merge_with_atomic_anchor",
        category="determinism",
        build=_deterministic_with_atomic_anchor,
        expected_core_substring="deterministic_forbids",
        concise_message=(
            "Deterministic merge was requested but every available grad-value "
            "lowering accumulates through relaxed cross-program atomics whose "
            "float-add order is not bitwise reproducible."
        ),
    ),
    UnsatCase(
        name="transactional_update_without_commit_lowering",
        category="anchor_capability",
        build=_transactional_without_commit_lowering,
        expected_core_substring="commit_capable",
        concise_message=(
            "The program declares a transactional boundary but no registered "
            "lowering can publish a versioned commit."
        ),
    ),
    UnsatCase(
        name="push_dispatch_missing_return_path",
        category="communication",
        build=_push_dispatch_missing_return,
        expected_core_substring="return",
        concise_message=(
            "Push dispatch moves payloads to expert owners, but no return path "
            "is planned; every dispatched route requires exactly one return."
        ),
    ),
)


def describe_unsat(case: UnsatCase, core_names: tuple[str, ...]) -> dict[str, object]:
    """Structured diagnostic record for one impossible case."""
    hit = case.expected_core_substring in " ".join(core_names)
    return {
        "case": case.name,
        "category": case.category,
        "status": "unsat",
        "core_names": list(core_names),
        "core_mapped": hit,
        "concise_message": case.concise_message,
        "diagnostic_code": DiagnosticCode.UNSAT_CONSTRAINTS.value,
        "severity": Severity.ERROR.value,
    }


__all__ = [
    "REPRESENTATIVE_UNSAT_CASES",
    "UnsatCase",
    "describe_unsat",
]
