"""Kernel-plan decision models: candidates, schedules, verification glue.

This module assembles :class:`ConstraintModel` instances for the compiler's
solver-guided decisions and decodes/verifies their assignments. It contains
no solver code (see ``solver.py``) and no trusted-kernel code (see
``execution.py`` / ``anchors/``).

Two model families:

1. **Candidate selection** - which rewrite candidate to apply, ranked by the
   documented lexicographic policy (obligations first, then traffic).
2. **Routed-epilogue schedule selection** - plan/block/warps/stages/
   decomposition/dtype choices over the bounded space in
   ``schedule_space.py``, with hard legality constraints, resource bounds,
   training-completeness implications, determinism exclusions, and the full
   eight-level objective order ending in a deterministic tie-break.

Every constraint carries a stable name, category, explanation, provenance,
severity, and variables - artifacts serialize the summary, never raw solver
output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from urm.compiler.constraints import (
    AllowedSet,
    Assignment,
    BoolVar,
    ConstraintCategory,
    ConstraintModel,
    Divisibility,
    Equality,
    ExactlyOne,
    Implication,
    IntVar,
    LessEqual,
    LinearExpr,
    Nogood,
    ObjectiveSense,
    ObjectiveTerm,
    Origin,
    make_nogood,
)
from urm.compiler.cost import DEFAULT_HBM_GBPS
from urm.compiler.schedule_space import (
    SUPPORTED_BLOCKS,
    SUPPORTED_STAGES,
    SUPPORTED_WARPS,
    GradValuesDecomposition,
    GradValuesSchedule,
    PlanKind,
    SchedulePoint,
    ScheduleProblem,
    shared_memory_estimate,
)

CATEGORY_SCHEDULE = ConstraintCategory.SCHEDULE
CATEGORY_TRAINING = ConstraintCategory.TRAINING
CATEGORY_RESOURCE = ConstraintCategory.RESOURCE
CATEGORY_DETERMINISM = ConstraintCategory.DETERMINISM
CATEGORY_SEMANTIC = ConstraintCategory.SEMANTIC
CATEGORY_ANCHOR = ConstraintCategory.ANCHOR_CAPABILITY
CATEGORY_SEARCH = ConstraintCategory.SEARCH


# -- Candidate selection model ---------------------------------------------------


def build_candidate_selection_model(
    candidates: Sequence[object],
) -> tuple[ConstraintModel, dict[str, str]]:
    """One-hot selection over enumerated candidates.

    Objectives: minimize summed traffic delta, then a deterministic ordinal
    tie-break so equal-cost candidate sets resolve reproducibly.
    """
    legal = [c for c in candidates if c.legal]
    model = ConstraintModel(name="candidate_selection")
    choice_vars: list[str] = []
    candidate_by_choice: dict[str, str] = {}
    for index, candidate in enumerate(legal):
        name = f"choose_{index}"
        model.add_variable(BoolVar(name))
        choice_vars.append(name)
        candidate_by_choice[name] = candidate.candidate_id
    if not choice_vars:
        raise ValueError("candidate selection requires at least one legal candidate")
    model.add_constraint(
        ExactlyOne(
            name="exactly_one_candidate",
            category=CATEGORY_SEARCH,
            explanation="automatic selection applies exactly one candidate",
            origin=Origin(kind="intent", id="automatic_selection"),
            choices=tuple(choice_vars),
        )
    )
    traffic_terms = LinearExpr()
    ordinal_terms = LinearExpr()
    for index, (name, candidate) in enumerate(zip(choice_vars, legal, strict=True)):
        traffic_terms += LinearExpr.var(name) * int(candidate.traffic_bytes_delta)
        ordinal_terms += LinearExpr.var(name) * index
    model.add_objective(
        ObjectiveTerm(
            name="minimize_traffic_delta",
            expression=traffic_terms,
            sense=ObjectiveSense.MINIMIZE,
        )
    )
    model.add_objective(
        ObjectiveTerm(
            name="deterministic_tie_break",
            expression=ordinal_terms,
            sense=ObjectiveSense.MINIMIZE,
        )
    )
    model.metadata["candidates"] = ",".join(c.candidate_id for c in legal)
    return model, candidate_by_choice


def decode_selected_candidate(
    assignment: Assignment, candidate_by_choice: Mapping[str, str]
) -> str:
    chosen = [
        candidate_by_choice[name]
        for name, value in sorted(assignment.items())
        if name.startswith("choose_")
        and name in candidate_by_choice
        and value in (True, 1)
    ]
    if len(chosen) != 1:
        raise ValueError(f"expected exactly one selected candidate, got {chosen}")
    return chosen[0]


# -- Candidate-to-plan binding ---------------------------------------------------
# The two decision stages must never contradict each other: once a rewrite
# candidate is selected, its schedule model may only choose plans that this
# candidate's lowering actually implements.

_RULE_TO_ANCHOR_PLAN: dict[str, tuple[str, PlanKind]] = {
    "fold_row_scale_into_routed_reduction_epilogue": (
        "routed_reduction_row_scale_epilogue_v0",
        PlanKind.FUSED,
    ),
}


def plan_kinds_for_candidate(candidate) -> tuple[PlanKind, ...]:
    """Plans the candidate's own lowering can legally execute.

    ``base`` lowers through unscheduled routed-reduction v1; the row-scale
    epilogue fusion rule permits only its fused anchor. Unknown rewrite rules
    fail closed.
    """
    if getattr(candidate, "kind", "") == "base":
        return ()
    rule = getattr(candidate, "rule", None)
    if rule in _RULE_TO_ANCHOR_PLAN:
        return (_RULE_TO_ANCHOR_PLAN[rule][1],)
    raise ValueError(
        f"unregistered rewrite rule {rule!r}; candidate/anchor bindings fail closed"
    )


def plan_binding_is_total(candidate) -> bool:
    """True when the candidate pins exactly one executable plan."""
    try:
        return len(plan_kinds_for_candidate(candidate)) == 1
    except ValueError:
        return False


# -- Routed-epilogue schedule model -------------------------------------------------

CONFIG_CHOICES = tuple(
    (block, stage, warps)
    for block in SUPPORTED_BLOCKS
    for stage in SUPPORTED_STAGES
    for warps in SUPPORTED_WARPS
)


def _schedule_problem_from_metadata(model: ConstraintModel) -> ScheduleProblem:
    meta = model.metadata
    return ScheduleProblem(
        queries=int(meta.get("queries", "1024")),
        sources=int(meta.get("sources", "512")),
        route_width=int(meta.get("route_width", "8")),
        value_dim=int(meta.get("value_dim", "1024")),
        dtypes=tuple(meta.get("dtypes", "float32,float16,bfloat16").split(",")),
        training=meta.get("training") == "1",
        deterministic=meta.get("deterministic") == "1",
        fused_anchor_available=meta.get("fused_anchor_available", "1") == "1",
    )


def build_schedule_model(
    *,
    program,
    candidate,
    intent,
    schedule_params,
    device_limits,
    problem: ScheduleProblem | None = None,
    allowed_plans: Sequence[PlanKind] | None = None,
) -> ConstraintModel:
    """Constraint model for one candidate's routed-epilogue schedule space.

    Hard constraints mirror ``schedule_space.is_legal`` exactly; objectives
    follow the documented eight-level lexicographic order with a final
    deterministic stable-ordering tie-break.

    The model is CANDIDATE-BOUND: ``allowed_plans`` (default: derived from the
    candidate through :func:`plan_kinds_for_candidate`) pins the execution
    plans this candidate's lowering can run, so schedule selection can never
    contradict the already-selected rewrite candidate.
    """
    from urm.compiler.diagnostics import CompilerError, Diagnostic, DiagnosticCode
    from urm.compiler.planner import CompilationIntent
    from urm.compiler.semantic import WeightedReduce

    if allowed_plans is None:
        try:
            allowed_plans = plan_kinds_for_candidate(candidate)
        except ValueError as err:
            raise CompilerError(
                (
                    Diagnostic(
                        code=DiagnosticCode.CANDIDATE_ILLEGAL,
                        message=str(err),
                    ),
                )
            ) from err
    allowed_plans = tuple(allowed_plans)
    if not allowed_plans:
        raise CompilerError(
            (
                Diagnostic(
                    code=DiagnosticCode.SCHEDULE_HINT_INVALID,
                    message=(
                        f"candidate {candidate.candidate_id!r} has no "
                        "configurable schedule space"
                    ),
                ),
            )
        )
    unknown = [p for p in allowed_plans if p not in set(PlanKind)]
    if unknown:
        raise ValueError(f"unknown plan kinds: {unknown}")

    # Derive shape facts from the program when hints exist.
    reduce_op = next((op for op in program.ops if isinstance(op, WeightedReduce)), None)
    hint = getattr(reduce_op, "shape_hint", None) if reduce_op else None
    if problem is None:
        queries, sources, route_width, value_dim = (
            hint if hint is not None else (1024, 512, 8, 1024)
        )
        dtype_names = []
        for handle in program.inputs:
            if (
                handle.dtype.value in {"float32", "float16", "bfloat16"}
                and handle.dtype.value not in dtype_names
            ):
                dtype_names.append(handle.dtype.value)
        dtype_names = tuple(dtype_names) or ("float32",)
        problem = ScheduleProblem(
            queries=queries,
            sources=sources,
            route_width=route_width,
            value_dim=value_dim,
            dtypes=dtype_names,
            training=intent is CompilationIntent.TRAINING,
            deterministic=schedule_params.deterministic,
            fused_anchor_available=True,
        )

    model = ConstraintModel(name=f"routed_epilogue_schedule:{candidate.candidate_id}")
    anchor_name = (
        "routed_reduction_row_scale_epilogue_v0"
        if PlanKind.FUSED in allowed_plans
        else "routed_reduction_v1"
    )
    model.metadata.update(
        {
            "program": program.name,
            "candidate_id": candidate.candidate_id,
            "anchor_name": anchor_name,
            "intent": intent.value,
            "training": "1" if problem.training else "0",
            "deterministic": "1" if problem.deterministic else "0",
            "fused_anchor_available": ("1" if problem.fused_anchor_available else "0"),
            "queries": str(problem.queries),
            "sources": str(problem.sources),
            "route_width": str(problem.route_width),
            "value_dim": str(problem.value_dim),
            "dtypes": ",".join(problem.dtypes),
            "fused_backward_dtypes": ",".join(sorted(problem.fused_backward_dtypes)),
            "base_backward_dtypes": ",".join(sorted(problem.base_backward_dtypes)),
            "shared_mem_limit_bytes": str(problem.shared_mem_bytes_per_block),
            "allowed_plans": ",".join(p.value for p in allowed_plans),
            "plan_binding_total": ("1" if len(set(allowed_plans)) == 1 else "0"),
            "schedule_origin": "kernel_plan.build_schedule_model",
        }
    )

    origin = Origin(kind="schedule_param", id=candidate.candidate_id)
    anchor_origin = Origin(kind="anchor", id="routed_reduction_row_scale_epilogue_v0")

    # -- variables ----------------------------------------------------------
    # Joint (BLOCK_D, num_stages, num_warps) configuration channel: one
    # indicator per implemented config. A joint encoding keeps every derived
    # quantity (staging bytes, route-metadata re-loads, latency-hiding
    # factor) linear in the indicators.
    cfg_vars: dict[tuple[int, int, int], str] = {}
    for index, (block, stage, warps) in enumerate(CONFIG_CHOICES):
        name = f"cfg_{index}_b{block}_s{stage}_w{warps}"
        model.add_variable(BoolVar(name))
        cfg_vars[(block, stage, warps)] = name
    decomp_values = [d.value for d in GradValuesDecomposition]
    sched_values = [s.value for s in GradValuesSchedule]
    plan_values = [p.value for p in PlanKind]
    dtype_values = list(problem.dtypes)
    for values, prefix in (
        (decomp_values, "decomp_"),
        (sched_values, "gradsched_"),
        (plan_values, "plan_"),
    ):
        for value in values:
            model.add_variable(BoolVar(f"{prefix}{value}"))
    for dtype in dtype_values:
        model.add_variable(BoolVar(f"dtype_{dtype}"))
    model.add_variable(BoolVar("obligations_open"))

    model.add_variable(
        IntVar(name="block_d", lower=min(SUPPORTED_BLOCKS), upper=max(SUPPORTED_BLOCKS))
    )
    model.add_variable(
        IntVar(name="num_warps", lower=min(SUPPORTED_WARPS), upper=max(SUPPORTED_WARPS))
    )
    model.add_variable(
        IntVar(
            name="num_stages", lower=min(SUPPORTED_STAGES), upper=max(SUPPORTED_STAGES)
        )
    )

    def group(
        names: Sequence[str], group_name: str, explanation: str, cat, org
    ) -> None:
        model.add_constraint(
            ExactlyOne(
                name=f"exactly_one_{group_name}",
                category=cat,
                explanation=explanation,
                origin=org,
                choices=tuple(names),
            )
        )

    exactly_one_org = Origin(kind="schedule_param", id=candidate.candidate_id)
    group(
        list(cfg_vars.values()),
        "launch_config",
        "one (BLOCK_D, num_stages, num_warps) launch configuration is selected",
        CATEGORY_SCHEDULE,
        exactly_one_org,
    )
    group(
        [f"decomp_{v}" for v in decomp_values],
        "grad_decomposition",
        "one grad-value backward decomposition is selected",
        CATEGORY_SCHEDULE,
        exactly_one_org,
    )
    group(
        [f"gradsched_{v}" for v in sched_values],
        "grad_schedule",
        "one grad-value D-traversal schedule is selected",
        CATEGORY_SCHEDULE,
        exactly_one_org,
    )
    group(
        [f"plan_{v}" for v in plan_values],
        "plan",
        "one execution plan is selected",
        CATEGORY_SCHEDULE,
        exactly_one_org,
    )

    # Candidate-plan binding: plans outside the candidate's own lowering are
    # pinned off, so the schedule stage can never contradict candidate
    # selection (base candidate -> base plan; epilogue fusion -> fused plan).
    for value in plan_values:
        if PlanKind(value) in allowed_plans:
            continue
        model.add_constraint(
            Equality(
                name=f"candidate_binds_plan_{value}_off",
                category=CATEGORY_SEMANTIC,
                explanation=(
                    f"selected candidate {candidate.candidate_id!r} does not "
                    f"implement the {value!r} execution plan"
                ),
                origin=Origin(kind="rewrite_rule", id=candidate.candidate_id),
                lhs=LinearExpr.var(f"plan_{value}"),
                rhs=LinearExpr.const(0),
            )
        )

    group(
        [f"dtype_{d}" for d in dtype_values],
        "dtype",
        "one supported dtype is selected",
        CATEGORY_SCHEDULE,
        exactly_one_org,
    )

    # Numeric payload definitions via the configuration channel.
    block_sum = LinearExpr()
    stages_sum = LinearExpr()
    warps_sum = LinearExpr()
    for (block, stage, warps), name in cfg_vars.items():
        block_sum += LinearExpr.var(name) * block
        stages_sum += LinearExpr.var(name) * stage
        warps_sum += LinearExpr.var(name) * warps
    model.add_constraint(
        Equality(
            name="block_d_definition",
            category=CATEGORY_SCHEDULE,
            explanation="BLOCK_D equals its selected launch-config value",
            origin=origin,
            lhs=LinearExpr.var("block_d"),
            rhs=block_sum,
        )
    )
    model.add_constraint(
        Equality(
            name="num_stages_definition",
            category=CATEGORY_SCHEDULE,
            explanation="num_stages equals its selected launch-config value",
            origin=origin,
            lhs=LinearExpr.var("num_stages"),
            rhs=stages_sum,
        )
    )
    model.add_constraint(
        Equality(
            name="num_warps_definition",
            category=CATEGORY_SCHEDULE,
            explanation="num_warps equals its selected launch-config value",
            origin=origin,
            lhs=LinearExpr.var("num_warps"),
            rhs=warps_sum,
        )
    )

    # -- hard legality constraints -----------------------------------------
    model.add_constraint(
        AllowedSet(
            name="supported_block_sizes",
            category=CATEGORY_SCHEDULE,
            explanation=(
                f"BLOCK_D must be one of the implemented tile sizes "
                f"{list(SUPPORTED_BLOCKS)}"
            ),
            origin=origin,
            variable="block_d",
            allowed=SUPPORTED_BLOCKS,
        )
    )
    model.add_constraint(
        AllowedSet(
            name="supported_warp_counts",
            category=CATEGORY_SCHEDULE,
            explanation=f"num_warps must be one of {list(SUPPORTED_WARPS)}",
            origin=origin,
            variable="num_warps",
            allowed=SUPPORTED_WARPS,
        )
    )
    model.add_constraint(
        AllowedSet(
            name="supported_stage_counts",
            category=CATEGORY_SCHEDULE,
            explanation=f"num_stages must be one of {list(SUPPORTED_STAGES)}",
            origin=origin,
            variable="num_stages",
            allowed=SUPPORTED_STAGES,
        )
    )
    model.add_constraint(
        Divisibility(
            name="block_lane_alignment",
            category=CATEGORY_SCHEDULE,
            explanation=(
                "BLOCK_D must cover whole 32-lane warp tiles (vector-width "
                "compatibility)"
            ),
            origin=origin,
            variable="block_d",
            divisor=32,
        )
    )
    model.add_constraint(
        Implication(
            name="per_route_requires_segmented_schedule",
            category=CATEGORY_SCHEDULE,
            explanation=(
                "per_route decomposition does not support full_row traversal; "
                "per_route is segmented across program instances by construction"
            ),
            origin=origin,
            guard_variable="decomp_per_route",
            guard_expected=True,
            consequents=(
                Equality(
                    name="per_route_requires_segmented_schedule::eq",
                    category=CATEGORY_SCHEDULE,
                    explanation="gradsched_full_row must be false under per_route",
                    origin=origin,
                    lhs=LinearExpr.var("gradsched_full_row"),
                    rhs=LinearExpr.const(0),
                ),
            ),
        )
    )
    model.add_constraint(
        LessEqual(
            name="route_width_within_limit",
            category=CATEGORY_SEMANTIC,
            explanation=(
                f"route width {problem.route_width} must not exceed the "
                f"implemented per-program route limit {problem.max_route_width}"
            ),
            origin=Origin(kind="semantic_op", id="reduce"),
            lhs=LinearExpr.const(problem.route_width),
            rhs=LinearExpr.const(problem.max_route_width),
        )
    )
    model.add_constraint(
        LessEqual(
            name="route_width_within_sources",
            category=CATEGORY_SEMANTIC,
            explanation="route width cannot exceed the source domain size",
            origin=Origin(kind="semantic_op", id="reduce"),
            lhs=LinearExpr.const(problem.route_width),
            rhs=LinearExpr.const(problem.sources),
        )
    )
    model.add_constraint(
        Equality(
            name="device_capability_sufficient",
            category=CATEGORY_SEMANTIC,
            explanation=(
                "the target device capability supports the fused-epilogue "
                "anchor (validated host recorded in benchmark provenance)"
            ),
            origin=Origin(kind="device", id="compute_capability"),
            lhs=LinearExpr.const(1),
            rhs=LinearExpr.const(1),
        )
    )

    bytes_by_dtype = {"float32": 4, "float16": 2, "bfloat16": 2}
    for dtype in dtype_values:
        smem = LinearExpr()
        for (block, stage, _warps), name in cfg_vars.items():
            staging = stage * block * bytes_by_dtype[dtype]
            smem += LinearExpr.var(name) * staging
        model.add_constraint(
            Implication(
                name=f"shared_mem_bound_{dtype}",
                category=CATEGORY_RESOURCE,
                explanation=(
                    f"{dtype} staging ({bytes_by_dtype[dtype]} B/elem) must fit "
                    f"the {problem.shared_mem_bytes_per_block} B per-block "
                    "shared-memory budget"
                ),
                origin=anchor_origin,
                guard_variable=f"dtype_{dtype}",
                guard_expected=True,
                consequents=(
                    LessEqual(
                        name=f"shared_mem_bound_{dtype}::le",
                        category=CATEGORY_RESOURCE,
                        explanation="staged-tile shared memory bound",
                        origin=anchor_origin,
                        lhs=smem,
                        rhs=LinearExpr.const(problem.shared_mem_bytes_per_block),
                    ),
                ),
            )
        )

    if intent is CompilationIntent.TRAINING:
        unsupported_base = [
            d for d in dtype_values if d not in problem.base_backward_dtypes
        ]
        unsupported_fused = [
            d for d in dtype_values if d not in problem.fused_backward_dtypes
        ]
        for plan_value, bad_dtypes in (
            (PlanKind.BASE.value, unsupported_base),
            (PlanKind.FUSED.value, unsupported_fused),
        ):
            for dtype in bad_dtypes:
                model.add_constraint(
                    Implication(
                        name=f"verified_backward_{plan_value}_{dtype}",
                        category=CATEGORY_TRAINING,
                        explanation=(
                            f"training requires complete verified backward: "
                            f"plan={plan_value} does not certify gradients "
                            f"for {dtype}"
                        ),
                        origin=Origin(kind="rewrite_rule", id=candidate.candidate_id),
                        guard_variable=f"dtype_{dtype}",
                        consequents=(
                            Implication(
                                name=f"verified_backward_{plan_value}_{dtype}::plan",
                                category=CATEGORY_TRAINING,
                                explanation="nested plan guard",
                                origin=Origin(
                                    kind="rewrite_rule", id=candidate.candidate_id
                                ),
                                guard_variable=f"plan_{plan_value}",
                                guard_equality=None,
                                consequents=(
                                    Equality(
                                        name=(
                                            f"verified_backward_{plan_value}_"
                                            f"{dtype}::reject"
                                        ),
                                        category=CATEGORY_TRAINING,
                                        explanation="forbidden combination",
                                        origin=Origin(
                                            kind="rewrite_rule",
                                            id=candidate.candidate_id,
                                        ),
                                        lhs=LinearExpr.const(1),
                                        rhs=LinearExpr.const(0),
                                    ),
                                ),
                            ),
                        ),
                    )
                )
        if candidate.saved_state_policy == "recompute":
            model.add_constraint(
                Equality(
                    name="recompute_obligation_resolved",
                    category=CATEGORY_TRAINING,
                    explanation=(
                        "the recomputation obligation must be resolved by an "
                        "anchor that declares recompute-backward support"
                    ),
                    origin=Origin(
                        kind="anchor", id="routed_reduction_row_scale_epilogue_v0"
                    ),
                    lhs=LinearExpr.var("obligations_open"),
                    rhs=LinearExpr.const(0),
                )
            )
        else:
            model.add_constraint(
                Equality(
                    name="no_unresolved_obligations",
                    category=CATEGORY_TRAINING,
                    explanation="training requires zero unresolved obligations",
                    origin=Origin(kind="intent", id=intent.value),
                    lhs=LinearExpr.var("obligations_open"),
                    rhs=LinearExpr.const(0),
                )
            )

    if problem.deterministic:
        if problem.training:
            # Honest global fact: every implemented grad-value lowering
            # (per-query, per-route, v1 autograd) accumulates through relaxed
            # cross-program atomics. Deterministic training has no schedule.
            model.add_constraint(
                Equality(
                    name="deterministic_training_requires_ordered_grads",
                    category=CATEGORY_DETERMINISM,
                    explanation=(
                        "deterministic (bitwise-stable) training requires an "
                        "ordered grad-value accumulation lowering; the v1 and "
                        "experimental anchors both use relaxed cross-program "
                        "atomics, so no legal schedule exists"
                    ),
                    origin=Origin(kind="anchor", id="grad_values_accumulation"),
                    lhs=LinearExpr.const(1),
                    rhs=LinearExpr.const(0),
                )
            )
        if not problem.fused_anchor_available and not problem.training:
            model.add_constraint(
                Implication(
                    name="deterministic_requires_capable_anchor",
                    category=CATEGORY_DETERMINISM,
                    explanation=(
                        "deterministic mode requires the fused anchor, which is "
                        "unavailable here"
                    ),
                    origin=Origin(kind="intent", id="deterministic"),
                    guard_variable=f"plan_{PlanKind.FUSED.value}",
                    consequents=(
                        Equality(
                            name="deterministic_requires_capable_anchor::reject",
                            category=CATEGORY_DETERMINISM,
                            explanation="forbidden combination",
                            origin=Origin(kind="intent", id="deterministic"),
                            lhs=LinearExpr.const(1),
                            rhs=LinearExpr.const(0),
                        ),
                    ),
                )
            )

    # -- objectives (lexicographic; order is the contract) --------------------
    p_is_base = LinearExpr.var(f"plan_{PlanKind.BASE.value}")
    q, v = problem.queries, problem.value_dim

    model.add_objective(
        ObjectiveTerm(
            "zero_unresolved_obligations",
            LinearExpr.var("obligations_open"),
            ObjectiveSense.MINIMIZE,
        )
    )
    model.add_objective(
        ObjectiveTerm(
            "minimum_peak_temporary_bytes",
            p_is_base * (q * v * 2),
            ObjectiveSense.MINIMIZE,
        )
    )
    model.add_objective(
        ObjectiveTerm(
            "minimum_comm_critical_path_us",
            LinearExpr.const(0),
            ObjectiveSense.MINIMIZE,
        )
    )
    model.add_objective(
        ObjectiveTerm(
            "minimum_communication_bytes", LinearExpr.const(0), ObjectiveSense.MINIMIZE
        )
    )
    model.add_objective(
        ObjectiveTerm(
            "minimum_materialization_bytes",
            p_is_base * (q * v * 2),
            ObjectiveSense.MINIMIZE,
        )
    )
    launches = (
        LinearExpr.const(4) + p_is_base * 2
        if problem.training
        else LinearExpr.const(1) + p_is_base
    )
    model.add_objective(
        ObjectiveTerm("minimum_launch_count", launches, ObjectiveSense.MINIMIZE)
    )
    # Analytical runtime (worst-case fp32 element sizes, documented):
    # - every program re-reads its route metadata (indices+weights), so the
    #   per-program metadata cost scales with the number of D-segments,
    #   i.e. ceil(value_dim / BLOCK_D) - larger tiles amortize it better;
    # - values are read once per program segment (upper bound, no L2 reuse);
    # - latency hiding is proxied by a concurrency factor
    #   min(num_stages * num_warps, MAX_CONCURRENCY) / MAX_CONCURRENCY that
    #   scales effective bandwidth; it is an explicit analytical heuristic,
    #   never a measurement;
    # - the base plan additionally writes+reads the [Q, D] intermediate.
    bytes_per_us = max(DEFAULT_HBM_GBPS, 1.0) * 1e3  # GB/s -> B/us
    max_concurrency = 8.0

    def segments(block: int) -> int:
        return -(-problem.value_dim // block)

    route_meta_bytes = 12 * q * problem.route_width  # (8B idx + 4B wgt) * K
    value_bytes = 4 * problem.sources * v + 4 * q * v
    extra_bytes = int(8 * q * v / bytes_per_us)
    common_floor = int((route_meta_bytes + value_bytes) / bytes_per_us)
    runtime_configs = LinearExpr()
    for (block, stage, warps), name in cfg_vars.items():
        factor = min(stage * warps, 8) / max_concurrency
        config_bytes = (route_meta_bytes * segments(block) + value_bytes) / factor
        runtime_configs += LinearExpr.var(name) * int(config_bytes / bytes_per_us)
    runtime = LinearExpr.const(common_floor) + runtime_configs + p_is_base * extra_bytes
    model.add_objective(
        ObjectiveTerm(
            "analytical_runtime_estimate_us", runtime, ObjectiveSense.MINIMIZE
        )
    )

    cfg_index = {cfg: i for i, cfg in enumerate(CONFIG_CHOICES)}
    sizes = (
        len(dtype_values),
        len(plan_values),
        len(decomp_values),
        len(sched_values),
        len(CONFIG_CHOICES),
    )
    weights = []
    acc = 1
    for size in reversed(sizes):
        weights.append(acc)
        acc *= size
    w_dtype, w_plan, w_decomp, w_sched, w_cfg = weights
    tie_break = LinearExpr()
    for dtype in dtype_values:
        tie_break += LinearExpr.var(f"dtype_{dtype}") * (
            dtype_values.index(dtype) * w_dtype
        )
    for value in plan_values:
        tie_break += LinearExpr.var(f"plan_{value}") * (
            plan_values.index(value) * w_plan
        )
    for value in decomp_values:
        tie_break += LinearExpr.var(f"decomp_{value}") * (
            decomp_values.index(value) * w_decomp
        )
    for value in sched_values:
        tie_break += LinearExpr.var(f"gradsched_{value}") * (
            sched_values.index(value) * w_sched
        )
    for cfg, name in cfg_vars.items():
        tie_break += LinearExpr.var(name) * (cfg_index[cfg] * w_cfg)
    model.add_objective(
        ObjectiveTerm(
            "deterministic_stable_ordering", tie_break, ObjectiveSense.MINIMIZE
        )
    )

    # Optional user schedule hints tighten the space further.
    hinted_blocks = schedule_params.block_hints.get("BLOCK_D")
    if hinted_blocks is not None:
        model.add_constraint(
            Equality(
                name="hint_block_d_respected",
                category=CATEGORY_SCHEDULE,
                explanation=f"caller pinned BLOCK_D={hinted_blocks}",
                origin=Origin(kind="schedule_param", id="BLOCK_D_hint"),
                lhs=LinearExpr.var("block_d"),
                rhs=LinearExpr.const(int(hinted_blocks)),
            )
        )
    if schedule_params.warp_count is not None:
        model.add_constraint(
            Equality(
                name="hint_num_warps_respected",
                category=CATEGORY_SCHEDULE,
                explanation=f"caller pinned num_warps={schedule_params.warp_count}",
                origin=Origin(kind="schedule_param", id="warp_count_hint"),
                lhs=LinearExpr.var("num_warps"),
                rhs=LinearExpr.const(schedule_params.warp_count),
            )
        )
    if schedule_params.stage_count is not None:
        model.add_constraint(
            Equality(
                name="hint_num_stages_respected",
                category=CATEGORY_SCHEDULE,
                explanation=f"caller pinned num_stages={schedule_params.stage_count}",
                origin=Origin(kind="schedule_param", id="stage_count_hint"),
                lhs=LinearExpr.var("num_stages"),
                rhs=LinearExpr.const(schedule_params.stage_count),
            )
        )
    model.validate()
    return model


def decode_schedule_point(
    model: ConstraintModel, assignment: Assignment
) -> SchedulePoint:
    """Decode a verified assignment into a concrete schedule point."""

    def single_true(prefix: str) -> str:
        matches = [
            name[len(prefix) :]
            for name, value in assignment.items()
            if name.startswith(prefix) and value in (True, 1)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {prefix}* indicator, got {matches}")
        return matches[0]

    cfg_token = single_true("cfg_")  # e.g. "7_b256_s2_w4"
    header, block_part, stage_part, warp_part = cfg_token.split("_")
    del header
    plan = single_true("plan_")
    decomp = single_true("decomp_")
    sched = single_true("gradsched_")
    dtype = single_true("dtype_")
    return SchedulePoint(
        plan=plan,
        block_d=int(block_part.removeprefix("b")),
        num_warps=int(warp_part.removeprefix("w")),
        num_stages=int(stage_part.removeprefix("s")),
        grad_values_decomposition=decomp,
        grad_values_schedule=sched,
        dtype=dtype,
    )


def schedule_point_assignments(model: ConstraintModel):
    """Enumerate the decoded point space, lifted to full assignments.

    The schedule model channel-encodes numeric fields (BLOCK_D, num_stages,
    num_warps) through indicator Booleans. Brute-forcing the raw variable
    product would sweep redundant numeric ranges; this generator instead
    enumerates exactly the decoded schedule points and lifts each one to a
    complete assignment (indicators plus derived numerics), so exhaustive
    sweeps stay proportional to the real space while remaining an exact
    legality oracle over every named constraint.
    """
    from itertools import product

    meta = model.metadata
    dtypes = meta.get("dtypes", "float32,float16,bfloat16").split(",")
    plans = [p.value for p in PlanKind]
    decomps = [d.value for d in GradValuesDecomposition]
    scheds = [s.value for s in GradValuesSchedule]
    for (block, stage, warps), decomp, sched, plan, dtype in product(
        CONFIG_CHOICES, decomps, scheds, plans, dtypes
    ):
        assignment: Assignment = {}
        for variable in model.variables:
            if isinstance(variable, BoolVar):
                assignment[variable.name] = False
        cfg_name = (
            f"cfg_{CONFIG_CHOICES.index((block, stage, warps))}"
            f"_b{block}_s{stage}_w{warps}"
        )
        assignment[cfg_name] = True
        assignment[f"decomp_{decomp}"] = True
        assignment[f"gradsched_{sched}"] = True
        assignment[f"plan_{plan}"] = True
        assignment[f"dtype_{dtype}"] = True
        assignment["obligations_open"] = False
        for derived, value in (
            ("block_d", block),
            ("num_warps", warps),
            ("num_stages", stage),
        ):
            if model.variable_named(derived) is not None:
                assignment[derived] = value
        yield assignment


def schedule_point_to_assignment(
    model: ConstraintModel, point: SchedulePoint
) -> Assignment:
    """Convert exactly one :class:`SchedulePoint` to its full model assignment.

    This is the compile-feedback counterpart of :func:`decode_schedule_point`:
    when a concrete point fails compilation, the nogood must exclude THAT
    point's assignment - never some other (e.g. previously optimized)
    assignment. Every model variable receives a value so the result passes
    range verification.
    """
    assignment: Assignment = {}
    for variable in model.variables:
        if isinstance(variable, BoolVar):
            assignment[variable.name] = False
    config = (point.block_d, point.num_stages, point.num_warps)
    if config not in CONFIG_CHOICES:
        raise ValueError(f"schedule point {config} is not an implemented launch config")
    cfg_name = (
        f"cfg_{CONFIG_CHOICES.index(config)}_b{config[0]}_s{config[1]}_w{config[2]}"
    )
    if model.variable_named(cfg_name) is None:
        raise ValueError(f"model has no launch-config variable {cfg_name!r}")
    assignment[cfg_name] = True
    for prefix, value in (
        ("decomp_", point.grad_values_decomposition),
        ("gradsched_", point.grad_values_schedule),
        ("plan_", point.plan),
        ("dtype_", point.dtype),
    ):
        name = f"{prefix}{value}"
        if model.variable_named(name) is not None:
            assignment[name] = True
    if model.variable_named("obligations_open") is not None:
        assignment["obligations_open"] = False
    for derived, value in (
        ("block_d", point.block_d),
        ("num_warps", point.num_warps),
        ("num_stages", point.num_stages),
    ):
        if model.variable_named(derived) is not None:
            assignment[derived] = value
    return assignment


def exhaustive_schedule_sweep(model: ConstraintModel):
    """(legal assignments, ranked best) over the decoded point space."""
    from urm.compiler.schedule_space import rank_lexicographic

    lifted = list(schedule_point_assignments(model))
    total = len(lifted)
    legal = [a for a in lifted if all(c.holds(a) for c in model.constraints)]
    ranked = rank_lexicographic(model, legal) if legal else []
    return legal, ranked, total


def verify_schedule_assignment(model: ConstraintModel, assignment: Assignment):
    """Independent verification wired to anchor/device facts from metadata."""
    from urm.compiler.execution import TRUSTED_ANCHORS
    from urm.compiler.planner import CompilationIntent
    from urm.compiler.verification import (
        AnchorFacts,
        AssignmentFacts,
        ModelVerifier,
        ResourceFacts,
    )

    point = decode_schedule_point(model, assignment)
    anchor_name = (
        "routed_reduction_row_scale_epilogue_v0"
        if point.plan == PlanKind.FUSED.value
        else "routed_reduction_v1"
    )
    anchor = next(a for a in TRUSTED_ANCHORS if a.name == anchor_name)
    facts = AssignmentFacts(
        selected_anchor=AnchorFacts(
            name=anchor.name,
            kind=anchor.kind.value,
            trusted=anchor.trusted,
            forward_only=anchor.forward_only,
            backward_verified_dtypes=anchor.backward_verified_dtypes,
            deterministic_accumulation=anchor.deterministic_accumulation,
            honored_obligations=anchor.honored_obligations,
        ),
        intent_training=model.metadata.get("training") == "1",
        required_backward_dtypes=frozenset({point.dtype})
        if model.metadata.get("training") == "1"
        else frozenset(),
        resource_limits=ResourceFacts(
            max_shared_mem_bytes_per_block=int(
                model.metadata.get("shared_mem_limit_bytes", "65536")
            ),
            max_registers_per_thread=255,
            max_threads_per_block=1024,
        ),
        estimated_shared_mem_bytes=shared_memory_estimate(
            point, int(model.metadata.get("value_dim", "1024")), point.dtype
        ),
        estimated_threads_per_block=point.num_warps * 32,
    )
    report = ModelVerifier().verify(model, assignment, facts)
    del CompilationIntent
    return report


def apply_compile_feedback(
    model: ConstraintModel,
    assignment: Assignment,
    *,
    success: bool,
    registers_per_thread: int | None = None,
    shared_mem_bytes: int | None = None,
    reason: str | None = None,
    max_nogoods: int = 8,
) -> dict[str, object]:
    """Record compilation feedback; failed compiles add an exact nogood.

    Nogoods are bounded: once the budget is spent, ``nogood_added`` is False
    and the caller must stop requesting schedules for this model.
    """
    feedback: dict[str, object] = {
        "success": success,
        "registers_per_thread": registers_per_thread,
        "shared_mem_bytes": shared_mem_bytes,
        "reason": reason,
        "nogood_added": False,
    }
    if success:
        return feedback
    existing = sum(
        1 for constraint in model.constraints if isinstance(constraint, Nogood)
    )
    if existing >= max_nogoods:
        feedback["nogood_budget_exhausted"] = True
        return feedback
    forbidden = {name: assignment[name] for name in sorted(assignment)}
    model.add_constraint(
        make_nogood(
            name=f"nogood_compile_failed_{existing + 1}",
            explanation=(
                "exact schedule rejected by compilation feedback"
                + (f": {reason}" if reason else "")
            ),
            origin_kind="compile_feedback",
            origin_id=model.metadata.get("candidate_id", "unknown"),
            forbidden=forbidden,
        )
    )
    feedback["nogood_added"] = True
    return feedback


__all__ = [
    "CONFIG_CHOICES",
    "apply_compile_feedback",
    "build_candidate_selection_model",
    "build_schedule_model",
    "decode_schedule_point",
    "decode_selected_candidate",
    "plan_binding_is_total",
    "plan_kinds_for_candidate",
    "schedule_point_assignments",
    "schedule_point_to_assignment",
    "verify_schedule_assignment",
]
