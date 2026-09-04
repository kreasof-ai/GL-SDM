"""Compilation matrix over existing URM presets (NAS-facing API exercise).

Builds typed semantic programs from every canonical `MixerSpec` preset,
validates them under an explicit intent, enumerates rewrite/lowering
candidates, builds constraint models, runs the solver passes when the
optional extra is installed, compiles what is supported, and records
structured diagnostics for what is not.

Metrics are named for what they actually measure:

- ``routing_skeleton_compile_rate`` - fraction of presets whose COARSE
  routing skeleton maps onto compiled routed-reduction semantics. This says
  nothing about dense attention, MoE expert GEMMs, or sparse attention being
  fully compiled.
- ``full_architecture_compile_rate`` - fraction of presets whose complete
  family detail (expert GEMMs, scan equations, page updates) has trusted
  lowerings in this repository. Expected to be far smaller than the skeleton
  rate until family adapters land.
- ``native_lowering_rate`` / ``upstream_adapter_rate`` - where executed work
  goes: compiler-generated anchors versus upstream kernels (FA/FLA/original SDM).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/compilation-matrix.json")


def selection_for(spec) -> str:
    from urm.ir import RoutingKind

    return {
        RoutingKind.DENSE: "dense",
        RoutingKind.BLOCK_SPARSE: "block_sparse",
        RoutingKind.TOP_K: "top_k",
        RoutingKind.PRODUCT_KEY: "top_k",
        RoutingKind.THRESHOLD: "threshold",
        RoutingKind.KERNELIZED_RECURRENCE: "kernelized_recurrence",
    }[spec.routing]


def domain_for(spec):
    from urm.compiler.semantic import LogicalDomain
    from urm.ir import Domain

    return {
        Domain.SEQUENCE: LogicalDomain.SEQUENCE,
        Domain.EXPERT: LogicalDomain.EXPERT,
        Domain.PARAMETER_BLOCK: LogicalDomain.PARAMETER_BLOCK,
        Domain.RECURRENT_STATE: LogicalDomain.RECURRENT_STATE,
        Domain.MEMORY_PAGE: LogicalDomain.MEMORY_PAGE,
    }[spec.source_domain]


# Presets whose FULL architecture detail is lowered by trusted code in this
# repository today. The coarse routing skeleton compiling does NOT imply this.
FULLY_LOWERED_FAMILIES = {
    "routed_top_k",
    "sparse_delta_memory",
}  # native routed reduction or pinned external family adapter end-to-end


def build_program(spec):
    """Translate one MixerSpec into a semantic compiler program.

    Returns (program | None, architecture_params dict, decline_reason | None).
    """
    from urm.compiler.semantic import (
        DType,
        ScoreNormalization,
        SelectionKind,
        routed_reduction_program,
        row_scaled_routed_reduction_program,
    )
    from urm.ir import MutationKind, Normalization, RoutingKind

    architecture = {
        "preset": spec.name,
        "routing": selection_for(spec),
        "query_domain": spec.query_domain.value,
        "source_domain": spec.source_domain.value,
        "normalization": spec.normalization.value,
        "mutation": spec.mutation.value,
        "residency": spec.residency.value,
        "collision_policy": spec.collision_policy.value,
        # Honest coverage labels per preset:
        "family_detail_lowered": spec.name in FULLY_LOWERED_FAMILIES,
        "requires_native_gpu_probe": spec.name == "sparse_delta_memory",
    }

    if spec.name == "sparse_delta_memory":
        from urm.compiler.semantic import (
            SDMExecutionMode,
            SparseStateExecutionMode,
            sparse_delta_memory_program,
            sparse_route_selection_program,
            sparse_state_mixer_program,
        )

        return (
            (
                sparse_delta_memory_program(
                    name=spec.name,
                    parallel=1,
                    sequence=128,
                    slots_per_partition=4096,
                    value_dim=256,
                    writes=64,
                    reads=64,
                    dtype=DType.BFLOAT16,
                    mode=SDMExecutionMode.TRAINING,
                ),
                sparse_state_mixer_program(
                    name="sparse_state_mixer_kernel_only",
                    parallel=1,
                    sequence=128,
                    slots_per_partition=4096,
                    value_dim=256,
                    writes=64,
                    reads=64,
                    dtype=DType.BFLOAT16,
                    mode=SparseStateExecutionMode.TRAINING,
                ),
                sparse_route_selection_program(
                    name="sparse_route_selection",
                    parallel=1,
                    sequence=128,
                    source_extent=4096,
                    route_width=64,
                    dtype=DType.BFLOAT16,
                ),
            ),
            architecture,
            None,
        )

    if selection_for(spec) == "kernelized_recurrence":
        return (
            None,
            architecture,
            (
                "ordered recurrence is represented but its scan equation stays "
                "backend-owned; compilation requires selecting a family adapter "
                "(FLA gated delta-rule adapter exists outside this matrix)"
            ),
        )

    if spec.mutation is not MutationKind.NONE:
        return (
            None,
            architecture,
            (
                "stateful presets require the state-planning slice "
                "(transactional commit planning lands with the SDM adapter)"
            ),
        )

    if spec.routing in {RoutingKind.TOP_K, RoutingKind.PRODUCT_KEY}:
        selection = SelectionKind.TOP_K
        top_k = spec.top_k
    else:
        selection = SelectionKind(
            "block_sparse" if spec.routing is RoutingKind.BLOCK_SPARSE else "dense"
        )
        top_k = None

    normalization = {
        Normalization.NONE: ScoreNormalization.NONE,
        Normalization.SOFTMAX: ScoreNormalization.SOFTMAX,
        Normalization.SIGMOID: ScoreNormalization.SIGMOID,
        Normalization.L1: ScoreNormalization.L1,
    }[spec.normalization]

    common = {
        "queries": 1024,
        "route_width": max(1, min(top_k or 8, 64)),
        "sources": 256,
        "value_dim": 512,
        "value_dtype": DType.FLOAT32,
        "source_domain": domain_for(spec),
        "selection": selection,
        "normalization": normalization,
        "top_k": top_k,
    }
    try:
        materialized = routed_reduction_program(name=spec.name, **common)
        scaled = row_scaled_routed_reduction_program(
            name=f"{spec.name}__row_scaled", **common
        )
    except ValueError as error:
        return None, architecture, f"program construction rejected: {error}"
    return (materialized, scaled), architecture, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--probe",
        choices=["auto", "off", "required"],
        default="auto",
        help="Probe mode: 'auto' (probe if CUDA available), 'off' (CPU-safe, no probe), 'required' (fail if no CUDA)",
    )
    args = parser.parse_args()

    from provenance import provenance

    from urm.compiler.diagnostics import CompilerError
    from urm.compiler.planner import CompilationIntent, UrmCompiler
    from urm.presets import CATALOG as ALL_PRESETS

    probe_mode = args.probe
    probe = None
    probing_active = False

    if probe_mode == "off":
        probing_active = False
        probe = None
    elif probe_mode == "required":
        try:
            import torch
            import triton
        except ImportError as exc:
            raise RuntimeError(
                f"Probe mode 'required' failed: torch/triton dependency missing: {exc}"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Probe mode 'required' failed: CUDA is unavailable on this host"
            )
        from urm.compiler.anchors.routed_reduction_epilogue import (
            make_triton_compile_probe,
        )

        probe = make_triton_compile_probe()
        probing_active = True
    elif probe_mode == "auto":
        try:
            import torch
            import triton  # noqa: F401

            if torch.cuda.is_available():
                from urm.compiler.anchors.routed_reduction_epilogue import (
                    make_triton_compile_probe,
                )

                probe = make_triton_compile_probe()
                probing_active = True
        except ImportError:
            probe = None
            probing_active = False

    intent = CompilationIntent.TRAINING
    compiler = UrmCompiler(compile_probe=probe)
    rows: list[dict[str, object]] = []
    solver_totals = {
        "solver_time_ms": 0.0,
        "infeasible_candidates": 0,
        "candidates_ranked_out_by_objective": 0,
        "compile_probe_failures": 0,
        "nogoods_added": 0,
        "verified_models": 0,
        "schedule_models_verified": 0,
        "unsat_categories": [],
        "unsat_codes": set(),
    }
    compile_failures = 0

    for preset in ALL_PRESETS:
        built, architecture, decline = build_program(preset)
        if built is None:
            rows.append(
                {
                    "architecture_params": architecture,
                    "schedule_params": {},
                    "valid_semantic_programs": 0,
                    "candidates": [],
                    "rewrite_accepted": 0,
                    "rewrite_rejected": 0,
                    "compiled": False,
                    "decline_reason": decline,
                    "escape_hatch_count": 0,
                }
            )
            continue
        if architecture.get("requires_native_gpu_probe") and probe_mode == "off":
            rows.append(
                {
                    "architecture_params": architecture,
                    "schedule_params": {},
                    "valid_semantic_programs": len(built),
                    "compiled_programs": 0,
                    "candidates": [],
                    "rewrite_accepted": 0,
                    "rewrite_rejected": 0,
                    "rejected_by_training_intent": 0,
                    "compiled": False,
                    "decline_reason": (
                        "native sparse-memory capability probe is disabled in "
                        "CPU-safe --probe off mode"
                    ),
                    "escape_hatch_count": 0,
                }
            )
            continue
        candidates_total = 0
        accepted = 0
        rejected_by_preconditions = 0
        rejected_by_intent = 0
        compiled_programs = 0
        anchors: list[str] = []
        costs: list[dict[str, object]] = []
        traces: list[dict[str, object]] = []
        selected_ids: list[str] = []
        policies: list[str] = []
        schedule_decisions: list[dict[str, object] | None] = []
        for program in built:
            summaries = compiler.enumerate_candidates(program, intent)
            candidates_total += len(summaries)
            rejected_by_preconditions += sum(
                1
                for s in summaries
                if not s.legal and s.reason_code != "intent_conflict"
            )
            rejected_by_intent += sum(
                1
                for s in summaries
                if not s.legal and s.reason_code == "intent_conflict"
            )
            try:
                result = compiler.compile(program, intent=intent)
            except CompilerError as error:
                rejected_by_intent += 1
                compile_failures += 1
                for diagnostic in error.diagnostics:
                    solver_totals["unsat_codes"].add(diagnostic.code.value)
                if any(
                    d.code.value in ("unsat_constraints", "intent_conflict")
                    for d in error.diagnostics
                ):
                    solver_totals["infeasible_candidates"] += 1
                traces.append({"program": program.name, "error": str(error)})
                continue
            compiled_programs += 1
            accepted += sum(1 for a in result.trace.attempts if a.outcome == "accepted")
            anchors.extend(result.trace.anchors)
            costs.append(result.cost.to_dict())
            traces.append(result.trace.to_dict())
            selected_ids.append(result.selected_candidate_id)
            policies.append(result.selection_policy.value)
            schedule_decisions.append(
                result.schedule_decision.to_dict()
                if result.schedule_decision is not None
                else None
            )
            decision = result.schedule_decision
            if decision is not None:
                solver_totals["solver_time_ms"] += float(
                    decision.solver_statistics.get("wall_ms", 0.0)
                )
                solver_totals["compile_probe_failures"] += (
                    decision.compile_failures_observed
                )
                solver_totals["nogoods_added"] += decision.nogoods_added
                all_verified = bool(decision.attempts) and all(
                    a.verified for a in decision.attempts
                )
                if all_verified:
                    solver_totals["verified_models"] += 1
                    solver_totals["schedule_models_verified"] += 1
            # Candidates the solver objective ranked out during auto-selection.
            if result.selection_policy.value == "solver_guided":
                solver_totals["candidates_ranked_out_by_objective"] += sum(
                    1
                    for r in result.rejected_alternatives
                    if r.reason_code == "rejected_by_solver_objective"
                )
            if result.solver_statistics:
                solver_totals["solver_time_ms"] += float(
                    result.solver_statistics.get("wall_ms", 0.0)
                )
        rows.append(
            {
                "architecture_params": architecture,
                "schedule_params": {
                    "anchor_overrides": {},
                    "block_hints": {},
                    "deterministic": False,
                },
                "valid_semantic_programs": len(built),
                "compiled_programs": compiled_programs,
                "candidates": [
                    {
                        "candidate_id": summary.candidate_id,
                        "kind": summary.kind,
                        "legal": summary.legal,
                        "reason_code": summary.reason_code,
                        "backward_verified": summary.backward_verified,
                        "equivalence_class": summary.equivalence_class,
                    }
                    for summary in compiler.enumerate_candidates(built[-1], intent)
                ],
                "selected_candidate_ids": selected_ids,
                "selection_policies": policies,
                "schedule_decisions": schedule_decisions,
                "rewrite_accepted": accepted,
                "rewrite_rejected": rejected_by_preconditions,
                "rejected_by_training_intent": rejected_by_intent,
                "compiled": compiled_programs > 0,
                "anchors_selected": sorted(set(anchors)),
                "anchor_dispatch_counts": {
                    name: anchors.count(name) for name in sorted(set(anchors))
                },
                "estimated_costs": costs,
                "traces": traces,
                "escape_hatch_count": 0,
                "measured_performance_pointer": (
                    "results/sparse-delta-memory/benchmark.json; "
                    "results/sparse-state-mixer/confirmation.json; "
                    "results/sparse-memory-e2e/confirmation.json"
                    if preset.name == "sparse_delta_memory"
                    else (
                        "results/final/*-forward.json (routed-reduction v1 is the "
                        "only native family lowering on this host)"
                        if selection_for(preset)
                        in {"top_k", "dense", "block_sparse", "threshold"}
                        else None
                    )
                ),
            }
        )

    presets = len(rows)
    valid_programs = sum(row["valid_semantic_programs"] for row in rows)
    # Constraint-model hashes come from the verified schedule decisions
    # themselves (ScheduleDecision.model_hash = model.summary_hash()).
    model_hashes = sorted(
        {
            str(row_decision["model_hash"])
            for row in rows
            for row_decision in row.get("schedule_decisions", [])
            if row_decision
        }
    )
    total_candidates = sum(
        row["rewrite_accepted"]
        + row["rewrite_rejected"]
        + row.get("rejected_by_training_intent", 0)
        for row in rows
    )
    compiled_rows = sum(1 for row in rows if row["compiled"])
    fully_lowered = sum(
        1
        for row in rows
        if row["compiled"] and row["architecture_params"].get("family_detail_lowered")
    )
    # Where executed work goes for the compiled programs.
    NATIVE_PREFIXES = (
        "routed_reduction",
        "urm_native_sparse_state_mixer",
        "urm_native_sparse_route_selection",
        "urm_native_sparse_memory_e2e",
    )
    UPSTREAM_ANCHORS = {
        "flash_attention_adapter",
        "fla_gated_delta_rule_adapter",
        "facebook_sparse_delta_memory_183e7df_external_adapter",
        "facebook_sparse_delta_memory_183e7df_precomputed_route_adapter",
    }
    compiled_programs_total = sum(row.get("compiled_programs", 0) for row in rows)
    native_programs = sum(
        count
        for row in rows
        for anchor_name, count in row.get("anchor_dispatch_counts", {}).items()
        if str(anchor_name).startswith(NATIVE_PREFIXES)
    )
    upstream_programs = sum(
        count
        for row in rows
        for anchor_name, count in row.get("anchor_dispatch_counts", {}).items()
        if anchor_name in UPSTREAM_ANCHORS
    )
    cmd_str = (
        f"python benchmarks/compilation_matrix.py --probe {args.probe}"
        if args.probe != "auto" or "--probe" in sys.argv
        else "python benchmarks/compilation_matrix.py"
    )
    if args.output != DEFAULT_OUTPUT:
        cmd_str += f" --output {args.output}"

    matrix_provenance = provenance(
        cmd_str,
        {
            "presets": [r["architecture_params"]["preset"] for r in rows],
            "intent": intent.value,
            "probe_mode": probe_mode,
            "probing_active": probing_active,
        },
        include_gpu=(probe_mode != "off"),
    )
    matrix_provenance["probe_mode"] = probe_mode
    matrix_provenance["probing_active"] = probing_active
    matrix_provenance["constraint_model_hash"] = (
        model_hashes[0] if model_hashes else "not_applicable_static_analysis"
    )
    matrix_provenance["constraint_model_hashes"] = model_hashes

    artifact = {
        "schema_version": 2,
        "generated_utc": datetime.now(UTC).isoformat(),
        "provenance": matrix_provenance,
        "summary": {
            "presets_evaluated": presets,
            "valid_semantic_programs": valid_programs,
            "generated_candidates": total_candidates,
            "rewrite_accepted": sum(row["rewrite_accepted"] for row in rows),
            "rewrite_rejected": sum(row["rewrite_rejected"] for row in rows),
            "candidates_rejected_by_imperative_checks": sum(
                row["rewrite_rejected"] for row in rows
            ),
            "infeasible_candidates": solver_totals["infeasible_candidates"],
            "candidates_ranked_out_by_objective": solver_totals[
                "candidates_ranked_out_by_objective"
            ],
            "compile_probe_failures": (
                solver_totals["compile_probe_failures"] if probing_active else None
            ),
            "nogoods_added": (
                solver_totals["nogoods_added"] if probing_active else None
            ),
            "verified_models": solver_totals["verified_models"],
            "schedule_models_verified": solver_totals["schedule_models_verified"],
            "unsat_categories": sorted(solver_totals["unsat_codes"]),
            "solver_time_ms": round(solver_totals["solver_time_ms"], 3),
            "compile_failures": compile_failures,
            # Renamed, honestly-scoped metrics:
            "routing_skeleton_compile_rate": round(compiled_rows / presets, 4),
            "full_architecture_compile_rate": round(fully_lowered / presets, 4),
            "native_lowering_rate": (
                round(native_programs / compiled_programs_total, 4)
                if compiled_programs_total
                else 0.0
            ),
            "upstream_adapter_rate": (
                round(upstream_programs / compiled_programs_total, 4)
                if compiled_programs_total
                else 0.0
            ),
            "escape_hatch_count": sum(row["escape_hatch_count"] for row in rows),
            "metric_notes": {
                "routing_skeleton_compile_rate": (
                    "fraction of presets whose coarse ROUTING SKELETON maps to "
                    "compiled routed reduction; dense attention, MoE expert "
                    "GEMMs and sparse attention are NOT fully compiled merely "
                    "because their skeleton routes map"
                ),
                "full_architecture_compile_rate": (
                    "fraction of presets whose COMPLETE family detail has "
                    "trusted lowerings in this repository today"
                ),
            },
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
