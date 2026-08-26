"""Compilation matrix over existing URM presets (NAS-facing API exercise).

Builds typed semantic programs from every canonical `MixerSpec` preset,
validates them, enumerates rewrite/lowering candidates, compiles what is
supported, and records structured diagnostics for what is not. Produces one
compact JSON artifact distinguishing architecture parameters from schedule
parameters. Escape-hatch count must remain zero by construction.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/compilation-matrix.json")

# Presets that carry no compiler-facing detail record beyond routing/state
# semantics map directly; family-specific detail specs (expert widths, scan
# equations) stay backend-owned and are recorded as architecture metadata.


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


def build_program(spec):
    """Translate one MixerSpec into a semantic compiler program.

    Returns (program | None, architecture_params dict, decline_reason | None).
    """
    from urm.compiler.semantic import (
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
    }

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
        "value_dtype": "float32",
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
    args = parser.parse_args()

    from urm.compiler.diagnostics import CompilerError
    from urm.compiler.planner import ScheduleParams, UrmCompiler
    from urm.presets import CATALOG as ALL_PRESETS

    compiler = UrmCompiler()
    rows: list[dict[str, object]] = []
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
        candidates_total = 0
        accepted = 0
        rejected = 0
        compiled_programs = 0
        anchors: list[str] = []
        costs: list[dict[str, object]] = []
        traces: list[dict[str, object]] = []
        for program in built:
            summaries = compiler.enumerate_candidates(program)
            candidates_total += len(summaries)
            rejected += sum(1 for s in summaries if not s.legal)
            try:
                result = compiler.compile(program, schedule_params=ScheduleParams())
            except CompilerError as error:
                rejected += 1
                traces.append({"program": program.name, "error": str(error)})
                continue
            compiled_programs += 1
            accepted += sum(1 for a in result.trace.attempts if a.outcome == "accepted")
            anchors.extend(result.trace.anchors)
            costs.append(result.cost.to_dict())
            traces.append(result.trace.to_dict())
        rows.append(
            {
                "architecture_params": architecture,
                # Schedule params stay a separate namespace even when empty.
                "schedule_params": {"anchor_overrides": {}, "block_hints": {}},
                "valid_semantic_programs": len(built),
                "compiled_programs": compiled_programs,
                "candidates": [
                    {
                        "rule": summary.rule,
                        "subject_op": summary.subject_op,
                        "legal": summary.legal,
                        "reason_code": summary.reason_code,
                    }
                    for summary in compiler.enumerate_candidates(built[-1])
                ],
                "rewrite_accepted": accepted,
                "rewrite_rejected": rejected,
                "compiled": compiled_programs > 0,
                "anchors_selected": sorted(set(anchors)),
                "estimated_costs": costs,
                "traces": traces,
                "escape_hatch_count": 0,
                "measured_performance_pointer": (
                    "results/final/*-forward.json (routed-reduction v1 is the "
                    "only family with a runnable native lowering on this host)"
                    if selection_for(preset)
                    in {"top_k", "dense", "block_sparse", "threshold"}
                    else None
                ),
            }
        )

    valid_programs = sum(row["valid_semantic_programs"] for row in rows)
    total_candidates = sum(
        row["rewrite_accepted"] + row["rewrite_rejected"] for row in rows
    )
    compiled_rows = sum(1 for row in rows if row["compiled"])
    artifact = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "summary": {
            "presets_evaluated": len(rows),
            "valid_semantic_programs": valid_programs,
            "generated_candidates": total_candidates,
            "rewrite_accepted": sum(row["rewrite_accepted"] for row in rows),
            "rewrite_rejected": sum(row["rewrite_rejected"] for row in rows),
            "compile_success_rate": round(compiled_rows / len(rows), 4),
            "escape_hatch_count": sum(row["escape_hatch_count"] for row in rows),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))


if __name__ == "__main__":
    main()
