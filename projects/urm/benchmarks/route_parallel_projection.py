"""Mode-specific conservative screening; never an integration/performance pass."""

import argparse
import json
import statistics
from pathlib import Path

from pretraining_projection import project_state_replacement
from provenance import write_artifact
from route_parallel_experiment import base_artifact


def project(baseline, performance):
    if not baseline["complete"]:
        raise ValueError("all frozen baseline seeds must finish")
    reports = {}
    for mode, evidence in baseline["modes"].items():
        ratio = evidence["performance"]["geometric_mean_optimizer_step_ratio"]
        profiles = [
            p
            for pair in evidence["pairs"]
            for p in pair["native"]["state_profiles"]["passes"]
        ]
        fraction = statistics.median(p["state_kernel_fraction"] for p in profiles)
        baseline_kernel_invocation = statistics.median(
            (p["forward_kernel_ms"] + p["backward_kernel_ms"]) / 48 for p in profiles
        )
        reports[mode] = {}
        for kind in ("route_global", "route_resident"):
            rows = [
                r
                for r in performance["measurements"]
                if r["implementation"] == kind and r["mode"] == mode
            ]
            if len(rows) != 3:
                raise ValueError("three matched candidate timing seeds are required")
            candidate_complete = statistics.median(
                r["combined"]["median_wall_ms"] for r in rows
            )
            speedup = baseline_kernel_invocation / candidate_complete
            projection = project_state_replacement(ratio, fraction, speedup)
            reports[mode][kind] = {
                **projection,
                "baseline_kernel_per_invocation_ms": baseline_kernel_invocation,
                "candidate_complete_stage_ms": candidate_complete,
                "fraction_min_max": [
                    min(p["state_kernel_fraction"] for p in profiles),
                    max(p["state_kernel_fraction"] for p in profiles),
                ],
                "optimistic_threefold_assumption": project_state_replacement(
                    ratio, fraction, 3
                ),
                "scope": "remove_actual_native_state_kernels_only; charge_candidate_complete_stage; leave_native_copy/cast/allocation_overhead_in_remainder",
                "screening_only": True,
                "integration_authorized": False,
            }
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline, performance, correctness = [
        json.loads(path.read_text())
        for path in (args.baseline, args.performance, args.correctness)
    ]
    commits = {
        row["clean_source_commit"] for row in (baseline, performance, correctness)
    }
    if len(commits) != 1 or None in commits:
        raise ValueError("evidence must identify one clean implementation commit")
    artifact = base_artifact(
        "ordered_route_parallel_mode_projections",
        {
            "baseline": str(args.baseline),
            "performance": str(args.performance),
            "correctness": str(args.correctness),
        },
        False,
    )
    artifact["projections"] = project(baseline, performance)
    artifact["numerical_acceptance"] = correctness["accepted_against_native_and_oracle"]
    artifact["recommendation"] = (
        "independent_review_before_any_integration"
        if artifact["numerical_acceptance"]
        and all(
            p["projection_gate_passed"]
            for mode in artifact["projections"].values()
            for p in mode.values()
        )
        else "inspect_measured_gap_or_numerical_failure; no_broad_tuning"
    )
    artifact["full_model_acceptance"] = (
        "not_run_for_candidates; production_integration_requires_reviewer_assessment_first"
    )
    write_artifact(args.output, artifact)


if __name__ == "__main__":
    main()
