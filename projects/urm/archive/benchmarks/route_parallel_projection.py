"""Mode-specific conservative screening; never an integration/performance pass."""

import argparse
import json
import statistics
from pathlib import Path

from pretraining_projection import project_state_replacement
from provenance import write_artifact
from route_parallel_experiment import base_artifact


def project(baseline, performance):
    if not baseline["complete"] or not performance["complete"]:
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
                r["public_autograd_combined"]["median_wall_ms"] for r in rows
            )
            speedup = baseline_kernel_invocation / candidate_complete
            projection = project_state_replacement(ratio, fraction, speedup)
            conservative_speedup = min(
                (p["forward_kernel_ms"] + p["backward_kernel_ms"]) / 48
                for p in profiles
            ) / max(max(r["public_autograd_combined"]["wall_ms"]) for r in rows)
            sensitivity = project_state_replacement(
                evidence["performance"]["hierarchical_ratio_ci95"]["upper"],
                min(p["state_kernel_fraction"] for p in profiles),
                conservative_speedup,
            )
            reports[mode][kind] = {
                **projection,
                "baseline_kernel_per_invocation_ms": baseline_kernel_invocation,
                "candidate_complete_stage_ms": candidate_complete,
                "conservative_sensitivity_not_confidence_interval": sensitivity,
                "sensitivity_scope": "baseline ratio CI upper, smallest observed fraction/kernel cost, largest observed candidate complete cost; not a joint statistical bound",
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
    parser.add_argument("--captures", type=Path, required=True)
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
    captures = json.loads(args.captures.read_text())
    if not captures["complete"] or captures["clean_source_commit"] not in commits:
        raise ValueError("capture source mismatch")
    for evidence in baseline["modes"].values():
        for pair in evidence["pairs"]:
            capture = next(
                x for x in captures["captures"] if x["seed"] == pair["native"]["seed"]
            )
            if any(
                row["matched_initial"] != capture["matched_initial"]
                for row in pair.values()
            ):
                raise ValueError(
                    "operand capture and actual model benchmark did not start matched"
                )
    artifact = base_artifact(
        "ordered_route_parallel_mode_projections",
        {
            "baseline": str(args.baseline),
            "performance": str(args.performance),
            "correctness": str(args.correctness),
            "captures": str(args.captures),
        },
        False,
    )
    artifact["projections"] = project(baseline, performance)
    artifact["numerical_acceptance"] = correctness["accepted_against_native_and_oracle"]
    artifact["capture_initial_conditions_match_both_baseline_backends_and_modes"] = True
    artifact["recommendation"] = (
        "independent_review_before_any_integration"
        if artifact["numerical_acceptance"]
        and any(
            all(
                artifact["projections"][mode][kind]["projection_gate_passed"]
                for mode in artifact["projections"]
            )
            for kind in ("route_global", "route_resident")
        )
        else "inspect_measured_gap_or_numerical_failure; no_broad_tuning"
    )
    artifact["full_model_acceptance"] = (
        "not_run_for_candidates; production_integration_requires_reviewer_assessment_first"
    )
    write_artifact(args.output, artifact)


if __name__ == "__main__":
    main()
