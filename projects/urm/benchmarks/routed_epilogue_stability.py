"""Cross-run empirical stability runner for routed-epilogue schedule selection.

Orchestrates multiple independent fresh-process executions of
``benchmarks/routed_epilogue_selection.py``, captures operating conditions
before and after measurement, preserves raw CUDA-event samples, computes
cross-run rank correlations, percentage drifts, confidence intervals, a
noise-aware winning set, and robust solver/heuristic regret classifications.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/solver/routed-epilogue-stability.json")
DEFAULT_RUNS = 5
DEFAULT_BASE_SEED = 17
DEFAULT_PRACTICAL_MARGIN_PCT = 2.5
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42

REGRET_TARGET_PCT = 10.0


def validate_child_provenance_invariants(
    child_artifacts: Sequence[dict[str, object]],
) -> None:
    """Fail closed if any child run disagrees on fundamental environmental invariants."""
    if len(child_artifacts) < 2:
        return
    invariant_keys = (
        "git_revision",
        "dirty_tree",
        "gpu",
        "driver",
        "cuda",
        "pytorch",
        "triton",
        "constraint_model_hash",
    )
    for i in range(len(child_artifacts)):
        for j in range(i + 1, len(child_artifacts)):
            prov_i = child_artifacts[i]["provenance"]
            prov_j = child_artifacts[j]["provenance"]
            for key in invariant_keys:
                val_i = prov_i.get(key)
                val_j = prov_j.get(key)
                if val_i != val_j:
                    raise RuntimeError(
                        f"Child provenance invariant violation for '{key}': "
                        f"run {i} has {val_i!r} but run {j} has {val_j!r}"
                    )
            prob_i = child_artifacts[i]["problem"]
            prob_j = child_artifacts[j]["problem"]
            if prob_i != prob_j:
                raise RuntimeError(
                    f"Child problem configuration mismatch between run {i} and run {j}: "
                    f"{prob_i} != {prob_j}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--practical-margin", type=float, default=DEFAULT_PRACTICAL_MARGIN_PCT
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required: this benchmark measures GPU schedules")

    from measurement import (
        bootstrap_ci,
        quantile,
        spearman_rank_correlation,
    )
    from provenance import provenance, utc_now, write_artifact

    project_root = Path(__file__).parents[1]
    selection_script = project_root / "benchmarks" / "routed_epilogue_selection.py"

    run_records: list[dict[str, object]] = []
    child_artifacts: list[dict[str, object]] = []
    seeds = [args.base_seed + i * 100 for i in range(args.runs)]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for run_idx, seed in enumerate(seeds):
            temp_output = temp_dir_path / f"run_{run_idx}.json"
            cmd = [
                sys.executable,
                str(selection_script),
                "--output",
                str(temp_output),
                "--seed",
                str(seed),
                "--run-id",
                str(run_idx),
            ]
            print(
                f"Executing independent run {run_idx + 1}/{args.runs} (seed={seed})..."
            )
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(project_root),
                check=False,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"Child run {run_idx} failed with exit code {res.returncode}:\n{res.stderr}"
                )
            if not temp_output.exists():
                raise RuntimeError(
                    f"Child run {run_idx} did not write expected artifact at {temp_output}"
                )
            child_artifact = json.loads(temp_output.read_text(encoding="utf-8"))
            child_artifacts.append(child_artifact)
            emp = child_artifact["empirical"]
            regret_info = emp["regret"]

            # Locate solver and heuristic samples in child run
            z3_selected = emp["solver_selected"]
            heur_selected = emp["heuristic_selected"]
            solver_sample = next(
                (s for s in emp["samples"] if s["schedule"] == z3_selected), None
            )
            heur_sample = next(
                (s for s in emp["samples"] if s["schedule"] == heur_selected), None
            )

            run_records.append(
                {
                    "run_id": run_idx,
                    "seed": seed,
                    "best_point": emp["best_point"],
                    "best_median_ms": emp["best_median_ms"],
                    "solver_selected": z3_selected,
                    "solver_median_ms": (
                        solver_sample["median_ms"] if solver_sample else None
                    ),
                    "solver_regret_pct": regret_info["z3_vs_best_pct"],
                    "heuristic_selected": heur_selected,
                    "heuristic_median_ms": (
                        heur_sample["median_ms"] if heur_sample else None
                    ),
                    "heuristic_regret_pct": regret_info["heuristic_vs_best_pct"],
                    "measured_points": emp["measured_points"],
                    "operating_conditions_before": emp.get(
                        "operating_conditions", {}
                    ).get("before"),
                    "operating_conditions_after": emp.get(
                        "operating_conditions", {}
                    ).get("after"),
                    "samples_by_key": {
                        json.dumps(s["schedule"], sort_keys=True): s
                        for s in emp["samples"]
                    },
                }
            )

    # Validate provenance invariants across all child runs
    validate_child_provenance_invariants(child_artifacts)

    problem = child_artifacts[0]["problem"]
    model_hash = child_artifacts[0]["provenance"]["constraint_model_hash"]

    # All unique schedule keys present across runs
    common_keys = set(run_records[0]["samples_by_key"].keys())
    for r in run_records[1:]:
        common_keys &= set(r["samples_by_key"].keys())
    sorted_common_keys = sorted(common_keys)

    # Compute ranks per run for each schedule
    for r in run_records:
        sorted_samples_run = sorted(
            r["samples_by_key"].values(), key=lambda s: s["median_ms"]
        )
        r["rank_by_key"] = {
            json.dumps(s["schedule"], sort_keys=True): rank + 1
            for rank, s in enumerate(sorted_samples_run)
        }

    # Per-schedule aggregate stability
    schedule_records: list[dict[str, object]] = []
    for sched_key in sorted_common_keys:
        sched_dict = json.loads(sched_key)
        run_medians = [r["samples_by_key"][sched_key]["median_ms"] for r in run_records]
        raw_samples_by_run = [
            r["samples_by_key"][sched_key]["raw_samples_ms"] for r in run_records
        ]
        ranks_in_runs = [r["rank_by_key"][sched_key] for r in run_records]

        mean_med = sum(run_medians) / len(run_medians)
        std_med = (
            math.sqrt(
                sum((m - mean_med) ** 2 for m in run_medians)
                / max(1, len(run_medians) - 1)
            )
            if len(run_medians) > 1
            else 0.0
        )
        cv = (std_med / mean_med * 100.0) if mean_med > 0 else 0.0

        ci_lower, ci_upper = bootstrap_ci(
            run_medians,
            statistic=lambda s: quantile(s, 0.5),
            num_resamples=args.bootstrap_resamples,
            confidence=0.95,
            seed=args.bootstrap_seed,
        )

        schedule_records.append(
            {
                "schedule": sched_dict,
                "median_of_medians_ms": round(quantile(run_medians, 0.5), 4),
                "min_median_ms": round(min(run_medians), 4),
                "max_median_ms": round(max(run_medians), 4),
                "cv_pct": round(cv, 2),
                "ci95_lower_ms": round(ci_lower, 4),
                "ci95_upper_ms": round(ci_upper, 4),
                "rank_per_run": ranks_in_runs,
                "raw_samples_by_run": raw_samples_by_run,
                "per_run_medians_ms": [round(m, 4) for m in run_medians],
            }
        )

    schedule_records.sort(
        key=lambda x: (
            x["median_of_medians_ms"],
            json.dumps(x["schedule"], sort_keys=True),
        )
    )
    for rank, rec in enumerate(schedule_records, 1):
        rec["aggregate_rank"] = rank

    # Cross-run grid statistics
    pairwise_spearman: list[float] = []
    top1_agreements = 0
    pairwise_top5_jaccard: list[float] = []

    for i in range(len(run_records)):
        for j in range(i + 1, len(run_records)):
            vec_i = [
                run_records[i]["samples_by_key"][k]["median_ms"]
                for k in sorted_common_keys
            ]
            vec_j = [
                run_records[j]["samples_by_key"][k]["median_ms"]
                for k in sorted_common_keys
            ]
            spearman_rho = spearman_rank_correlation(vec_i, vec_j)
            pairwise_spearman.append(round(spearman_rho, 4))

            # Top 1 agreement
            if run_records[i]["best_point"] == run_records[j]["best_point"]:
                top1_agreements += 1

            # Top 5 Jaccard
            top5_i = {k for k, rk in run_records[i]["rank_by_key"].items() if rk <= 5}
            top5_j = {k for k, rk in run_records[j]["rank_by_key"].items() if rk <= 5}
            jaccard = len(top5_i & top5_j) / max(1, len(top5_i | top5_j))
            pairwise_top5_jaccard.append(round(jaccard, 4))

    all_drifts = [
        ((rec["max_median_ms"] - rec["min_median_ms"]) / rec["min_median_ms"]) * 100.0
        for rec in schedule_records
        if rec["min_median_ms"] > 0
    ]
    median_drift = round(quantile(all_drifts, 0.5), 2) if all_drifts else 0.0
    max_drift = round(max(all_drifts), 2) if all_drifts else 0.0

    # Winning exploratory candidate set construction
    rep_best = schedule_records[0]
    rep_best_med = rep_best["median_of_medians_ms"]
    exploratory_entries = []

    for rec in schedule_records:
        slowdown = round(
            ((rec["median_of_medians_ms"] - rep_best_med) / rep_best_med) * 100.0,
            2,
        )
        ci_overlap = rec["ci95_lower_ms"] <= rep_best["ci95_upper_ms"]
        in_exploratory_set = bool(slowdown <= args.practical_margin or ci_overlap)
        if in_exploratory_set:
            exploratory_entries.append(
                {
                    "schedule": rec["schedule"],
                    "aggregate_rank": rec["aggregate_rank"],
                    "median_of_medians_ms": rec["median_of_medians_ms"],
                    "ci95_lower_ms": rec["ci95_lower_ms"],
                    "ci95_upper_ms": rec["ci95_upper_ms"],
                    "slowdown_vs_best_pct": slowdown,
                    "in_exploratory_set": in_exploratory_set,
                }
            )

    is_winner_stable = (
        len({json.dumps(r["best_point"], sort_keys=True) for r in run_records}) == 1
    )

    # Robust regret classification
    def classify_regret(
        regret_values: Sequence[float],
    ) -> dict[str, object]:
        med_reg = round(quantile(regret_values, 0.5), 2)
        worst_reg = round(max(regret_values), 2)
        ci_lo, ci_hi = bootstrap_ci(
            regret_values,
            statistic=lambda s: quantile(s, 0.5),
            num_resamples=args.bootstrap_resamples,
            confidence=0.95,
            seed=args.bootstrap_seed,
        )
        ci_lo = round(ci_lo, 2)
        ci_hi = round(ci_hi, 2)
        if ci_hi <= REGRET_TARGET_PCT:
            classification = "pass"
        elif ci_lo > REGRET_TARGET_PCT:
            classification = "fail"
        else:
            classification = "inconclusive"
        return {
            "per_run_regret_pct": [round(r, 2) for r in regret_values],
            "median_regret_pct": med_reg,
            "worst_regret_pct": worst_reg,
            "ci95_lower_pct": ci_lo,
            "ci95_upper_pct": ci_hi,
            "target_pct": REGRET_TARGET_PCT,
            "classification": classification,
        }

    solver_regret_values = [float(r["solver_regret_pct"]) for r in run_records]
    heuristic_regret_values = [float(r["heuristic_regret_pct"]) for r in run_records]

    solver_robust = classify_regret(solver_regret_values)
    heuristic_robust = classify_regret(heuristic_regret_values)

    # Strip raw mapping for cleaner run records
    serializable_runs = []
    for r in run_records:
        serializable_runs.append(
            {
                "run_id": r["run_id"],
                "seed": r["seed"],
                "best_point": r["best_point"],
                "best_median_ms": r["best_median_ms"],
                "solver_selected": r["solver_selected"],
                "solver_median_ms": r["solver_median_ms"],
                "solver_regret_pct": r["solver_regret_pct"],
                "heuristic_selected": r["heuristic_selected"],
                "heuristic_median_ms": r["heuristic_median_ms"],
                "heuristic_regret_pct": r["heuristic_regret_pct"],
                "measured_points": r["measured_points"],
                "operating_conditions_before": r["operating_conditions_before"],
                "operating_conditions_after": r["operating_conditions_after"],
            }
        )

    stability_artifact = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "provenance": {
            **provenance(
                f"python benchmarks/routed_epilogue_stability.py --runs {args.runs}",
                {
                    "problem": problem,
                    "runs": args.runs,
                    "base_seed": args.base_seed,
                    "practical_margin": args.practical_margin,
                },
            ),
            "constraint_model_hash": model_hash,
        },
        "problem": problem,
        "stability_config": {
            "runs": args.runs,
            "base_seed": args.base_seed,
            "seeds": seeds,
            "rounds_per_run": 4,
            "samples_per_round": 5,
            "warmup_runs": 5,
            "practical_equivalence_margin_pct": args.practical_margin,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "runs": serializable_runs,
        "cross_run_statistics": {
            "schedules_measured_in_all_runs": len(schedule_records),
            "pairwise_spearman_correlations": pairwise_spearman,
            "min_spearman_correlation": (
                min(pairwise_spearman) if pairwise_spearman else 1.0
            ),
            "median_spearman_correlation": (
                round(quantile(pairwise_spearman, 0.5), 4) if pairwise_spearman else 1.0
            ),
            "max_spearman_correlation": (
                max(pairwise_spearman) if pairwise_spearman else 1.0
            ),
            "top1_agreement_count": top1_agreements,
            "pairwise_top5_jaccard": pairwise_top5_jaccard,
            "median_top5_jaccard": (
                round(quantile(pairwise_top5_jaccard, 0.5), 4)
                if pairwise_top5_jaccard
                else 1.0
            ),
            "mean_top5_jaccard": (
                round(
                    sum(pairwise_top5_jaccard) / max(1, len(pairwise_top5_jaccard)), 4
                )
                if pairwise_top5_jaccard
                else 1.0
            ),
            "median_abs_drift_pct": median_drift,
            "max_abs_drift_pct": max_drift,
        },
        "winning_set": {
            "representative_best": rep_best["schedule"],
            "representative_best_median_ms": rep_best["median_of_medians_ms"],
            "is_winner_stable_across_runs": is_winner_stable,
            "practical_equivalence_margin_pct": args.practical_margin,
            "exploratory_candidate_set": exploratory_entries,
            "note": (
                "exploratory_candidate_set is derived from discovery marginal CI overlap; "
                "it is not a paired statistical equivalence result. Canonical paired "
                "equivalence is established by benchmarks/routed_epilogue_confirmation.py"
            ),
        },
        "robust_regret": {
            "solver": solver_robust,
            "heuristic": heuristic_robust,
        },
        "per_schedule_stability": schedule_records,
    }

    write_artifact(args.output, stability_artifact)
    summary = {
        "runs": args.runs,
        "spearman_median": stability_artifact["cross_run_statistics"][
            "median_spearman_correlation"
        ],
        "top5_jaccard_median": stability_artifact["cross_run_statistics"][
            "median_top5_jaccard"
        ],
        "winner_stable": is_winner_stable,
        "representative_best": rep_best["schedule"],
        "solver_regret_median": solver_robust["median_regret_pct"],
        "solver_regret_ci95": [
            solver_robust["ci95_lower_pct"],
            solver_robust["ci95_upper_pct"],
        ],
        "solver_classification": solver_robust["classification"],
        "heuristic_regret_median": heuristic_robust["median_regret_pct"],
        "heuristic_regret_ci95": [
            heuristic_robust["ci95_lower_pct"],
            heuristic_robust["ci95_upper_pct"],
        ],
        "heuristic_classification": heuristic_robust["classification"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
