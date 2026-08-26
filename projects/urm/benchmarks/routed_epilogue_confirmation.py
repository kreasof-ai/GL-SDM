"""Paired adaptive confirmation benchmark for routed-epilogue schedule selection.

Implements discovery followed by confirmation:
- Freezes a deterministic shortlist and discovery reference schedule from discovery.
- Measures candidates in paired randomized blocks (AB / BA order) against the reference.
- Tracks thermal/clock drift with pre/post block drift sentinels.
- Uses adaptive block sampling for candidates near the practical equivalence boundary.
- Evaluates paired equivalence via two-level hierarchical bootstrap (runs x blocks).
- Validates provenance invariants across all child runs before aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/solver/routed-epilogue-confirmation.json")
DEFAULT_RUNS = 5
DEFAULT_BASE_SEED = 101
DEFAULT_MIN_BLOCKS = 8
DEFAULT_MAX_BLOCKS = 16
DEFAULT_SAMPLES_PER_PAIR = 5
DEFAULT_WARMUP_RUNS = 5
DEFAULT_PRACTICAL_MARGIN_PCT = 2.5
DEFAULT_SENTINEL_DRIFT_THRESHOLD_PCT = 15.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 42

PROBLEM = {
    "queries": 1024,
    "sources": 512,
    "route_width": 8,
    "value_dim": 1024,
    "dtype": "bfloat16",
    "training": True,
    "deterministic": False,
}

# Frozen discovery shortlist (aggregate top candidates from discovery sweep)
DISCOVERY_SHORTLIST: tuple[dict[str, object], ...] = (
    {
        "plan": "fused",
        "block_d": 256,
        "num_warps": 4,
        "num_stages": 2,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 128,
        "num_warps": 4,
        "num_stages": 2,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 256,
        "num_warps": 8,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 256,
        "num_warps": 4,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 128,
        "num_warps": 4,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 256,
        "num_warps": 8,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "full_row",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 64,
        "num_warps": 4,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
    {
        "plan": "fused",
        "block_d": 256,
        "num_warps": 2,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "segmented",
        "dtype": "bfloat16",
    },
)

# Fixed discovery reference schedule (never re-selected from confirmation data)
DISCOVERY_REFERENCE_SCHEDULE: dict[str, object] = DISCOVERY_SHORTLIST[0]


def compute_shortlist_hash(
    shortlist: Sequence[dict[str, object]],
    reference: dict[str, object],
    problem: dict[str, object],
) -> str:
    canonical = json.dumps(
        {"shortlist": shortlist, "reference": reference, "problem": problem},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


SHORTLIST_HASH = compute_shortlist_hash(
    DISCOVERY_SHORTLIST, DISCOVERY_REFERENCE_SCHEDULE, PROBLEM
)


def run_single_confirmation(
    *,
    seed: int,
    run_id: int,
    output_path: Path,
    min_blocks: int = DEFAULT_MIN_BLOCKS,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    sentinel_drift_threshold_pct: float = DEFAULT_SENTINEL_DRIFT_THRESHOLD_PCT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> dict[str, object]:
    """Execute one fresh-process confirmation run with paired blocks and drift sentinel."""
    import random

    import epilogue_schedules as sched
    import torch
    from measurement import (
        capture_gpu_operating_conditions,
        collect_cuda_samples,
        quantile,
    )
    from provenance import provenance, utc_now, write_artifact

    from urm.compiler.kernel_plan import SchedulePoint

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required: confirmation benchmark measures GPU schedules")

    indices, weights, values, row_scale = sched.make_inputs(
        PROBLEM["queries"],
        PROBLEM["route_width"],
        PROBLEM["sources"],
        PROBLEM["value_dim"],
        PROBLEM["dtype"],
    )

    def make_workload(point_dict: dict[str, object]) -> Callable[[], None]:
        point = SchedulePoint(**point_dict)

        def run() -> None:
            output, _fwd_info = sched.forward_launch(
                point, indices, weights, values, row_scale
            )
            _grads, _bwd_info = sched.backward_launch(
                point, indices, weights, values, row_scale, output
            )

        return run

    ref_workload = make_workload(DISCOVERY_REFERENCE_SCHEDULE)
    cand_workloads = [make_workload(c) for c in DISCOVERY_SHORTLIST]

    current_seed = seed
    for retry_idx in range(max_retries + 1):
        gpu_before = capture_gpu_operating_conditions()

        # Measure start sentinel
        for _ in range(warmup_runs):
            ref_workload()
        torch.cuda.synchronize()
        sentinel_start_raw = collect_cuda_samples(
            ref_workload, samples_per_round=samples_per_pair
        )
        sentinel_start_med = quantile(sentinel_start_raw, 0.5)

        paired_blocks_records: list[dict[str, object]] = []
        candidate_indices = list(range(len(DISCOVERY_SHORTLIST)))

        # Execute blocks adaptively
        block_count = min_blocks
        b = 0
        while b < block_count:
            block_rng = random.Random(current_seed + (b + 1) * 1000)
            shuffled_candidates = list(candidate_indices)
            block_rng.shuffle(shuffled_candidates)

            for cand_idx in shuffled_candidates:
                cand_sched = DISCOVERY_SHORTLIST[cand_idx]
                c_workload = cand_workloads[cand_idx]
                direction = "AB" if block_rng.random() < 0.5 else "BA"

                # Warmup pair
                for _ in range(warmup_runs):
                    ref_workload()
                    c_workload()
                torch.cuda.synchronize()

                if direction == "AB":
                    ref_raw = collect_cuda_samples(
                        ref_workload, samples_per_round=samples_per_pair
                    )
                    cand_raw = collect_cuda_samples(
                        c_workload, samples_per_round=samples_per_pair
                    )
                else:
                    cand_raw = collect_cuda_samples(
                        c_workload, samples_per_round=samples_per_pair
                    )
                    ref_raw = collect_cuda_samples(
                        ref_workload, samples_per_round=samples_per_pair
                    )

                c_med = quantile(cand_raw, 0.5)
                r_med = quantile(ref_raw, 0.5)
                paired_ratio = c_med / r_med
                paired_log_ratio = math.log(paired_ratio)
                paired_slowdown_pct = (paired_ratio - 1.0) * 100.0

                paired_blocks_records.append(
                    {
                        "block_id": b,
                        "candidate_index": cand_idx,
                        "candidate": cand_sched,
                        "direction": direction,
                        "candidate_median_ms": round(c_med, 6),
                        "reference_median_ms": round(r_med, 6),
                        "paired_ratio": round(paired_ratio, 6),
                        "paired_log_ratio": round(paired_log_ratio, 6),
                        "paired_slowdown_pct": round(paired_slowdown_pct, 4),
                        "candidate_raw_samples_ms": [round(s, 6) for s in cand_raw],
                        "reference_raw_samples_ms": [round(s, 6) for s in ref_raw],
                    }
                )
            b += 1

            # Adaptive check at min_blocks boundary
            if b == min_blocks and block_count < max_blocks:
                # Check if any candidate is ambiguous near 2.5% boundary
                ambiguous = False
                for c_idx in candidate_indices:
                    logs = [
                        r["paired_log_ratio"]
                        for r in paired_blocks_records
                        if r["candidate_index"] == c_idx
                    ]
                    mean_log = sum(logs) / len(logs)
                    slowdown = (math.exp(mean_log) - 1.0) * 100.0
                    if 1.8 <= slowdown <= 3.2:
                        ambiguous = True
                        break
                if ambiguous:
                    block_count = min(max_blocks, block_count + 4)

        # Measure end sentinel
        for _ in range(warmup_runs):
            ref_workload()
        torch.cuda.synchronize()
        sentinel_end_raw = collect_cuda_samples(
            ref_workload, samples_per_round=samples_per_pair
        )
        sentinel_end_med = quantile(sentinel_end_raw, 0.5)
        sentinel_drift_pct = (
            abs(sentinel_end_med - sentinel_start_med) / sentinel_start_med * 100.0
        )
        gpu_after = capture_gpu_operating_conditions()

        if (
            sentinel_drift_pct > sentinel_drift_threshold_pct
            and retry_idx < max_retries
        ):
            print(
                f"Run {run_id} sentinel drift {sentinel_drift_pct:.2f}% "
                f"> {sentinel_drift_threshold_pct}%; retrying ({retry_idx + 1}/{max_retries})..."
            )
            current_seed += 9999
            continue

        # Successful or final run
        run_artifact = {
            "schema_version": 1,
            "generated_utc": utc_now(),
            "provenance": {
                **provenance(
                    f"python benchmarks/routed_epilogue_confirmation.py --single-run --seed {current_seed} --run-id {run_id}",
                    {
                        "problem": PROBLEM,
                        "shortlist_hash": SHORTLIST_HASH,
                        "seed": current_seed,
                        "run_id": run_id,
                    },
                ),
                "shortlist_hash": SHORTLIST_HASH,
            },
            "run_metadata": {
                "run_id": run_id,
                "seed": current_seed,
                "initial_seed": seed,
                "retries_performed": retry_idx,
                "total_blocks_executed": block_count,
                "samples_per_pair": samples_per_pair,
                "warmup_runs": warmup_runs,
                "sentinel_drift": {
                    "start_median_ms": round(sentinel_start_med, 6),
                    "end_median_ms": round(sentinel_end_med, 6),
                    "drift_pct": round(sentinel_drift_pct, 4),
                    "threshold_pct": sentinel_drift_threshold_pct,
                    "passed": bool(sentinel_drift_pct <= sentinel_drift_threshold_pct),
                },
                "operating_conditions_before": gpu_before,
                "operating_conditions_after": gpu_after,
            },
            "problem": PROBLEM,
            "shortlist_hash": SHORTLIST_HASH,
            "discovery_reference_schedule": DISCOVERY_REFERENCE_SCHEDULE,
            "shortlist": list(DISCOVERY_SHORTLIST),
            "paired_blocks": paired_blocks_records,
        }
        write_artifact(output_path, run_artifact)
        return run_artifact

    raise RuntimeError(
        f"Run {run_id} failed after {max_retries} retries due to persistent sentinel drift"
    )


def validate_child_provenance_invariants(
    child_artifacts: Sequence[dict[str, object]],
) -> None:
    """Fail closed if any child run disagrees on fundamental environmental invariants."""
    if len(child_artifacts) < 2:
        return
    invariant_keys = (
        "git_revision",
        "dirty_tree",
        "shortlist_hash",
        "gpu",
        "driver",
        "cuda",
        "pytorch",
        "triton",
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
            # Check problem configuration agreement
            prob_i = child_artifacts[i]["problem"]
            prob_j = child_artifacts[j]["problem"]
            if prob_i != prob_j:
                raise RuntimeError(
                    f"Child problem configuration mismatch between run {i} and run {j}: "
                    f"{prob_i} != {prob_j}"
                )


def run_confirmation_suite(
    *,
    runs: int = DEFAULT_RUNS,
    base_seed: int = DEFAULT_BASE_SEED,
    output_path: Path = DEFAULT_OUTPUT,
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN_PCT,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Orchestrate multiple independent subprocess confirmation runs and aggregate."""
    from measurement import hierarchical_bootstrap_paired_slowdown, quantile
    from provenance import provenance, utc_now, write_artifact

    project_root = Path(__file__).parents[1]
    confirmation_script = (
        project_root / "benchmarks" / "routed_epilogue_confirmation.py"
    )

    child_artifacts: list[dict[str, object]] = []
    seeds = [base_seed + i * 100 for i in range(runs)]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for run_idx, seed in enumerate(seeds):
            temp_output = temp_dir_path / f"confirm_run_{run_idx}.json"
            cmd = [
                sys.executable,
                str(confirmation_script),
                "--single-run",
                "--output",
                str(temp_output),
                "--seed",
                str(seed),
                "--run-id",
                str(run_idx),
            ]
            print(
                f"Executing paired confirmation run {run_idx + 1}/{runs} (seed={seed})..."
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
                    f"Confirmation child run {run_idx} failed (code {res.returncode}):\n{res.stderr}"
                )
            if not temp_output.exists():
                raise RuntimeError(
                    f"Confirmation child run {run_idx} produced no artifact at {temp_output}"
                )
            child_artifact = json.loads(temp_output.read_text(encoding="utf-8"))
            child_artifacts.append(child_artifact)

    # 1. Validate child provenance invariants across all children
    validate_child_provenance_invariants(child_artifacts)

    # 2. Extract paired log ratios per candidate across runs
    candidate_count = len(DISCOVERY_SHORTLIST)
    candidate_paired_log_ratios: list[list[list[float]]] = [
        [] for _ in range(candidate_count)
    ]
    candidate_raw_samples: list[list[list[float]]] = [
        [] for _ in range(candidate_count)
    ]

    for run_artifact in child_artifacts:
        blocks = run_artifact["paired_blocks"]
        for cand_idx in range(candidate_count):
            cand_blocks = [b for b in blocks if b["candidate_index"] == cand_idx]
            log_ratios = [b["paired_log_ratio"] for b in cand_blocks]
            raws = [s for b in cand_blocks for s in b["candidate_raw_samples_ms"]]
            candidate_paired_log_ratios[cand_idx].append(log_ratios)
            candidate_raw_samples[cand_idx].append(raws)

    # 3. Hierarchical bootstrap paired analysis per candidate
    evaluated_candidates: list[dict[str, object]] = []

    for cand_idx, cand_sched in enumerate(DISCOVERY_SHORTLIST):
        log_ratios_by_run = candidate_paired_log_ratios[cand_idx]
        med_slowdown, ci_lo, ci_hi = hierarchical_bootstrap_paired_slowdown(
            log_ratios_by_run,
            num_resamples=bootstrap_resamples,
            confidence=0.95,
            seed=bootstrap_seed + cand_idx,
        )

        is_equiv = bool(ci_hi <= practical_margin)
        is_reference = bool(cand_sched == DISCOVERY_REFERENCE_SCHEDULE)

        # Collect per-run median candidate latencies
        per_run_meds: list[float] = []
        for run_artifact in child_artifacts:
            cand_blocks = [
                b
                for b in run_artifact["paired_blocks"]
                if b["candidate_index"] == cand_idx
            ]
            all_cand_raw = [
                s for b in cand_blocks for s in b["candidate_raw_samples_ms"]
            ]
            if all_cand_raw:
                per_run_meds.append(round(quantile(all_cand_raw, 0.5), 6))

        evaluated_candidates.append(
            {
                "candidate_index": cand_idx,
                "schedule": cand_sched,
                "is_discovery_reference": is_reference,
                "median_paired_slowdown_pct": round(med_slowdown, 4),
                "ci95_lower_slowdown_pct": round(ci_lo, 4),
                "ci95_upper_slowdown_pct": round(ci_hi, 4),
                "is_practically_equivalent": is_equiv,
                "per_run_medians_ms": per_run_meds,
                "paired_blocks_measured_total": sum(len(r) for r in log_ratios_by_run),
            }
        )

    # Sort evaluated candidates: equivalent set first (sorted by median slowdown), then rest
    evaluated_candidates.sort(
        key=lambda c: (
            not c["is_practically_equivalent"],
            c["median_paired_slowdown_pct"],
            json.dumps(c["schedule"], sort_keys=True),
        )
    )
    for rank, c in enumerate(evaluated_candidates, 1):
        c["confirmation_rank"] = rank

    equivalent_set = [c for c in evaluated_candidates if c["is_practically_equivalent"]]

    # Robust deployment representative selection
    if equivalent_set:
        representative_choice = equivalent_set[0]["schedule"]
        deployment_status = "confirmed_equivalent"
    else:
        # Conservative fallback to discovery reference schedule
        representative_choice = DISCOVERY_REFERENCE_SCHEDULE
        deployment_status = "conservative_fallback_measurement_limited"

    # Solver schedule status in confirmation
    solver_schedule = {
        "plan": "fused",
        "block_d": 256,
        "num_warps": 8,
        "num_stages": 1,
        "grad_values_decomposition": "per_query",
        "grad_values_schedule": "full_row",
        "dtype": "bfloat16",
    }
    solver_eval = next(
        (c for c in evaluated_candidates if c["schedule"] == solver_schedule),
        None,
    )

    confirmation_artifact = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "provenance": {
            **provenance(
                f"python benchmarks/routed_epilogue_confirmation.py --runs {runs}",
                {
                    "problem": PROBLEM,
                    "shortlist_hash": SHORTLIST_HASH,
                    "runs": runs,
                    "base_seed": base_seed,
                    "practical_margin_pct": practical_margin,
                },
            ),
            "shortlist_hash": SHORTLIST_HASH,
        },
        "problem": PROBLEM,
        "confirmation_config": {
            "runs": runs,
            "base_seed": base_seed,
            "seeds": seeds,
            "shortlist_size": len(DISCOVERY_SHORTLIST),
            "shortlist_hash": SHORTLIST_HASH,
            "discovery_reference_schedule": DISCOVERY_REFERENCE_SCHEDULE,
            "min_blocks_per_run": DEFAULT_MIN_BLOCKS,
            "max_blocks_per_run": DEFAULT_MAX_BLOCKS,
            "samples_per_pair": DEFAULT_SAMPLES_PER_PAIR,
            "warmup_runs": DEFAULT_WARMUP_RUNS,
            "practical_equivalence_margin_pct": practical_margin,
            "sentinel_drift_threshold_pct": DEFAULT_SENTINEL_DRIFT_THRESHOLD_PCT,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
        },
        "runs": [
            {
                "run_id": r["run_metadata"]["run_id"],
                "seed": r["run_metadata"]["seed"],
                "total_blocks_executed": r["run_metadata"]["total_blocks_executed"],
                "sentinel_drift": r["run_metadata"]["sentinel_drift"],
                "operating_conditions_before": r["run_metadata"][
                    "operating_conditions_before"
                ],
                "operating_conditions_after": r["run_metadata"][
                    "operating_conditions_after"
                ],
            }
            for r in child_artifacts
        ],
        "deployment_decision": {
            "status": deployment_status,
            "practical_equivalence_margin_pct": practical_margin,
            "representative_schedule": representative_choice,
            "conservative_fallback_schedule": DISCOVERY_REFERENCE_SCHEDULE,
            "equivalent_set_count": len(equivalent_set),
            "shortlist_count": len(DISCOVERY_SHORTLIST),
            "solver_selected_schedule": solver_schedule,
            "solver_is_in_equivalent_set": (
                solver_eval["is_practically_equivalent"] if solver_eval else False
            ),
            "solver_paired_slowdown_pct": (
                solver_eval["median_paired_slowdown_pct"] if solver_eval else None
            ),
            "solver_ci95_upper_slowdown_pct": (
                solver_eval["ci95_upper_slowdown_pct"] if solver_eval else None
            ),
        },
        "evaluated_shortlist": evaluated_candidates,
    }

    write_artifact(output_path, confirmation_artifact)
    summary = {
        "runs": runs,
        "shortlist_hash": SHORTLIST_HASH[:12],
        "equivalent_set_size": len(equivalent_set),
        "deployment_status": deployment_status,
        "representative_schedule": representative_choice,
        "solver_in_equivalent_set": (
            solver_eval["is_practically_equivalent"] if solver_eval else False
        ),
        "solver_ci95_upper_slowdown_pct": (
            solver_eval["ci95_upper_slowdown_pct"] if solver_eval else None
        ),
    }
    print(json.dumps(summary, indent=2))
    return confirmation_artifact


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
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--run-id", type=int, default=0)
    args = parser.parse_args()

    if args.single_run:
        run_single_confirmation(
            seed=args.seed,
            run_id=args.run_id,
            output_path=args.output,
            min_blocks=DEFAULT_MIN_BLOCKS,
            max_blocks=DEFAULT_MAX_BLOCKS,
            samples_per_pair=DEFAULT_SAMPLES_PER_PAIR,
            warmup_runs=DEFAULT_WARMUP_RUNS,
            sentinel_drift_threshold_pct=DEFAULT_SENTINEL_DRIFT_THRESHOLD_PCT,
            max_retries=DEFAULT_MAX_RETRIES,
        )
    else:
        run_confirmation_suite(
            runs=args.runs,
            base_seed=args.base_seed,
            output_path=args.output,
            practical_margin=args.practical_margin,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )


if __name__ == "__main__":
    main()
