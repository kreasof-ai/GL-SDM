"""Solver-guided routed-epilogue schedule selection with empirical grounding.

Pipeline exercised end-to-end on the committed problem:

    semantic program -> UrmCompiler.compile() end-to-end
      (candidate selection -> candidate-bound schedule model -> Z3
       optimization -> independent verification -> serialized decision)
    -> explicit feasibility/optimization cross-check on the same model
    -> exhaustive sweep agreement check
    -> empirical measurement of the deduplicated legal fused schedule grid
       (raw samples, seeded interleaved rounds, genuine median AND p95)
    -> compiler-driven vs direct dispatch overhead measurement

Z3 is successful if it removes illegal schedules and produces explainable,
independently verified plans. It is NOT required to predict the fastest
schedule; empirical regret of the selected schedule is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import torch

DEFAULT_OUTPUT = Path("results/compiler/solver/routed-epilogue-selection.json")

PROBLEM = {
    "queries": 1024,
    "sources": 512,
    "route_width": 8,
    "value_dim": 1024,
    "dtype": "bfloat16",
}

# Measurement methodology (documented in the artifact, tested in
# tests/test_measurement.py).
MEASUREMENT = {
    "quantile_definition": (
        "type-7 linear interpolation over raw CUDA-event samples "
        "(rank = q*(n-1); median == statistics.median for every n)"
    ),
    "warmup_runs": 5,
    "rounds": 4,
    "samples_per_round": 5,
    "seed": 17,
}

REGRET_TARGET_PCT = 10.0
OVERHEAD_GATE_PCT = 5.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-empirical", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required: this benchmark measures GPU schedules")

    import epilogue_schedules as sched
    from measurement import measure_schedules_interleaved, summarize_samples
    from provenance import provenance, utc_now, write_artifact

    from urm.compiler.anchors.routed_reduction_epilogue import (
        RoutedEpilogueLaunchConfig,
        _extract_resource_usage,
        make_triton_compile_probe,
    )
    from urm.compiler.kernel_plan import (
        decode_schedule_point,
        exhaustive_schedule_sweep,
        schedule_point_to_assignment,
        verify_schedule_assignment,
    )
    from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler
    from urm.compiler.schedule_space import (
        SUPPORTED_BLOCKS,
        SUPPORTED_STAGES,
        SUPPORTED_WARPS,
        PlanKind,
        SchedulePoint,
        heuristic_schedule,
        legal_schedules,
    )
    from urm.compiler.semantic import DType, row_scaled_routed_reduction_program
    from urm.compiler.solver import FeasibilityPass, OptimizationPass, z3_version

    intent = CompilationIntent.TRAINING
    program = row_scaled_routed_reduction_program(
        queries=PROBLEM["queries"],
        route_width=PROBLEM["route_width"],
        sources=PROBLEM["sources"],
        value_dim=PROBLEM["value_dim"],
        value_dtype=DType.BFLOAT16,
    )
    compiler = UrmCompiler(compile_probe=make_triton_compile_probe())
    candidates = compiler.enumerate_candidates(program, intent)
    fused = next(c for c in candidates if c.kind == "rewrite")
    model = compiler.build_constraints(
        program, fused.candidate_id, intent, schedule_params=ScheduleParams()
    )

    # -- explicit feasibility pass -------------------------------------------
    started = time.perf_counter()
    feasibility = FeasibilityPass().run(model)
    feasibility_ms = (time.perf_counter() - started) * 1000.0
    assert feasibility.status.value == "sat", feasibility.diagnostics

    # -- explicit optimization pass ------------------------------------------
    started = time.perf_counter()
    optimized = OptimizationPass().run(model)
    optimization_ms = (time.perf_counter() - started) * 1000.0
    assert optimized.status.value == "sat"

    # -- END-TO-END compile(): the selected schedule enters the real path ------
    compiled_result = compiler.compile(program, intent=intent)
    decision = compiled_result.schedule_decision
    assert decision is not None, "compile() must carry a schedule decision"
    assert all(attempt.verified for attempt in decision.attempts)
    z3_point = decode_schedule_point(model, optimized.assignment)

    # -- independent verification of the explicit optimum -----------------------
    report = verify_schedule_assignment(model, optimized.assignment)
    assert report.ok, report.failures[:3]

    # -- exhaustive agreement over the decoded point space ----------------------
    legal_assignments, ranked, total_points = exhaustive_schedule_sweep(model)
    agree_optima = ranked[0][0] == optimized.assignment
    reference_problem = model_to_problem(model)
    # Like-for-like legality: the committed model is CANDIDATE-BOUND, so the
    # reference enumeration must be restricted to the same allowed plans.
    allowed_plans = set(model.metadata.get("allowed_plans", "base,fused").split(","))
    reference_legal = [
        point
        for point in legal_schedules(reference_problem)
        if point.plan in allowed_plans
    ]
    reference_legal_set = {point.stable_key for point in reference_legal}
    model_sweep_legal_set = {
        decode_schedule_point(model, assignment).stable_key
        for assignment in legal_assignments
    }
    exact_set_agreement = reference_legal_set == model_sweep_legal_set

    notes = [
        (
            "exhaustive sweep enumerates the decoded point space and re-checks "
            "every named constraint imperatively; it is the legality oracle"
        ),
        (
            "reference legality is restricted to the selected candidate's "
            "allowed plans so the comparison is like-for-like against the "
            "candidate-bound model"
        ),
        (
            "empirical numbers are computed from RAW CUDA-event samples with a "
            "tested type-7 percentile; analytical objective values are never "
            "presented as measurements"
        ),
        (
            "schedules are deduplicated by stable key before measurement and "
            "measured in seeded, shuffled interleaved rounds"
        ),
    ]

    # -- unsat example under deterministic training (same shape) ---------------
    det_model = compiler.build_constraints(
        program,
        fused.candidate_id,
        intent,
        schedule_params=ScheduleParams(deterministic=True),
    )
    det_feasibility = FeasibilityPass().run(det_model)
    unsat_example = None
    if det_feasibility.status.value == "unsat":
        names = list(det_feasibility.unsat_core_names)
        unsat_example = {
            "core_names": names,
            "concise_message": (
                "No legal deterministic training schedule: every implemented "
                "grad-value lowering accumulates through relaxed cross-program "
                "atomics."
            ),
        }

    artifact: dict[str, object] = {
        "schema_version": 2,
        "generated_utc": utc_now(),
        "provenance": {
            **provenance("python benchmarks/routed_epilogue_selection.py", PROBLEM),
            "constraint_model_hash": model.summary_hash(),
        },
        "problem": {
            **PROBLEM,
            "training": True,
            "deterministic": False,
        },
        "space_size": {
            "total_points": total_points,
            "legal_points": len(legal_assignments),
            "pruned_by_constraints": total_points - len(legal_assignments),
        },
        "legality": {
            "reference_impl": "schedule_space.legal_schedules",
            "reference_legal_points": len(reference_legal),
            "model_sweep_legal_points": len(legal_assignments),
            "reference_matches_model_sweep": exact_set_agreement,
            "exact_set_agreement": exact_set_agreement,
            "solver_assignments_checked": 1,
            "agreement": bool(exact_set_agreement and agree_optima),
            "legality_accuracy": 1.0 if exact_set_agreement else 0.0,
        },
        "unsat_core_example": unsat_example,
        "z3_selection": {
            "status": optimized.status.value,
            "selected_schedule": z3_point.as_dict(),
            "matches_compiled_decision": (
                decision.schedule_point.stable_key == z3_point.stable_key
            ),
            "solve_ms": round(feasibility_ms + optimization_ms, 3),
            "feasibility_solve_ms": round(feasibility_ms, 3),
            "optimization_solve_ms": round(optimization_ms, 3),
            "objective_values": list(optimized.objective_values),
            "verified": report.ok,
            "verification_failures": [failure.to_dict() for failure in report.failures],
            "solver_version": z3_version(),
            "model_hash": model.summary_hash(),
        },
        "schedule_decision": decision.to_dict(),
        "measurement": {
            **MEASUREMENT,
            "deduplicated_before_measurement": True,
        },
        "heuristic_schedule": heuristic_schedule(reference_problem).as_dict(),
        "compile_feedback": [],
        "empirical": None,
        "dispatch_overhead": None,
        "notes": notes,
    }

    if args.skip_empirical:
        write_artifact(args.output, artifact)
        print(
            json.dumps({k: artifact[k] for k in ("space_size", "legality")}, indent=2)
        )
        return

    # -- empirical measurement of the deduplicated fused grid -------------------
    indices, weights, values, row_scale = sched.make_inputs(
        PROBLEM["queries"],
        PROBLEM["route_width"],
        PROBLEM["sources"],
        PROBLEM["value_dim"],
        PROBLEM["dtype"],
    )

    dtype_name = PROBLEM["dtype"]
    grid_points: list[SchedulePoint] = []
    # Full (block, warps, stages) space at per-query/segmented traversal ...
    for block in SUPPORTED_BLOCKS:
        for warps in SUPPORTED_WARPS:
            for stage in SUPPORTED_STAGES:
                grid_points.append(
                    SchedulePoint(
                        plan=PlanKind.FUSED.value,
                        block_d=block,
                        num_warps=warps,
                        num_stages=stage,
                        grad_values_decomposition="per_query",
                        grad_values_schedule="segmented",
                        dtype=dtype_name,
                    )
                )
    # ... plus decomposition/traversal variants at the selected tile.
    for decomp in ("per_query", "per_route"):
        for traversal in ("full_row", "segmented"):
            if decomp == "per_route" and traversal == "full_row":
                continue  # per_route is segmented by construction; full_row is invalid
            grid_points.append(
                SchedulePoint(
                    plan=z3_point.plan,
                    block_d=z3_point.block_d,
                    num_warps=z3_point.num_warps,
                    num_stages=z3_point.num_stages,
                    grad_values_decomposition=decomp,
                    grad_values_schedule=traversal,
                    dtype=dtype_name,
                )
            )

    legal_keys = {point.stable_key for point in reference_legal}
    unique_points = []
    seen_keys: set[str] = set()
    for point in grid_points:
        if point.stable_key not in legal_keys or point.stable_key in seen_keys:
            continue
        seen_keys.add(point.stable_key)
        unique_points.append(point)

    resource_by_key: dict[str, dict[str, int | None]] = {}

    def workload(point: SchedulePoint):
        def run() -> None:
            output, fwd_info = sched.forward_launch(
                point, indices, weights, values, row_scale
            )
            _grads, bwd_info = sched.backward_launch(
                point, indices, weights, values, row_scale, output
            )
            if point.stable_key not in resource_by_key:
                fwd_res = _extract_resource_usage(fwd_info.kernel, fwd_info.handle)
                all_kres = [fwd_res]
                for name, handle in bwd_info.extra_handles:
                    all_kres.append(_extract_resource_usage(name, handle))
                known_regs = [
                    k.registers_per_thread
                    for k in all_kres
                    if k.registers_per_thread is not None
                ]
                known_smem = [
                    k.shared_mem_bytes
                    for k in all_kres
                    if k.shared_mem_bytes is not None
                ]
                resource_by_key[point.stable_key] = {
                    "registers_per_thread": max(known_regs) if known_regs else None,
                    "shared_mem_bytes": max(known_smem) if known_smem else None,
                }

        return run

    # Priming pass doubles as compile-feedback capture per point.
    feedback_records: list[dict[str, object]] = []
    failed_keys: set[str] = set()
    workloads: dict[str, Callable[[], None]] = {}
    for point in unique_points:
        try:
            workload(point)()
        except Exception as error:  # noqa: BLE001 - recorded nogood path
            failed_keys.add(point.stable_key)
            record = compile_feedback_record(
                model,
                schedule_point_to_assignment(model, point),
                reason=str(error)[:200],
            )
            feedback_records.append(record)
            continue
        workloads[point.stable_key] = workload(point)

    measured = measure_schedules_interleaved(
        workloads,
        warmup_runs=MEASUREMENT["warmup_runs"],
        rounds=MEASUREMENT["rounds"],
        samples_per_round=MEASUREMENT["samples_per_round"],
        seed=MEASUREMENT["seed"],
    )

    samples: list[dict[str, object]] = []
    key_to_point = {p.stable_key: p for p in unique_points}
    for stable_key, raw in measured.items():
        stats = summarize_samples(raw)
        point = key_to_point[stable_key]
        res_info = resource_by_key.get(stable_key, {})
        samples.append(
            {
                "schedule": point.as_dict(),
                "median_ms": round(stats["median_ms"], 4),
                "p95_ms": round(stats["p95_ms"], 4),
                "min_ms": round(stats["min_ms"], 4),
                "sample_count": stats["sample_count"],
                "registers_per_thread": res_info.get("registers_per_thread"),
                "shared_mem_bytes": res_info.get("shared_mem_bytes"),
            }
        )
    samples.sort(key=lambda s: json.dumps(s["schedule"], sort_keys=True))

    best = min(samples, key=lambda s: s["median_ms"])

    def sample_for(point: SchedulePoint):
        return next(
            (s for s in samples if s["schedule"] == point.as_dict()),
            None,
        )

    def regret(sample) -> float:
        if sample is None or not best["median_ms"]:
            return float("nan")
        return round(
            100.0 * (sample["median_ms"] - best["median_ms"]) / best["median_ms"],
            2,
        )

    solver_regret = regret(sample_for(z3_point))
    heuristic_point = heuristic_schedule(reference_problem)
    heuristic_sample = sample_for(SchedulePoint(**heuristic_point.as_dict()))
    heuristic_regret = regret(heuristic_sample)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    compile_failures_observed = len(failed_keys)
    failed_points_excluded_from_measurement = (
        compile_failures_observed if len(samples) > 0 and failed_keys else 0
    )
    artifact["compile_feedback"] = feedback_records
    artifact["empirical"] = {
        "gpu": properties.name,
        "measured_points": len(samples),
        "compile_failures_observed": compile_failures_observed,
        "failed_points_excluded_from_measurement": (
            failed_points_excluded_from_measurement
        ),
        "nogoods_added": sum(
            1 for record in feedback_records if record.get("nogood_added")
        ),
        "best_point": best["schedule"],
        "best_median_ms": best["median_ms"],
        "solver_selected": z3_point.as_dict(),
        "heuristic_selected": heuristic_point.as_dict(),
        "regret": {
            "z3_vs_best_pct": solver_regret,
            "heuristic_vs_best_pct": heuristic_regret,
            "target_pct": REGRET_TARGET_PCT,
            "within_target": bool(
                not math.isnan(solver_regret) and solver_regret <= REGRET_TARGET_PCT
            ),
            "note": (
                "regret is computed within the measured fused grid against the "
                "genuine median; the base (unfused) plan was already compared "
                "in results/compiler/routed-scale-epilogue/benchmark.json"
            ),
        },
        "samples": samples,
    }

    # -- compiler-driven execution vs direct execution --------------------------
    plan_payload = compiled_result.plan.to_dict()
    step_config = next(
        (
            step["launch_config"]
            for step in plan_payload["steps"]
            if step["kind"] == "anchor_dispatch" and step["launch_config"]
        ),
        None,
    )
    if step_config is not None:
        direct_config = RoutedEpilogueLaunchConfig.from_point(z3_point)
        # Decoding the serialized plan into an executable configuration is
        # COMPILATION: it happens once, outside every timed region. The timed
        # region compares steady-state dispatch through the prepared plan
        # against a direct production launch.
        prepared_config = sched.prepare_plan_step(step_config)
        assert direct_config == prepared_config, (
            "direct and decoded configurations must be operationally equal"
        )

        # Warm up both dispatch paths before timing
        for _ in range(20):
            sched.launch_prepared_step(
                direct_config, indices, weights, values, row_scale
            )
            sched.launch_prepared_step(
                prepared_config, indices, weights, values, row_scale
            )
        torch.cuda.synchronize()

        BATCH_SIZE = 50
        ROUNDS = 20
        import random

        from measurement import quantile

        def time_batched_launch(cfg: RoutedEpilogueLaunchConfig) -> float:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(BATCH_SIZE):
                sched.launch_prepared_step(cfg, indices, weights, values, row_scale)
            end_ev.record()
            torch.cuda.synchronize()
            return start_ev.elapsed_time(end_ev) / BATCH_SIZE

        rng = random.Random(MEASUREMENT["seed"] + 1)
        direct_times: list[float] = []
        driven_times: list[float] = []
        deltas: list[float] = []

        for _ in range(ROUNDS):
            order = [("direct", direct_config), ("driven", prepared_config)]
            rng.shuffle(order)
            results = {}
            for name, cfg in order:
                results[name] = time_batched_launch(cfg)
            direct_times.append(results["direct"])
            driven_times.append(results["driven"])
            deltas.append(results["driven"] - results["direct"])

        direct_stats = summarize_samples(direct_times)
        driven_stats = summarize_samples(driven_times)
        delta_mean = sum(deltas) / len(deltas)
        delta_median = quantile(deltas, 0.5)
        delta_std = (sum((x - delta_mean) ** 2 for x in deltas) / len(deltas)) ** 0.5

        overhead_pct = round(
            100.0
            * (driven_stats["median_ms"] - direct_stats["median_ms"])
            / direct_stats["median_ms"],
            3,
        )
        artifact["dispatch_overhead"] = {
            "direct_median_ms": round(direct_stats["median_ms"], 4),
            "compiler_driven_median_ms": round(driven_stats["median_ms"], 4),
            "overhead_pct": overhead_pct,
            "gate_pct": OVERHEAD_GATE_PCT,
            "gate_pass": bool(overhead_pct <= OVERHEAD_GATE_PCT),
            "delta_mean_ms": round(delta_mean, 6),
            "delta_median_ms": round(delta_median, 6),
            "delta_std_ms": round(delta_std, 6),
            "paired_samples_count": ROUNDS,
            "batch_launches_per_sample": BATCH_SIZE,
            "note": (
                "dispatch equivalence evaluates steady-state execution through "
                "the decoded launch configuration from the serialized executable "
                "plan versus direct configuration launch over repeated batched "
                "launches with paired randomized sampling"
            ),
        }

    write_artifact(args.output, artifact)
    summary = {
        "legality": artifact["legality"],
        "selected": z3_point.as_dict(),
        "regret": artifact["empirical"]["regret"],
        "dispatch_overhead": artifact["dispatch_overhead"],
    }
    print(json.dumps(summary, indent=2))


def compile_feedback_record(
    model, assignment, *, reason: str | None
) -> dict[str, object]:
    """Record compile feedback for THIS failed assignment (exact nogood)."""
    from urm.compiler.kernel_plan import apply_compile_feedback

    return apply_compile_feedback(
        model,
        assignment,
        success=False,
        reason=reason,
        max_nogoods=64,
    )


def model_to_problem(model):
    from urm.compiler.schedule_space import ScheduleProblem

    meta = model.metadata
    return ScheduleProblem(
        queries=int(meta["queries"]),
        sources=int(meta["sources"]),
        route_width=int(meta["route_width"]),
        value_dim=int(meta["value_dim"]),
        dtypes=tuple(meta["dtypes"].split(",")),
        training=meta["training"] == "1",
        deterministic=meta["deterministic"] == "1",
        fused_anchor_available=meta.get("fused_anchor_available", "1") == "1",
    )


if __name__ == "__main__":
    main()
