"""Solver-guided routed-epilogue schedule selection with empirical grounding.

Pipeline exercised end-to-end on the committed problem:

    semantic program -> candidate enumeration -> constraint model
      -> Z3 feasibility (tracked, named assertions)
      -> bounded lexicographic optimization
      -> independent imperative verification
      -> exhaustive sweep agreement check
      -> empirical measurement of the legal fused schedule grid on GPU
      -> compile feedback capture

Z3 is successful if it removes illegal schedules and produces explainable,
independently verified plans. It is NOT required to predict the fastest
schedule; empirical regret of the selected schedule is reported separately.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
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


def cuda_median_ms(fn, warmup: int = 5, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-empirical", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required: this benchmark measures GPU schedules")

    import epilogue_schedules as sched
    from provenance import provenance, utc_now, write_artifact

    from urm.compiler.diagnostics import CompilerError  # noqa: F401 - parity with tests
    from urm.compiler.kernel_plan import (
        apply_compile_feedback,
        decode_schedule_point,
        exhaustive_schedule_sweep,
        verify_schedule_assignment,
    )
    from urm.compiler.planner import (
        CompilationIntent,
        ScheduleParams,
        UrmCompiler,
    )
    from urm.compiler.schedule_space import (
        SUPPORTED_BLOCKS,
        SUPPORTED_STAGES,
        SUPPORTED_WARPS,
        PlanKind,
        SchedulePoint,
        heuristic_schedule,
        legal_schedules,
    )
    from urm.compiler.semantic import row_scaled_routed_reduction_program
    from urm.compiler.solver import FeasibilityPass, OptimizationPass, z3_version

    intent = CompilationIntent.TRAINING
    program = row_scaled_routed_reduction_program(
        queries=PROBLEM["queries"],
        route_width=PROBLEM["route_width"],
        sources=PROBLEM["sources"],
        value_dim=PROBLEM["value_dim"],
        value_dtype=__import__(
            "urm.compiler.semantic", fromlist=["DType"]
        ).DType.BFLOAT16,
    )
    compiler = UrmCompiler()
    candidates = compiler.enumerate_candidates(program, intent)
    fused = next(c for c in candidates if c.kind == "rewrite")
    model = compiler.build_constraints(
        program, fused.candidate_id, intent, schedule_params=ScheduleParams()
    )

    # -- feasibility pass ----------------------------------------------------
    started = time.perf_counter()
    feasibility = FeasibilityPass().run(model)
    feasibility_ms = (time.perf_counter() - started) * 1000.0
    assert feasibility.status.value == "sat", feasibility.diagnostics

    # -- optimization pass ----------------------------------------------------
    started = time.perf_counter()
    optimized = OptimizationPass().run(model)
    optimization_ms = (time.perf_counter() - started) * 1000.0
    assert optimized.status.value == "sat"

    # -- independent verification ----------------------------------------------
    report = verify_schedule_assignment(model, optimized.assignment)
    assert report.ok, report.failures[:3]
    z3_point = decode_schedule_point(model, optimized.assignment)

    # -- exhaustive agreement over the decoded point space ----------------------
    legal_assignments, ranked, total_points = exhaustive_schedule_sweep(model)
    agree_optima = ranked[0][0] == optimized.assignment
    reference_problem = model_to_problem(model)
    reference_legal = legal_schedules(reference_problem)

    notes = [
        (
            "exhaustive sweep enumerates the decoded point space and re-checks "
            "every named constraint imperatively; it is the legality oracle"
        ),
        (
            "empirical numbers are measured medians on the committed GPU host; "
            "analytical objective values are never presented as measurements"
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
        "schema_version": 1,
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
            "reference_matches_model_sweep": (
                len(reference_legal) == len(legal_assignments)
            ),
            "z3_sat_assignments": 1,
            "agreement": bool(
                len(reference_legal) == len(legal_assignments) and agree_optima
            ),
            "legality_accuracy": (
                1.0 if len(reference_legal) == len(legal_assignments) else 0.0
            ),
        },
        "unsat_core_example": unsat_example,
        "z3_selection": {
            "status": optimized.status.value,
            "selected_schedule": z3_point.as_dict(),
            "solve_ms": round(feasibility_ms + optimization_ms, 3),
            "feasibility_solve_ms": round(feasibility_ms, 3),
            "optimization_solve_ms": round(optimization_ms, 3),
            "objective_values": list(optimized.objective_values),
            "verified": report.ok,
            "verification_failures": [failure.to_dict() for failure in report.failures],
            "solver_version": z3_version(),
            "model_hash": model.summary_hash(),
        },
        "heuristic_schedule": heuristic_schedule(reference_problem).as_dict(),
        "compile_feedback": [],
        "empirical": None,
        "notes": notes,
    }

    if args.skip_empirical:
        write_artifact(args.output, artifact)
        print(
            json.dumps({k: artifact[k] for k in ("space_size", "legality")}, indent=2)
        )
        return

    # -- empirical measurement of the fused schedule grid -----------------------
    indices, weights, values, row_scale = sched.make_inputs(
        PROBLEM["queries"],
        PROBLEM["route_width"],
        PROBLEM["sources"],
        PROBLEM["value_dim"],
        PROBLEM["dtype"],
    )

    def measure(point: SchedulePoint) -> tuple[float, float, dict[str, int | None]]:
        def run() -> None:
            output, _handle = sched.forward_launch(
                point, indices, weights, values, row_scale
            )
            sched.backward_launch(point, indices, weights, values, row_scale, output)

        median_ms = cuda_median_ms(run)

        # Compile feedback from the most recent forward launch handle.
        _output, handle = sched.forward_launch(
            point, indices, weights, values, row_scale
        )
        feedback = sched.compile_feedback_for(handle)
        return median_ms, median_ms, feedback

    # Stratified grid: full (block, warps, stages) space at the chosen dtype
    # and traversal; then decomposition/traversal variants at the chosen tile.
    dtype_name = PROBLEM["dtype"]
    samples: list[dict[str, object]] = []
    compile_failures_avoided = 0
    feedback_records: list[dict[str, object]] = []
    legal_keys = {point.stable_key for point in legal_schedules(reference_problem)}
    for plan in (PlanKind.FUSED.value,):
        for block in SUPPORTED_BLOCKS:
            for warps in SUPPORTED_WARPS:
                for stage in SUPPORTED_STAGES:
                    point = SchedulePoint(
                        plan=plan,
                        block_d=block,
                        num_warps=warps,
                        num_stages=stage,
                        grad_values_decomposition="per_query",
                        grad_values_schedule="segmented",
                        dtype=dtype_name,
                    )
                    if point.stable_key not in legal_keys:
                        continue
                    try:
                        median, p95, fb = measure(point)
                    except Exception as error:  # noqa: BLE001 - recorded nogood path
                        record = apply_compile_feedback(
                            model,
                            optimized.assignment,
                            success=False,
                            reason=str(error)[:200],
                        )
                        feedback_records.append(record)
                        if not record.get("nogood_added"):
                            compile_failures_avoided += 1
                        continue
                    samples.append(
                        {
                            "schedule": point.as_dict(),
                            "median_ms": round(median, 4),
                            "p95_ms": round(p95, 4),
                            "registers_per_thread": fb["registers_per_thread"],
                            "shared_mem_bytes": fb["shared_mem_bytes"],
                        }
                    )
    for decomp in ("per_query", "per_route"):
        for traversal in ("full_row", "segmented"):
            point = SchedulePoint(
                plan=z3_point.plan,
                block_d=z3_point.block_d,
                num_warps=z3_point.num_warps,
                num_stages=z3_point.num_stages,
                grad_values_decomposition=decomp,
                grad_values_schedule=traversal,
                dtype=dtype_name,
            )
            if point.stable_key in legal_keys:
                median, p95, _fb = measure(point)
                samples.append(
                    {
                        "schedule": point.as_dict(),
                        "median_ms": round(median, 4),
                        "p95_ms": round(p95, 4),
                        "registers_per_thread": None,
                        "shared_mem_bytes": None,
                    }
                )

    best = min(samples, key=lambda s: s["median_ms"])
    z3_sample = next(
        (
            s
            for s in samples
            if all(
                s["schedule"][key] == value for key, value in z3_point.as_dict().items()
            )
        ),
        None,
    )
    heuristic_point = heuristic_schedule(reference_problem)
    heuristic_sample = next(
        (
            s
            for s in samples
            if all(
                s["schedule"][key] == value
                for key, value in heuristic_point.as_dict().items()
            )
        ),
        None,
    )

    def regret(sample) -> float:
        if sample is None or not best["median_ms"]:
            return float("nan")
        return round(
            100.0 * (sample["median_ms"] - best["median_ms"]) / best["median_ms"],
            2,
        )

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    artifact["empirical"] = {
        "gpu": properties.name,
        "measured_points": len(samples),
        "compile_failures_avoided": compile_failures_avoided,
        "best_point": best["schedule"],
        "best_median_ms": best["median_ms"],
        "regret": {
            "z3_vs_best_pct": regret(z3_sample),
            "heuristic_vs_best_pct": regret(heuristic_sample),
            "note": (
                "regret is computed within the measured fused grid; the base "
                "(unfused) plan was already compared in "
                "results/compiler/routed-scale-epilogue/benchmark.json"
            ),
        },
        "samples": samples,
    }
    artifact["compile_feedback"] = feedback_records

    write_artifact(args.output, artifact)
    print(
        json.dumps(
            {
                "legality": artifact["legality"],
                "selected": z3_point.as_dict(),
                "regret": artifact["empirical"]["regret"],
            },
            indent=2,
        )
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
