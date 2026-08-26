"""Materialized vs fused row-scale epilogue comparison (CODA-inspired plan).

Compares three schedules for

    output[q, d] = row_scale[q] * sum_k weights[q, k] * values[indices[q, k], d]

  A  materialized   : trusted routed-reduction v1 writes `base`; a second op
                      reads it, applies row_scale, writes output.
  B  fused          : experimental typed epilogue folds row_scale into the
                      reduction mainloop; `base` is never materialized.
  C  fused_fullrow  : variant of B forcing a full-row tile configuration.
                      Retained in the artifact whatever the outcome; classified
                      rejected when slower than B's heuristic launch.

Host-bound and GPU-bound shapes are measured separately. Wall time and CUDA
device-span time are recorded separately; launch counts and traffic deltas are
analytic models, clearly marked as estimates.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_OUTPUT = Path("results/compiler/routed-scale-epilogue/benchmark.json")

CASES = {
    "decode_hostbound": {
        "queries": 64,
        "route_width": 2,
        "sources": 256,
        "value_dim": 128,
        "dtype": "bfloat16",
        "regime": "host_bound",
    },
    "mid_balanced": {
        "queries": 1024,
        "route_width": 8,
        "sources": 512,
        "value_dim": 1024,
        "dtype": "float16",
        "regime": "gpu_bound",
    },
    "prefill_gpu_bound": {
        "queries": 4096,
        "route_width": 8,
        "sources": 1024,
        "value_dim": 2048,
        "dtype": "bfloat16",
        "regime": "gpu_bound",
    },
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata() -> dict[str, object]:
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": version("triton"),
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability()),
    }


def make_inputs(case: dict[str, object]):
    import torch

    dtype = getattr(torch, str(case["dtype"]))
    generator = torch.Generator(device="cuda").manual_seed(int(case["sources"]))
    indices = torch.randint(
        int(case["sources"]),
        (int(case["queries"]), int(case["route_width"])),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    weights = torch.randn(
        (int(case["queries"]), int(case["route_width"])),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    values = torch.randn(
        (int(case["sources"]), int(case["value_dim"])),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    scale = torch.randn(
        (int(case["queries"]),), device="cuda", dtype=dtype, generator=generator
    )
    return indices, weights, values, scale


def analytic_traffic(
    q: int, s: int, k: int, d: int, bytes_per_elt: int
) -> dict[str, int]:
    """Documented static byte bounds (upper bound: no L2 duplicate reuse)."""
    route_bytes = q * k * (8 + bytes_per_elt)
    gather_upper = q * k * d * bytes_per_elt
    base_io = q * d * bytes_per_elt  # write in plan A, read again in plan A
    return {
        "shared_bytes": route_bytes + gather_upper,
        "materialized_extra_bytes": 2 * base_io,  # write base, read base
        "scale_vector_bytes": q * bytes_per_elt,
    }


def timed(fn, warmup: int, samples: int) -> dict[str, float]:
    """Event device span per call plus wall time from one clock."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    device_spans: list[float] = []
    wall_start = time.perf_counter()
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        device_spans.append(start.elapsed_time(end))
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / samples
    return {
        "wall_ms_per_call": wall_ms,
        "median_device_ms": statistics.median(device_spans),
        "p95_device_ms": percentile(device_spans, 0.95),
        "min_device_ms": min(device_spans),
    }


def run_case(
    name: str, case: dict[str, object], warmup: int, samples: int, mode: str
) -> dict[str, object]:
    import torch

    from urm.compiler.anchors import routed_reduce_row_scale
    from urm.triton_kernels.routed_reduce import routed_reduce

    indices, weights, values, scale = make_inputs(case)
    q, k = indices.shape
    s, d = values.shape
    bytes_per_elt = weights.element_size()

    if mode == "backward":
        weights.requires_grad_(True)
        values.requires_grad_(True)
        scale.requires_grad_(True)
        grad_generator = torch.Generator(device="cuda").manual_seed(999)
        output_gradient = torch.randn(
            (q, d), device="cuda", dtype=values.dtype, generator=grad_generator
        )

    def plan_materialized() -> torch.Tensor:
        # Two-op plan: trusted v1 writes `base`; a second op applies row_scale.
        if mode == "backward":
            weights.grad = None
            values.grad = None
            base = routed_reduce(indices, weights, values)
            out = base * scale[:, None]
            out.backward(output_gradient)
            return out
        return routed_reduce(indices, weights, values) * scale[:, None]

    def plan_fused() -> torch.Tensor:
        if mode == "backward":
            weights.grad = None
            values.grad = None
            scale.grad = None
            out = routed_reduce_row_scale(indices, weights, values, scale)
            out.backward(output_gradient)
            return out
        return routed_reduce_row_scale(indices, weights, values, scale)

    measurements: dict[str, object] = {}
    plans = {
        "A_materialized_v1_plus_scale": plan_materialized,
        "B_fused_row_scale_epilogue": plan_fused,
    }
    for label, fn in plans.items():
        stats = timed(fn, warmup=warmup, samples=samples)
        stats["dispatch_share_wall_minus_device"] = max(
            0.0, stats["wall_ms_per_call"] - stats["median_device_ms"]
        )
        measurements[label] = stats

    traffic = analytic_traffic(q, s, k, d, bytes_per_elt)
    launches = {
        "A_forward_launches": 2,
        "B_forward_launches": 1,
        "A_backward_launches_approximate": 6,
        "B_backward_launches_exact": 3,
    }
    return {
        "case": {**case, "name": name},
        "mode": mode,
        "analytic_traffic_bytes_upper_bound_no_l2_reuse": traffic,
        "launch_counts_documented_model": launches,
        "measurements": measurements,
        "avoided_materialization_bytes_estimate": traffic["materialized_extra_bytes"],
    }


def correctness_by_dtype() -> dict[str, object]:
    import torch

    from urm.compiler.anchors import routed_reduce_row_scale

    report: dict[str, object] = {}
    for dtype_name, dtype in (
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ):
        indices, weights, values, scale = make_inputs(
            {
                "queries": 128,
                "route_width": 4,
                "sources": 64,
                "value_dim": 96,
                "dtype": "float32",
            }
        )
        weights = weights.to(dtype)
        values = values.to(dtype)
        scale = scale.to(dtype)
        fused = routed_reduce_row_scale(indices, weights, values, scale)
        reference = (
            torch.einsum("qk,qkd->qd", weights.float(), values.float()[indices.long()])
            * scale.float()[:, None]
        )
        gap = (fused.float() - reference).abs().max().item()
        report[dtype_name] = {
            "max_abs_error_vs_eager_reference": gap,
            "reference_max_abs": reference.abs().max().item(),
            # Same envelopes as the rewrite contract / GPU differential tests:
            # assert_close semantics, gap <= atol + rtol*|reference|.
            "atol": {"float32": 1e-5, "float16": 1.5e-2, "bfloat16": 2e-2}[dtype_name],
            "rtol": 2e-2,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cases = [
        run_case(name, case, args.warmup, args.samples, mode="forward")
        for name, case in CASES.items()
    ]
    training = [
        run_case(name, case, args.warmup, args.samples, mode="backward")
        for name, case in CASES.items()
    ]

    # Schedule C: full-row tile variant, measured and retained either way.
    schedule_c = evaluate_fullrow_variant()

    artifact = {
        "schema_version": 1,
        "environment": environment_metadata(),
        "semantics": (
            "output[q,d] = row_scale[q] * sum_k w[q,k]*V[idx[q,k],d]; "
            "plan A materializes base, plan B folds the scale into a typed "
            "routed-reduction epilogue"
        ),
        "forward_cases": cases,
        "backward_cases": training,
        "schedule_C_fullrow_variant": schedule_c,
        "correctness_by_dtype": correctness_by_dtype(),
        "estimates_are_analytical_not_measured_counters": True,
    }
    rendered = json.dumps(artifact, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({"written": str(args.output)}))


def evaluate_fullrow_variant() -> dict[str, object]:
    """Full-row tile fused variant vs heuristic launch (retained either way)."""

    import torch
    import triton

    from urm.compiler.anchors.routed_reduction_epilogue import (
        _forward_launch,
        _rrs_forward_kernel,
    )

    indices, weights, values, scale = make_inputs(CASES["mid_balanced"])
    q, k = indices.shape
    d = values.shape[1]
    output = torch.empty_like(values[:q])

    def launch(block_d: int, num_warps: int):
        grid = (q, triton.cdiv(d, block_d))
        _rrs_forward_kernel[grid](
            indices,
            weights,
            values,
            scale,
            output,
            ROUTE_WIDTH=k,
            VALUE_DIM=d,
            BLOCK_D=block_d,
            EVEN_D=d % block_d == 0,
            num_warps=num_warps,
        )

    heuristic_block, heuristic_warps = _forward_launch(d, q)
    fullrow_block = triton.next_power_of_2(d)
    configs = {
        "heuristic": (heuristic_block, heuristic_warps),
        "fullrow_16warps": (fullrow_block, 16),
    }
    timings: dict[str, float] = {}

    def measure(label: str, block_d: int, warps: int) -> float:
        stats = timed(lambda: launch(block_d, warps), warmup=10, samples=30)
        return stats["median_device_ms"]

    for label, (block_d, warps) in configs.items():
        timings[label] = measure(label, block_d, warps)
    winner = min(timings, key=timings.get)
    rejected = [label for label in timings if label != winner]
    return {
        "median_device_ms": timings,
        "accepted_schedule": f"heuristic ({winner})"
        if winner == "heuristic"
        else "fullrow_16warps",
        "rejected_schedules_retained": [
            {
                "schedule": label,
                "median_device_ms": timings[label],
                "reason": "slower than accepted schedule",
            }
            for label in rejected
        ],
    }


if __name__ == "__main__":
    main()
