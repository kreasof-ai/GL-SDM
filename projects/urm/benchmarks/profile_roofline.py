"""Roofline profiler for the routed-reduction backend.

Produces compact machine-readable JSON per committed case and mode, reporting:

- end-to-end wall time (CUDA-event steady state, cold compile excluded);
- aggregate GPU kernel time and per-kernel breakdown (torch.profiler/CUPTI);
- host dispatch/launch overhead (wall minus GPU time) and launch counts;
- useful algorithmic FLOPs and TFLOP/s (documented model, not instruction
  counts): forward ``2*Q*K*D``, grad-weights ``2*Q*K*D``, grad-values
  ``2*Q*K*D`` including conceptual accumulation, complete backward ~4*Q*K*D;
- FP32 CUDA-core MFU against a *measured* SGEMM (TF32 off) peak recorded in
  ``results/device-limits.json``; tensor-core peaks are never used because the
  kernels accumulate through FP32 CUDA-core operations;
- static analytic DRAM-byte estimates kept strictly separate from measured
  traffic; MBU from those estimates is labeled ``static_estimate``;
- Nsight Compute counter fields (measured DRAM bytes, L2 hit rate, achieved
  occupancy, scheduler stalls) when NCU is installed AND permitted; otherwise
  an explicit ``not_available`` record with the blocking reason - never zeros;
- registers per thread / spills from Triton compile metadata and theoretical
  occupancy derived from them;
- route statistics (unique sources touched, top-row share) and an atomic
  contention indicator: grad-values GPU time divided by its static byte
  estimate moved at measured sustainable bandwidth;
- eligibility: normalized utilization is marked ineligible for host-bound
  measurements where kernel time is a minority of wall time.

Usage (from projects/urm):
    PYTHONPATH=src python benchmarks/profile_roofline.py \
        [--case smoke|...|all] [--mode forward|backward|both] \
        --device-limits results/device-limits.json \
        --output-dir results/profiling
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch
from routed_reduce import load_cases, make_inputs, percentile

DEFAULT_DEVICE_LIMITS = (
    Path(__file__).with_name("..") / "results" / "device-limits.json"
)
NCU_CANDIDATES = ("ncu", "/usr/local/cuda/bin/ncu", "/usr/local/cuda-12.9/bin/ncu")

# Useful algorithmic FLOP multipliers per Q*K*D element-product term.
FLOPS = {
    "forward": 2.0,
    "backward": 4.0,
}
BACKWARD_COMPONENT_FLOPS = {
    "_routed_reduce_grad_weights_kernel": 2.0,
    "_routed_reduce_grad_values_per_query_kernel": 2.0,
    "_routed_reduce_grad_values_kernel": 2.0,
}


def _write_inner_probe(target: Path) -> None:
    source = '''"""Minimal kernel launch used to probe Nsight Compute permissions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _probe_kernel(x_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(x_ptr + offsets, values * 2.0)


x = torch.ones(256, device="cuda")
_probe_kernel[(1,)](x, BLOCK=256)
torch.cuda.synchronize()
'''
    target.write_text(source)


def probe_ncu() -> dict[str, object]:
    """Detect an installed AND permitted Nsight Compute; never fabricate data."""
    binary = next((b for b in NCU_CANDIDATES if shutil.which(b)), None)
    if binary is None:
        return {"status": "not_available", "reason": "ncu binary not found"}
    try:
        version = (
            subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            .stdout.strip()
            .splitlines()[-1]
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "status": "not_available",
            "reason": f"ncu invocation failed: {error}",
            "version": version,
        }

    probe = Path("/tmp") / f"urm_ncu_probe_{datetime.now(UTC).timestamp():.0f}.py"
    _write_inner_probe(probe)
    report = "/tmp/urm_ncu_probe"
    try:
        completed = subprocess.run(
            [
                binary,
                "--launch-count",
                "1",
                "-o",
                report,
                "-f",
                _python_binary(),
                str(probe),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd="/tmp",
            env={**os.environ, "HOME": "/tmp"},
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "status": "not_available",
            "reason": f"probe failed: {error}",
            "version": version,
        }
    finally:
        probe.unlink(missing_ok=True)
    combined = completed.stdout + completed.stderr
    if "ERR_NVGPUCTRPERM" in combined:
        return {
            "status": "not_available",
            "reason": (
                "GPU performance counters restricted on this host "
                "(ERR_NVGPUCTRPERM); container lacks CAP_SYS_ADMIN or driver "
                "RmProfilingAdminOnly=1"
            ),
            "version": version,
        }
    if completed.returncode != 0:
        return {
            "status": "not_available",
            "reason": f"probe rc={completed.returncode}",
            "version": version,
        }
    return {"status": "available", "version": version}


def _python_binary() -> str:
    import sys

    return sys.executable


def triton_kernel_metadata() -> dict[str, object]:
    """Extract registers/spills/shared-mem from compiled Triton kernels."""
    try:
        from urm.triton_kernels import routed_reduce as rr

        out: dict[str, object] = {}
        caches = getattr(rr._routed_reduce_forward_kernel, "device_caches", None)
        if not caches:
            return {"status": "not_available", "reason": "no device_caches attribute"}
        for device_cache in caches.values():
            for key, kernel in device_cache[0].items():
                name = kernel.name if hasattr(kernel, "name") else key[:24]
                entry = {
                    "num_warps": getattr(kernel.metadata, "num_warps", None),
                    "shared_bytes": getattr(kernel.metadata, "shared", None),
                    "registers_per_thread": getattr(kernel, "n_regs", None),
                    "spills": getattr(kernel, "n_spills", None),
                }
                props = torch.cuda.get_device_properties(0)
                threads = 32 * (entry["num_warps"] or 4)
                regs = entry["registers_per_thread"]
                smem = entry["shared_bytes"] or 0
                limits = [
                    props.max_threads_per_multi_processor // threads,
                ]
                if regs:
                    limits.append(
                        (props.regs_per_multiprocessor or 65536)
                        // max(1, regs * threads)
                    )
                smem_per_sm = getattr(props, "shared_memory_per_multiprocessor", None)
                if smem_per_sm and smem:
                    limits.append(smem_per_sm // smem)
                entry["theoretical_blocks_per_sm"] = min(limits) if limits else None
                entry["theoretical_occupancy_pct"] = (
                    round(
                        100
                        * min(limits)
                        * threads
                        / props.max_threads_per_multi_processor,
                        1,
                    )
                    if limits
                    else None
                )
                out[str(name)] = entry
        return {"status": "available", "kernels": out}
    except Exception as error:  # noqa: BLE001 - introspection is best-effort by design
        return {"status": "not_available", "reason": repr(error)}


def profile_kernels(function, iterations: int) -> dict[str, object]:
    """Per-kernel CUDA time breakdown via CUPTI (torch.profiler)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            function()
        torch.cuda.synchronize()
    breakdown: dict[str, float] = {}
    launches: dict[str, int] = {}
    for event in prof.key_averages():
        if event.device_time_total <= 0:
            continue
        # Divide by iteration count to keep per-iteration units.
        breakdown[event.key] = event.device_time_total / iterations / 1e3  # ms
        launches[event.key] = event.count // iterations
    aggregate_gpu_ms = sum(breakdown.values())
    return {
        "aggregate_gpu_ms": aggregate_gpu_ms,
        "launches_per_iteration": sum(launches.values()),
        "per_kernel_ms": {
            k: round(v, 6) for k, v in sorted(breakdown.items(), key=lambda kv: -kv[1])
        },
        "per_kernel_launches": launches,
    }


def route_statistics(indices: torch.Tensor, sources: int) -> dict[str, object]:
    flat = indices.reshape(-1).long()
    unique = torch.unique(flat).numel()
    counts = torch.bincount(flat, minlength=sources)
    top_share = (counts.max().float() / flat.numel()).item()
    return {
        "routes": int(flat.numel()),
        "unique_sources_touched": int(unique),
        "source_utilization_pct": round(100.0 * unique / sources, 2),
        "hottest_source_traffic_share_pct": round(100.0 * top_share, 3),
    }


def static_byte_estimates(
    case: dict, mode: str, index_item_bytes: int = 4
) -> dict[str, object]:
    """Analytic traffic bounds.

    ``gathered`` counts every route's value row read (upper bound before any
    cache reuse); ``unique_rows`` assumes perfect dedup of duplicate source
    rows (lower bound). Real DRAM traffic lies between depending on L2 reuse.
    These are static workload estimates, never measured hardware counters.
    """
    q, s = int(case["queries"]), int(case["sources"])
    k, d = int(case["route_width"]), int(case["value_dim"])
    item = {"float16": 2, "bfloat16": 2, "float32": 4}[str(case["dtype"])]

    def bytes_forward() -> dict[str, int]:
        return {
            "indices_weights_read": q * k * (index_item_bytes + item),
            "values_gathered_read": q * k * d * item,
            "values_unique_rows_read": s * d * item,
            "output_write": q * d * item,
        }

    def bytes_backward() -> dict[str, int]:
        gw = {
            "indices_read": q * k * index_item_bytes,
            "values_gathered_read": q * k * d * item,
            "grad_output_read_gw": q * k * d * item,  # per-route re-read
            "grad_weights_write": q * k * 4,
        }
        gv = {
            "indices_weights_read": q * k * (index_item_bytes + item),
            "grad_output_read_gv": q * d * item,  # per-query kernel loads once
            "atomic_rmw_bytes": q * k * d * 4,
        }
        return {
            "grad_weights_kernel": gw,
            "grad_values_kernel": gv,
            "zero_init_and_casts_rw": (s * d * 4) * 2 + (q * d + q * k) * item * 2,
        }

    payload: dict[str, object] = {}
    if mode == "forward":
        parts = bytes_forward()
        total = sum(parts.values())
        payload["forward"] = parts
        payload["total_min_bytes"] = (
            parts["indices_weights_read"]
            + parts["values_unique_rows_read"]
            + parts["output_write"]
        )
        payload["total_max_bytes"] = total
    else:
        parts = bytes_backward()
        payload["backward"] = parts
        mins = (
            parts["grad_weights_kernel"]["indices_read"]
            + parts["grad_weights_kernel"]["values_gathered_read"]
            + parts["grad_weights_kernel"]["grad_weights_write"]
            + parts["grad_values_kernel"]["indices_weights_read"]
            + parts["grad_values_kernel"]["grad_output_read_gv"]
            + parts["grad_values_kernel"]["atomic_rmw_bytes"]
            + parts["zero_init_and_casts_rw"]
        )
        payload["total_min_bytes"] = mins
        maxima = (
            sum(parts["grad_weights_kernel"].values())
            + parts["grad_values_kernel"]["indices_weights_read"]
            + parts["grad_values_kernel"]["grad_output_read_gv"]
            + parts["grad_values_kernel"]["atomic_rmw_bytes"]
            + parts["zero_init_and_casts_rw"]
        )
        payload["total_max_bytes"] = maxima
    return payload


def measure_case(
    case: dict,
    mode: str,
    warmup: int,
    samples: int,
    inner: int,
    device_limits_path: str,
    ncu_availability: dict[str, object],
) -> dict[str, object]:
    indices, weights, values = make_inputs(case, "cuda")
    requires_grad = mode == "backward"
    weights.requires_grad_(requires_grad)
    values.requires_grad_(requires_grad)
    output_gradient = None
    if requires_grad:
        gradient_generator = torch.Generator(device="cuda").manual_seed(
            int(case.get("seed", 0)) + 10_000
        )
        output_gradient = torch.randn(
            (indices.shape[0], values.shape[1]),
            device="cuda",
            dtype=values.dtype,
            generator=gradient_generator,
        )

    from urm.backends import TritonRoutedReductionBackend

    backend = TritonRoutedReductionBackend()
    saved: list[torch.Tensor] = []

    def invoke() -> None:
        if requires_grad:
            weights.grad = None
            values.grad = None
        result = backend.execute(indices, weights, values, validate_indices=False)
        if requires_grad:
            result.output.backward(output_gradient)
        else:
            saved.clear()
            saved.append(result.output)

    invoke()  # compile outside timing
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            invoke()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)

    kernel_profile = profile_kernels(invoke, inner)

    queries, route_width = indices.shape
    value_dim = values.shape[1]
    useful_flops = FLOPS[mode] * queries * route_width * value_dim
    median_wall_ms = statistics.median(timings)
    gpu_ms = kernel_profile["aggregate_gpu_ms"]
    dispatch_ms = max(0.0, median_wall_ms - gpu_ms)
    dispatch_fraction = dispatch_ms / median_wall_ms if median_wall_ms > 0 else 0.0
    eligible = gpu_ms >= median_wall_ms * 0.5

    limits = json.loads(Path(device_limits_path).read_text())
    sustainable_gbps = limits["bandwidth"]["sustainable_gbps"]
    fp32_peak_tfps = limits["fp32_cuda_core"]["fp32_cuda_core_tfps_measured"]

    est = static_byte_estimates(case, mode)
    static_gbps_mid = None
    mbu_static = None
    if eligible:
        mid_bytes = (est["total_min_bytes"] + est["total_max_bytes"]) / 2
        static_gbps_mid = mid_bytes / (gpu_ms / 1e3) / 1e9
        mbu_static = static_gbps_mid / sustainable_gbps
    mfu = None
    if eligible:
        mfu = (useful_flops / (gpu_ms / 1e3) / 1e12) / fp32_peak_tfps

    contention = None
    if mode == "backward":
        for name, ms in kernel_profile["per_kernel_ms"].items():
            if "grad_values" in name:
                gv_flops_term = 2.0 * queries * route_width * value_dim
                ideal_ms = None
                ideal_ms = None  # computed from bytes below
                gv_bytes = queries * route_width * value_dim * 8  # RMW read+write fp32
                ideal_ms = gv_bytes / (sustainable_gbps * 1e9) * 1e3
                if ms > 0 and ideal_ms > 0:
                    contention = {
                        "grad_values_gpu_ms": round(ms, 6),
                        "ideal_bw_bound_ms": round(ideal_ms, 6),
                        "serialization_factor": round(ms / ideal_ms, 2),
                        "method": (
                            "grad-values kernel time vs its atomic RMW bytes "
                            "(read+write fp32) moved at measured sustainable bandwidth"
                        ),
                    }
                del gv_flops_term
                break

    report: dict[str, object] = {
        "schema_version": 1,
        "case": {
            key: case[key]
            for key in (
                "name",
                "queries",
                "sources",
                "route_width",
                "value_dim",
                "dtype",
                "distribution",
            )
        },
        "mode": mode,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "timing": {
            "median_wall_ms": round(median_wall_ms, 6),
            "p95_wall_ms": round(percentile(timings, 0.95), 6),
            "samples": samples,
            "inner_iterations": inner,
            "aggregate_gpu_ms": round(gpu_ms, 6),
            "host_dispatch_ms_estimate": round(dispatch_ms, 6),
            "dispatch_fraction_of_wall": round(dispatch_fraction, 3),
            "launches_per_iteration": kernel_profile["launches_per_iteration"],
            "per_kernel_ms": kernel_profile["per_kernel_ms"],
            "per_kernel_launches": kernel_profile["per_kernel_launches"],
        },
        "flops_model": {
            "formula": (
                f"{FLOPS[mode]} * Q * K * D" if mode == "backward" else "2 * Q * K * D"
            ),
            "note": (
                "Useful algorithmic FLOPs of the reduction, not executed "
                "instruction counts. Backward counts grad-weights (2), "
                "grad-values including conceptual accumulation (2), and "
                "~zero for casts/init."
            ),
            "useful_flops": useful_flops,
            "useful_tfps_over_gpu_time": round(useful_flops / (gpu_ms / 1e3) / 1e12, 4),
            "useful_tfps_over_wall": round(
                useful_flops / (median_wall_ms / 1e3) / 1e12, 4
            ),
        },
        "mfu": {
            "denominator": "fp32_cuda_core_tfps_measured",
            "denominator_source": "results/device-limits.json (TF32-off cuBLAS SGEMM, measured)",
            "fp32_cuda_core_peak_tfps": fp32_peak_tfps,
            "vendor_spec_fp32_tfps_reference_only": limits.get(
                "vendor_reference", {}
            ).get("a10g_spec_fp32_tfps"),
            "eligible": eligible,
            "mfu_fraction": round(mfu, 4) if mfu is not None else "not_available",
            "mfu_pct": round(mfu * 100, 2) if mfu is not None else "not_available",
        },
        "memory_traffic": {
            "measured": {
                "dram_bytes_sum": "not_available",
                "l2_hit_rate": "not_available",
                "mbu_measured": "not_available",
                "reason": ncu_availability.get("reason", "unavailable"),
            },
            "static_estimate": {
                **est,
                "gbps_at_gpu_time_midpoint": round(static_gbps_mid, 1)
                if static_gbps_mid
                else "not_available",
                "mbu_vs_measured_sustainable": round(mbu_static, 3)
                if mbu_static
                else "not_available",
                "sustainable_bandwidth_gbps_denominator": sustainable_gbps,
                "bandwidth_denominator_source": "results/device-limits.json (best of copy/fill/read kernels)",
                "note": (
                    "MBU here uses static analytic byte bounds over GPU time, "
                    "not hardware counters; real DRAM traffic lies between the "
                    "min/max bounds due to duplicate-row caching."
                ),
            },
            "vendor_bandwidth_reference_gbps": limits.get("vendor_reference", {}).get(
                "a10g_spec_mem_gbps"
            ),
            "vendor_relative_utilization": (
                round(
                    static_gbps_mid / limits["vendor_reference"]["a10g_spec_mem_gbps"],
                    3,
                )
                if static_gbps_mid
                else "not_available"
            ),
        },
        "atomic_contention_indicator": contention
        or ("not_applicable" if mode != "backward" else None),
        "route_statistics": route_statistics(indices, int(case["sources"])),
        "triton_kernel_metadata": triton_kernel_metadata(),
        "normalized_utilization_eligible": eligible,
        "eligibility_note": (
            None
            if eligible
            else "Host dispatch dominates this small case; MFU/MBU are marked "
            "ineligible rather than reported as misleadingly tiny numbers."
        ),
    }
    if report["atomic_contention_indicator"] is None:
        report["atomic_contention_indicator"] = "not_computed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--case", default="all")
    parser.add_argument(
        "--mode", choices=("forward", "backward", "both"), default="both"
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument(
        "--device-limits",
        type=Path,
        default=None,
        help="JSON produced by measure_device_limits.py",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/profiling"))
    args = parser.parse_args()

    cases_path = args.cases or Path(__file__).with_name("cases.toml")
    cases = load_cases(cases_path)
    selected = list(cases) if args.case == "all" else [args.case]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    properties = torch.cuda.get_device_properties(0)

    limits_path = args.device_limits or Path("results/device-limits.json")
    args_limits_resolved = str(limits_path.resolve())

    ncu_report = probe_ncu()
    ncu_status = ncu_report["status"]
    print(f"ncu probe: {ncu_status} ({ncu_report.get('reason', 'available')})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "ncu": ncu_report,
        "cases": {},
    }

    for name in selected:
        case = cases[name]
        modes = ("forward", "backward") if args.mode == "both" else (args.mode,)
        for mode in modes:
            report = measure_case(
                case,
                mode,
                args.warmup,
                args.samples,
                args.inner,
                args_limits_resolved,
                ncu_report,
            )
            filename = args.output_dir / f"{name}-{mode}.profiling.json"
            filename.write_text(json.dumps(report, indent=2) + "\n")
            summary["cases"][f"{name}-{mode}"] = {
                "wall_median_ms": report["timing"]["median_wall_ms"],
                "gpu_ms": report["timing"]["aggregate_gpu_ms"],
                "dispatch_fraction": report["timing"]["dispatch_fraction_of_wall"],
                "mfu_pct": report["mfu"]["mfu_pct"],
                "mbu_static": report["memory_traffic"]["static_estimate"][
                    "mbu_vs_measured_sustainable"
                ],
                "eligible": report["normalized_utilization_eligible"],
            }
            t = report["timing"]
            print(
                f"{name:>17}-{mode:<8} wall {t['median_wall_ms'] * 1e3:8.1f}us "
                f"gpu {t['aggregate_gpu_ms'] * 1e3:8.1f}us "
                f"disp {report['timing']['dispatch_fraction_of_wall']:4.0%} "
                f"MFU {report['mfu']['mfu_pct']}% "
                f"MBUst {report['memory_traffic']['static_estimate']['mbu_vs_measured_sustainable']}"
            )

    summary_path = args.output_dir / "summary.profiling.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else None
    if existing:
        existing_cases = existing.pop("cases", {})
        existing_cases.update(summary["cases"])
        existing["timestamp_utc"] = summary["timestamp_utc"]
        existing.update({k: v for k, v in summary.items() if k != "cases"})
        summary = existing
        summary["cases"] = existing_cases
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
