"""Measure sustainable DRAM bandwidth and FP32 CUDA-core peak on this device.

These two numbers are the MFU/MBU denominators used by profile_roofline.py.
Both are measured empirically in the same software environment as the routed
reduction benchmarks, not taken from marketing sheets. The vendor datasheet
figure is recorded separately as a reference point only.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from datetime import UTC, datetime

import torch


def _events_bench(fn, warmup: int = 5, samples: int = 20, inner: int = 10) -> float:
    """Median CUDA-event milliseconds per fn() call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / inner)
    return statistics.median(times)


def measure_bandwidth(total_bytes_per_buffer: int = 2**30) -> dict[str, float]:
    """Sustainable HBM bandwidth from copy/fill/read kernels on 1 GiB buffers."""
    elements = total_bytes_per_buffer // 4
    src = torch.randn(elements, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    results: dict[str, float] = {}

    # Device-to-device copy: moves 2x buffer (read + write).
    ms = _events_bench(lambda: dst.copy_(src))
    results["copy_gbps"] = (2 * total_bytes_per_buffer) / (ms / 1e3) / 1e9

    # Fill: writes 1x buffer.
    ms = _events_bench(lambda: dst.fill_(0.5))
    results["fill_gbps"] = total_bytes_per_buffer / (ms / 1e3) / 1e9

    # Read-dominated: sum reduction reads 1x buffer, writes ~nothing.
    ms = _events_bench(lambda: torch.sum(src))
    results["read_gbps"] = total_bytes_per_buffer / (ms / 1e3) / 1e9

    results["sustainable_gbps"] = max(results.values())
    return results


def measure_fp32_cuda_core_peak(
    matrix: int = 8192, tf32_off: bool = True
) -> dict[str, float]:
    """Best SGEMM throughput with TF32 disabled => FP32 CUDA-core FMA path."""
    if tf32_off:
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a = torch.randn(matrix, matrix, device="cuda", dtype=torch.float32)
    b = torch.randn(matrix, matrix, device="cuda", dtype=torch.float32)
    flops_per_call = 2 * matrix**3

    # Warm up cuBLAS algorithm selection; excluded from steady state by design.
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    samples = []
    for _ in range(15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.matmul(a, b)
        end.record()
        end.synchronize()
        samples.append(flops_per_call / (start.elapsed_time(end) / 1e3) / 1e12)
    return {
        "fp32_cuda_core_tfps_measured": max(samples),
        "sgemm_matrix": matrix,
        "tf32_disabled": tf32_off,
    }


def measure_bf16_tensor_core_peak(matrix: int = 8192) -> dict[str, float]:
    """Best BF16 GEMM throughput; the denominator for MMA-based kernels
    (e.g., dense attention comparators), never used for routed reduction."""
    a = torch.randn(matrix, matrix, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(matrix, matrix, device="cuda", dtype=torch.bfloat16)
    flops_per_call = 2 * matrix**3
    for _ in range(3):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    samples = []
    for _ in range(15):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.matmul(a, b)
        end.record()
        end.synchronize()
        samples.append(flops_per_call / (start.elapsed_time(end) / 1e3) / 1e12)
    return {"bf16_tensor_core_tfps_measured": max(samples), "sgemm_matrix": matrix}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    properties = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "bandwidth": measure_bandwidth(),
        "fp32_cuda_core": measure_fp32_cuda_core_peak(),
        "bf16_tensor_core": measure_bf16_tensor_core_peak(),
        "vendor_reference": {
            "a10g_spec_fp32_tfps": 31.2,
            "a10g_spec_mem_gbps": 600,
            "source": "NVIDIA A10G datasheet; reference only, not used as denominator",
        },
        "notes": (
            "Bandwidth denominators are the best of measured copy/fill/read "
            "kernels on 1 GiB buffers in this exact software environment. The "
            "MFU denominator is measured TF32-disabled SGEMM (cuBLAS FP32 FMA "
            "path on CUDA cores); tensor-core peaks are never used because the "
            "routed-reduction kernels accumulate through FP32 CUDA-core ops."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
