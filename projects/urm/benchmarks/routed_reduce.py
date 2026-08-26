"""Correctness and steady-state benchmark for routed reduction backends."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from urm.backends import TorchRoutedReductionBackend, TritonRoutedReductionBackend

DEFAULT_CASES = Path(__file__).with_name("cases.toml")


def load_cases(path: Path) -> dict[str, dict[str, object]]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return {case["name"]: case for case in data["routed_reduction"]["case"]}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_inputs(case: dict[str, object], device: str) -> tuple[object, object, object]:
    import torch

    dtype = getattr(torch, str(case["dtype"]))
    queries = int(case["queries"])
    sources = int(case["sources"])
    route_width = int(case["route_width"])
    value_dim = int(case["value_dim"])
    generator = torch.Generator(device=device).manual_seed(int(case.get("seed", 0)))
    distribution = case.get("distribution", "uniform")
    if distribution == "uniform":
        indices = torch.randint(
            sources,
            (queries, route_width),
            device=device,
            dtype=torch.int32,
            generator=generator,
        )
    elif distribution == "skewed":
        samples = torch.rand((queries, route_width), device=device, generator=generator)
        indices = torch.floor(samples.pow(4) * sources).to(torch.int32)
    elif distribution == "recurrent_reuse":
        hot_sources = max(1, min(sources, route_width * 2))
        indices = torch.randint(
            hot_sources,
            (queries, route_width),
            device=device,
            dtype=torch.int32,
            generator=generator,
        )
    else:
        raise ValueError(f"unknown route distribution: {distribution}")
    weights = torch.randn(
        (queries, route_width), device=device, dtype=dtype, generator=generator
    )
    values = torch.randn(
        (sources, value_dim), device=device, dtype=dtype, generator=generator
    )
    return indices.contiguous(), weights.contiguous(), values.contiguous()


def environment_metadata() -> dict[str, object]:
    import torch

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": package_version("triton"),
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": properties.total_memory,
    }


def benchmark_callable(
    function: Callable[[], object], warmup: int, samples: int, inner: int
) -> tuple[float, list[float], int]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    compilation_started = time.perf_counter()
    function()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - compilation_started) * 1000.0
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)
    return cold_ms, timings, torch.cuda.max_memory_allocated()


def run_backend(
    backend_name: str,
    indices: object,
    weights: object,
    values: object,
    warmup: int,
    samples: int,
    inner: int,
    mode: str,
    output_gradient: object | None,
) -> tuple[dict[str, object], dict[str, object]]:
    backend = {
        "torch": TorchRoutedReductionBackend(),
        "triton": TritonRoutedReductionBackend(),
    }[backend_name]

    latest_result: object | None = None

    def invoke() -> object:
        nonlocal latest_result
        if mode == "backward":
            weights.grad = None
            values.grad = None
        latest_result = backend.execute(
            indices, weights, values, validate_indices=False
        )
        if mode == "backward":
            latest_result.output.backward(output_gradient)
        return latest_result.output

    cold_ms, timings, peak_memory = benchmark_callable(
        invoke, warmup=warmup, samples=samples, inner=inner
    )
    assert latest_result is not None
    output = latest_result.output
    queries, route_width = indices.shape
    sources, value_dim = values.shape
    del sources
    useful_flops = 2 * queries * route_width * value_dim
    median_ms = statistics.median(timings)
    metrics: dict[str, object] = {
        "backend": backend.name,
        "mode": mode,
        "cold_compile_or_first_call_ms": cold_ms,
        "median_ms": median_ms,
        "p95_ms": percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "samples": samples,
        "inner_iterations": inner,
        "useful_tflops": useful_flops / (median_ms * 1e9),
        "peak_allocated_bytes": peak_memory,
        "kernel_metadata": latest_result.metadata,
    }
    artifacts = {"output": output.detach().clone()}
    if mode == "backward":
        artifacts["grad_weights"] = weights.grad.detach().clone()
        artifacts["grad_values"] = values.grad.detach().clone()
    return metrics, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--case", default="smoke")
    parser.add_argument(
        "--backend", choices=("torch", "triton", "both"), default="both"
    )
    parser.add_argument("--mode", choices=("forward", "backward"), default="forward")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--inner", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.list_cases:
        print("\n".join(cases))
        return
    try:
        case = cases[args.case]
    except KeyError as error:
        raise SystemExit(f"unknown case {args.case!r}; use --list-cases") from error

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the GPU benchmark")
    indices, weights, values = make_inputs(case, "cuda")
    output_gradient = None
    if args.mode == "backward":
        weights.requires_grad_(True)
        values.requires_grad_(True)
        gradient_generator = torch.Generator(device="cuda").manual_seed(
            int(case.get("seed", 0)) + 10_000
        )
        output_gradient = torch.randn(
            (indices.shape[0], values.shape[1]),
            device="cuda",
            dtype=values.dtype,
            generator=gradient_generator,
        )
    selected = ("torch", "triton") if args.backend == "both" else (args.backend,)
    measurements: list[dict[str, object]] = []
    artifacts: dict[str, dict[str, object]] = {}
    for backend_name in selected:
        measurement, artifacts[backend_name] = run_backend(
            backend_name,
            indices,
            weights,
            values,
            warmup=args.warmup,
            samples=args.samples,
            inner=args.inner,
            mode=args.mode,
            output_gradient=output_gradient,
        )
        measurements.append(measurement)

    correctness: dict[str, object] | None = None
    if "torch" in artifacts and "triton" in artifacts:
        correctness = {}
        for artifact_name in artifacts["torch"]:
            difference = (
                artifacts["torch"][artifact_name].float()
                - artifacts["triton"][artifact_name].float()
            ).abs()
            correctness[artifact_name] = {
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
            }

    result = {
        "schema_version": 1,
        "case": case,
        "environment": environment_metadata(),
        "correctness": correctness,
        "measurements": measurements,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
