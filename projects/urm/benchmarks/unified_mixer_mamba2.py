"""Parity and paired plan-overhead profile for pinned Mamba-2 SSD."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import mamba_ssm
import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.unified_mixer import (
    MixerBackend,
    MixerIntent,
    compile_mixer,
    named_mixer_recipe,
)

EXPECTED_MAMBA_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
SHAPE = {"batch": 1, "sequence": 256, "heads": 4, "head_dim": 16, "groups": 1, "state_dim": 64}
CHUNK_SIZE = 64


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not find Git root for Mamba source {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_MAMBA_REVISION:
        raise RuntimeError(
            f"loaded Mamba source must match {EXPECTED_MAMBA_REVISION}, got {revision}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("the pinned Mamba source checkout must be clean")
    return source, revision


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    b, t, h, p, g, n = (
        SHAPE["batch"],
        SHAPE["sequence"],
        SHAPE["heads"],
        SHAPE["head_dim"],
        SHAPE["groups"],
        SHAPE["state_dim"],
    )
    values = {
        "x": torch.randn((b, t, h, p), device="cuda", generator=generator) * 0.1,
        "dt": torch.rand((b, t, h), device="cuda", generator=generator) * 0.1 + 0.01,
        "A": -torch.rand((h,), device="cuda", generator=generator) - 0.1,
        "B": torch.randn((b, t, g, n), device="cuda", generator=generator) * 0.1,
        "C": torch.randn((b, t, g, n), device="cuda", generator=generator) * 0.1,
        "initial_states": torch.randn(
            (b, h, p, n), device="cuda", generator=generator
        )
        * 0.1,
    }
    return {name: value.requires_grad_() for name, value in values.items()}


def _direct(inputs: dict[str, torch.Tensor]):
    return mamba_chunk_scan_combined(
        inputs["x"],
        inputs["dt"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        chunk_size=CHUNK_SIZE,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=inputs["initial_states"],
        dt_softplus=False,
        return_final_states=True,
    )


def _compiled(plan, inputs: dict[str, torch.Tensor]):
    result = plan.execute(**inputs)
    return result.output, result.final_state


def _loss(output: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean() + state.float().square().mean()


def _clear(inputs: dict[str, torch.Tensor]) -> None:
    for tensor in inputs.values():
        tensor.grad = None


def _forward_backward(call, inputs: dict[str, torch.Tensor]):
    _clear(inputs)
    output, state = call(inputs)
    _loss(output, state).backward()
    return output, state, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs: dict[str, torch.Tensor], backward: bool):
    _clear(inputs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    output, state = call(inputs)
    if backward:
        _loss(output, state).backward()
    end_event.record()
    torch.cuda.synchronize()
    return (
        time.perf_counter() - start_wall,
        start_event.elapsed_time(end_event) / 1000,
    )


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _measure_pair(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup):
    cold_direct = _time_one(direct, direct_inputs, backward=False)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False)
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(direct, direct_inputs, backward)
            _time_one(compiled, compiled_inputs, backward)
    measurements = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall = [], []
        direct_device, compiled_device, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, device = _time_one(direct, direct_inputs, backward)
                    direct_wall.append(wall)
                    direct_device.append(device)
                else:
                    wall, device = _time_one(compiled, compiled_inputs, backward)
                    compiled_wall.append(wall)
                    compiled_device.append(device)
            i = len(overhead)
            overhead.append((compiled_wall[i] - direct_wall[i]) / direct_wall[i])
        median_overhead = statistics.median(overhead)
        measurements[mode] = {
            "direct_wall": _summary(direct_wall),
            "compiled_wall": _summary(compiled_wall),
            "direct_device": _summary(direct_device),
            "compiled_device": _summary(compiled_device),
            "paired_compiled_overhead_fraction": {
                "median": median_overhead,
                "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {
                    "limit_fraction": 0.10,
                    "pass": abs(median_overhead) <= 0.10,
                },
            },
            "pair_order": order,
        }
    return {
        "first_direct_forward_call_ms": {
            "wall": cold_direct[0] * 1000,
            "device": cold_direct[1] * 1000,
        },
        "first_compiled_forward_call_after_direct_autotune_ms": {
            "wall": cold_compiled[0] * 1000,
            "device": cold_compiled[1] * 1000,
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def run(pairs: int, warmup: int, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the Mamba-2 profile requires CUDA")
    source, revision = _source_identity()
    operands = _inputs(seed=64044)
    direct_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    compiled_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe("mamba2_ssm_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    direct = _direct
    compiled = lambda values: _compiled(plan, values)
    direct_result = _forward_backward(direct, direct_inputs)
    compiled_result = _forward_backward(compiled, compiled_inputs)
    output_error = (direct_result[0].float() - compiled_result[0].float()).abs().max().item()
    state_error = (direct_result[1].float() - compiled_result[1].float()).abs().max().item()
    gradient_errors = {
        name: (left.float() - right.float()).abs().max().item()
        for name, left, right in zip(
            operands, direct_result[2], compiled_result[2], strict=True
        )
    }
    for actual, expected in zip(direct_result[:2], compiled_result[:2], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    for actual, expected in zip(direct_result[2], compiled_result[2], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    performance = _measure_pair(
        direct, compiled, direct_inputs, compiled_inputs, pairs, warmup
    )
    root = next(parent for parent in source.parents if (parent / ".git").exists())
    source_paths = (
        "mamba_ssm/ops/triton/ssd_combined.py",
        "mamba_ssm/ops/triton/ssd_chunk_state.py",
        "mamba_ssm/ops/triton/ssd_state_passing.py",
        "mamba_ssm/ops/triton/ssd_chunk_scan.py",
        "mamba_ssm/ops/triton/ssd_bmm.py",
    )
    case = {
        "architecture_ids": ["arch-044"],
        "semantic_scope": "Mamba-2 SSD scan; input projections, short convolution, dt bias/softplus, skip and output gate excluded",
        "shape": {**SHAPE, "dtype": "float32", "chunk_size": CHUNK_SIZE},
        "upstream_callable": "mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined",
        "compiled_anchor": plan.anchor,
        "compiler_plan_build_ms": plan_build_ms,
        "parity": {
            "status": "pass",
            "output_max_abs_error": output_error,
            "final_state_max_abs_error": state_error,
            "input_gradient_max_abs_errors": gradient_errors,
            "tolerances": {
                "output_atol": 1e-7,
                "state_atol": 1e-7,
                "gradient_atol": 1e-7,
            },
        },
        "performance": performance,
    }
    config = {
        "recipe": "mamba2_ssm_core",
        "pairs": pairs,
        "warmup": warmup,
        "shape": SHAPE,
        "chunk_size": CHUNK_SIZE,
        "dtype": "float32",
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K2 library plan with pinned Mamba-2 SSD source",
        "upstream": {
            "repository": "https://github.com/state-spaces/mamba",
            "revision": revision,
            "module_version": getattr(mamba_ssm, "__version__", None),
            "loaded_module": str(source),
            "kernel_source_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in source_paths
            },
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/mamba-checkout:src python benchmarks/unified_mixer_mamba2.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one direct Mamba-2 source call or one compiled plan call, optionally followed by output/state backward",
            "sampling": "paired interleaved source/plan calls with synchronization and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of paired (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "both calls execute the same pinned Mamba-2 SSD operator; timings measure compiler-plan and runtime-validation overhead",
        },
        "cases": {"mamba2_ssm_core": case},
    }
    write_artifact(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/unified-mixer/mamba2-k2.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
