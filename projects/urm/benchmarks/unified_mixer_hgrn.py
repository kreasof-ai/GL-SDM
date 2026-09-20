"""Parity and paired HGRN profiles against the pinned FLA recurrent operator."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import fla
import torch
from fla.ops.hgrn import fused_recurrent_hgrn

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.unified_mixer import (
    MixerBackend,
    MixerIntent,
    compile_mixer,
    named_mixer_recipe,
)

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
BATCH = 1
SEQUENCE = 1024
CHANNELS = 1024


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(fla)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not find Git root for loaded FLA source {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_FLA_REVISION:
        raise RuntimeError(
            f"loaded FLA source must match {EXPECTED_FLA_REVISION}, got {revision} at {source}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("the pinned FLA source checkout must be clean")
    return source, revision


def _source_hashes(source: Path) -> dict[str, str]:
    root = next(parent for parent in source.parents if (parent / ".git").exists())
    paths = (
        "fla/ops/hgrn/fused_recurrent.py",
        "fla/ops/hgrn/chunk.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randn((BATCH, SEQUENCE, CHANNELS), device="cuda", generator=generator)
        * 0.2
    )
    log_decay = (
        -torch.rand((BATCH, SEQUENCE, CHANNELS), device="cuda", generator=generator)
        * 0.05
    )
    initial_state = (
        torch.randn((BATCH, CHANNELS), device="cuda", generator=generator) * 0.1
    )
    return {
        "x": x.requires_grad_(),
        "log_decay": log_decay.requires_grad_(),
        "initial_state": initial_state.requires_grad_(),
    }


def _direct(inputs: dict[str, torch.Tensor]):
    return fused_recurrent_hgrn(
        inputs["x"],
        inputs["log_decay"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
    )


def _compiled(plan, inputs: dict[str, torch.Tensor]):
    result = plan.execute(**inputs)
    return result.output, result.final_state.squeeze(-1)


def _loss(output: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean() + state.float().square().mean()


def _forward_backward(call, inputs: dict[str, torch.Tensor]):
    for tensor in inputs.values():
        tensor.grad = None
    output, state = call(inputs)
    _loss(output, state).backward()
    return output, state, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs: dict[str, torch.Tensor], *, backward: bool):
    for tensor in inputs.values():
        tensor.grad = None
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
    return time.perf_counter() - start_wall, start_event.elapsed_time(end_event) / 1000


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
            _time_one(direct, direct_inputs, backward=backward)
            _time_one(compiled, compiled_inputs, backward=backward)
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
                    wall, device = _time_one(direct, direct_inputs, backward=backward)
                    direct_wall.append(wall)
                    direct_device.append(device)
                else:
                    wall, device = _time_one(
                        compiled, compiled_inputs, backward=backward
                    )
                    compiled_wall.append(wall)
                    compiled_device.append(device)
            pair_index = len(overhead)
            overhead.append(
                (compiled_wall[pair_index] - direct_wall[pair_index])
                / direct_wall[pair_index]
            )
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
                "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
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
        raise RuntimeError("the HGRN unified mixer profile requires CUDA")
    source, revision = _source_identity()
    operands = _inputs(seed=61227)
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
        named_mixer_recipe("hgrn_ssm_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    direct = _direct
    compiled = lambda values: _compiled(plan, values)
    direct_result = _forward_backward(direct, direct_inputs)
    compiled_result = _forward_backward(compiled, compiled_inputs)
    output_error = (direct_result[0] - compiled_result[0]).abs().max().item()
    state_error = (direct_result[1] - compiled_result[1]).abs().max().item()
    gradient_errors = {
        name: (left - right).abs().max().item()
        for name, left, right in zip(
            operands, direct_result[2], compiled_result[2], strict=True
        )
    }
    value_atol = 2e-6
    gradient_atol = 2e-6
    torch.testing.assert_close(
        compiled_result[0], direct_result[0], atol=value_atol, rtol=2e-5
    )
    torch.testing.assert_close(
        compiled_result[1], direct_result[1], atol=value_atol, rtol=2e-5
    )
    for actual, expected in zip(compiled_result[2], direct_result[2], strict=True):
        torch.testing.assert_close(actual, expected, atol=gradient_atol, rtol=2e-5)
    performance = _measure_pair(
        direct, compiled, direct_inputs, compiled_inputs, pairs, warmup
    )
    config = {
        "recipe": "hgrn_ssm_core",
        "pairs": pairs,
        "warmup": warmup,
        "shape": [BATCH, SEQUENCE, CHANNELS],
        "dtype": "float32",
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K2 HGRN library plan with direct pinned FLA operator",
        "upstream": {
            "repository": "https://github.com/fla-org/flash-linear-attention",
            "revision": revision,
            "module_version": getattr(fla, "__version__", None),
            "loaded_module": str(source),
            "kernel_source_sha256": _source_hashes(source),
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/fla-checkout:src python benchmarks/unified_mixer_hgrn.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one source call or one unified library plan call, optionally followed by output and final-state backward",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "both paths invoke the same pinned FLA fused recurrent HGRN operator; measurements capture plan dispatch overhead",
        },
        "cases": {
            "hgrn_ssm_core": {
                "architecture_ids": ["arch-023"],
                "semantic_scope": "vector-state gated recurrent core; projections, input gate production, normalization and output projection excluded",
                "shape": {
                    "batch": BATCH,
                    "sequence": SEQUENCE,
                    "channels": CHANNELS,
                    "dtype": "float32",
                },
                "upstream_callable": "fla.ops.hgrn.fused_recurrent_hgrn",
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": output_error,
                    "final_state_max_abs_error": state_error,
                    "input_gradient_max_abs_errors": gradient_errors,
                    "tolerances": {
                        "output_atol": value_atol,
                        "state_atol": value_atol,
                        "gradient_atol": gradient_atol,
                        "relative_tolerance": 2e-5,
                    },
                },
                "performance": performance,
            }
        },
    }
    write_artifact(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/unified-mixer/hgrn-k2.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
