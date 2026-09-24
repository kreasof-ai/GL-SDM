"""Parity and paired K1 profiles against the pinned FLA AttnRes operator."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import fla
import torch
from fla.ops.attnres import fused_attnres

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from benchmarks.recipe_catalog import load_kernel_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
DEPTH, BATCH, SEQUENCE, WIDTH = 8, 1, 256, 128
DTYPE = torch.bfloat16


def _source_identity() -> tuple[Path, str]:
    source = Path(fla.__file__).resolve()
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
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ):
        raise RuntimeError("the pinned FLA source checkout must be clean")
    return source, revision


def _inputs(seed: int) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    residuals = tuple(
        torch.randn(
            BATCH,
            SEQUENCE,
            WIDTH,
            device="cuda",
            dtype=DTYPE,
            generator=generator,
        ).requires_grad_()
        for _ in range(DEPTH)
    )
    return {
        "query": torch.randn(WIDTH, device="cuda", dtype=DTYPE, generator=generator).requires_grad_(),
        "rms_weight": torch.randn(WIDTH, device="cuda", dtype=DTYPE, generator=generator).requires_grad_(),
        "residuals": residuals,
    }


def _clone_inputs(inputs: dict[str, object]) -> dict[str, object]:
    return {
        name: tuple(item.detach().clone().requires_grad_() for item in value)
        if name == "residuals"
        else value.detach().clone().requires_grad_()
        for name, value in inputs.items()
    }


def _direct(inputs: dict[str, object]) -> torch.Tensor:
    return fused_attnres(
        query=inputs["query"],
        residuals=inputs["residuals"],
        rms_weight=inputs["rms_weight"],
        rms_eps=1e-6,
        scale=1.0,
        checkpoint_level=1,
    )


def _prepare(inputs: dict[str, object]) -> dict[str, torch.Tensor]:
    residuals = inputs["residuals"]
    stacked = torch.stack(residuals, dim=2).float()  # [B,T,L,D]
    key = stacked * torch.rsqrt(stacked.square().mean(dim=-1, keepdim=True) + 1e-6)
    query = inputs["query"].float() * inputs["rms_weight"].float()
    flat_count = BATCH * SEQUENCE
    return {
        "query": query.view(1, 1, 1, WIDTH).expand(flat_count, -1, -1, -1),
        "key": key.reshape(flat_count, DEPTH, 1, WIDTH),
        "value": stacked.reshape(flat_count, DEPTH, 1, WIDTH),
    }


def _compiled(plan, inputs: dict[str, object]) -> torch.Tensor:
    return plan.execute(**inputs).output


def _equation(plan, inputs: dict[str, object]) -> torch.Tensor:
    output = plan.execute(**_prepare(inputs)).output
    return output.reshape(BATCH, SEQUENCE, WIDTH).to(inputs["query"].dtype)


def _grads(call, inputs: dict[str, object]):
    for value in inputs["residuals"]:
        value.grad = None
    inputs["query"].grad = None
    inputs["rms_weight"].grad = None
    output = call(inputs)
    output.float().square().mean().backward()
    gradients = {
        "query": inputs["query"].grad,
        "rms_weight": inputs["rms_weight"].grad,
        "residuals": tuple(value.grad for value in inputs["residuals"]),
    }
    return output, gradients


def _time_one(call, inputs, backward: bool) -> tuple[float, float]:
    for value in inputs["residuals"]:
        value.grad = None
    inputs["query"].grad = None
    inputs["rms_weight"].grad = None
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    output = call(inputs)
    if backward:
        output.float().square().mean().backward()
    stop.record()
    torch.cuda.synchronize()
    return time.perf_counter() - wall_start, start.elapsed_time(stop) / 1000


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _measure(direct, compiled, inputs, pairs: int, warmup: int):
    _time_one(direct, _clone_inputs(inputs), False)
    _time_one(compiled, _clone_inputs(inputs), False)
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(direct, _clone_inputs(inputs), backward)
            _time_one(compiled, _clone_inputs(inputs), backward)
    result = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall, paired = [], [], []
        for index in range(pairs):
            for name in (("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")):
                wall, _ = _time_one(
                    direct if name == "direct" else compiled,
                    _clone_inputs(inputs),
                    backward,
                )
                (direct_wall if name == "direct" else compiled_wall).append(wall)
            paired.append((compiled_wall[-1] - direct_wall[-1]) / direct_wall[-1])
        median = statistics.median(paired)
        result[mode] = {
            "direct_wall": _summary(direct_wall),
            "compiled_wall": _summary(compiled_wall),
            "paired_compiled_overhead_fraction": {
                "median": median,
                "p95": quantile(paired, 0.95),
                "raw_samples": paired,
                "gate": {"limit_fraction": 0.10, "pass": median <= 0.10},
            },
        }
    return result


def run(pairs: int, warmup: int, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the AttnRes profile requires CUDA")
    source, revision = _source_identity()
    operands = _inputs(7199)
    plan_started = time.perf_counter()
    plan = compile_mixer(
        load_kernel_recipe("attnres_depth_core"),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    equation_plan = compile_mixer(
        load_kernel_recipe("attnres_depth_core"),
        backend=MixerBackend.REFERENCE,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    direct_inputs, compiled_inputs = _clone_inputs(operands), _clone_inputs(operands)
    direct_result, direct_grads = _grads(_direct, direct_inputs)
    compiled_result, compiled_grads = _grads(
        lambda values: _compiled(plan, values), compiled_inputs
    )
    output_error = (direct_result.float() - compiled_result.float()).abs().max().item()
    gradient_errors = {}
    for name in ("query", "rms_weight"):
        gradient_errors[name] = (
            direct_grads[name].float() - compiled_grads[name].float()
        ).abs().max().item()
    gradient_errors["residuals"] = max(
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(
            direct_grads["residuals"], compiled_grads["residuals"], strict=True
        )
    )
    torch.testing.assert_close(compiled_result.float(), direct_result.float(), atol=2e-2, rtol=2e-2)
    for name in ("query", "rms_weight"):
        torch.testing.assert_close(
            compiled_grads[name].float(), direct_grads[name].float(), atol=2e-2, rtol=2e-2
        )
    for left, right in zip(direct_grads["residuals"], compiled_grads["residuals"], strict=True):
        torch.testing.assert_close(left.float(), right.float(), atol=2e-2, rtol=2e-2)
    equation_result, equation_grads = _grads(
        lambda values: _equation(equation_plan, values), _clone_inputs(operands)
    )
    equation_output_error = (
        direct_result.float() - equation_result.float()
    ).abs().max().item()
    equation_gradient_errors = {
        name: (
            direct_grads[name].float() - equation_grads[name].float()
        ).abs().max().item()
        for name in ("query", "rms_weight")
    }
    equation_gradient_errors["residuals"] = max(
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(
            direct_grads["residuals"], equation_grads["residuals"], strict=True
        )
    )
    torch.testing.assert_close(
        equation_result.float(), direct_result.float(), atol=2e-2, rtol=2e-2
    )
    for name in ("query", "rms_weight"):
        torch.testing.assert_close(
            equation_grads[name].float(), direct_grads[name].float(), atol=2e-2, rtol=2e-2
        )
    for left, right in zip(direct_grads["residuals"], equation_grads["residuals"], strict=True):
        torch.testing.assert_close(left.float(), right.float(), atol=2e-2, rtol=2e-2)
    performance = _measure(
        _direct, lambda values: _compiled(plan, values), operands, pairs, warmup
    )
    cases = {
        "attnres_depth_core": {
            "architecture_ids": ["arch-054"],
            "semantic_scope": "AttnRes residual-depth softmax; RMS-normalized keys, learned query and residual value sum included; surrounding transformer layer excluded",
            "shape": {
                "depth": DEPTH,
                "batch": BATCH,
                "sequence": SEQUENCE,
                "width": WIDTH,
                "dtype": "bfloat16",
                "kernel_compute_dtype": "float32",
            },
            "upstream_callable": "fla.ops.attnres.fused.fused_attnres",
            "compiled_anchor": plan.anchor,
            "compiler_plan_build_ms": plan_build_ms,
            "parity": {
                "status": "pass",
                "output_max_abs_error": output_error,
                "input_gradient_max_abs_errors": gradient_errors,
                "reference_equation": {
                    "status": "pass",
                    "output_max_abs_error": equation_output_error,
                    "input_gradient_max_abs_errors": equation_gradient_errors,
                },
                "tolerances": {"output_atol": 2e-2, "gradient_atol": 2e-2},
            },
            "performance": {"measurements": performance},
        }
    }
    config = {"depth": DEPTH, "batch": BATCH, "sequence": SEQUENCE, "width": WIDTH, "dtype": "bfloat16", "pairs": pairs, "warmup": warmup}
    write_artifact(
        output_path,
        {
            "schema_version": 1,
            "generated_utc": datetime.now(UTC).isoformat(),
            "purpose": "compare AttnRes residual-depth aggregation with a unified K1 library plan",
            "upstream": {
                "repository": "https://github.com/fla-org/flash-linear-attention",
                "revision": revision,
                "loaded_module": str(source),
            },
            "provenance": provenance(
                "PYTHONPATH=/path/to/flash-linear-attention:src:benchmarks python benchmarks/unified_mixer_attnres.py",
                config,
            ),
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "methodology": {
                "timed_work": "pinned FLA fused AttnRes versus its unified compiler adapter on preallocated leaf inputs",
                "sampling": "paired alternating direct/compiled calls with synchronized wall and CUDA event timing",
                "warmup": warmup,
                "pairs": pairs,
                "overhead_gate_fraction": 0.10,
                "interpretation": "the compiler library plan invokes the pinned fused AttnRes operator; a separate reference-equation check covers RMSNorm, query scaling and generic K1 depth softmax",
            },
            "cases": cases,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/attnres-k1.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
