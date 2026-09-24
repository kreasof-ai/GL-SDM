"""Pinned BDH strict-past attention parity and paired K2 plan profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_BDH_REVISION = "2b0d7a45b058d4309c84a10e0768d541fe18bdc2"
BATCH, SEQUENCE, HEADS, DIM, VALUE_DIM = 1, 64, 2, 32, 16


def _source_identity() -> tuple[object, Path, str, str]:
    source_module = importlib.import_module("bdh")
    source = Path(source_module.__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify the BDH source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if revision != EXPECTED_BDH_REVISION or dirty:
        raise RuntimeError(
            "BDH profiling requires the clean pinned source revision "
            f"{EXPECTED_BDH_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    return source_module, source, revision, source_hash


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        BATCH, SEQUENCE, HEADS, DIM, device="cuda", generator=generator
    ) * 0.1
    value = torch.randn(
        BATCH, SEQUENCE, HEADS, VALUE_DIM, device="cuda", generator=generator
    ) * 0.1
    query.requires_grad_()
    value.requires_grad_()
    return {"query": query, "key": query, "value": value}


def _clone(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    query = values["query"].detach().clone().requires_grad_()
    value = values["value"].detach().clone().requires_grad_()
    return {"query": query, "key": query, "value": value}


def _direct(module, values: dict[str, torch.Tensor]):
    query = values["query"].transpose(1, 2)
    value = values["value"].transpose(1, 2)
    return module(query, query, value).transpose(1, 2)


def _compiled(plan, values: dict[str, torch.Tensor]):
    return plan.execute(**values).output


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _differentiate(call, values: dict[str, torch.Tensor]):
    for name in ("query", "value"):
        values[name].grad = None
    output = call(values)
    _loss(output).backward()
    return output, {name: values[name].grad for name in ("query", "value")}


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def _compare_grads(actual, expected, *, atol: float, rtol: float) -> dict[str, float]:
    errors = {}
    for name in actual:
        errors[name] = _max_error(actual[name], expected[name])
        torch.testing.assert_close(
            actual[name], expected[name], atol=atol, rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )
    return errors


def _time_one(call, values, *, backward: bool):
    for name in ("query", "value"):
        values[name].grad = None
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output = call(values)
    if backward:
        _loss(output).backward()
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


def _profile(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup):
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
            result = {}
            for backend in (first, second):
                result[backend] = _time_one(
                    direct if backend == "direct" else compiled,
                    direct_inputs if backend == "direct" else compiled_inputs,
                    backward=backward,
                )
            direct_wall.append(result["direct"][0])
            direct_device.append(result["direct"][1])
            compiled_wall.append(result["compiled"][0])
            compiled_device.append(result["compiled"][1])
            overhead.append(
                (result["compiled"][0] - result["direct"][0])
                / result["direct"][0]
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
        "first_compiled_forward_call_after_direct_ms": {
            "wall": cold_compiled[0] * 1000,
            "device": cold_compiled[1] * 1000,
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def run(pairs: int, warmup: int, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the BDH unified mixer profile requires CUDA")
    source_module, source, revision, source_hash = _source_identity()
    module = source_module.Attention(
        source_module.BDHConfig(
            n_embd=HEADS * DIM,
            n_head=HEADS,
            mlp_internal_dim_multiplier=1,
            dropout=0.0,
        )
    ).to("cuda").eval()
    operands = _inputs(seed=63063)

    build_start = time.perf_counter()
    recipe = named_mixer_recipe("bdh_attention_core")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="float32"
    )
    plan_build_ms = (time.perf_counter() - build_start) * 1000
    direct = lambda values: _direct(module, values)
    compiled = lambda values: _compiled(library_plan, values)
    reference = lambda values: _compiled(reference_plan, values)

    upstream_inputs = _clone(operands)
    reference_inputs = _clone(operands)
    library_inputs = _clone(operands)
    upstream_output, upstream_gradients = _differentiate(direct, upstream_inputs)
    reference_output, reference_gradients = _differentiate(reference, reference_inputs)
    library_output, library_gradients = _differentiate(compiled, library_inputs)

    equation_output_error = _max_error(reference_output, upstream_output)
    library_output_error = _max_error(library_output, upstream_output)
    equation_gradient_errors = _compare_grads(
        reference_gradients, upstream_gradients, atol=3e-5, rtol=3e-5
    )
    library_gradient_errors = _compare_grads(
        library_gradients, upstream_gradients, atol=0.0, rtol=0.0
    )
    torch.testing.assert_close(reference_output, upstream_output, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(library_output, upstream_output, atol=0.0, rtol=0.0)

    direct_inputs, compiled_inputs = _clone(operands), _clone(operands)
    performance = _profile(
        direct, compiled, direct_inputs, compiled_inputs, pairs, warmup
    )
    config = {
        "recipe": "bdh_attention_core",
        "pairs": pairs,
        "warmup": warmup,
        "shape": [BATCH, SEQUENCE, HEADS, DIM, VALUE_DIM],
        "dtype": "float32",
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare the unified K2 BDH attention plans with pinned BDH Attention.forward",
        "upstream": {
            "repository": "https://github.com/pathwaycom/bdh",
            "revision": revision,
            "module_version": None,
            "loaded_module": str(source),
            "kernel_source_sha256": {"bdh.py": source_hash},
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/bdh-checkout:src python benchmarks/unified_mixer_bdh.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one source BDH Attention.forward call or one compiled library plan call, optionally followed by output backward",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "both paths invoke the same pinned BDH Attention.forward implementation; measurements capture plan dispatch overhead",
        },
        "cases": {
            "bdh_attention_core": {
                "architecture_ids": ["arch-063"],
                "semantic_scope": "rotary strict-past unnormalized attention; BDH projections, normalization, feedforward gates, dropout and block composition excluded",
                "shape": {
                    "batch": BATCH,
                    "sequence": SEQUENCE,
                    "heads": HEADS,
                    "query_key_dim": DIM,
                    "value_dim": VALUE_DIM,
                    "dtype": "float32",
                },
                "upstream_callable": "bdh.Attention.forward",
                "compiled_anchor": library_plan.anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": library_output_error,
                    "input_gradient_max_abs_errors": library_gradient_errors,
                    "tolerances": {"output_atol": 0.0, "gradient_atol": 0.0, "relative_tolerance": 0.0},
                },
                "reference_equation_parity": {
                    "status": "pass",
                    "output_max_abs_error": equation_output_error,
                    "input_gradient_max_abs_errors": equation_gradient_errors,
                    "tolerances": {"output_atol": 3e-5, "gradient_atol": 3e-5, "relative_tolerance": 3e-5},
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
        "--output", type=Path, default=Path("results/unified-mixer/bdh-k2.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
