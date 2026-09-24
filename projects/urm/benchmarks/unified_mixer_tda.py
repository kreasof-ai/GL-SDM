"""Pinned TDA Triton parity and paired K1 plan profile."""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from measurement import quantile
from provenance import provenance, write_artifact
from benchmarks.comparators.tda import tda_attention_adapter, tda_source_identity
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from benchmarks.recipe_catalog import load_kernel_recipe

BATCH, SEQUENCE, HEADS, DIM = 1, 64, 2, 32
NAMES = ("query_a", "query_b", "key_a", "key_b", "value")


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = {
        name: torch.randn(
            BATCH, SEQUENCE, HEADS, DIM, device="cuda", dtype=torch.float32,
            generator=generator,
        ).mul_(0.1).requires_grad_()
        for name in NAMES
    }
    values["beta"] = torch.tensor(0.7, device="cuda")
    values["lambda_weight"] = torch.tensor(0.35, device="cuda")
    return values


def _clone(values):
    return {
        name: tensor.detach().clone().requires_grad_(tensor.requires_grad)
        for name, tensor in values.items()
    }


def _upstream(values):
    import importlib

    source = importlib.import_module("triton_threshold_attention")
    return source.differential_threshold_rela_triton(
        *(values[name].transpose(1, 2).contiguous() for name in NAMES),
        values["beta"], values["lambda_weight"], relu_power=2.0, normalize=True,
    ).transpose(1, 2)


def _compiled(plan, values):
    return plan.execute(**values).output


def _loss(output):
    return output.square().mean()


def _differentiate(call, values):
    for name in NAMES:
        values[name].grad = None
    output = call(values)
    _loss(output).backward()
    return output, {name: values[name].grad for name in NAMES}


def _max_error(left, right):
    return (left.float() - right.float()).abs().max().item()


def _timing(call, values, backward):
    for name in NAMES:
        values[name].grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    begin_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin_event.record()
    output = call(values)
    if backward:
        _loss(output).backward()
    end_event.record()
    torch.cuda.synchronize()
    return time.perf_counter() - start, begin_event.elapsed_time(end_event) / 1000


def _summary(values):
    return {
        "sample_count": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": quantile(values, 0.95) * 1000,
        "raw_samples_ms": [value * 1000 for value in values],
    }


def _profile(upstream, compiled, upstream_inputs, compiled_inputs, pairs, warmup):
    cold_upstream = _timing(upstream, upstream_inputs, False)
    cold_compiled = _timing(compiled, compiled_inputs, False)
    for _ in range(warmup):
        for backward in (False, True):
            _timing(upstream, upstream_inputs, backward)
            _timing(compiled, compiled_inputs, backward)
    modes = {}
    for name, backward in (("forward", False), ("forward_backward", True)):
        raw = {backend: [] for backend in ("upstream", "compiled")}
        events = {backend: [] for backend in ("upstream", "compiled")}
        overhead, order = [], []
        for index in range(pairs):
            first, second = (
                ("upstream", "compiled") if index % 2 == 0 else ("compiled", "upstream")
            )
            order.append(first + second)
            samples = {}
            for backend in (first, second):
                samples[backend] = _timing(
                    upstream if backend == "upstream" else compiled,
                    upstream_inputs if backend == "upstream" else compiled_inputs,
                    backward,
                )
            for backend in ("upstream", "compiled"):
                raw[backend].append(samples[backend][0])
                events[backend].append(samples[backend][1])
            overhead.append((samples["compiled"][0] - samples["upstream"][0]) / samples["upstream"][0])
        median = statistics.median(overhead)
        modes[name] = {
            "upstream_wall": _summary(raw["upstream"]),
            "compiled_wall": _summary(raw["compiled"]),
            "upstream_device": _summary(events["upstream"]),
            "compiled_device": _summary(events["compiled"]),
            "paired_compiled_overhead_fraction": {
                "median": median,
                "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median <= 0.10},
            },
            "pair_order": order,
        }
    return {
        "first_upstream_forward_call_ms": {
            "wall": cold_upstream[0] * 1000, "device": cold_upstream[1] * 1000
        },
        "first_compiled_forward_call_after_upstream_ms": {
            "wall": cold_compiled[0] * 1000, "device": cold_compiled[1] * 1000
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": modes,
    }


def run(pairs: int, warmup: int, output_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("the TDA unified mixer profile requires CUDA")
    identity = tda_source_identity()
    values = _inputs(68068)
    recipe = load_kernel_recipe("tda_attention_core")
    build_start = time.perf_counter()
    library_plan = compile_mixer(
        recipe, backend=MixerBackend.LIBRARY, intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    plan_build_ms = (time.perf_counter() - build_start) * 1000
    upstream_inputs, reference_inputs, compiled_inputs = (
        _clone(values), _clone(values), _clone(values)
    )
    upstream_result = _differentiate(_upstream, upstream_inputs)
    reference_result = _differentiate(
        lambda data: reference_plan.execute(**data).output, reference_inputs
    )
    compiled = lambda data: _compiled(library_plan, data)
    compiled_result = _differentiate(compiled, compiled_inputs)
    output_atol, gradient_atol, rtol = 2e-4, 2e-4, 1e-2
    torch.testing.assert_close(reference_result[0], upstream_result[0], atol=output_atol, rtol=rtol)
    reference_errors = {}
    for name in NAMES:
        reference_errors[name] = _max_error(reference_result[1][name], upstream_result[1][name])
        torch.testing.assert_close(
            reference_result[1][name], upstream_result[1][name],
            atol=gradient_atol, rtol=rtol,
        )
    torch.testing.assert_close(compiled_result[0], upstream_result[0], atol=0.0, rtol=0.0)
    compiled_errors = {}
    for name in NAMES:
        compiled_errors[name] = _max_error(compiled_result[1][name], upstream_result[1][name])
        torch.testing.assert_close(
            compiled_result[1][name], upstream_result[1][name], atol=0.0, rtol=0.0
        )
    profile = _profile(_upstream, compiled, _clone(values), _clone(values), pairs, warmup)
    config = {"recipe": "tda_attention_core", "pairs": pairs, "warmup": warmup,
              "shape": [BATCH, SEQUENCE, HEADS, DIM], "dtype": "float32",
              "beta": 0.7, "lambda_weight": 0.35, "relu_power": 2.0}
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare the unified K1 TDA plan against the pinned threshold differential Triton source",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["source_path"],
            "kernel_source_sha256": {"triton_threshold_attention.py": identity["source_sha256"]},
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/TDA-checkout:src python benchmarks/unified_mixer_tda.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__, "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "the pinned two-branch TDA Triton call or its compiler library-plan adapter, optionally followed by output backward",
            "sampling": "paired interleaved calls with alternating order, synchronized wall and CUDA event timing",
            "warmup": warmup, "pairs": pairs,
            "overhead": "median of per-pair (compiled-upstream)/upstream fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "both timed paths invoke the same pinned TDA Triton implementation; measurements capture plan/layout dispatch overhead",
        },
        "cases": {
            "tda_attention_core": {
                "architecture_ids": ["arch-068"],
                "semantic_scope": "two normalized causal threshold-rectified score reductions combined by a clamped scalar lambda; beta/lambda generation and full projections excluded",
                "shape": {"batch": BATCH, "sequence": SEQUENCE, "heads": HEADS,
                          "key_dim": DIM, "value_dim": DIM, "dtype": "float32",
                          "beta": 0.7, "lambda_weight": 0.35, "relu_power": 2.0},
                "upstream_callable": "triton_threshold_attention.differential_threshold_rela_triton",
                "compiled_anchor": library_plan.anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": _max_error(compiled_result[0], upstream_result[0]),
                    "input_gradient_max_abs_errors": compiled_errors,
                    "tolerances": {"output_atol": 0.0, "gradient_atol": 0.0, "relative_tolerance": 0.0},
                },
                "reference_equation_parity": {
                    "status": "pass",
                    "output_max_abs_error": _max_error(reference_result[0], upstream_result[0]),
                    "input_gradient_max_abs_errors": reference_errors,
                    "tolerances": {"output_atol": output_atol, "gradient_atol": gradient_atol,
                                   "relative_tolerance": rtol},
                },
                "performance": profile,
            }
        },
    }
    write_artifact(output_path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/tda-k1.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
