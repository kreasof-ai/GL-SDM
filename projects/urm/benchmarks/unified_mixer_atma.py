"""Parity and paired K1 profiles against pinned ATMA Polar Triton kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from kernel import polar_triton
from model.blocks import polar_reduce

from provenance import provenance, write_artifact
from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_ATMA_REVISION = "28bb3de8afbe7c0b00115e0fbff36afc9ad49c11"
RECIPES = ("polar_attention_core", "foveal_sparse_polar_attention_core")
ARCHITECTURES = {
    "polar_attention_core": "arch-064",
    "foveal_sparse_polar_attention_core": "arch-065",
}


def _revision() -> str | None:
    root = Path(polar_triton.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _inputs(recipe: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch, heads, sequence, dim = 1, 2, 64, 16

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator) * 0.2

    values = {
        "query": rand(batch, heads, sequence, dim).requires_grad_(),
        "key": rand(batch, heads, sequence, dim).requires_grad_(),
        "value": rand(batch, heads, sequence, dim).requires_grad_(),
        "n_keys": torch.arange(1, sequence + 1, device="cuda", dtype=torch.float32),
        "v_null": rand(heads, dim).requires_grad_(),
        "null_base": (rand(heads) + 2.0).requires_grad_(),
        "null_slope_raw": (rand(heads) + 0.5).requires_grad_(),
        "len_gain_raw": (rand(heads) - 1.0).requires_grad_(),
        "mag_beta_raw": (rand(heads) - 1.5).requires_grad_(),
    }
    if recipe == "foveal_sparse_polar_attention_core":
        page_size = 16
        values["page_indices"] = torch.zeros(
            (batch, sequence // page_size, 2), device="cuda", dtype=torch.int32
        )
        values["page_counts"] = torch.tensor(
            [[0, 0, 1, 2]], device="cuda", dtype=torch.int32
        )
        values["page_indices"][0, 2, 0] = 0
        values["page_indices"][0, 3, :2] = torch.tensor(
            [0, 1], device="cuda", dtype=torch.int32
        )
    return values


def _clone(values):
    return {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in values.items()
    }


def _call(recipe: str, values):
    if recipe == "polar_attention_core":
        return polar_triton.polar_attention(
            values["query"], values["key"], values["value"], values["n_keys"],
            v_null=values["v_null"], null_base=values["null_base"],
            null_slope_raw=values["null_slope_raw"], len_gain_raw=values["len_gain_raw"],
            mag_beta_raw=values["mag_beta_raw"],
        )
    return polar_triton.polar_attention_sparse(
        values["query"], values["key"], values["value"],
        values["page_indices"], values["page_counts"],
        page_size=16, local_window=16,
        v_null=values["v_null"], null_base=values["null_base"],
        null_slope_raw=values["null_slope_raw"], len_gain_raw=values["len_gain_raw"],
        mag_beta_raw=values["mag_beta_raw"],
    )


def _compiled(plan, values):
    result = plan.execute(**values, **({"page_size": 16, "local_window": 16} if plan.spec.name == "foveal_sparse_polar_attention_core" else {}))
    return result.output, result.auxiliary_output


def _oracle(recipe: str, values):
    query, key, value = values["query"], values["key"], values["value"]
    batch, heads, sequence, dim = query.shape
    positions = torch.arange(sequence, device=query.device)
    allowed = positions[None, :] <= positions[:, None]
    if recipe == "foveal_sparse_polar_attention_core":
        allowed = allowed & (positions[None, :] > positions[:, None] - 16)
        allowed = allowed.unsqueeze(0).expand(batch, -1, -1).clone()
        for query_page in range(sequence // 16):
            start, stop = query_page * 16, (query_page + 1) * 16
            for slot in range(int(values["page_counts"][0, query_page])):
                key_page = int(values["page_indices"][0, query_page, slot])
                key_start, key_stop = key_page * 16, (key_page + 1) * 16
                allowed[0, start:stop, key_start:key_stop] = True
    else:
        allowed = allowed.unsqueeze(0)
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / (dim**0.5)
    scores = scores.masked_fill(~allowed[:, None], -torch.inf)
    return polar_reduce(
        scores, value, values["n_keys"],
        v_null=values["v_null"], null_base=values["null_base"],
        null_slope_raw=values["null_slope_raw"], len_gain_raw=values["len_gain_raw"],
        mag_beta_raw=values["mag_beta_raw"],
    )


def _loss(result, direction_weight, magnitude_weight):
    return (result[0] * direction_weight).sum() + (result[1] * magnitude_weight).sum()


def _differentiate(call, values, direction_weight, magnitude_weight):
    for tensor in values.values():
        if tensor.is_floating_point():
            tensor.grad = None
    result = call(values)
    _loss(result, direction_weight, magnitude_weight).backward()
    return result


def _max_error(left, right):
    return (left.float() - right.float()).abs().max().item()


def _grad_errors(left, right):
    errors = {}
    for name in left:
        if left[name].requires_grad:
            errors[name] = _max_error(left[name].grad, right[name].grad)
    return errors


def _time_one(call, values, weights, backward):
    for tensor in values.values():
        if tensor.is_floating_point():
            tensor.grad = None
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call(values)
    if backward:
        _loss(result, *weights).backward()
    end.record()
    torch.cuda.synchronize()
    return time.perf_counter() - start_wall, start.elapsed_time(end) / 1000


def _summary(values):
    ordered = sorted(values)
    return {
        "sample_count": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": ordered[max(0, int(0.95 * (len(ordered) - 1)))] * 1000,
        "raw_samples_ms": [value * 1000 for value in values],
    }


def _profile(direct, compiled, direct_inputs, compiled_inputs, weights, pairs, warmup):
    for _ in range(warmup):
        _time_one(direct, direct_inputs, weights, False)
        _time_one(compiled, compiled_inputs, weights, False)
        _time_one(direct, direct_inputs, weights, True)
        _time_one(compiled, compiled_inputs, weights, True)
    measurements = {}
    for label, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall, direct_device, compiled_device = [], [], [], []
        overhead, order = [], []
        for index in range(pairs):
            first, second = ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            order.append(first + second)
            pair = {}
            for name in (first, second):
                call, inputs = (direct, direct_inputs) if name == "direct" else (compiled, compiled_inputs)
                pair[name] = _time_one(call, inputs, weights, backward)
            direct_wall.append(pair["direct"][0])
            direct_device.append(pair["direct"][1])
            compiled_wall.append(pair["compiled"][0])
            compiled_device.append(pair["compiled"][1])
            overhead.append((pair["compiled"][0] - pair["direct"][0]) / pair["direct"][0])
        median = statistics.median(overhead)
        measurements[label] = {
            "direct_wall": _summary(direct_wall),
            "compiled_wall": _summary(compiled_wall),
            "direct_device": _summary(direct_device),
            "compiled_device": _summary(compiled_device),
            "paired_compiled_overhead_fraction": {
                "median": median,
                "p95": sorted(overhead)[max(0, int(0.95 * (len(overhead) - 1)))],
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median <= 0.10},
            },
            "pair_order": order,
        }
    return {"warmup_calls_per_backend_per_mode": warmup, "paired_samples_per_mode": pairs, "measurements": measurements}


def run(pairs: int, warmup: int, output: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("the ATMA Polar profile requires CUDA")
    revision = _revision()
    if revision != EXPECTED_ATMA_REVISION:
        raise RuntimeError(f"loaded ATMA source must match {EXPECTED_ATMA_REVISION}, got {revision!r}")
    cases = {}
    for recipe in RECIPES:
        operands = _inputs(recipe, 1830 if recipe == RECIPES[0] else 1831)
        direct_inputs, compiled_inputs = _clone(operands), _clone(operands)
        plan = compile_mixer(
            named_mixer_recipe(recipe), backend=MixerBackend.LIBRARY,
            intent=MixerIntent.TRAINING, dtype="float32",
        )
        reference_plan = compile_mixer(
            named_mixer_recipe(recipe), intent=MixerIntent.TRAINING, dtype="float32"
        )
        reference_inputs, oracle_inputs = _clone(operands), _clone(operands)
        weights = (
            torch.randn((1, 2, 64, 16), device="cuda", generator=torch.Generator(device="cuda").manual_seed(9001)),
            torch.randn((1, 2, 64), device="cuda", generator=torch.Generator(device="cuda").manual_seed(9002)),
        )
        reference_result = _differentiate(
            lambda values: (lambda result: (result.output, result.auxiliary_output))(
                reference_plan.execute(**values, **({"page_size": 16, "local_window": 16} if recipe == RECIPES[1] else {}))
            ), reference_inputs, *weights,
        )
        oracle_result = _differentiate(lambda values: _oracle(recipe, values), oracle_inputs, *weights)
        torch.testing.assert_close(reference_result[0], oracle_result[0], atol=3e-3, rtol=3e-3)
        torch.testing.assert_close(reference_result[1], oracle_result[1], atol=3e-3, rtol=3e-3)
        equation_gradient_errors = _grad_errors(reference_inputs, oracle_inputs)
        for name, error in equation_gradient_errors.items():
            torch.testing.assert_close(reference_inputs[name].grad, oracle_inputs[name].grad, atol=3e-3, rtol=3e-3, msg=lambda m: f"{name}: {m}")

        direct = lambda values, recipe=recipe: _call(recipe, values)
        compiled = lambda values, plan=plan: _compiled(plan, values)
        profile = _profile(direct, compiled, direct_inputs, compiled_inputs, weights, pairs, warmup)
        upstream_result = _differentiate(direct, direct_inputs, *weights)
        compiled_result = _differentiate(compiled, compiled_inputs, *weights)
        torch.testing.assert_close(compiled_result[0], upstream_result[0], atol=0, rtol=0)
        torch.testing.assert_close(compiled_result[1], upstream_result[1], atol=0, rtol=0)
        adapter_gradient_errors = _grad_errors(compiled_inputs, direct_inputs)
        for name, error in adapter_gradient_errors.items():
            torch.testing.assert_close(compiled_inputs[name].grad, direct_inputs[name].grad, atol=0, rtol=0, msg=lambda m: f"{name}: {m}")
        case = {
            "architecture_ids": [ARCHITECTURES[recipe]],
            "semantic_scope": "ATMA Polar direction and bounded-magnitude reduction; Foveal also consumes caller-supplied local and selected remote page routes. Projection, GQA expansion, convolutions, route scoring and output/count projections are excluded.",
            "shape": {"batch": 1, "sequence": 64, "heads": 2, "query_heads": 2, "key_value_heads": 2, "key_dim": 16, "value_dim": 16, "dtype": "float32", **({"page_size": 16, "local_window": 16, "remote_capacity": 2} if recipe == RECIPES[1] else {})},
            "intent_modes": ["training_forward", "training_forward_backward"],
            "compiled_anchor": plan.anchor,
            "parity": {
                "status": "pass",
                "output_max_abs_error": _max_error(compiled_result[0], upstream_result[0]),
                "auxiliary_output_max_abs_error": _max_error(compiled_result[1], upstream_result[1]),
                "input_gradient_max_abs_errors": adapter_gradient_errors,
                "tolerances": {"output_atol": 0.0, "auxiliary_output_atol": 0.0, "gradient_atol": 0.0},
            },
            "reference_equation_parity": {
                "status": "pass",
                "upstream_reference_callable": "model.blocks.polar_reduce",
                "output_max_abs_error": _max_error(reference_result[0], oracle_result[0]),
                "auxiliary_output_max_abs_error": _max_error(reference_result[1], oracle_result[1]),
                "input_gradient_max_abs_errors": equation_gradient_errors,
                "tolerances": {"output_atol": 3e-3, "auxiliary_output_atol": 3e-3, "gradient_atol": 3e-3},
            },
            "architecture_entrypoint_parity": None,
            "performance": profile,
        }
        cases[recipe] = case

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K1 plans with exact pinned ATMA Polar Triton kernels",
        "upstream": {
            "repository": None,
            "local_path": "/home/sagemaker-user/atma",
            "revision": revision,
            "loaded_module": polar_triton.__file__,
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/atma:src python benchmarks/unified_mixer_atma.py",
            {"recipes": RECIPES, "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one forward call or one forward plus backward on preallocated leaf inputs",
            "sampling": "paired interleaved direct/compiled calls with synchronized wall and CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct wall-time fractions",
            "overhead_gate_fraction": 0.10,
            "overhead_gate_rule": "median compiled slowdown must not exceed 10%",
            "interpretation": "the library anchor calls the same ATMA Triton operator; timings measure compiler-plan and validation overhead",
        },
        "cases": cases,
    }
    write_artifact(output, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/atma-polar-k1.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
