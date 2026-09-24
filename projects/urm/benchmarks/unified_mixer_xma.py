"""Pinned XMA nonlinear-recurrence parity and paired-plan profiles."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import xma
from xma import KernelBackend

from provenance import provenance, write_artifact
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_XMA_REVISION = "384ed0a7bd82ced1f40609603dd541cac5416844"
RECIPES = ("rnn_core", "gru_core", "m2rnn_core")


def _revision() -> str | None:
    root = Path(xma.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _inputs(recipe: str, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", generator=generator) * 0.1

    batch, sequence, heads, dim = 1, 64, 1, 16
    if recipe == "rnn_core":
        return {
            "query": rand(batch, sequence, heads, dim).requires_grad_(),
            "weight": rand(heads, dim, dim).requires_grad_(),
            "initial_state": rand(batch, heads, dim).requires_grad_(),
        }
    if recipe == "gru_core":
        return {
            "query": rand(batch, sequence, heads, dim).requires_grad_(),
            "weight": rand(heads, dim, dim).requires_grad_(),
            "forget_input": rand(batch, sequence, heads, dim).requires_grad_(),
            "forget_weight": rand(heads, dim, dim).requires_grad_(),
            "reset_input": rand(batch, sequence, heads, dim).requires_grad_(),
            "reset_weight": rand(heads, dim, dim).requires_grad_(),
            "initial_state": rand(batch, heads, dim).requires_grad_(),
        }
    if recipe == "m2rnn_core":
        value_dim = 16
        return {
            "query": rand(batch, sequence, heads, dim).requires_grad_(),
            "key": rand(batch, sequence, heads, dim).requires_grad_(),
            "value": rand(batch, sequence, heads, value_dim).requires_grad_(),
            "weight": rand(heads, value_dim, value_dim).requires_grad_(),
            "forget_input": torch.sigmoid(
                rand(batch, sequence, heads)
            ).requires_grad_(),
            "initial_state": rand(batch, heads, dim, value_dim).requires_grad_(),
        }
    raise ValueError(f"unknown XMA recipe {recipe!r}")


def _upstream(recipe: str, values: dict[str, torch.Tensor], backend: KernelBackend):
    if recipe == "rnn_core":
        from xma.layers.rnn import rnn

        return rnn(
            values["query"], values["weight"],
            input_state=values["initial_state"], kernel_backend=backend,
        )
    if recipe == "gru_core":
        from xma.layers.gru import gru

        return gru(
            values["query"], values["weight"], values["forget_input"],
            values["forget_weight"], values["reset_input"], values["reset_weight"],
            input_state=values["initial_state"], kernel_backend=backend,
        )
    if recipe == "m2rnn_core":
        from xma.layers.m2rnn import m2rnn

        return m2rnn(
            values["query"], values["key"], values["value"], values["weight"],
            values["forget_input"], input_state=values["initial_state"],
            kernel_backend=backend,
        )
    raise ValueError(f"unknown XMA recipe {recipe!r}")


def _compiled(plan, values: dict[str, torch.Tensor]):
    result = plan.execute(**values)
    return result.output, result.final_state


def _clone(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone().requires_grad_(value.requires_grad)
        for name, value in values.items()
    }


def _loss(result) -> torch.Tensor:
    output, state = result
    return output.float().square().mean() + state.float().square().mean()


def _max_error(left, right) -> float:
    return (left.float() - right.float()).abs().max().item()


def _compare_gradients(left, right, *, atol: float, rtol: float) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name in left:
        actual, expected = left[name].grad, right[name].grad
        errors[name] = _max_error(actual, expected)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")
    return errors


def _differentiate(call, values: dict[str, torch.Tensor]):
    for tensor in values.values():
        tensor.grad = None
    result = call(values)
    _loss(result).backward()
    return result


def _time_one(call, values: dict[str, torch.Tensor], *, backward: bool):
    for tensor in values.values():
        tensor.grad = None
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    result = call(values)
    if backward:
        _loss(result).backward()
    end_event.record()
    torch.cuda.synchronize()
    return time.perf_counter() - start_wall, start_event.elapsed_time(end_event) / 1000


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": sorted(values)[max(0, int(0.95 * (len(values) - 1)))] * 1000,
        "raw_samples_ms": [value * 1000 for value in values],
    }


def _profile(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup):
    for _ in range(warmup):
        _time_one(direct, direct_inputs, backward=False)
        _time_one(compiled, compiled_inputs, backward=False)
        _time_one(direct, direct_inputs, backward=True)
        _time_one(compiled, compiled_inputs, backward=True)
    modes = {}
    for label, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall = [], []
        direct_device, compiled_device = [], []
        overhead, order = [], []
        for index in range(pairs):
            first, second = ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            order.append(first + second)
            pair = {}
            for name in (first, second):
                if name == "direct":
                    pair[name] = _time_one(direct, direct_inputs, backward=backward)
                else:
                    pair[name] = _time_one(compiled, compiled_inputs, backward=backward)
            direct_wall.append(pair["direct"][0])
            direct_device.append(pair["direct"][1])
            compiled_wall.append(pair["compiled"][0])
            compiled_device.append(pair["compiled"][1])
            overhead.append((pair["compiled"][0] - pair["direct"][0]) / pair["direct"][0])
        median_overhead = statistics.median(overhead)
        modes[label] = {
            "direct_wall": _summary(direct_wall),
            "compiled_wall": _summary(compiled_wall),
            "direct_device": _summary(direct_device),
            "compiled_device": _summary(compiled_device),
            "paired_compiled_overhead_fraction": {
                "median": median_overhead,
                "p95": sorted(overhead)[max(0, int(0.95 * (len(overhead) - 1)))],
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
            },
            "pair_order": order,
        }
    return {"warmup_calls_per_backend_per_mode": warmup, "paired_samples_per_mode": pairs, "measurements": modes}


def run(recipe: str, pairs: int, warmup: int, output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the XMA recurrent profile requires CUDA")
    revision = _revision()
    if revision != EXPECTED_XMA_REVISION:
        raise RuntimeError(f"loaded XMA source must match {EXPECTED_XMA_REVISION}, got {revision!r}")

    operands = _inputs(recipe, seed=72100 + RECIPES.index(recipe))
    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe(recipe), backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING, dtype="float32",
    )
    build_ms = (time.perf_counter() - plan_started) * 1000
    direct_inputs, compiled_inputs = _clone(operands), _clone(operands)
    direct = lambda values: _upstream(recipe, values, KernelBackend.triton)
    compiled = lambda values: _compiled(plan, values)

    reference_inputs, equation_inputs = _clone(operands), _clone(operands)
    reference_plan = compile_mixer(
        named_mixer_recipe(recipe), intent=MixerIntent.TRAINING, dtype="float32"
    )
    reference = _differentiate(
        lambda values: _compiled(reference_plan, values), reference_inputs
    )
    equation = _differentiate(
        lambda values: _upstream(recipe, values, KernelBackend.torch), equation_inputs
    )
    torch.testing.assert_close(reference[0], equation[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(reference[1], equation[1], atol=3e-6, rtol=3e-6)
    equation_gradient_errors = _compare_gradients(
        reference_inputs, equation_inputs, atol=3e-6, rtol=3e-6
    )

    profile = _profile(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup)
    source_result = _differentiate(direct, direct_inputs)
    compiled_result = _differentiate(compiled, compiled_inputs)
    torch.testing.assert_close(compiled_result[0], source_result[0], atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(compiled_result[1], source_result[1], atol=3e-6, rtol=3e-6)
    adapter_gradient_errors = _compare_gradients(
        compiled_inputs, direct_inputs, atol=3e-6, rtol=3e-6
    )
    parity = {
        "status": "pass",
        "output_max_abs_error": _max_error(compiled_result[0], source_result[0]),
        "final_state_max_abs_error": _max_error(compiled_result[1], source_result[1]),
        "input_gradient_max_abs_errors": adapter_gradient_errors,
        "tolerances": {"output_atol": 3e-6, "state_atol": 3e-6, "gradient_atol": 3e-6},
    }
    equation_parity = {
        "status": "pass",
        "upstream_reference_backend": "XMA KernelBackend.torch",
        "output_max_abs_error": _max_error(reference[0], equation[0]),
        "final_state_max_abs_error": _max_error(reference[1], equation[1]),
        "input_gradient_max_abs_errors": equation_gradient_errors,
        "tolerances": {"output_atol": 3e-6, "state_atol": 3e-6, "gradient_atol": 3e-6},
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified mixer K2 plan with pinned XMA Triton nonlinear recurrent operators",
        "upstream": {
            "repository": "https://github.com/open-lm-engine/accelerated-model-architectures",
            "revision": revision,
            "loaded_module": xma.__file__,
            "backend": "KernelBackend.triton",
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/pinned-xma:src python benchmarks/unified_mixer_xma.py",
            {"recipe": recipe, "pairs": pairs, "warmup": warmup},
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
            "interpretation": "the library anchor calls the same XMA Triton operator; timings measure compiler-plan and validation overhead",
        },
        "cases": {
            recipe: {
                "architecture_ids": {
                    "rnn_core": ["arch-060"],
                    "gru_core": ["arch-061"],
                    "m2rnn_core": ["arch-062"],
                }[recipe],
                "semantic_scope": "Fixed-length single-head nonlinear recurrence; XMA projections, head replication, packed variable-length sequence support and gradient clipping remain external.",
                "shape": {"batch": 1, "sequence": 64, "heads": 1, "key_dim": 16, "value_dim": 16, "dtype": "float32"},
                "intent_modes": ["training_forward", "training_forward_backward"],
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": build_ms,
                "parity": parity,
                "reference_equation_parity": equation_parity,
                "architecture_entrypoint_parity": None,
                "performance": profile,
            }
        },
    }
    write_artifact(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    output = args.output or Path(f"results/unified-mixer/{args.recipe.replace('_core', '')}-xma-k2.json")
    run(args.recipe, args.pairs, args.warmup, output)


if __name__ == "__main__":
    main()
