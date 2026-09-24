"""Pinned FLA MoM per-route recurrence parity and K2 plan profile."""

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
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

from measurement import quantile
from provenance import provenance, write_artifact
from benchmarks.comparators.fla_gated_delta import fla_version
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from benchmarks.recipe_catalog import load_kernel_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
BATCH, SEQUENCE, HEADS, KEY_DIM, VALUE_DIM = 2, 64, 2, 32, 16


def _source_identity() -> tuple[Path, str, dict[str, str]]:
    package_source = Path(inspect.getfile(fla)).resolve()
    repository = next(
        (parent for parent in package_source.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        raise RuntimeError(f"could not identify FLA source checkout for {package_source}")
    identity = fla_version()
    revision = identity.get("source_revision")
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
    if revision != EXPECTED_FLA_REVISION or dirty:
        raise RuntimeError(
            "MoM profiling requires the clean pinned FLA source revision "
            f"{EXPECTED_FLA_REVISION}; got {revision!r} with dirty={bool(dirty)}"
        )
    source_paths = (
        "fla/layers/mom.py",
        "fla/ops/gated_delta_rule/chunk.py",
        "fla/modules/l2norm.py",
    )
    hashes = {
        path: hashlib.sha256((repository / path).read_bytes()).hexdigest()
        for path in source_paths
    }
    return package_source, revision, hashes


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        BATCH, SEQUENCE, HEADS, KEY_DIM, device="cuda", dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    key = torch.randn_like(query, generator=generator) * 0.1
    value = torch.randn(
        BATCH, SEQUENCE, HEADS, VALUE_DIM, device="cuda", dtype=torch.bfloat16,
        generator=generator,
    ) * 0.1
    beta = torch.sigmoid(
        torch.randn(BATCH, SEQUENCE, HEADS, device="cuda", generator=generator)
    ).to(torch.bfloat16)
    log_decay = -torch.rand(
        BATCH, SEQUENCE, HEADS, device="cuda", generator=generator
    ) * 0.03
    return {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "beta": beta.requires_grad_(),
        "log_decay": log_decay.requires_grad_(),
    }


def _clone(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in values.items()
    }


def _direct(values: dict[str, torch.Tensor]):
    return chunk_gated_delta_rule(
        values["query"],
        values["key"],
        values["value"],
        values["log_decay"],
        values["beta"],
        scale=1.0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=False,
        state_v_first=True,
    )


def _compiled(plan, values: dict[str, torch.Tensor]):
    result = plan.execute(**values)
    return result.output, result.final_state


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _differentiate(call, values: dict[str, torch.Tensor]):
    for tensor in values.values():
        tensor.grad = None
    output, state = call(values)
    _loss(output).backward()
    gradients = {name: values[name].grad for name in values}
    return output, state, gradients


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def _compare_gradients(actual, expected, *, atol: float, rtol: float):
    errors = {}
    for name in actual:
        errors[name] = _max_error(actual[name], expected[name])
        torch.testing.assert_close(
            actual[name], expected[name], atol=atol, rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )
    return errors


def _time_one(call, values, *, backward: bool):
    for tensor in values.values():
        tensor.grad = None
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output, _ = call(values)
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
        direct_wall, compiled_wall, direct_device, compiled_device = [], [], [], []
        overhead, order = [], []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            results = {}
            for backend in (first, second):
                results[backend] = _time_one(
                    direct if backend == "direct" else compiled,
                    direct_inputs if backend == "direct" else compiled_inputs,
                    backward=backward,
                )
            direct_wall.append(results["direct"][0])
            compiled_wall.append(results["compiled"][0])
            direct_device.append(results["direct"][1])
            compiled_device.append(results["compiled"][1])
            overhead.append(
                (results["compiled"][0] - results["direct"][0])
                / results["direct"][0]
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
        "first_direct_forward_call_ms": {"wall": cold_direct[0] * 1000, "device": cold_direct[1] * 1000},
        "first_compiled_forward_call_after_direct_ms": {"wall": cold_compiled[0] * 1000, "device": cold_compiled[1] * 1000},
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def run(pairs: int, warmup: int, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the MoM unified mixer profile requires CUDA")
    source, revision, source_hashes = _source_identity()
    operands = _inputs(seed=51051)
    plan_start = time.perf_counter()
    recipe = load_kernel_recipe("mom_selected_memory_core")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    plan_build_ms = (time.perf_counter() - plan_start) * 1000
    compiled = lambda values: _compiled(library_plan, values)
    reference = lambda values: _compiled(reference_plan, values)

    direct_inputs, reference_inputs, compiled_inputs = (
        _clone(operands), _clone(operands), _clone(operands)
    )
    direct_result = _differentiate(_direct, direct_inputs)
    reference_result = _differentiate(reference, reference_inputs)
    compiled_result = _differentiate(compiled, compiled_inputs)
    output_atol = 1e-2
    state_atol = 1e-2
    gradient_atol = 2e-2
    torch.testing.assert_close(reference_result[0], direct_result[0], atol=output_atol, rtol=1e-2)
    torch.testing.assert_close(reference_result[1], direct_result[1], atol=state_atol, rtol=1e-2)
    torch.testing.assert_close(compiled_result[0], direct_result[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(compiled_result[1], direct_result[1], atol=0.0, rtol=0.0)
    equation_gradient_errors = _compare_gradients(
        reference_result[2], direct_result[2], atol=gradient_atol, rtol=2e-2
    )
    compiled_gradient_errors = _compare_gradients(
        compiled_result[2], direct_result[2], atol=0.0, rtol=0.0
    )

    direct_inputs, compiled_inputs = _clone(operands), _clone(operands)
    performance = _profile(_direct, compiled, direct_inputs, compiled_inputs, pairs, warmup)
    config = {
        "recipe": "mom_selected_memory_core",
        "pairs": pairs,
        "warmup": warmup,
        "shape": [BATCH, SEQUENCE, HEADS, KEY_DIM, VALUE_DIM],
        "dtype": "bfloat16",
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K2 MoM per-route plan with pinned FLA chunk_gated_delta_rule",
        "upstream": {
            "repository": "https://github.com/fla-org/flash-linear-attention",
            "revision": revision,
            "module_version": getattr(fla, "__version__", None),
            "loaded_module": str(source),
            "kernel_source_sha256": source_hashes,
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/fla-checkout:src python benchmarks/unified_mixer_mom.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one source chunk_gated_delta_rule call or one unified library plan call, optionally followed by output backward",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "both paths invoke the same pinned FLA chunk operator with MoM's Q/K normalization and V-first state flags; measurements capture plan dispatch overhead",
        },
        "cases": {
            "mom_selected_memory_core": {
                "architecture_ids": ["arch-051"],
                "semantic_scope": "one already routed memory stream per batch item; router scoring, top-k dispatch/merge, memory projections, convolutions and full layer/cache excluded",
                "shape": {
                    "routed_memory_streams": BATCH,
                    "sequence": SEQUENCE,
                    "heads": HEADS,
                    "key_dim": KEY_DIM,
                    "value_dim": VALUE_DIM,
                    "dtype": "bfloat16",
                },
                "upstream_callable": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule (as called by fla.layers.mom.MoM.forward)",
                "compiled_anchor": library_plan.anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": _max_error(compiled_result[0], direct_result[0]),
                    "final_state_max_abs_error": _max_error(compiled_result[1], direct_result[1]),
                    "input_gradient_max_abs_errors": compiled_gradient_errors,
                    "tolerances": {"output_atol": 0.0, "state_atol": 0.0, "gradient_atol": 0.0, "relative_tolerance": 0.0},
                },
                "reference_equation_parity": {
                    "status": "pass",
                    "output_max_abs_error": _max_error(reference_result[0], direct_result[0]),
                    "final_state_max_abs_error": _max_error(reference_result[1], direct_result[1]),
                    "input_gradient_max_abs_errors": equation_gradient_errors,
                    "tolerances": {"output_atol": output_atol, "state_atol": state_atol, "gradient_atol": gradient_atol, "relative_tolerance": 2e-2},
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
        "--output", type=Path, default=Path("results/unified-mixer/mom-k2.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
