"""Pinned CAT FlexAttention parity and paired K1 plan profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from measurement import quantile
from provenance import provenance, write_artifact
from benchmarks.comparators.fla_gated_delta import fla_version
from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
BATCH, CHUNK_SIZE, HEADS, DIM = 1, 8, 2, 16
BLOCK_SIZE = 2 + CHUNK_SIZE
SEQUENCE = 2 * BLOCK_SIZE + 2


def _source_identity() -> tuple[object, Path, str, str]:
    source_module = importlib.import_module("fla.models.cat.modeling_cat")
    source = Path(inspect.getfile(source_module)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify FLA source checkout for {source}")
    revision = fla_version().get("source_revision")
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
            "CAT profiling requires the clean pinned FLA source revision "
            f"{EXPECTED_FLA_REVISION}; got {revision!r} with dirty={bool(dirty)}"
        )
    return (
        source_module,
        source,
        revision,
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    positions = torch.arange(SEQUENCE, device="cuda")
    mask_mod = _CAT.get_cat_mask_mod(BLOCK_SIZE)
    attention_mask = mask_mod(
        None, None, positions[:, None], positions[None, :]
    ).view(1, 1, SEQUENCE, SEQUENCE)
    return {
        name: torch.randn(
            BATCH,
            SEQUENCE,
            HEADS,
            DIM,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        ).mul_(0.1).requires_grad_()
        for name in ("query", "key", "value")
    } | {"attention_mask": attention_mask}


def _clone(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_()
        if name != "attention_mask"
        else tensor
        for name, tensor in values.items()
    }


def _direct(values: dict[str, torch.Tensor]):
    return _CAT.flex_attention_compiled(
        values["query"].transpose(1, 2),
        values["key"].transpose(1, 2),
        values["value"].transpose(1, 2),
        block_mask=_CAT_BLOCK_MASK,
    ).transpose(1, 2)


def _compiled(plan, values: dict[str, torch.Tensor]):
    return plan.execute(**values).output


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _differentiate(call, values: dict[str, torch.Tensor]):
    for name in ("query", "key", "value"):
        values[name].grad = None
    output = call(values)
    _loss(output).backward()
    return output, {name: values[name].grad for name in ("query", "key", "value")}


def _max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def _gradient_errors(actual, expected, *, atol: float, rtol: float):
    errors = {}
    for name in actual:
        errors[name] = _max_error(actual[name], expected[name])
        torch.testing.assert_close(
            actual[name], expected[name], atol=atol, rtol=rtol,
            msg=lambda message: f"{name}: {message}",
        )
    return errors


def _time_one(call, values, *, backward: bool):
    for name in ("query", "key", "value"):
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
        direct_wall, compiled_wall, direct_device, compiled_device = [], [], [], []
        overhead, order = [], []
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
            compiled_wall.append(result["compiled"][0])
            direct_device.append(result["direct"][1])
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
        raise RuntimeError("the CAT unified mixer profile requires CUDA")
    global _CAT, _CAT_BLOCK_MASK
    _CAT, source, revision, source_hash = _source_identity()
    mask_mod = _CAT.get_cat_mask_mod(BLOCK_SIZE)
    _CAT_BLOCK_MASK = _CAT.create_block_mask_compiled(
        mask_mod, B=None, H=None, Q_LEN=SEQUENCE, KV_LEN=SEQUENCE
    )
    operands = _inputs(seed=66066)
    build_start = time.perf_counter()
    recipe = named_mixer_recipe("cat_attention_core")
    library_plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    plan_build_ms = (time.perf_counter() - build_start) * 1000
    direct = _direct
    compiled = lambda values: _compiled(library_plan, values)
    reference = lambda values: _compiled(reference_plan, values)

    direct_inputs, reference_inputs, compiled_inputs = (
        _clone(operands), _clone(operands), _clone(operands)
    )
    direct_result = _differentiate(direct, direct_inputs)
    reference_result = _differentiate(reference, reference_inputs)
    compiled_result = _differentiate(compiled, compiled_inputs)
    atol = rtol = 2e-2
    torch.testing.assert_close(reference_result[0], direct_result[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(compiled_result[0], direct_result[0], atol=atol, rtol=rtol)
    reference_gradient_errors = _gradient_errors(
        reference_result[1], direct_result[1], atol=atol, rtol=rtol
    )
    compiled_gradient_errors = _gradient_errors(
        compiled_result[1], direct_result[1], atol=atol, rtol=rtol
    )

    direct_inputs, compiled_inputs = _clone(operands), _clone(operands)
    performance = _profile(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup)
    config = {
        "recipe": "cat_attention_core",
        "pairs": pairs,
        "warmup": warmup,
        "shape": [BATCH, SEQUENCE, HEADS, DIM],
        "chunk_size": CHUNK_SIZE,
        "block_size_including_special_tokens": BLOCK_SIZE,
        "dtype": "bfloat16",
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K1 CAT attention plans with pinned FLA FlexAttention",
        "upstream": {
            "repository": "https://github.com/fla-org/flash-linear-attention",
            "revision": revision,
            "module_version": None,
            "loaded_module": str(source),
            "kernel_source_sha256": {"fla/models/cat/modeling_cat.py": source_hash},
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/fla-checkout:src python benchmarks/unified_mixer_cat.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one pinned CAT flex_attention_compiled call or one K1 library plan call, optionally followed by output backward",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "upstream uses compiled FlexAttention while the unified K1 library plan uses PyTorch SDPA; CAT compression and frontend projections are outside the measured slice",
        },
        "cases": {
            "cat_attention_core": {
                "architecture_ids": ["arch-066"],
                "semantic_scope": "causal CAT mask over two decoder blocks plus the final special-token pair; compression, adaptive/separator token construction, rotary transform and Q/K/V projections excluded",
                "shape": {
                    "batch": BATCH,
                    "sequence": SEQUENCE,
                    "heads": HEADS,
                    "key_dim": DIM,
                    "value_dim": DIM,
                    "chunk_size": CHUNK_SIZE,
                    "block_size_including_special_tokens": BLOCK_SIZE,
                    "dtype": "bfloat16",
                },
                "upstream_callable": "fla.models.cat.modeling_cat.flex_attention_compiled",
                "compiled_anchor": library_plan.anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": _max_error(compiled_result[0], direct_result[0]),
                    "input_gradient_max_abs_errors": compiled_gradient_errors,
                    "tolerances": {"output_atol": atol, "gradient_atol": atol, "relative_tolerance": rtol},
                },
                "reference_equation_parity": {
                    "status": "pass",
                    "output_max_abs_error": _max_error(reference_result[0], direct_result[0]),
                    "input_gradient_max_abs_errors": reference_gradient_errors,
                    "tolerances": {"output_atol": atol, "gradient_atol": atol, "relative_tolerance": rtol},
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
        "--output", type=Path, default=Path("results/unified-mixer/cat-k1.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
