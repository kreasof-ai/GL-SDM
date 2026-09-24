"""Parity and paired K1 profiles against pinned FlashAttention 2 source."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import flash_attn
import torch
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_FLASH_REVISION = "1bda8f9290cd48d030f1516f0e680cd464ef3554"
RECIPES = ("mha", "mqa", "gqa", "mla_attention_core")
SEQUENCE = 64
QUERY_HEADS = 4
KEY_DIM = 32
DTYPE = torch.bfloat16


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(flash_attn)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(
            f"could not find Git root for loaded FlashAttention source {source}"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_FLASH_REVISION:
        raise RuntimeError(
            f"loaded FlashAttention source must match {EXPECTED_FLASH_REVISION}, got {revision} at {source}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("the pinned FlashAttention source checkout must be clean")
    return source, revision


def _kernel_source_hashes(source: Path) -> dict[str, str]:
    root = next(parent for parent in source.parents if (parent / ".git").exists())
    paths = (
        "csrc/flash_attn/flash_api.cpp",
        "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
        "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def _inputs(recipe: str, seed: int) -> dict[str, torch.Tensor]:
    kv_heads = {"mha": 4, "mqa": 1, "gqa": 2, "mla_attention_core": 2}[recipe]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        (1, SEQUENCE, QUERY_HEADS, KEY_DIM),
        device="cuda",
        dtype=DTYPE,
        generator=generator,
    )
    key = torch.randn(
        (1, SEQUENCE, kv_heads, KEY_DIM),
        device="cuda",
        dtype=DTYPE,
        generator=generator,
    )
    value_dim = 16 if recipe == "mla_attention_core" else KEY_DIM
    value = torch.randn(
        (1, SEQUENCE, kv_heads, KEY_DIM),
        device="cuda",
        dtype=DTYPE,
        generator=generator,
    )
    if value_dim < KEY_DIM:
        value = torch.cat(
            (value[..., :value_dim], torch.zeros_like(value[..., value_dim:])), dim=-1
        )
    return {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
    }


def _direct(inputs: dict[str, torch.Tensor], *, crop_mla_value: bool = False):
    output = flash_attn_func(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        dropout_p=0.0,
        softmax_scale=KEY_DIM**-0.5,
        causal=True,
    )
    # FLA's MLA layer pads V to qk_head_dim for FlashAttention and then drops
    # the padding, retaining only v_head_dim in the layer output.
    return output[..., :16] if crop_mla_value else output


def _compiled(plan, inputs: dict[str, torch.Tensor], *, crop_mla_value: bool = False):
    # Keep PyTorch SDPA on its FlashAttention backend for a same-family comparison.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        output = plan.execute(**inputs).output
        return output[..., :16] if crop_mla_value else output


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _forward_backward(call, inputs: dict[str, torch.Tensor]):
    for tensor in inputs.values():
        tensor.grad = None
    output = call(inputs)
    _loss(output).backward()
    return output, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs: dict[str, torch.Tensor], *, backward: bool):
    for tensor in inputs.values():
        tensor.grad = None
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    output = call(inputs)
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


def run(
    pairs: int, warmup: int, output_path: Path, only: str | None = None
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the FlashAttention unified mixer profile requires CUDA")
    source, revision = _source_identity()
    import flash_attn_2_cuda

    cases = {}
    aliases = {"mla": "mla_attention_core"}
    only = aliases.get(only, only)
    selected_recipes = (only,) if only is not None else RECIPES
    if any(recipe not in RECIPES for recipe in selected_recipes):
        raise ValueError(f"unknown recipe selection: {selected_recipes}")
    for index, recipe in enumerate(selected_recipes):
        operands = _inputs(recipe, seed=44091 + index)
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
            named_mixer_recipe(recipe),
            backend=MixerBackend.LIBRARY,
            intent=MixerIntent.TRAINING,
            dtype="bfloat16",
        )
        plan_build_ms = (time.perf_counter() - plan_started) * 1000
        crop_mla_value = recipe == "mla_attention_core"
        direct = lambda values, crop=crop_mla_value: _direct(
            values, crop_mla_value=crop
        )
        compiled = lambda values, plan=plan, crop=crop_mla_value: _compiled(
            plan, values, crop_mla_value=crop
        )

        parity_direct = _forward_backward(direct, direct_inputs)
        parity_compiled = _forward_backward(compiled, compiled_inputs)
        output_error = (
            (parity_direct[0].float() - parity_compiled[0].float()).abs().max().item()
        )
        gradient_errors = {
            name: (left.float() - right.float()).abs().max().item()
            for name, left, right in zip(
                operands, parity_direct[1], parity_compiled[1], strict=True
            )
        }
        value_tolerance = 2e-2
        gradient_tolerance = 2e-2
        torch.testing.assert_close(
            parity_compiled[0].float(),
            parity_direct[0].float(),
            atol=value_tolerance,
            rtol=value_tolerance,
        )
        for direct_grad, compiled_grad in zip(
            parity_direct[1], parity_compiled[1], strict=True
        ):
            torch.testing.assert_close(
                compiled_grad.float(),
                direct_grad.float(),
                atol=gradient_tolerance,
                rtol=gradient_tolerance,
            )

        performance = _measure_pair(
            direct, compiled, direct_inputs, compiled_inputs, pairs, warmup
        )
        cases[recipe] = {
            "architecture_ids": {
                "mha": ["arch-001", "arch-014"],
                "mqa": ["arch-002"],
                "gqa": ["arch-003"],
                "mla_attention_core": ["arch-004"],
            }[recipe],
            "semantic_scope": (
                "causal softmax attention kernel; positional transforms, projections and cache ABI excluded"
                if recipe != "mla_attention_core"
                else "MLA causal attention after latent expansion and RoPE concatenation; V is padded to qk_head_dim for FlashAttention then cropped to v_head_dim; projections and cache ABI excluded"
            ),
            "shape": {
                "batch": 1,
                "sequence": SEQUENCE,
                "query_heads": QUERY_HEADS,
                "key_value_heads": {"mha": 4, "mqa": 1, "gqa": 2, "mla_attention_core": 2}[recipe],
                "key_dim": KEY_DIM,
                "value_dim": 16 if recipe == "mla_attention_core" else KEY_DIM,
                "flash_value_dim": KEY_DIM,
                "dtype": "bfloat16",
            },
            "upstream_callable": (
                "fla.layers.mla.MultiheadLatentAttention.flash_attn_func"
                if recipe == "mla_attention_core"
                else "flash_attn.flash_attn_interface.flash_attn_func"
            ),
            "compiled_anchor": plan.anchor,
            "compiler_plan_build_ms": plan_build_ms,
            "parity": {
                "status": "pass",
                "output_max_abs_error": output_error,
                "input_gradient_max_abs_errors": gradient_errors,
                "tolerances": {
                    "output_atol": value_tolerance,
                    "output_rtol": value_tolerance,
                    "gradient_atol": gradient_tolerance,
                    "gradient_rtol": gradient_tolerance,
                },
            },
            "performance": performance,
        }

    config = {
        "recipes": selected_recipes,
        "pairs": pairs,
        "warmup": warmup,
        "dtype": "bfloat16",
        "sequence": SEQUENCE,
        "head_dim": KEY_DIM,
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K1 library plan with direct pinned FlashAttention operator",
        "upstream": {
            "repository": "https://github.com/Dao-AILab/flash-attention",
            "revision": revision,
            "module_version": getattr(flash_attn, "__version__", None),
            "distribution_version": importlib.metadata.version("flash_attn"),
            "loaded_module": str(source),
            "extension_binary": str(Path(flash_attn_2_cuda.__file__).resolve()),
            "extension_sha256": hashlib.sha256(
                Path(flash_attn_2_cuda.__file__).read_bytes()
            ).hexdigest(),
            "extension_build_scope": "locally narrowed dispatch; exact pinned BF16 causal D=32 forward/backward kernel source units only",
            "kernel_source_sha256": _kernel_source_hashes(source),
        },
        "provenance": provenance(
            "CUDA_HOME=/path/to/cuda-toolkit PYTHONPATH=/path/to/flash-attention:src python benchmarks/unified_mixer_flash.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one forward or one forward plus backward on preallocated leaf tensors",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "direct FlashAttention source is compared with the unified library plan forced to PyTorch's flash SDPA backend",
        },
        "cases": cases,
    }
    write_artifact(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/unified-mixer/flash-k1.json")
    )
    parser.add_argument("--only", choices=RECIPES)
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output, args.only)


if __name__ == "__main__":
    main()
