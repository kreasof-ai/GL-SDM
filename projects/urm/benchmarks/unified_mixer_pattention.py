"""Profile TokenFormer's softmax Pattention mode against K1."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from benchmarks.recipe_catalog import compile_named_recipe, load_kernel_recipe

EXPECTED_REVISION = "4d56c73f407635e62f6df16b97dc897b4477129e"
BATCH, SEQUENCE, PARAMETER_TOKENS, KEY_DIM, VALUE_DIM = 1, 128, 256, 32, 32
DTYPE = torch.bfloat16


def _source_identity() -> tuple[Path, str, str]:
    source = Path("/tmp/urm-pattention-pinned/megatron/model/tokenformer.py").resolve()
    repository = source.parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"TokenFormer source must match {EXPECTED_REVISION}, got {revision}")
    if subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ):
        raise RuntimeError("the pinned TokenFormer source checkout must be clean")
    return source, revision, hashlib.sha256(source.read_bytes()).hexdigest()


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        BATCH, SEQUENCE, KEY_DIM, device="cuda", dtype=DTYPE, generator=generator
    ) * 0.05
    key = torch.randn(
        PARAMETER_TOKENS, KEY_DIM, device="cuda", dtype=DTYPE, generator=generator
    ) * 0.05
    value = torch.randn(
        PARAMETER_TOKENS, VALUE_DIM, device="cuda", dtype=DTYPE, generator=generator
    ) * 0.002
    return {
        "query": query.requires_grad_(),
        "key_param_tokens": key.requires_grad_(),
        "value_param_tokens": value.requires_grad_(),
    }


def _clone(inputs):
    return {name: value.detach().clone().requires_grad_() for name, value in inputs.items()}


def _direct(inputs):
    # Mirrors the pinned Pattention.forward softmax branch exactly with no
    # mask or dropout. The source's exp/L1 normalization equals S * softmax.
    query = inputs["query"]
    key = inputs["key_param_tokens"]
    value = inputs["value_param_tokens"]
    scores = query @ key.transpose(-2, -1)
    unnormalized = torch.exp(scores)
    weights = unnormalized / torch.norm(unnormalized, p=1, dim=-1, keepdim=True)
    weights = weights * PARAMETER_TOKENS
    return weights @ value


def _compiled(plan, inputs):
    q = inputs["query"].unsqueeze(2)
    k = inputs["key_param_tokens"].view(1, PARAMETER_TOKENS, 1, KEY_DIM)
    v = inputs["value_param_tokens"].view(1, PARAMETER_TOKENS, 1, VALUE_DIM)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = plan.execute(query=q, key=k, value=v)["output"]
    return output.squeeze(2) * PARAMETER_TOKENS


def _result(call, inputs):
    for tensor in inputs.values():
        tensor.grad = None
    output = call(inputs)
    output.square().mean().backward()
    return output, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs, backward):
    for tensor in inputs.values():
        tensor.grad = None
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    output = call(inputs)
    if backward:
        output.square().mean().backward()
    stop.record()
    torch.cuda.synchronize()
    return time.perf_counter() - wall_start, start.elapsed_time(stop) / 1000


def _summary(values):
    return {
        "sample_count": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": quantile(values, 0.95) * 1000,
        "raw_samples_ms": [value * 1000 for value in values],
    }


def _measure(direct, compiled, inputs, pairs, warmup):
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(direct, _clone(inputs), backward)
            _time_one(compiled, _clone(inputs), backward)
    result = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        direct_samples, compiled_samples, overhead = [], [], []
        for index in range(pairs):
            for name in (("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")):
                wall, _ = _time_one(
                    direct if name == "direct" else compiled,
                    _clone(inputs),
                    backward,
                )
                (direct_samples if name == "direct" else compiled_samples).append(wall)
            overhead.append((compiled_samples[-1] - direct_samples[-1]) / direct_samples[-1])
        median = statistics.median(overhead)
        result[mode] = {
            "direct_wall": _summary(direct_samples),
            "compiled_wall": _summary(compiled_samples),
            "paired_compiled_overhead_fraction": {
                "median": median,
                "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median <= 0.10},
            },
        }
    return result


def run(pairs: int, warmup: int, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TokenFormer Pattention profile requires CUDA")
    source, revision, source_hash = _source_identity()
    inputs = _inputs(8127)
    plan_started = time.perf_counter()
    plan = compile_named_recipe(
        "pattention_core", target="library", intent="training"
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    direct_inputs, compiled_inputs = _clone(inputs), _clone(inputs)
    direct_result, direct_grads = _result(_direct, direct_inputs)
    compiled_result, compiled_grads = _result(
        lambda values: _compiled(plan, values), compiled_inputs
    )
    output_error = (direct_result - compiled_result).abs().max().item()
    gradient_errors = {
        name: (direct - compiled).abs().max().item()
        for name, direct, compiled in zip(inputs, direct_grads, compiled_grads, strict=True)
    }
    torch.testing.assert_close(compiled_result, direct_result, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(compiled_grads, direct_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    direct = _direct
    compiled = lambda values: _compiled(plan, values)
    performance = _measure(direct, compiled, inputs, pairs, warmup)
    cases = {
        "pattention_core": {
            "architecture_ids": ["arch-057"],
            "semantic_scope": "TokenFormer Pattention with norm_activation_type='softmax', parameter token construction supplied, no mask or dropout; GELU/L2 normalization and MoE routing excluded",
            "shape": {
                "batch": BATCH,
                "sequence": SEQUENCE,
                "parameter_tokens": PARAMETER_TOKENS,
                "query_dim": KEY_DIM,
                "value_dim": VALUE_DIM,
                "dtype": "bfloat16",
            },
            "upstream_callable": "megatron.model.tokenformer.Pattention.forward (softmax branch)",
            "compiled_anchor": plan.anchor,
            "compiler_plan_build_ms": plan_build_ms,
            "parity": {
                "status": "pass",
                "output_max_abs_error": output_error,
                "input_gradient_max_abs_errors": gradient_errors,
                "tolerances": {"output_atol": 2e-2, "gradient_atol": 2e-2},
            },
            "performance": {"measurements": performance},
        }
    }
    config = {
        "mode": "softmax",
        "batch": BATCH,
        "sequence": SEQUENCE,
        "parameter_tokens": PARAMETER_TOKENS,
        "query_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "dtype": "bfloat16",
        "pairs": pairs,
        "warmup": warmup,
    }
    write_artifact(
        output_path,
        {
            "schema_version": 1,
            "generated_utc": datetime.now(UTC).isoformat(),
            "purpose": "profile the softmax mode of TokenFormer Pattention against the unified K1 parameter-axis plan",
            "upstream": {
                "repository": "https://github.com/Haiyang-W/TokenFormer",
                "revision": revision,
                "loaded_module": str(source),
                "source_sha256": source_hash,
            },
            "provenance": provenance(
                "PYTHONPATH=/path/to/TokenFormer:src:benchmarks python benchmarks/unified_mixer_pattention.py",
                config,
            ),
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "methodology": {
                "timed_work": "pinned Pattention softmax equation versus compiler K1 SDPA with the architecture's S output multiplier",
                "sampling": "paired alternating direct/compiled calls with synchronized wall and CUDA event timing",
                "warmup": warmup,
                "pairs": pairs,
                "overhead_gate_fraction": 0.10,
                "interpretation": "Pattention.forward is reproduced from the pinned source's softmax branch; parameter token creation, projections and routing are excluded",
            },
            "cases": cases,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/pattention-k1.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
