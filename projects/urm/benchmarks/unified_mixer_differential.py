"""Pinned Differential Transformer V1 parity and paired K1 plan profile."""

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
from urm.compiler.unified_mixer import (
    MixerBackend,
    MixerIntent,
    compile_mixer,
    named_mixer_recipe,
)

EXPECTED_REVISION = "50224e387211f15ac6a3b2685730b9a0c850f145"
BATCH, SEQUENCE, HEADS, DIM = 1, 64, 2, 16
WIDTH = 2 * HEADS * DIM


class SliceProjection(torch.nn.Module):
    def __init__(self, start: int, width: int):
        super().__init__()
        self.start, self.width = start, width

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., self.start : self.start + self.width]


def _source():
    module = importlib.import_module("multihead_diffattn")
    source = Path(inspect.getfile(module)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify Microsoft source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git", "-C", str(repository), "status", "--porcelain",
            "--untracked-files=no",
        ], text=True,
    )
    if revision != EXPECTED_REVISION or dirty:
        raise RuntimeError(
            f"Differential profile requires clean {EXPECTED_REVISION}; "
            f"got {revision} with dirty={bool(dirty)}"
        )
    return module, source, revision, hashlib.sha256(source.read_bytes()).hexdigest()


def _make_source_layer(module):
    # Zero RoPE is the frontend boundary condition for this K1-core comparison.
    module.apply_rotary_emb = lambda value, *args, **kwargs: value
    torch.manual_seed(67067)
    layer = module.MultiheadDiffAttn(
        embed_dim=WIDTH, depth=2, num_heads=HEADS
    ).to("cuda").eval()
    layer.q_proj = SliceProjection(0, WIDTH)
    layer.k_proj = SliceProjection(WIDTH, WIDTH)
    layer.v_proj = SliceProjection(2 * WIDTH, WIDTH)
    layer.subln = torch.nn.Identity()
    layer.out_proj = torch.nn.Identity()
    return layer


def _inputs(seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        name: torch.randn(
            BATCH, SEQUENCE, WIDTH, device="cuda", dtype=torch.float32,
            generator=generator,
        ).mul_(0.1).requires_grad_()
        for name in ("query_raw", "key_raw", "value_raw")
    }


def _clone(values):
    return {name: tensor.detach().clone().requires_grad_() for name, tensor in values.items()}


def _lambda(layer):
    first = torch.exp(torch.sum(layer.lambda_q1 * layer.lambda_k1).float())
    second = torch.exp(torch.sum(layer.lambda_q2 * layer.lambda_k2).float())
    return (first - second + layer.lambda_init).detach()


def _branches(raw, coefficient):
    query = raw["query_raw"].view(BATCH, SEQUENCE, HEADS, 2, DIM)
    key = raw["key_raw"].view(BATCH, SEQUENCE, HEADS, 2, DIM)
    return {
        "query_a": query[..., 0, :],
        "query_b": query[..., 1, :],
        "key_a": key[..., 0, :],
        "key_b": key[..., 1, :],
        "value": raw["value_raw"].view(BATCH, SEQUENCE, HEADS, 2 * DIM),
        "lambda_weight": coefficient,
    }


def _upstream(layer, raw):
    joined = torch.cat((raw["query_raw"], raw["key_raw"], raw["value_raw"]), dim=-1)
    return layer(joined, (None, None)) / (1.0 - layer.lambda_init)


def _compiled(plan, raw, coefficient):
    return plan.execute(**_branches(raw, coefficient)).output.flatten(2)


def _loss(output):
    return output.square().mean()


def _differentiate(call, values):
    for tensor in values.values():
        tensor.grad = None
    output = call(values)
    _loss(output).backward()
    return output, {name: tensor.grad for name, tensor in values.items()}


def _max_error(left, right):
    return (left.float() - right.float()).abs().max().item()


def _time(call, values):
    for tensor in values.values():
        tensor.grad = None
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    output = call(values)
    _loss(output).backward()
    end.record()
    torch.cuda.synchronize()
    return time.perf_counter() - start_wall, begin.elapsed_time(end) / 1000


def _summary(values):
    return {
        "sample_count": len(values), "median_ms": statistics.median(values) * 1000,
        "p95_ms": quantile(values, 0.95) * 1000,
        "raw_samples_ms": [value * 1000 for value in values],
    }


def _profile(upstream, compiled, source_inputs, compiled_inputs, pairs, warmup):
    measurements = {}
    for _ in range(warmup):
        _time(upstream, source_inputs)
        _time(compiled, compiled_inputs)
    for mode in ("forward", "forward_backward"):
        if mode == "forward":
            def time_mode(call, data):
                for tensor in data.values(): tensor.grad = None
                torch.cuda.synchronize()
                start = time.perf_counter()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(); call(data); end.record(); torch.cuda.synchronize()
                return time.perf_counter() - start, begin.elapsed_time(end) / 1000
        else:
            time_mode = _time
        samples = {"upstream": [], "compiled": []}
        device = {"upstream": [], "compiled": []}
        overhead, order = [], []
        for index in range(pairs):
            first, second = (
                ("upstream", "compiled") if index % 2 == 0 else ("compiled", "upstream")
            )
            order.append(first + second)
            pair = {}
            for name in (first, second):
                pair[name] = time_mode(
                    upstream if name == "upstream" else compiled,
                    source_inputs if name == "upstream" else compiled_inputs,
                )
            for name in samples:
                samples[name].append(pair[name][0])
                device[name].append(pair[name][1])
            overhead.append((pair["compiled"][0] - pair["upstream"][0]) / pair["upstream"][0])
        median = statistics.median(overhead)
        measurements[mode] = {
            "upstream_wall": _summary(samples["upstream"]),
            "compiled_wall": _summary(samples["compiled"]),
            "upstream_device": _summary(device["upstream"]),
            "compiled_device": _summary(device["compiled"]),
            "paired_compiled_overhead_fraction": {
                "median": median, "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median <= 0.10},
            },
            "pair_order": order,
        }
    return {"warmup_calls_per_backend_per_mode": warmup,
            "paired_samples_per_mode": pairs, "measurements": measurements}


def run(pairs: int, warmup: int, output_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("the Differential Attention unified mixer profile requires CUDA")
    source_module, source, revision, source_hash = _source()
    layer = _make_source_layer(source_module)
    coefficient = _lambda(layer)
    values = _inputs(67067)
    build_start = time.perf_counter()
    recipe = named_mixer_recipe("differential_attention_core")
    library_plan = compile_mixer(
        recipe, backend=MixerBackend.LIBRARY, intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    reference_plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, dtype="float32")
    plan_build_ms = (time.perf_counter() - build_start) * 1000
    upstream = lambda raw: _upstream(layer, raw)
    compiled = lambda raw: _compiled(library_plan, raw, coefficient)
    reference = lambda raw: _compiled(reference_plan, raw, coefficient)
    upstream_raw, reference_raw, compiled_raw = _clone(values), _clone(values), _clone(values)
    upstream_result = _differentiate(upstream, upstream_raw)
    reference_result = _differentiate(reference, reference_raw)
    compiled_result = _differentiate(compiled, compiled_raw)
    atol, rtol = 2e-6, 2e-5
    torch.testing.assert_close(reference_result[0], upstream_result[0], atol=atol, rtol=rtol)
    torch.testing.assert_close(compiled_result[0], upstream_result[0], atol=atol, rtol=rtol)
    reference_errors, compiled_errors = {}, {}
    for name in values:
        reference_errors[name] = _max_error(reference_result[1][name], upstream_result[1][name])
        compiled_errors[name] = _max_error(compiled_result[1][name], upstream_result[1][name])
        torch.testing.assert_close(reference_result[1][name], upstream_result[1][name], atol=atol, rtol=rtol)
        torch.testing.assert_close(compiled_result[1][name], upstream_result[1][name], atol=atol, rtol=rtol)
    profile = _profile(upstream, compiled, _clone(values), _clone(values), pairs, warmup)
    config = {"recipe": "differential_attention_core", "pairs": pairs, "warmup": warmup,
              "shape": [BATCH, SEQUENCE, HEADS, DIM], "dtype": "float32",
              "variant": "microsoft_differential_transformer_v1", "source_depth": 2,
              "rotary": "identity boundary condition", "normalization": "disabled at the K1 slice"}
    payload = {
        "schema_version": 1, "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified Differential Transformer V1 K1 plans with the pinned Microsoft V1 module",
        "upstream": {"repository": "https://github.com/microsoft/unilm", "revision": revision,
                     "loaded_module": str(source),
                     "kernel_source_sha256": {"Diff-Transformer/multihead_diffattn.py": source_hash}},
        "provenance": provenance(
            "PYTHONPATH=/path/to/unilm/Diff-Transformer:/path/to/flash-attention:src python benchmarks/unified_mixer_differential.py",
            config,
        ),
        "hardware": {"gpu": torch.cuda.get_device_name(0),
                     "compute_capability": list(torch.cuda.get_device_capability(0)),
                     "torch": torch.__version__, "cuda": torch.version.cuda},
        "methodology": {
            "timed_work": "pinned V1 source forward with preprojected branch slices and output norm/projection disabled, or the unified two-SDPA K1 plan, with optional output backward",
            "sampling": "paired interleaved calls with alternating order; synchronized wall and CUDA event timing",
            "warmup": warmup, "pairs": pairs,
            "overhead": "median of per-pair (compiled-upstream)/upstream fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "V1 source forward is the upstream implementation. RoPE is set to identity and V1 post-normalization/output projection are disabled at the declared K1 slice boundary.",
        },
        "cases": {"differential_attention_core": {
            "architecture_ids": ["arch-067"],
            "semantic_scope": "pinned Differential Transformer V1: two causal softmax reductions, lambda_full subtraction; preprojected branch inputs, identity RoPE, and no per-head RMSNorm/output projection",
            "shape": {"batch": BATCH, "sequence": SEQUENCE, "heads": HEADS,
                      "key_dim": DIM, "value_dim": 2 * DIM, "dtype": "float32",
                      "source_depth": 2},
            "upstream_callable": "Diff-Transformer.multihead_diffattn.MultiheadDiffAttn.forward",
            "compiled_anchor": library_plan.anchor, "compiler_plan_build_ms": plan_build_ms,
            "parity": {"status": "pass",
                       "output_max_abs_error": _max_error(compiled_result[0], upstream_result[0]),
                       "input_gradient_max_abs_errors": compiled_errors,
                       "tolerances": {"output_atol": atol, "gradient_atol": atol, "relative_tolerance": rtol}},
            "reference_equation_parity": {"status": "pass",
                       "output_max_abs_error": _max_error(reference_result[0], upstream_result[0]),
                       "input_gradient_max_abs_errors": reference_errors,
                       "tolerances": {"output_atol": atol, "gradient_atol": atol, "relative_tolerance": rtol}},
            "performance": profile,
        }},
    }
    write_artifact(output_path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path("results/unified-mixer/differential-k1.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
