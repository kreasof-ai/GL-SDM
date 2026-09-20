"""Pinned TPA/Tucker source parity and paired K1 plan profiles."""

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
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe


def _identity(module_name: str, expected_revision: str):
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    repository = next(parent for parent in source.parents if (parent / ".git").exists())
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != expected_revision or dirty:
        raise RuntimeError(f"expected clean source revision {expected_revision}, got {revision}")
    return module, {
        "repository": str(repository),
        "revision": revision,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _clone(values):
    if isinstance(values, tuple):
        return tuple(value.detach().clone().requires_grad_() for value in values)
    return values.detach().clone().requires_grad_()


def _max_error(left, right):
    return (left.float() - right.float()).abs().max().item()


def _timing(call, values, backward):
    torch.cuda.synchronize()
    start = time.perf_counter()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    output = call(values)
    if backward:
        output.float().square().mean().backward()
    end.record()
    torch.cuda.synchronize()
    return time.perf_counter() - start, begin.elapsed_time(end) / 1000


def _summary(samples):
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _profile(upstream, compiled, upstream_inputs, compiled_inputs, pairs, warmup, clear):
    cold_upstream = _timing(upstream, upstream_inputs, False)
    cold_compiled = _timing(compiled, compiled_inputs, False)
    for _ in range(warmup):
        for backward in (False, True):
            clear()
            _timing(upstream, upstream_inputs, backward)
            clear()
            _timing(compiled, compiled_inputs, backward)
    modes = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        wall = {"upstream": [], "compiled": []}
        device = {"upstream": [], "compiled": []}
        overhead, order = [], []
        for index in range(pairs):
            names = ("upstream", "compiled") if index % 2 == 0 else ("compiled", "upstream")
            order.append("".join(names))
            results = {}
            for name in names:
                clear()
                results[name] = _timing(
                    upstream if name == "upstream" else compiled,
                    upstream_inputs if name == "upstream" else compiled_inputs,
                    backward,
                )
            for name in ("upstream", "compiled"):
                wall[name].append(results[name][0])
                device[name].append(results[name][1])
            overhead.append(
                (results["compiled"][0] - results["upstream"][0]) / results["upstream"][0]
            )
        median = statistics.median(overhead)
        modes[mode] = {
            "upstream_wall": _summary(wall["upstream"]),
            "compiled_wall": _summary(wall["compiled"]),
            "upstream_device": _summary(device["upstream"]),
            "compiled_device": _summary(device["compiled"]),
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
            "wall": cold_upstream[0] * 1000,
            "device": cold_upstream[1] * 1000,
        },
        "first_compiled_forward_call_after_upstream_ms": {
            "wall": cold_compiled[0] * 1000,
            "device": cold_compiled[1] * 1000,
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": modes,
    }


def _run_tpa(pairs, warmup):
    source, identity = _identity(
        "model.T6", "c276c80d5ad807881dedb4707d8d3c20b4e97ec6"
    )
    layer = source.CausalSelfAttention(
        source.GPTConfig(n_embd=64, n_head=2, head_dim=32, rank=4, q_rank=4)
    ).cuda().eval()
    with torch.no_grad():
        layer.c_proj.weight.copy_(torch.eye(64, device="cuda"))
    base = torch.randn(1, 64, 64, device="cuda").mul_(0.1)
    recipe = named_mixer_recipe("tpa_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(x):
        return layer(x)

    def compiled(x):
        q, k, v = layer.c_qkv(x)
        out = plan.execute(query=q, key=k, value=v).output
        return layer.c_proj(out.contiguous().view(1, 64, 64))

    upstream_x, compiled_x = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_x), compiled(compiled_x)
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grad = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grad = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grad, upstream_grad, strict=True)
    ]
    for actual, expected in zip(compiled_grad, upstream_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    performance = _profile(
        upstream,
        compiled,
        _clone(base),
        _clone(base),
        pairs,
        warmup,
        lambda: layer.zero_grad(set_to_none=True),
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "batch": 1, "sequence": 64, "heads": 2, "head_dim": 32,
            "q_rank": 4, "rank": 4, "dtype": "float32",
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": errors,
        "performance": performance,
    }


def _run_tucker(pairs, warmup):
    source, identity = _identity(
        "src.attn.triton.tucker_attn",
        "c3e3d3cec991f4303b824c7fb7cbb95e3748d5c7",
    )
    shapes = ((4, 2048, 16), (4, 2048, 16), (4, 2048, 16), (4, 16, 16))
    base = tuple(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        for shape in shapes
    )
    layer = source.FlashAttentionTucker(causal=False, attn_autotune=False).cuda()
    recipe = named_mixer_recipe("tucker_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(values):
        return layer(*values, sm_scale=16**-0.5).transpose(1, 2)

    def compiled(values):
        return plan.execute(
            query=values[0], key=values[1], value=values[2], B_pre=values[3]
        ).output

    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_inputs), compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grad = torch.autograd.grad(
        upstream_output.float().square().mean(), upstream_inputs
    )
    compiled_grad = torch.autograd.grad(
        compiled_output.float().square().mean(), compiled_inputs
    )
    errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grad, upstream_grad, strict=True)
    ]
    for actual, expected in zip(compiled_grad, upstream_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-2)

    def clear():
        for values in (upstream_inputs, compiled_inputs):
            for value in values:
                value.grad = None

    performance = _profile(
        upstream, compiled, upstream_inputs, compiled_inputs, pairs, warmup, clear
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "batch": 4, "sequence": 2048, "heads": 4, "query_rank": 16,
            "key_rank": 16, "value_rank": 16, "dtype": "bfloat16",
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": errors,
        "performance": performance,
    }


def _run_longformer(pairs, warmup):
    from urm.adapters.longformer import (
        longformer_attention_adapter,
        longformer_source_identity,
    )

    identity = longformer_source_identity()
    batch, sequence, heads, dim, window = 1, 2048, 4, 32, 32
    base = tuple(
        torch.randn(batch, sequence, heads, dim, device="cuda").mul_(0.1)
        for _ in range(3)
    )
    recipe = named_mixer_recipe("longformer_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(values):
        return longformer_attention_adapter(
            *values, attention_window=window
        )[0]

    def compiled(values):
        return plan.execute(
            query=values[0], key=values[1], value=values[2],
            attention_window=window,
        ).output

    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_inputs), compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grad = torch.autograd.grad(upstream_output.square().mean(), upstream_inputs)
    compiled_grad = torch.autograd.grad(compiled_output.square().mean(), compiled_inputs)
    errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grad, upstream_grad, strict=True)
    ]
    for actual, expected in zip(compiled_grad, upstream_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-5)

    def clear():
        for values in (upstream_inputs, compiled_inputs):
            for value in values:
                value.grad = None

    performance = _profile(
        upstream, compiled, upstream_inputs, compiled_inputs, pairs, warmup, clear
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "batch": batch, "sequence": sequence, "heads": heads,
            "head_dim": dim, "one_sided_window": window, "dtype": "float32",
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": errors,
        "performance": performance,
    }


def _run_kata(pairs, warmup):
    from urm.adapters.kata import kata_attention_adapter, kata_source_identity

    source = importlib.import_module("kata.parallel_kata_attn")
    identity = kata_source_identity()
    batch, sequence, heads, dim, value_dim, groups = 4, 1024, 8, 64, 64, 4
    base = tuple(
        torch.randn(
            batch, sequence, heads, width, device="cuda", dtype=torch.bfloat16
        ).mul_(0.1)
        for width in (dim, dim, value_dim)
    )
    recipe = named_mixer_recipe("kata_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    reference_plan = compile_mixer(
        recipe, intent=MixerIntent.TRAINING, dtype="bfloat16"
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(values):
        return source.parallel_kata_attn(
            *values, num_groups=groups, use_triton_bwd=True
        )

    def compiled(values):
        return plan.execute(
            query=values[0], key=values[1], value=values[2], num_groups=groups
        ).output

    def reference(values):
        if isinstance(values, dict):
            return reference_plan.execute(**values)
        return reference_plan.execute(
            query=values[0], key=values[1], value=values[2], num_groups=groups
        ).output

    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output = upstream(upstream_inputs)
    compiled_output = compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grad = torch.autograd.grad(
        upstream_output.float().square().mean(), upstream_inputs
    )
    compiled_grad = torch.autograd.grad(
        compiled_output.float().square().mean(), compiled_inputs
    )
    # Keep the independent dense equation check at a small representative size;
    # its explicit (B,H,T,T,M) intermediate is not the profiled implementation.
    equation_groups = 2
    equation_base = tuple(
        torch.randn(1, 64, 2, 32, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        for _ in range(3)
    )
    equation_upstream_inputs, reference_inputs = _clone(equation_base), _clone(equation_base)
    equation_upstream_output = source.parallel_kata_attn(
        *equation_upstream_inputs,
        num_groups=equation_groups,
        use_triton_bwd=True,
    )
    reference_output = reference(
        {
            "query": reference_inputs[0],
            "key": reference_inputs[1],
            "value": reference_inputs[2],
            "num_groups": equation_groups,
        }
    )
    adapter_errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grad, upstream_grad, strict=True)
    ]
    equation_grad = torch.autograd.grad(
        equation_upstream_output.float().square().mean(), equation_upstream_inputs
    )
    reference_grad = torch.autograd.grad(
        reference_output.output.float().square().mean(), reference_inputs
    )
    equation_errors = [
        _max_error(actual, expected)
        for actual, expected in zip(reference_grad, equation_grad, strict=True)
    ]
    for actual, expected in zip(compiled_grad, upstream_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        reference_output.output, equation_upstream_output, atol=2e-2, rtol=2e-2
    )
    for actual, expected in zip(reference_grad, equation_grad, strict=True):
        torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)

    def clear():
        for values in (upstream_inputs, compiled_inputs):
            for value in values:
                value.grad = None

    performance = _profile(
        upstream, compiled, upstream_inputs, compiled_inputs, pairs, warmup, clear
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "batch": batch, "sequence": sequence, "heads": heads,
            "key_dim": dim, "value_dim": value_dim, "num_groups": groups,
            "dtype": "bfloat16",
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": adapter_errors,
        "equation_output_error": _max_error(
            reference_output.output, equation_upstream_output
        ),
        "equation_gradient_errors": equation_errors,
        "performance": performance,
    }


def _run_conformer(pairs, warmup):
    source, identity = _identity(
        "espnet2.legacy.nets.pytorch_backend.transformer.attention",
        "2950325ea62c8052f448aaf11affdabe169ec8ab",
    )
    batch, sequence, heads, dim = 2, 1024, 4, 32
    width = heads * dim
    layer = source.MultiHeadedAttention(
        n_head=heads,
        n_feat=width,
        dropout_rate=0.0,
        qk_norm=False,
        use_flash_attn=False,
        causal=False,
        use_sdpa=True,
    ).cuda().eval()
    base = torch.randn(batch, sequence, width, device="cuda").mul_(0.1)
    recipe = named_mixer_recipe("conformer_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(x):
        return layer(x, x, x, mask=None)

    def compiled(x):
        query, key, value = layer.forward_qkv(x, x, x)
        output = plan.execute(
            query=query.transpose(1, 2).contiguous(),
            key=key.transpose(1, 2).contiguous(),
            value=value.transpose(1, 2).contiguous(),
        ).output
        return layer.linear_out(output.reshape(batch, sequence, width))

    upstream_x, compiled_x = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_x), compiled(compiled_x)
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grad = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grad = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grad, upstream_grad, strict=True)
    ]
    for actual, expected in zip(compiled_grad, upstream_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
    performance = _profile(
        upstream,
        compiled,
        _clone(base),
        _clone(base),
        pairs,
        warmup,
        lambda: layer.zero_grad(set_to_none=True),
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "batch": batch,
            "sequence": sequence,
            "heads": heads,
            "head_dim": dim,
            "dtype": "float32",
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": errors,
        "performance": performance,
    }


def _run_hopfield(pairs, warmup):
    source, identity = _identity(
        "hflayers.activation", "f56f929c95b77a070ae675ea4f56b6d54d36e730"
    )
    source_dir = Path(source.__file__).resolve().parent
    identity["source_sha256"] = {
        "activation.py": hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
        "functional.py": hashlib.sha256(
            (source_dir / "functional.py").read_bytes()
        ).hexdigest(),
    }
    batch, sequence, heads, dim = 2, 256, 4, 32
    width = heads * dim
    layer = source.HopfieldCore(
        embed_dim=width, num_heads=heads, dropout=0.0, bias=True
    ).cuda().eval()
    base = torch.randn(batch, sequence, width, device="cuda").mul_(0.1)
    recipe = named_mixer_recipe("hopfield_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(x):
        sequence_first = x.transpose(0, 1)
        output = layer(
            sequence_first,
            sequence_first,
            sequence_first,
            need_weights=False,
            scaling=1.0,
            update_steps_max=0,
        )[0]
        return output.transpose(0, 1).contiguous()

    def compiled(x):
        q, k, v = torch.nn.functional.linear(
            x, layer.in_proj_weight, layer.in_proj_bias
        ).chunk(3, dim=-1)
        q, k, v = (
            item.reshape(batch, sequence, heads, dim)
            for item in (q, k, v)
        )
        output = plan.execute(query=q, key=k, value=v).output
        return layer.out_proj(output.reshape(batch, sequence, width))

    upstream_x, compiled_x = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_x), compiled(compiled_x)
    torch.testing.assert_close(compiled_output, upstream_output, atol=1e-6, rtol=1e-6)
    upstream_grad = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_x, *layer.parameters())
    )
    compiled_grad = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_x, *layer.parameters())
    )
    errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grad, upstream_grad, strict=True)
    ]
    for actual, expected in zip(compiled_grad, upstream_grad, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    performance = _profile(
        upstream,
        compiled,
        _clone(base),
        _clone(base),
        pairs,
        warmup,
        lambda: layer.zero_grad(set_to_none=True),
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "batch": batch,
            "sequence": sequence,
            "heads": heads,
            "head_dim": dim,
            "dtype": "float32",
            "scaling": 1.0,
            "update_steps_max": 0,
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": errors,
        "performance": performance,
    }


def run(case: str, pairs: int, warmup: int, output_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("factorized attention profiles require CUDA")
    result = {
        "tpa": _run_tpa,
        "tucker": _run_tucker,
        "longformer": _run_longformer,
        "kata": _run_kata,
        "conformer": _run_conformer,
        "hopfield": _run_hopfield,
    }[case](pairs, warmup)
    identity, recipe, plan, performance = (
        result["identity"], result["recipe"], result["plan"], result["performance"]
    )
    case_name = {
        "tpa": "tpa_attention_core",
        "tucker": "tucker_attention_core",
        "longformer": "longformer_attention_core",
        "kata": "kata_attention_core",
        "conformer": "conformer_attention_core",
        "hopfield": "hopfield_attention_core",
    }[case]
    architecture_id = {
        "tpa": "arch-069", "tucker": "arch-070", "longformer": "arch-071",
        "kata": "arch-073",
        "conformer": "arch-075",
        "hopfield": "arch-078",
    }[case]
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": f"compare the unified {case.upper()} K1 plan against its pinned source attention callable",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["source_path"],
            "kernel_source_sha256": {
                **(
                    identity["source_sha256"]
                    if isinstance(identity["source_sha256"], dict)
                    else {Path(identity["source_path"]).name: identity["source_sha256"]}
                )
            },
        },
        "provenance": provenance(
            f"PYTHONPATH=/path/to/{case}-checkout:src python benchmarks/unified_mixer_factorized_attention.py --case {case}",
            {"case": case, "shape": result["shape"], "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": {
                "tpa": "complete TPA CausalSelfAttention including CP Q/K/V factorization, RoPE, attention and output projection",
                "tucker": "complete pinned Tucker fused factorized-query attention callable",
                "longformer": "pinned local-window sliding-chunks QK/PV attention core",
                "kata": "pinned KATA grouped squared-dot normalized-positive attention including Triton forward and backward",
                "conformer": "complete pinned Conformer MultiHeadedAttention call including Q/K/V and output projections",
                "hopfield": "pinned HopfieldCore single association update including Q/K/V and output projections",
            }[case],
            "sampling": "paired interleaved calls with alternating order and synchronized wall/CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": {
            case_name: {
                "architecture_ids": [architecture_id],
                "semantic_scope": recipe.component_scope,
                "shape": result["shape"],
                "upstream_callable": identity["source_path"],
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": result["build_ms"],
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": result["output_error"],
                    "input_and_parameter_gradient_max_abs_errors": result["gradient_errors"],
                    "tolerances": {
                        "output_atol": 1e-6 if case == "hopfield" else 0.0,
                        "gradient_atol": (
                            2e-6 if case == "hopfield"
                            else 1e-6 if case == "tucker"
                            else 1e-12 if case == "longformer"
                            else 0.0
                        ),
                        "relative_tolerance": (
                            2e-5 if case == "hopfield"
                            else 1e-2 if case == "tucker"
                            else 1e-5 if case == "longformer"
                            else 0.0
                        ),
                    },
                    "independent_reference": "tests/test_unified_mixer.py verifies reference K1 equation parity",
                    **(
                        {
                            "reference_equation_output_max_abs_error": result[
                                "equation_output_error"
                            ],
                            "reference_equation_gradient_max_abs_errors": result[
                                "equation_gradient_errors"
                            ],
                            "reference_equation_tolerance": {
                                "output_atol": 2e-2,
                                "gradient_atol": 2e-2,
                                "relative_tolerance": 2e-2,
                            },
                        }
                        if case == "kata"
                        else {}
                    ),
                },
                "performance": {
                    "comparison": "full source attention callable vs compiler library plan including required factorization/output stages",
                    "measurements": performance["measurements"],
                    "cold_calls": {
                        "upstream_forward": performance["first_upstream_forward_call_ms"],
                        "compiled_forward_after_upstream": performance[
                            "first_compiled_forward_call_after_upstream_ms"
                        ],
                    },
                },
            }
        },
    }
    write_artifact(output_path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=(
            "tpa", "tucker", "longformer", "kata", "conformer", "hopfield"
        ),
        required=True,
    )
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(f"results/unified-mixer/{args.case}-k1.json")
    run(args.case, args.pairs, args.warmup, output)


if __name__ == "__main__":
    main()
