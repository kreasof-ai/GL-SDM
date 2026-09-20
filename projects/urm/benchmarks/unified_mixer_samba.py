"""Pinned Samba no-PE attention branch parity and paired K1 profile."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from provenance import provenance, utc_now, write_artifact
from urm.adapters.samba import load_samba_attention, samba_source_root
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer, named_mixer_recipe
from unified_mixer_factorized_attention import _clone, _max_error, _profile


def _run(pairs: int, warmup: int):
    if not torch.cuda.is_available():
        raise RuntimeError("Samba profiles require CUDA")
    source_root = samba_source_root()
    if source_root is None:
        raise RuntimeError("the pinned Samba checkout is unavailable")
    source_class, identity = load_samba_attention(source_root)
    # This is the pinned Samba_421M_nope attention branch. The smaller profile
    # fixture keeps the exact source operations while making 21 paired runs fast.
    batch, sequence, heads, head_dim = 1, 512, 12, 128
    width = heads * head_dim
    config = SimpleNamespace(
        full_per_layer=1_000_000,
        head_size=head_dim,
        n_head=heads,
        n_query_groups=heads,
        bias=False,
        sc_attn=False,
        nope=True,
        local_window=2048,
    )
    layer = source_class(config, layer_idx=1, n_embd=width).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(53053)
    base = torch.randn(
        batch, sequence, width, device="cuda", generator=generator
    ).mul_(0.1)
    recipe = named_mixer_recipe("samba_attention_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(values):
        return layer(
            values,
            rope=(None, None),
            max_seq_length=sequence,
            mask=None,
        )[0]

    def compiled(values):
        qkv = layer.attn(values)
        q_per_kv = layer.n_head // layer.n_query_groups
        qkv = qkv.view(batch, sequence, layer.n_query_groups, q_per_kv + 2, head_dim)
        query, key, value = qkv.split((q_per_kv, 1, 1), dim=-2)
        query = query.reshape(batch, sequence, heads, head_dim)
        key = key.reshape(batch, sequence, heads, head_dim)
        value = value.reshape(batch, sequence, heads, head_dim)
        output = plan.execute(query=query, key=key, value=value).output
        return layer.proj(output.reshape(batch, sequence, width))

    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_inputs), compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=0.0, rtol=0.0)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), (upstream_inputs, *layer.parameters())
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), (compiled_inputs, *layer.parameters())
    )
    gradient_errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grads, upstream_grads, strict=True)
    ]
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def clear():
        upstream_inputs.grad = None
        compiled_inputs.grad = None
        layer.zero_grad(set_to_none=True)

    performance = _profile(
        upstream, compiled, upstream_inputs, compiled_inputs, pairs, warmup, clear
    )
    return {
        "identity": identity,
        "recipe": recipe,
        "plan": plan,
        "build_ms": build_ms,
        "shape": {
            "configuration": "Samba_421M_nope",
            "batch": batch,
            "sequence": sequence,
            "heads": heads,
            "head_dim": head_dim,
            "dtype": "float32",
            "attention_layer_index": 1,
            "hidden_size": width,
            "nope": True,
            "short_convolution": False,
            "local_window": 2048,
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": gradient_errors,
        "performance": performance,
    }


def run(pairs: int, warmup: int, output_path: Path):
    result = _run(pairs, warmup)
    identity, recipe, plan, performance = (
        result["identity"], result["recipe"], result["plan"], result["performance"]
    )
    payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "purpose": "compare URM K1 against the pinned Samba_421M_nope CausalSelfAttention branch",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["source_path"],
            "kernel_source_sha256": {
                Path(identity["source_path"]).name: identity["source_sha256"]
            },
            "callable_source": identity["callable_source"],
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/samba-checkout:src python benchmarks/unified_mixer_samba.py",
            {"shape": result["shape"], "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "pinned Samba CausalSelfAttention.forward including source QKV/output projections and the compiler causal K1 call",
            "sampling": "paired interleaved calls with alternating order and synchronized wall/CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": {
            "samba_attention_core": {
                "architecture_ids": ["arch-053"],
                "semantic_scope": recipe.component_scope,
                "shape": result["shape"],
                "upstream_callable": "CausalSelfAttention.forward (Samba_421M_nope attention branch)",
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": result["build_ms"],
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": result["output_error"],
                    "input_and_parameter_gradient_max_abs_errors": result["gradient_errors"],
                    "tolerances": {
                        "output_atol": 0.0,
                        "gradient_atol": 0.0,
                        "relative_tolerance": 0.0,
                    },
                    "independent_reference": "source and compiler use the same pinned no-PE causal attention branch; K1 reference equation is exercised by the named-recipe test",
                },
                "performance": {
                    "comparison": "complete pinned source attention branch including QKV/output projections vs source projections around compiler K1",
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
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/unified-mixer/samba-k1.json")
    )
    args = parser.parse_args()
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
