"""Pinned FwPKM selected-read parity and paired K1 profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import time
from pathlib import Path

import torch

from measurement import quantile
from provenance import provenance, utc_now, write_artifact
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe
from unified_mixer_factorized_attention import _clone, _max_error, _profile


SOURCE_REVISION = "b1c8e234b523d70245fa197eed4b80a985c413a8"
SOURCE_FILES = (
    "src/models/fwpkm/fwpkm.py",
    "src/models/fwpkm/pkm_legacy/xformer_embeddingbag.py",
    "src/models/fwpkm/xformer_embeddingbag_grad_wrapper.py",
)


def _source_identity(module):
    source_path = Path(module.__file__).resolve()
    repository = next(parent for parent in source_path.parents if (parent / ".git").exists())
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != SOURCE_REVISION or dirty:
        raise RuntimeError(f"expected clean FwPKM {SOURCE_REVISION}, got {revision}")
    return {
        "repository": str(repository),
        "revision": revision,
        "loaded_module": str(source_path),
        "kernel_source_sha256": {
            relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
            for relative in SOURCE_FILES
        },
    }


def _run(pairs: int, warmup: int):
    if not torch.cuda.is_available():
        raise RuntimeError("FwPKM profiles require CUDA")
    source = importlib.import_module("src.models.fwpkm.fwpkm")
    identity = _source_identity(source)
    batch, sequence, heads, key_dim, value_dim, topk, subsize = (
        4, 256, 2, 64, 64, 8, 256
    )
    layer = source.FastWeightProductKeyMemory(
        mem_k_dim=key_dim,
        mem_v_dim=value_dim,
        mem_heads=heads,
        mem_topk=topk,
        mem_n_subkeys=subsize,
        qk_score_type="idw",
        score_nonlinear="softmax",
        score_temperature=1.0,
        addr_loss=None,
    ).cuda().eval()
    layer.reset_parameters()
    generator = torch.Generator(device="cuda").manual_seed(56056)
    base = (
        torch.randn(
            batch, sequence, heads * key_dim, device="cuda", generator=generator
        ).mul_(0.1),
        layer.keys.detach().clone(),
        layer.values.detach().clone(),
    )
    recipe = named_mixer_recipe("fwpkm_memory_read_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def upstream(values):
        query, keys, memory_values = values
        return layer.retrieve_values(
            query, {"keys": keys, "values": memory_values}
        )["retireved_values"]

    def compiled(values):
        query, keys, memory_values = values
        scores, indices, *_ = layer.get_indices(
            query.reshape(batch * sequence, heads, key_dim), keys
        )
        selected_values = memory_values.index_select(
            0, indices.reshape(-1)
        ).reshape(batch * sequence, heads * topk, 1, value_dim)
        score_logits = (scores / layer.score_temperature).reshape(
            batch * sequence, heads * topk, 1, 1
        )
        unit_query = torch.ones(
            batch * sequence, 1, 1, 1, device=query.device, dtype=query.dtype
        )
        return plan.execute(
            query=unit_query, key=score_logits, value=selected_values
        ).output.reshape(batch * sequence, value_dim)

    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_inputs), compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=1e-6, rtol=1e-5)
    upstream_grads = torch.autograd.grad(
        upstream_output.square().mean(), upstream_inputs
    )
    compiled_grads = torch.autograd.grad(
        compiled_output.square().mean(), compiled_inputs
    )
    gradient_errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grads, upstream_grads, strict=True)
    ]
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

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
            "batch": batch,
            "sequence": sequence,
            "heads": heads,
            "key_dim_per_head": key_dim,
            "value_dim": value_dim,
            "topk_per_head": topk,
            "subkeys_per_half": subsize,
            "memory_slots": subsize**2,
            "qk_score_type": layer.qk_score_type,
            "score_nonlinear": layer.score_nonlinear,
            "score_temperature": layer.score_temperature,
            "dtype": "float32",
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
        "purpose": "compare the unified FwPKM K1 selected-memory-read plan against pinned FastWeightProductKeyMemory.retrieve_values",
        "upstream": identity,
        "provenance": provenance(
            "PYTHONPATH=/path/to/fwpkm-checkout:src python benchmarks/unified_mixer_fwpkm.py",
            {"shape": result["shape"], "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "pinned FwPKM product-key IDW scoring/top-k routing plus learned-value retrieval; fast-weight writes and chunk update are outside this K1 read profile",
            "sampling": "paired interleaved calls with alternating order and synchronized wall/CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": {
            "fwpkm_memory_read_core": {
                "architecture_ids": ["arch-056"],
                "semantic_scope": recipe.component_scope,
                "shape": result["shape"],
                "upstream_callable": "FastWeightProductKeyMemory.retrieve_values",
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": result["build_ms"],
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": result["output_error"],
                    "input_and_parameter_gradient_max_abs_errors": result["gradient_errors"],
                    "tolerances": {
                        "output_atol": 1e-6,
                        "gradient_atol": 2e-6,
                        "relative_tolerance": 2e-5,
                    },
                    "independent_reference": "the pinned-source test compares both dot-product and IDW product-key routing against the K1 read",
                },
                "performance": {
                    "comparison": "complete pinned retrieve_values call vs pinned route plus compiler K1 softmax/value reduction, including selected-value gather",
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
        "--output", type=Path, default=Path("results/unified-mixer/fwpkm-k1.json")
    )
    args = parser.parse_args()
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
