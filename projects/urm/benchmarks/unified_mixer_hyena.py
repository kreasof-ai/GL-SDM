"""Pinned Hyena implicit-filter FFT convolution parity and K2 profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import subprocess
import time
from pathlib import Path

import torch

from provenance import provenance, utc_now, write_artifact
from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe
from unified_mixer_factorized_attention import _clone, _max_error, _profile


SOURCE_REVISION = "02220c69d247e5473616cd053a443ad99fd2559b"
SOURCE_FILE = "standalone_hyena.py"


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
        raise RuntimeError(f"expected clean Safari {SOURCE_REVISION}, got {revision}")
    return {
        "repository": str(repository),
        "revision": revision,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256((repository / SOURCE_FILE).read_bytes()).hexdigest(),
    }


def _run(pairs: int, warmup: int):
    if not torch.cuda.is_available():
        raise RuntimeError("Hyena profiles require CUDA")
    source = importlib.import_module("standalone_hyena")
    identity = _source_identity(source)
    batch, sequence, width, filter_order = 1, 128, 32, 16
    layer = source.HyenaOperator(
        d_model=width,
        l_max=sequence,
        order=2,
        filter_order=filter_order,
        dropout=0.0,
        filter_dropout=0.0,
    ).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(77077)
    base = torch.randn(
        batch, sequence, width, device="cuda", generator=generator
    ).mul_(0.1)
    recipe = named_mixer_recipe("hyena_fftconv_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def compiled(values):
        projected = layer.in_proj(values).transpose(1, 2)
        filtered = layer.short_filter(projected)[..., :sequence]
        x0, x1, value = filtered.split(width, dim=1)
        raw_kernel = layer.filter_fn.filter(sequence)[0]
        mixer = plan.execute(
            query=(value * x1).transpose(1, 2),
            kernel=raw_kernel.transpose(0, 1).contiguous(),
            direct=layer.filter_fn.bias,
        ).output.transpose(1, 2)
        return layer.out_proj((mixer * x0).transpose(1, 2))

    upstream = layer
    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_inputs), compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=2e-6, rtol=2e-5)
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
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

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
            "batch": batch,
            "sequence": sequence,
            "d_model": width,
            "order": 2,
            "filter_order": filter_order,
            "dtype": "float32",
            "dropout": 0.0,
            "filter_dropout": 0.0,
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
        "purpose": "compare unified Hyena K2 FFT convolution against pinned Safari HyenaOperator.forward",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["source_path"],
            "kernel_source_sha256": {SOURCE_FILE: identity["source_sha256"]},
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/safari-checkout:src python benchmarks/unified_mixer_hyena.py",
            {"shape": result["shape"], "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "pinned HyenaOperator.forward including learned implicit-filter generation, short depthwise convolution, projections, multiplicative gating and source FFT; compiled path keeps the same source frontend around the unified K2 FFT convolution",
            "sampling": "paired interleaved calls with alternating order and synchronized wall/CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": {
            "hyena_fftconv_core": {
                "architecture_ids": ["arch-077"],
                "semantic_scope": recipe.component_scope,
                "shape": result["shape"],
                "upstream_callable": "HyenaOperator.forward(order=2)",
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": result["build_ms"],
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": result["output_error"],
                    "input_and_parameter_gradient_max_abs_errors": result["gradient_errors"],
                    "tolerances": {
                        "output_atol": 2e-6,
                        "gradient_atol": 3e-6,
                        "relative_tolerance": 3e-5,
                    },
                    "independent_reference": "tests/test_unified_mixer.py compares the K2 adapter with pinned HyenaOperator.forward",
                },
                "performance": {
                    "comparison": "complete pinned HyenaOperator.forward vs source implicit filter, short convolution, gating and projections around the independent K2 FFT core",
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
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/hyena-k2.json"))
    args = parser.parse_args()
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
