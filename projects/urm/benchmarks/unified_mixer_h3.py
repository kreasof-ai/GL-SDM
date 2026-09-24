"""Pinned H3 two-stage SSM FFT convolution parity and K2 profile."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import torch

from measurement import quantile
from provenance import provenance, utc_now, write_artifact
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe
from unified_mixer_factorized_attention import _clone, _max_error, _profile


SOURCE_REVISION = "5c4d06b5795405170387c80998b58d76179a8a1a"
SOURCE_FILES = (
    "src/models/ssm/h3.py",
    "src/models/ssm/ss_kernel.py",
    "src/models/ssm/ss_kernel_diag.py",
    "src/models/ssm/ss_kernel_shift.py",
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
        raise RuntimeError(f"expected clean H3 {SOURCE_REVISION}, got {revision}")
    files = {name: hashlib.sha256((repository / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {"repository": str(repository), "revision": revision, "source_path": str(source_path), "source_sha256": files}


def _run(pairs: int, warmup: int):
    if not torch.cuda.is_available():
        raise RuntimeError("H3 profiles require CUDA")
    import importlib

    source = importlib.import_module("src.models.ssm.h3")
    identity = _source_identity(source)
    batch, sequence, width, state = 1, 128, 64, 16
    layer = source.H3(
        d_model=width,
        d_state=state,
        l_max=sequence,
        head_dim=1,
        use_fast_fftconv=False,
    ).cuda().eval()
    generator = torch.Generator(device="cuda").manual_seed(76076)
    base = torch.randn(batch, sequence, width, device="cuda", generator=generator).mul_(0.1)
    recipe = named_mixer_recipe("h3_ssm_fft_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    def compiled(values):
        qkv_input = values.reshape(batch * sequence, width).transpose(0, 1)
        q, k, v = (
            weight @ qkv_input + bias.unsqueeze(-1)
            for weight, bias in (
                (layer.q_proj.weight, layer.q_proj.bias),
                (layer.k_proj.weight, layer.k_proj.bias),
                (layer.v_proj.weight, layer.v_proj.bias),
            )
        )
        q, k, v = (
            item.reshape(width, batch, sequence).permute(1, 2, 0).unsqueeze(-1)
            for item in (q, k, v)
        )
        read_kernel = layer.kernel(L=sequence, state=None, rate=1.0)[0].squeeze(0)
        key_kernel = layer.ssm_k_kernel(
            L=sequence, state=None, rate=1.0
        )[0].squeeze(0)
        mixer = plan.execute(
            query=q,
            key=k,
            value=v,
            ssm_kernel=read_kernel,
            ssm_k_kernel=key_kernel,
            ssm_k_direct=layer.ssm_k_D,
            skip=layer.D,
        ).output.squeeze(-1)
        return layer.output_linear(mixer)

    upstream = lambda values: layer(values)
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
            "state_dim": state,
            "head_dim": 1,
            "dtype": "float32",
            "use_fast_fftconv": False,
            "kernel_mode": "diag",
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
        "purpose": "compare the unified H3 K2 two-stage FFT convolution core against pinned H3.forward",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["source_path"],
            "kernel_source_sha256": identity["source_sha256"],
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/h3-checkout:src python benchmarks/unified_mixer_h3.py",
            {"shape": result["shape"], "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "pinned H3.forward including SSM filter generation, Q/K/V and output projections, multiplicative gating, and two PyTorch FFT convolutions; compiler call uses an independent K2 FFT-convolution equation",
            "sampling": "paired interleaved calls with alternating order and synchronized wall/CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": {
            "h3_ssm_fft_core": {
                "architecture_ids": ["arch-076"],
                "semantic_scope": recipe.component_scope,
                "shape": result["shape"],
                "upstream_callable": "H3.forward(use_fast_fftconv=False, head_dim=1)",
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
                    "independent_reference": "tests/test_unified_mixer.py compares the K2 FFT equation with pinned H3.forward",
                },
                "performance": {
                    "comparison": "complete pinned H3.forward vs source projections/filter generation around the independent K2 FFT core",
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
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/h3-k2.json"))
    args = parser.parse_args()
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
