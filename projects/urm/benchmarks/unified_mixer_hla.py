"""Pinned HLA paper-equation parity and second-order K2 profile."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import time
from pathlib import Path

import torch

from provenance import provenance, utc_now, write_artifact
from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe
from unified_mixer_factorized_attention import _clone, _max_error, _profile


SOURCE_REVISION = "484fef2bb40d4ed58f7656e545cf5ef64c40c962"
PAPER_SHA256 = "574242e6b6694e1ef87440f588cbe364517b76bcd9e771b2fb93b0d38feb822b"
PAPER_FILE = Path("/tmp/urm-comparator-pins/hla/HLA.pdf")


def _paper_identity():
    repository = PAPER_FILE.parent
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    checksum = hashlib.sha256(PAPER_FILE.read_bytes()).hexdigest()
    if revision != SOURCE_REVISION or dirty or checksum != PAPER_SHA256:
        raise RuntimeError("the pinned HLA project/paper identity has changed")
    return {
        "repository": str(repository),
        "revision": revision,
        "paper_path": str(PAPER_FILE.resolve()),
        "paper_sha256": checksum,
    }


def _dense_paper_equation(torch, query, key, value):
    """Dense masked equation (3.3), independently transcribed from the paper."""
    sequence = query.shape[1]
    scores = torch.matmul(
        query.transpose(1, 2), key.transpose(1, 2).transpose(-1, -2)
    )
    causal = torch.ones(sequence, sequence, dtype=torch.bool, device=query.device).tril()
    masked_affinity = scores.masked_fill(~causal, 0.0)
    second_order = torch.matmul(masked_affinity, masked_affinity.transpose(-1, -2))
    second_order = second_order.masked_fill(~causal, 0.0)
    return torch.matmul(second_order, value.transpose(1, 2)).transpose(1, 2)


def _run(pairs: int, warmup: int, sequence: int = 1024, key_dim: int = 16, heads: int = 8):
    if not torch.cuda.is_available():
        raise RuntimeError("HLA profiles require CUDA")
    identity = _paper_identity()
    batch, value_dim = 1, key_dim
    generator = torch.Generator(device="cuda").manual_seed(74074)
    base = tuple(
        torch.randn(
            batch, sequence, heads, width, device="cuda", generator=generator
        ).mul_(0.1)
        for width in (key_dim, key_dim, value_dim)
    )
    recipe = named_mixer_recipe("hla_second_order_core")
    build_start = time.perf_counter()
    plan = compile_mixer(
        recipe,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    build_ms = (time.perf_counter() - build_start) * 1000
    upstream = lambda values: _dense_paper_equation(torch, *values)
    compiled = lambda values: plan.execute(
        query=values[0], key=values[1], value=values[2]
    ).output

    upstream_inputs, compiled_inputs = _clone(base), _clone(base)
    upstream_output, compiled_output = upstream(upstream_inputs), compiled(compiled_inputs)
    torch.testing.assert_close(compiled_output, upstream_output, atol=2e-5, rtol=3e-5)
    upstream_grads = torch.autograd.grad(upstream_output.square().mean(), upstream_inputs)
    compiled_grads = torch.autograd.grad(compiled_output.square().mean(), compiled_inputs)
    gradient_errors = [
        _max_error(actual, expected)
        for actual, expected in zip(compiled_grads, upstream_grads, strict=True)
    ]
    for actual, expected in zip(compiled_grads, upstream_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)

    def clear():
        for tensor in (*upstream_inputs, *compiled_inputs):
            tensor.grad = None

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
            "key_dim": key_dim,
            "value_dim": value_dim,
            "dtype": "float32",
        },
        "output_error": _max_error(compiled_output, upstream_output),
        "gradient_errors": gradient_errors,
        "performance": performance,
    }


def run(pairs: int, warmup: int, output_path: Path, sequence: int, key_dim: int, heads: int):
    result = _run(pairs, warmup, sequence, key_dim, heads)
    identity, recipe, plan, performance = (
        result["identity"], result["recipe"], result["plan"], result["performance"]
    )
    payload = {
        "schema_version": 1,
        "generated_utc": utc_now(),
        "purpose": "compare unified HLA second-order causal K2 recurrence to the pinned paper's dense masked equation",
        "upstream": {
            "repository": identity["repository"],
            "revision": identity["revision"],
            "loaded_module": identity["paper_path"],
            "paper_sha256": identity["paper_sha256"],
        },
        "provenance": provenance(
            "python benchmarks/unified_mixer_hla.py",
            {"shape": result["shape"], "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "paper-derived dense masked second-order tensor equation (3.3) vs the exact streaming-summary K2 recurrence; upstream repository has no executable operator",
            "sampling": "paired interleaved calls with alternating order and synchronized wall/CUDA-event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead_gate_fraction": 0.10,
        },
        "cases": {
            "hla_second_order_core": {
                "architecture_ids": ["arch-074"],
                "semantic_scope": recipe.component_scope,
                "shape": result["shape"],
                "upstream_callable": "paper Equation (3.3), dense masked second-order tensor equation; independent paper-derived implementation",
                "compiled_anchor": plan.anchor,
                "compiler_plan_build_ms": result["build_ms"],
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": result["output_error"],
                    "input_and_parameter_gradient_max_abs_errors": result["gradient_errors"],
                    "tolerances": {
                        "output_atol": 2e-5,
                        "gradient_atol": 3e-5,
                        "relative_tolerance": 5e-5,
                    },
                    "independent_reference": "tests/test_unified_mixer.py compares the streaming recurrence to a dense transcription of pinned HLA paper Equation (3.3)",
                },
                "performance": {
                    "comparison": "paper-derived dense O(T^3) masked equation vs exact K2 streaming-statistic implementation",
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
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/hla-k2.json"))
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--key-dim", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    args = parser.parse_args()
    run(args.pairs, args.warmup, args.output, args.sequence, args.key_dim, args.heads)


if __name__ == "__main__":
    main()
