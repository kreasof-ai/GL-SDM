"""Parity and paired profiling for the pinned Mamba-3 SISO kernel core."""

from __future__ import annotations

import argparse
import inspect
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import mamba_ssm
import torch
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

from provenance import provenance, write_artifact
from unified_mixer_fla import (
    _clone_inputs,
    _compiled,
    _forward_backward,
    _measure_pair,
    _state_max_errors,
)
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_MAMBA_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
RECIPE = "mamba3_siso_core"


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not find Git root for loaded Mamba source {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_MAMBA_REVISION:
        raise RuntimeError(
            f"loaded Mamba source must match {EXPECTED_MAMBA_REVISION}, got {revision} at {source}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("the pinned Mamba source checkout must be clean")
    return source, revision


def _inputs(seed: int) -> dict[str, torch.Tensor | bool]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch, sequence, heads, key_dim, value_dim, angle_dim = 1, 64, 2, 16, 16, 4
    query = torch.nn.functional.normalize(
        torch.randn(
            batch, sequence, heads, key_dim,
            device="cuda", dtype=torch.bfloat16, generator=generator,
        ), dim=-1,
    )
    key = torch.nn.functional.normalize(
        torch.randn(
            batch, sequence, heads, key_dim,
            device="cuda", dtype=torch.bfloat16, generator=generator,
        ), dim=-1,
    )
    value = torch.randn(
        batch, sequence, heads, value_dim,
        device="cuda", dtype=torch.bfloat16, generator=generator,
    ) * 0.1
    dt = torch.rand(
        batch, heads, sequence, device="cuda", generator=generator
    ) * 0.08 + 0.01
    adt = -torch.rand(
        batch, heads, sequence, device="cuda", generator=generator
    ) * 4.0 * dt
    return {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
        "adt": adt.requires_grad_(),
        "dt": dt.requires_grad_(),
        "trap": torch.rand(
            batch, heads, sequence, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ).requires_grad_(),
        "query_bias": (
            torch.randn(
                heads, key_dim, device="cuda", dtype=torch.bfloat16,
                generator=generator,
            ) * 0.05
        ).requires_grad_(),
        "key_bias": (
            torch.randn(
                heads, key_dim, device="cuda", dtype=torch.bfloat16,
                generator=generator,
            ) * 0.05
        ).requires_grad_(),
        "angles": (
            torch.randn(
                batch, sequence, heads, angle_dim, device="cuda",
                dtype=torch.float32, generator=generator,
            ) * 0.1
        ).requires_grad_(),
    }


def _direct(inputs):
    output, angle_state, ssm_state, key_state, value_state = mamba3_siso_combined(
        inputs["query"].contiguous(),
        inputs["key"].contiguous(),
        inputs["value"].contiguous(),
        inputs["adt"].contiguous(),
        inputs["dt"].contiguous(),
        inputs["trap"].contiguous(),
        inputs["query_bias"].contiguous(),
        inputs["key_bias"].contiguous(),
        inputs["angles"].contiguous(),
        chunk_size=64,
        return_final_states=True,
    )
    return output, (angle_state, ssm_state, key_state, value_state)


def run(pairs: int, warmup: int, output_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Mamba-3 profile requires CUDA")
    source, revision = _source_identity()
    inputs = _inputs(seed=7745)
    direct_inputs = _clone_inputs(inputs)
    compiled_inputs = _clone_inputs(inputs)
    reference_inputs = _clone_inputs(inputs)
    plan = compile_mixer(
        named_mixer_recipe(RECIPE),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    reference_plan = compile_mixer(
        named_mixer_recipe(RECIPE),
        backend=MixerBackend.REFERENCE,
        intent=MixerIntent.TRAINING,
        dtype="bfloat16",
    )
    direct = _direct
    compiled = lambda values: _compiled(plan, values)
    reference = lambda values: _compiled(reference_plan, values)
    performance = _measure_pair(
        direct, compiled, direct_inputs, compiled_inputs, pairs, warmup
    )

    upstream_result = _forward_backward(direct, direct_inputs)
    adapter_inputs = _clone_inputs(inputs)
    adapter_result = _forward_backward(compiled, adapter_inputs)
    output_error = (
        upstream_result[0].float() - adapter_result[0].float()
    ).abs().max().item()
    state_errors = _state_max_errors(upstream_result[1], adapter_result[1])
    gradient_errors = {
        name: (
            0.0
            if left is None and right is None
            else float("inf")
            if left is None or right is None
            else (left.float() - right.float()).abs().max().item()
        )
        for name, left, right in zip(
            iter(inputs),
            upstream_result[2],
            adapter_result[2],
            strict=True,
        )
    }
    if output_error != 0 or any(error != 0 for error in state_errors + list(gradient_errors.values())):
        raise AssertionError("pinned Mamba-3 adapter differs from its direct source call")

    equation_result = _forward_backward(reference, reference_inputs)
    equation_output_error = (
        upstream_result[0].float() - equation_result[0].float()
    ).abs().max().item()
    equation_state_errors = _state_max_errors(
        upstream_result[1], equation_result[1]
    )
    equation_gradient_errors = {
        name: (
            0.0
            if left is None and right is None
            else float("inf")
            if left is None or right is None
            else (left.float() - right.float()).abs().max().item()
        )
        for name, left, right in zip(
            iter(inputs),
            upstream_result[2],
            equation_result[2],
            strict=True,
        )
    }
    torch.testing.assert_close(
        equation_result[0], upstream_result[0], atol=2e-2, rtol=2e-2
    )
    for actual, expected in zip(
        equation_result[1], upstream_result[1], strict=True
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-2, rtol=2e-2
        )
    for actual, expected in zip(
        equation_result[2], upstream_result[2], strict=True
    ):
        if actual is not None and expected is not None:
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=5e-2, rtol=5e-2
            )

    configuration = {
        "recipe": RECIPE,
        "shape": {
            "batch": 1,
            "sequence": 64,
            "heads": 2,
            "key_dim": 16,
            "value_dim": 16,
            "angle_dim": 4,
            "dtype": "bfloat16",
            "initial_state": "zero/omitted",
            "skip": "omitted",
            "gate": "omitted",
        },
        "pairs": pairs,
        "warmup": warmup,
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified mixer Mamba-3 SISO core against pinned Mamba Triton source",
        "upstream": {
            "repository": "https://github.com/state-spaces/mamba",
            "revision": revision,
            "loaded_module": str(source),
            "callable": "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined.mamba3_siso_combined",
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/mamba:src:benchmarks python benchmarks/unified_mixer_mamba3.py",
            configuration,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "Mamba-3 SISO forward or forward plus output-loss backward with final states enabled",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "overhead_gate_rule": "median compiled slowdown must not exceed 10%; speedups pass",
            "interpretation": "the library anchor calls the same pinned Triton operator; timings measure plan overhead, not a new kernel speedup",
        },
        "cases": {
            RECIPE: {
                "architecture_ids": ["arch-045"],
                "semantic_scope": "Mamba-3 SISO rotary angle accumulator and trapezoidal four-state SSM recurrence; model projections and dt/A frontend excluded",
                "shape": configuration["shape"],
                "upstream_callable": "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined.mamba3_siso_combined",
                "compiled_anchor": plan.anchor,
                "parity": {
                    "status": "pass",
                    "output_max_abs_error": output_error,
                    "final_state_max_abs_errors": state_errors,
                    "input_gradient_max_abs_errors": gradient_errors,
                    "tolerances": {"output_atol": 0.0, "state_atol": 0.0, "gradient_atol": 0.0},
                },
                "reference_equation_parity": {
                    "status": "pass",
                    "anchor": reference_plan.anchor,
                    "test": "tests/test_unified_mixer.py::test_mamba3_siso_core_matches_pinned_upstream_outputs_states_and_gradients",
                    "output_max_abs_error": equation_output_error,
                    "final_state_max_abs_errors": equation_state_errors,
                    "input_gradient_max_abs_errors": equation_gradient_errors,
                    "tolerances": {"output_atol": 2e-2, "state_atol": 2e-2, "gradient_atol": 5e-2},
                },
                "performance": performance,
            }
        },
    }
    write_artifact(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("results/unified-mixer/mamba3-k2.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
