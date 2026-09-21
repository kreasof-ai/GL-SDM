"""Qualify the native K2 diagonal recurrence against the pinned FLA HGRN operator.

This is a native-replacement qualification, not a dispatch-overhead measurement:
the ``compiled`` path executes the URM-native diagonal recurrence kernel
(``urm_native_diagonal_recurrence_v1``) and the ``direct`` path executes the
pinned upstream FLA ``fused_recurrent_hgrn`` operator. The two share no kernel.

Correctness is verified before performance: the native kernel is compared
against both the upstream callable and an independent eager oracle for outputs,
final state, and every input gradient. A numerical failure disqualifies the
workload regardless of speed. Performance is the paired median
``(native - direct) / direct`` fraction with a bootstrap confidence interval,
judged against the frozen production-matrix budget.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import fla
import torch
from fla.ops.hgrn import chunk_hgrn, fused_recurrent_hgrn

from measurement import (
    bootstrap_ci,
    capture_gpu_operating_conditions,
    quantile,
)
from provenance import provenance, write_artifact
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
# Frozen production-matrix budget for k2-diagonal-recurrence.
SLOWDOWN_BUDGET_FRACTION = 0.10
# Correctness tolerances (fp32 native kernel vs bf16-capable upstream).
OUTPUT_ATOL = 2e-5
GRADIENT_ATOL = 2e-5
RELATIVE_TOLERANCE = 2e-4


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(fla)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        # FLA installed as a wheel: record the module version, no git checkout.
        return source, getattr(fla, "__version__", "unknown")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    return source, revision


def _source_hashes(source: Path) -> dict[str, str]:
    root = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if root is None:
        return {}
    paths = ("fla/ops/hgrn/fused_recurrent.py", "fla/ops/hgrn/chunk.py")
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
        if (root / path).exists()
    }


def _oracle_hgrn(x, log_decay, initial_state):
    """Independent eager oracle for the diagonal gated recurrence (fp32)."""
    state = initial_state.float().clone()
    outputs = []
    for token in range(x.shape[1]):
        state = torch.exp(log_decay[:, token].float()) * state + x[:, token].float()
        outputs.append(state)
    output = torch.stack(outputs, dim=1)
    return output, state


def _inputs(seed: int, batch: int, sequence: int, channels: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((batch, sequence, channels), device="cuda", generator=generator) * 0.2
    log_decay = (
        -torch.rand((batch, sequence, channels), device="cuda", generator=generator) * 0.05
    )
    initial_state = torch.randn((batch, channels), device="cuda", generator=generator) * 0.1
    return {
        "x": x.requires_grad_(),
        "log_decay": log_decay.requires_grad_(),
        "initial_state": initial_state.requires_grad_(),
    }


def _direct(inputs):
    """Exact upstream reference path (sequential fused recurrence)."""
    output, state = fused_recurrent_hgrn(
        inputs["x"], inputs["log_decay"],
        initial_state=inputs["initial_state"], output_final_state=True,
    )
    return output, state


def _competitive(inputs):
    """Competitive upstream performance baseline (chunked parallel kernel).

    Per the comparison policy, the performance baseline is the fastest
    compatible upstream kernel, not a slow reference. ``chunk_hgrn`` is FLA's
    chunked parallel HGRN; it uses a different accumulation order than the exact
    sequential recurrence, so it is the performance comparator while
    ``fused_recurrent_hgrn`` remains the exact correctness comparator.
    """
    output, state = chunk_hgrn(
        inputs["x"], inputs["log_decay"],
        initial_state=inputs["initial_state"], output_final_state=True,
    )
    return output, state


def _compiled(plan, inputs):
    result = plan.execute(
        x=inputs["x"], log_decay=inputs["log_decay"], initial_state=inputs["initial_state"]
    )
    return result.output, result.final_state.squeeze(-1)


def _loss(output, state):
    return output.float().square().mean() + state.float().square().mean()


def _forward_backward(call, inputs):
    for tensor in inputs.values():
        tensor.grad = None
    output, state = call(inputs)
    _loss(output, state).backward()
    return output, state, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs, *, backward: bool, block: int = 1):
    """Time a block of ``block`` invocations, returning per-call wall and device time.

    Timing a block (sync only at the block boundaries) lets the CPU dispatch of
    later calls overlap with the GPU execution of earlier ones, which hides the
    per-call mistiming overhead that would otherwise dominate short kernels. This
    measures the true ordinary-invocation throughput; the block still includes
    every per-call dispatch and allocation. Per the contract, device (kernel) and
    wall (ordinary invocation) are reported separately.
    """
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(block):
        for tensor in inputs.values():
            tensor.grad = None
        output, state = call(inputs)
        if backward:
            _loss(output, state).backward()
    end_event.record()
    torch.cuda.synchronize()
    return (
        (time.perf_counter() - start_wall) / block,
        start_event.elapsed_time(end_event) / 1000 / block,
    )


def _summary(samples):
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _measure_pair(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup, block):
    cold_direct = _time_one(direct, direct_inputs, backward=False, block=block)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False, block=block)
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(direct, direct_inputs, backward=backward, block=block)
            _time_one(compiled, compiled_inputs, backward=backward, block=block)
    measurements = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, _ = _time_one(direct, direct_inputs, backward=backward, block=block)
                    direct_wall.append(wall)
                else:
                    wall, _ = _time_one(compiled, compiled_inputs, backward=backward, block=block)
                    compiled_wall.append(wall)
            pair_index = len(overhead)
            overhead.append(
                (compiled_wall[pair_index] - direct_wall[pair_index])
                / direct_wall[pair_index]
            )
        median_overhead = statistics.median(overhead)
        ci_lower, ci_upper = bootstrap_ci(overhead, num_resamples=2000)
        measurements[mode] = {
            "direct_wall": _summary(direct_wall),
            "compiled_wall": _summary(compiled_wall),
            "paired_native_overhead_fraction": {
                "median": median_overhead,
                "p95": quantile(overhead, 0.95),
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
                "raw_samples": overhead,
                "gate": {
                    "limit_fraction": SLOWDOWN_BUDGET_FRACTION,
                    # The confidence bound, not the point estimate, must meet budget.
                    "pass": ci_upper <= SLOWDOWN_BUDGET_FRACTION,
                },
            },
            "pair_order": order,
        }
    return {
        "first_direct_forward_call_ms": {"wall": cold_direct[0] * 1000, "device": cold_direct[1] * 1000},
        "first_compiled_forward_call_after_warmup_ms": {"wall": cold_compiled[0] * 1000, "device": cold_compiled[1] * 1000},
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def run(pairs: int, warmup: int, batch: int, sequence: int, channels: int, output_path: Path, block: int = 1) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("native K2 HGRN qualification requires CUDA")
    source, revision = _source_identity()
    operands = _inputs(seed=61227, batch=batch, sequence=sequence, channels=channels)
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe("hgrn_ssm_core"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    native_anchor = plan.anchor

    # --- Correctness first: native vs upstream AND native vs independent oracle.
    direct_result = _forward_backward(_direct, direct_inputs)
    compiled_result = _forward_backward(lambda i: _compiled(plan, i), compiled_inputs)
    oracle_output, oracle_state = _oracle_hgrn(
        operands["x"], operands["log_decay"], operands["initial_state"]
    )

    output_error_upstream = (direct_result[0] - compiled_result[0]).abs().max().item()
    state_error_upstream = (direct_result[1] - compiled_result[1]).abs().max().item()
    output_error_oracle = (oracle_output - compiled_result[0]).abs().max().item()
    state_error_oracle = (oracle_state - compiled_result[1]).abs().max().item()
    gradient_errors_upstream = {
        name: (left - right).abs().max().item()
        for name, left, right in zip(operands, direct_result[2], compiled_result[2], strict=True)
    }

    correctness_pass = True
    try:
        torch.testing.assert_close(compiled_result[0], direct_result[0], atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1], direct_result[1], atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        # Native vs independent oracle (looser: oracle is fp32 sequential).
        torch.testing.assert_close(compiled_result[0], oracle_output, atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1], oracle_state, atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        for actual, expected in zip(compiled_result[2], direct_result[2], strict=True):
            torch.testing.assert_close(actual, expected, atol=GRADIENT_ATOL, rtol=RELATIVE_TOLERANCE)
    except AssertionError:
        correctness_pass = False

    # --- Performance only after correctness, against the competitive comparator.
    competitive_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    performance = _measure_pair(_competitive, lambda i: _compiled(plan, i), competitive_inputs, compiled_inputs, pairs, warmup, block)
    fwd_gate = performance["measurements"]["forward"]["paired_native_overhead_fraction"]["gate"]["pass"]
    fb_gate = performance["measurements"]["forward_backward"]["paired_native_overhead_fraction"]["gate"]["pass"]

    if not correctness_pass:
        verdict = "numeric_failed"
    elif fwd_gate and fb_gate:
        verdict = "qualified"
    else:
        verdict = "correct_below_target"

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "qualify URM-native K2 diagonal recurrence against the pinned FLA HGRN operator (native replacement, not dispatch overhead)",
        "matrix_workload": "k2-diagonal-recurrence",
        "verdict": verdict,
        "native_anchor": native_anchor,
        "upstream": {
            "repository": "https://github.com/fla-org/flash-linear-attention",
            "expected_revision": EXPECTED_FLA_REVISION,
            "loaded_revision_or_version": revision,
            "loaded_module": str(source),
            "kernel_source_sha256": _source_hashes(source),
        },
        "provenance": provenance(
            "PYTHONPATH=src python benchmarks/qualify_native_k2_hgrn.py",
            {"recipe": "hgrn_ssm_core", "pairs": pairs, "warmup": warmup,
             "shape": [batch, sequence, channels], "dtype": "float32"},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_operating_conditions": capture_gpu_operating_conditions(),
        "methodology": {
            "comparison": "URM-native diagonal recurrence kernel vs pinned FLA HGRN; the two share no kernel",
            "correctness_comparator": "fla.ops.hgrn.fused_recurrent_hgrn (exact sequential recurrence) plus an independent eager oracle",
            "performance_comparator": "fla.ops.hgrn.chunk_hgrn (competitive chunked parallel kernel); the fastest compatible upstream kernel is the performance baseline, not a slow reference",
            "timed_work": "one native plan call or one upstream call, optionally followed by output and final-state backward",
            "sampling": "paired interleaved native/direct calls, order alternates, synchronized wall and CUDA event timing; each timed unit is a block of invocations to amortize per-call mistiming overhead",
            "warmup": warmup,
            "pairs": pairs,
            "block": block,
            "overhead": "median of per-pair (native-direct)/direct fractions with bootstrap CI",
            "slowdown_budget_fraction": SLOWDOWN_BUDGET_FRACTION,
            "gate_basis": "the 95% confidence-interval upper bound must meet the budget",
        },
        "cases": {
            "hgrn_ssm_core": {
                "semantic_scope": "diagonal gated recurrence core; projections and output projection excluded",
                "shape": {"batch": batch, "sequence": sequence, "channels": channels, "dtype": "float32"},
                "upstream_callable": "fla.ops.hgrn.fused_recurrent_hgrn",
                "performance_comparator_callable": "fla.ops.hgrn.chunk_hgrn",
                "native_anchor": native_anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass" if correctness_pass else "fail",
                    "output_max_abs_error_vs_upstream": output_error_upstream,
                    "final_state_max_abs_error_vs_upstream": state_error_upstream,
                    "output_max_abs_error_vs_oracle": output_error_oracle,
                    "final_state_max_abs_error_vs_oracle": state_error_oracle,
                    "input_gradient_max_abs_errors_vs_upstream": gradient_errors_upstream,
                    "tolerances": {
                        "output_atol": OUTPUT_ATOL,
                        "gradient_atol": GRADIENT_ATOL,
                        "relative_tolerance": RELATIVE_TOLERANCE,
                    },
                },
                "performance": performance,
            }
        },
    }
    write_artifact(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=1024)
    parser.add_argument("--block", type=int, default=10,
                        help="invocations per timed unit; amortizes per-call mistiming overhead")
    parser.add_argument("--output", type=Path, default=Path("results/qualification/native-k2-hgrn.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    payload = run(args.pairs, args.warmup, args.batch, args.sequence, args.channels, args.output, args.block)
    case = payload["cases"]["hgrn_ssm_core"]
    fwd = case["performance"]["measurements"]["forward"]["paired_native_overhead_fraction"]
    fb = case["performance"]["measurements"]["forward_backward"]["paired_native_overhead_fraction"]
    print(f"verdict: {payload['verdict']}")
    print(f"parity: {case['parity']['status']}  (output err vs upstream {case['parity']['output_max_abs_error_vs_upstream']:.2e})")
    print(f"forward  overhead: median {fwd['median']*100:+.2f}%  ci95 [{fwd['ci95_lower']*100:+.2f}%, {fwd['ci95_upper']*100:+.2f}%]  gate {'PASS' if fwd['gate']['pass'] else 'FAIL'}")
    print(f"fwd+bwd  overhead: median {fb['median']*100:+.2f}%  ci95 [{fb['ci95_lower']*100:+.2f}%, {fb['ci95_upper']*100:+.2f}%]  gate {'PASS' if fb['gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
