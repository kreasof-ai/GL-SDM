"""Qualify the native K2 matrix-state gated-delta recurrence against FLA.

This is a native-replacement qualification, not a dispatch-overhead measurement:
the ``compiled`` path executes the URM-native matrix-state recurrence kernel
(``urm_native_matrix_state_recurrence_v1``) and the ``direct`` path executes the
pinned upstream FLA gated-delta operator. The two share no kernel.

Correctness is verified before performance: the native kernel is compared
against both the exact upstream callable (``fused_recurrent_gated_delta_rule``)
and an independent eager oracle for outputs, final state, and every input
gradient. A numerical failure disqualifies the workload regardless of speed.
Performance is measured against the competitive upstream comparator
(``chunk_gated_delta_rule``, FLA's chunked parallel kernel), reported as the
paired median ``(native - direct) / direct`` fraction with a bootstrap
confidence interval, judged against the frozen production-matrix budget.
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
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule

from measurement import (
    bootstrap_ci,
    capture_gpu_operating_conditions,
    quantile,
)
from provenance import provenance, write_artifact
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
# Frozen production-matrix budget for k2-gated-delta-recurrence.
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
    paths = (
        "fla/ops/gated_delta_rule/fused_recurrent.py",
        "fla/ops/gated_delta_rule/chunk.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in paths
        if (root / path).exists()
    }


def _oracle_gated_delta(q, k, v, g, beta, scale, initial_state):
    """Independent eager oracle for the gated-delta matrix recurrence (fp32).

    Unlike FLA's exact ``fused_recurrent_gated_delta_rule`` - which has no
    backward pass - this oracle is pure differentiable PyTorch, so it provides
    the independent gradient reference the exact upstream cannot.
    """
    state = initial_state.float().clone()  # [B,H,K,V]
    outputs = []
    for token in range(q.shape[1]):
        decay = torch.exp(g[:, token].float())[:, :, None, None]  # head decay
        state = decay * state
        retrieved = (state * k[:, token].float()[:, :, :, None]).sum(dim=2)
        delta = beta[:, token].float()[:, :, None] * (v[:, token].float() - retrieved)
        state = state + k[:, token].float()[:, :, :, None] * delta[:, :, None, :]
        outputs.append(scale * (state * q[:, token].float()[:, :, :, None]).sum(dim=2))
    output = torch.stack(outputs, dim=1)
    return output, state


def _inputs(seed, batch, sequence, heads, key_dim, value_dim):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, sequence, heads, key_dim), device="cuda", generator=generator)
    k = torch.nn.functional.normalize(
        torch.randn((batch, sequence, heads, key_dim), device="cuda", generator=generator),
        dim=-1,
    )
    v = torch.randn((batch, sequence, heads, value_dim), device="cuda", generator=generator)
    g = -torch.rand((batch, sequence, heads), device="cuda", generator=generator) * 0.3
    beta = torch.rand((batch, sequence, heads), device="cuda", generator=generator)
    initial_state = torch.randn(
        (batch, heads, key_dim, value_dim), device="cuda", generator=generator
    ) * 0.1
    return {
        "q": q.requires_grad_(),
        "k": k.requires_grad_(),
        "v": v.requires_grad_(),
        "g": g.requires_grad_(),
        "beta": beta.requires_grad_(),
        "initial_state": initial_state.requires_grad_(),
    }


def _direct(inputs, scale):
    """Exact upstream reference path (sequential fused recurrence)."""
    output, state = fused_recurrent_gated_delta_rule(
        inputs["q"], inputs["k"], inputs["v"], g=inputs["g"], beta=inputs["beta"],
        scale=scale, initial_state=inputs["initial_state"], output_final_state=True,
    )
    return output, state


def _competitive(inputs, scale):
    """Competitive upstream performance baseline (chunked parallel kernel).

    Per the comparison policy, the performance baseline is the fastest
    compatible upstream kernel, not a slow reference. ``chunk_gated_delta_rule``
    is FLA's chunked parallel gated-delta kernel; it uses a different
    accumulation order than the exact sequential recurrence, so it is the
    performance comparator while ``fused_recurrent_gated_delta_rule`` remains
    the exact correctness comparator.
    """
    output, state = chunk_gated_delta_rule(
        inputs["q"], inputs["k"], inputs["v"], inputs["g"], inputs["beta"],
        scale=scale, initial_state=inputs["initial_state"], output_final_state=True,
    )
    return output, state


def _compiled(plan, inputs):
    result = plan.execute(
        query=inputs["q"], key=inputs["k"], value=inputs["v"],
        log_decay=inputs["g"], beta=inputs["beta"],
        initial_state=inputs["initial_state"],
    )
    return result.output, result.final_state


def _loss(output, state):
    return output.float().square().mean() + state.float().square().mean()


def _forward_backward(call, inputs):
    for tensor in inputs.values():
        tensor.grad = None
    output, state = call(inputs)
    _loss(output, state).backward()
    return output, state, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs, *, backward: bool, block: int = 1):
    """Time a block of ``block`` invocations (sync at block boundaries) so CPU
    dispatch overlaps GPU execution, hiding per-call mistiming overhead."""
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


def run(pairs, warmup, batch, sequence, heads, key_dim, value_dim, output_path, block=1):
    if not torch.cuda.is_available():
        raise RuntimeError("native K2 gated-delta qualification requires CUDA")
    source, revision = _source_identity()
    # Compare equivalent work: the URM native/reference gated-delta recurrence
    # applies read scale 1.0 (its plain-recurrence convention), so the upstream
    # comparator is run with scale=1.0 as well. FLA's default of key_dim**-0.5 is
    # a different equation; matching the scale keeps the comparison equivalent.
    scale = 1.0
    operands = _inputs(90210, batch, sequence, heads, key_dim, value_dim)
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe("gated_delta_net"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    native_anchor = plan.anchor

    # --- Correctness first. Forward output/state are compared against the exact
    # upstream (fused_recurrent) AND the independent oracle. Gradients are
    # compared against the independent oracle, because the exact upstream
    # (fused_recurrent_gated_delta_rule) does not implement a backward pass.
    direct_output, direct_state = _direct(
        {n: t.detach() for n, t in operands.items()}, scale
    )
    oracle_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    oracle_result = _forward_backward(
        lambda i: _oracle_gated_delta(i["q"], i["k"], i["v"], i["g"], i["beta"], scale, i["initial_state"]),
        oracle_inputs,
    )
    compiled_result = _forward_backward(lambda i: _compiled(plan, i), compiled_inputs)

    output_error_upstream = (direct_output - compiled_result[0]).abs().max().item()
    state_error_upstream = (direct_state - compiled_result[1]).abs().max().item()
    output_error_oracle = (oracle_result[0] - compiled_result[0]).abs().max().item()
    state_error_oracle = (oracle_result[1] - compiled_result[1]).abs().max().item()
    gradient_errors_oracle = {
        name: (left - right).abs().max().item()
        for name, left, right in zip(operands, oracle_result[2], compiled_result[2], strict=True)
    }

    correctness_pass = True
    try:
        # Forward vs exact upstream.
        torch.testing.assert_close(compiled_result[0], direct_output, atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1], direct_state, atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        # Forward vs independent oracle.
        torch.testing.assert_close(compiled_result[0], oracle_result[0], atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1], oracle_result[1], atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        # Gradients vs independent oracle (exact upstream has no backward).
        for actual, expected in zip(compiled_result[2], oracle_result[2], strict=True):
            torch.testing.assert_close(actual, expected, atol=GRADIENT_ATOL, rtol=RELATIVE_TOLERANCE)
    except AssertionError:
        correctness_pass = False

    # --- Performance only after correctness, against the competitive comparator.
    competitive_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    performance = _measure_pair(
        lambda i: _competitive(i, scale), lambda i: _compiled(plan, i),
        competitive_inputs, compiled_inputs, pairs, warmup, block,
    )
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
        "purpose": "qualify URM-native K2 matrix-state gated-delta recurrence against the pinned FLA gated-delta operator (native replacement, not dispatch overhead)",
        "matrix_workload": "k2-gated-delta-recurrence",
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
            "PYTHONPATH=src python benchmarks/qualify_native_k2_gated_delta.py",
            {"recipe": "gated_delta_net", "pairs": pairs, "warmup": warmup,
             "shape": [batch, sequence, heads, key_dim, value_dim], "dtype": "float32"},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_operating_conditions": capture_gpu_operating_conditions(),
        "methodology": {
            "comparison": "URM-native matrix-state recurrence kernel vs pinned FLA gated-delta; the two share no kernel",
            "correctness_comparator": "forward output/state vs fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule (exact sequential recurrence) and an independent eager oracle; gradients vs the oracle because the exact upstream implements no backward pass",
            "performance_comparator": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule (competitive chunked parallel kernel); the fastest compatible upstream kernel is the performance baseline, not a slow reference",
            "timed_work": "one native plan call or one upstream call, optionally followed by output and final-state backward",
            "sampling": "paired interleaved native/direct calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (native-direct)/direct fractions with bootstrap CI",
            "slowdown_budget_fraction": SLOWDOWN_BUDGET_FRACTION,
            "gate_basis": "the 95% confidence-interval upper bound must meet the budget",
        },
        "cases": {
            "gated_delta_net": {
                "semantic_scope": "matrix-state gated-delta recurrence core; projections and output projection excluded",
                "shape": {"batch": batch, "sequence": sequence, "heads": heads,
                          "key_dim": key_dim, "value_dim": value_dim, "dtype": "float32"},
                "upstream_callable": "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule",
                "performance_comparator_callable": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
                "native_anchor": native_anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass" if correctness_pass else "fail",
                    "output_max_abs_error_vs_upstream": output_error_upstream,
                    "final_state_max_abs_error_vs_upstream": state_error_upstream,
                    "output_max_abs_error_vs_oracle": output_error_oracle,
                    "final_state_max_abs_error_vs_oracle": state_error_oracle,
                    "input_gradient_max_abs_errors_vs_oracle": gradient_errors_oracle,
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
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--block", type=int, default=10,
                        help="invocations per timed unit; amortizes per-call mistiming overhead")
    parser.add_argument("--output", type=Path, default=Path("results/qualification/native-k2-gated-delta.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    payload = run(args.pairs, args.warmup, args.batch, args.sequence, args.heads, args.key_dim, args.value_dim, args.output, args.block)
    case = payload["cases"]["gated_delta_net"]
    fwd = case["performance"]["measurements"]["forward"]["paired_native_overhead_fraction"]
    fb = case["performance"]["measurements"]["forward_backward"]["paired_native_overhead_fraction"]
    print(f"verdict: {payload['verdict']}")
    print(f"parity: {case['parity']['status']}  (output err vs upstream {case['parity']['output_max_abs_error_vs_upstream']:.2e})")
    print(f"forward  overhead: median {fwd['median']*100:+.2f}%  ci95 [{fwd['ci95_lower']*100:+.2f}%, {fwd['ci95_upper']*100:+.2f}%]  gate {'PASS' if fwd['gate']['pass'] else 'FAIL'}")
    print(f"fwd+bwd  overhead: median {fb['median']*100:+.2f}%  ci95 [{fb['ci95_lower']*100:+.2f}%, {fb['ci95_upper']*100:+.2f}%]  gate {'PASS' if fb['gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
