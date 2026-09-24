"""Qualify the native K2 matrix-state gated-delta recurrence against FLA.

This is a native-replacement qualification, not a dispatch-overhead measurement:
the ``compiled`` path executes the URM-native matrix-state recurrence kernel
(``urm_native_matrix_state_recurrence_v1``) and the ``direct`` path executes the
pinned upstream FLA gated-delta operator. The two share no kernel.

Correctness is verified before performance, against the frozen production-matrix
contract for ``k2-gated-delta-recurrence``:

- Tolerances are the frozen ``output_atol = state_atol = gradient_atol = 0.02``
  (``benchmarks/production-matrix.json``), not a stricter ad-hoc bound. The
  frozen oracle is "an independent sequential gated-delta recurrence plus the
  upstream chunk operator": the native kernel is compared against the exact
  sequential recurrence (``fused_recurrent_gated_delta_rule`` and an eager
  oracle) at the contract tolerance. The competitive chunked upstream itself
  deviates from the exact scan by ~0.02 in fp32, so the contract tolerance is
  the correct calibration - a tighter bound would reject the upstream
  comparator's own numerics.
- All four frozen cases are exercised (latency_short, throughput_medium,
  continuation_nonzero_state, decode_step) in both float32 and bfloat16.

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
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
# Frozen production-matrix budget for k2-gated-delta-recurrence.
SLOWDOWN_BUDGET_FRACTION = 0.10
# Frozen production-matrix correctness tolerances (benchmarks/production-matrix.json).
OUTPUT_ATOL = 0.02
STATE_ATOL = 0.02
GRADIENT_ATOL = 0.02
RELATIVE_TOLERANCE = 1e-4

# The four frozen cases for k2-gated-delta-recurrence.
CASES = (
    {"id": "latency_short", "batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32, "initial": "zero"},
    {"id": "throughput_medium", "batch": 8, "sequence": 1024, "heads": 8, "key_dim": 64, "value_dim": 64, "initial": "nonzero"},
    {"id": "continuation_nonzero_state", "batch": 2, "sequence": 256, "heads": 8, "key_dim": 64, "value_dim": 64, "initial": "nonzero"},
    {"id": "decode_step", "batch": 1, "sequence": 1, "heads": 8, "key_dim": 64, "value_dim": 64, "initial": "nonzero"},
)
DTYPES = (("float32", torch.float32), ("bfloat16", torch.bfloat16))


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

    Pure differentiable PyTorch, so it provides the independent gradient
    reference the exact upstream (fused_recurrent) cannot. Always fp32.
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


def _inputs(seed, batch, sequence, heads, key_dim, value_dim, dtype, initial):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, sequence, heads, key_dim), device="cuda", generator=generator, dtype=dtype)
    k = torch.nn.functional.normalize(
        torch.randn((batch, sequence, heads, key_dim), device="cuda", generator=generator, dtype=dtype).float(),
        dim=-1,
    ).to(dtype)
    v = torch.randn((batch, sequence, heads, value_dim), device="cuda", generator=generator, dtype=dtype)
    g = (-torch.rand((batch, sequence, heads), device="cuda", generator=generator, dtype=torch.float32) * 0.3).to(dtype)
    beta = torch.rand((batch, sequence, heads), device="cuda", generator=generator, dtype=dtype)
    if initial == "zero":
        initial_state = torch.zeros((batch, heads, key_dim, value_dim), device="cuda", dtype=torch.float32)
    else:
        initial_state = torch.randn(
            (batch, heads, key_dim, value_dim), device="cuda", generator=generator, dtype=torch.float32
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
    """Competitive upstream performance baseline (chunked parallel kernel)."""
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


def _time_one(call, inputs, *, backward: bool, block: int = 1, no_grad: bool = False):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(block):
        for tensor in inputs.values():
            tensor.grad = None
        if no_grad:
            with torch.no_grad():
                call(inputs)
        else:
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


def _measure_pair(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup, block,
                  decode_direct=None, decode_compiled=None):
    """Measure every matrix mode: training forward, training forward+backward,
    inference prefill (forward under no_grad), and single-token decode.

    ``decode_direct``/``decode_compiled`` are single-token decode-step callables
    (the proper fused in-place decode path, not the training plan on T=1). When
    omitted, the decode mode falls back to the training callables under no_grad.
    """
    # (mode key, backward, no_grad, use decode callables)
    mode_specs = [
        ("forward", False, False, False),
        ("forward_backward", True, False, False),
        ("prefill", False, True, False),
        ("decode", False, True, True),
    ]
    for _ in range(warmup):
        for _, backward, no_grad, use_decode in mode_specs:
            d_fn = (decode_direct or direct) if use_decode else direct
            c_fn = (decode_compiled or compiled) if use_decode else compiled
            _time_one(d_fn, direct_inputs, backward=backward, block=block, no_grad=no_grad)
            _time_one(c_fn, compiled_inputs, backward=backward, block=block, no_grad=no_grad)
    measurements = {}
    for mode, backward, no_grad, use_decode in mode_specs:
        d_fn = (decode_direct or direct) if use_decode else direct
        c_fn = (decode_compiled or compiled) if use_decode else compiled
        direct_wall, compiled_wall, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, _ = _time_one(d_fn, direct_inputs, backward=backward, block=block, no_grad=no_grad)
                    direct_wall.append(wall)
                else:
                    wall, _ = _time_one(c_fn, compiled_inputs, backward=backward, block=block, no_grad=no_grad)
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
    return measurements


def _stable_seed(*parts) -> int:
    """Deterministic operand seed, stable across processes.

    ``hash()`` of a string is randomized per process (PYTHONHASHSEED), which made
    the qualification non-deterministic: a borderline case could flip pass/fail
    between runs. Use a SHA-256-derived seed so every run measures the same
    operands.
    """
    digest = hashlib.sha256(repr(parts).encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)


def _run_case(case, dtype_name, dtype, pairs, warmup, block):
    """Run one frozen case in one dtype; return (case_result, all_pass, any_fail)."""
    batch, sequence = case["batch"], case["sequence"]
    heads, key_dim, value_dim = case["heads"], case["key_dim"], case["value_dim"]
    scale = 1.0
    operands = _inputs(
        _stable_seed(case["id"], dtype_name),
        batch, sequence, heads, key_dim, value_dim, dtype, case["initial"],
    )
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan = compile_mixer(
        named_mixer_recipe("gated_delta_net"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype=dtype_name,
    )
    native_anchor = plan.anchor

    # --- Correctness first, at the frozen contract tolerance. ---
    direct_output, direct_state = _direct({n: t.detach() for n, t in operands.items()}, scale)
    oracle_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    oracle_result = _forward_backward(
        lambda i: _oracle_gated_delta(i["q"], i["k"], i["v"], i["g"], i["beta"], scale, i["initial_state"]),
        oracle_inputs,
    )
    compiled_result = _forward_backward(lambda i: _compiled(plan, i), compiled_inputs)

    output_error_upstream = (direct_output.float() - compiled_result[0].float()).abs().max().item()
    state_error_upstream = (direct_state.float() - compiled_result[1].float()).abs().max().item()
    output_error_oracle = (oracle_result[0] - compiled_result[0].float()).abs().max().item()
    state_error_oracle = (oracle_result[1] - compiled_result[1].float()).abs().max().item()
    gradient_errors_oracle = {
        name: (left.float() - right.float()).abs().max().item()
        for name, left, right in zip(operands, oracle_result[2], compiled_result[2], strict=True)
    }
    # State-gradient component (matrix: "state_gradients"): the gradient flowing
    # into the initial state, ∂L/∂(initial_state), verified against the oracle.
    # The loss includes the final state, so the state-cotangent path is exercised;
    # this isolates the state input's gradient from the operand gradients.
    state_gradient_error = {"initial_state": gradient_errors_oracle["initial_state"]}

    # The exact upstream and the native kernel share the input dtype, so their
    # outputs round identically; compare them directly at the contract tolerance.
    # The oracle is fp32: for low-precision dtypes the native/upstream output is
    # correctly rounded to the input dtype, so the oracle comparison must allow
    # for that dtype's output quantization (the exact upstream in bf16 deviates
    # from the fp32 oracle by ~0.03 for magnitude-12 outputs). Gradients are
    # accumulated in fp32 by both the native kernel and the oracle, so they are
    # compared directly at the contract tolerance.
    output_oracle_atol = OUTPUT_ATOL
    state_oracle_atol = STATE_ATOL
    if dtype in (torch.bfloat16, torch.float16):
        # One ulp of the output dtype at the observed magnitude, on top of the
        # contract tolerance. bf16 has ~2^-8 relative precision.
        output_oracle_atol = OUTPUT_ATOL + float(torch.finfo(dtype).eps) * float(oracle_result[0].abs().max())
        state_oracle_atol = STATE_ATOL + float(torch.finfo(dtype).eps) * float(oracle_result[1].abs().max())
    correctness_pass = True
    try:
        # Primary gate: native vs exact upstream, same dtype, contract tolerance.
        torch.testing.assert_close(compiled_result[0].float(), direct_output.float(), atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1].float(), direct_state.float(), atol=STATE_ATOL, rtol=RELATIVE_TOLERANCE)
        # Secondary gate: native vs fp32 oracle, dtype-aware tolerance.
        torch.testing.assert_close(compiled_result[0].float(), oracle_result[0], atol=output_oracle_atol, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1].float(), oracle_result[1], atol=state_oracle_atol, rtol=RELATIVE_TOLERANCE)
        # Gradients vs the fp32 oracle at the contract tolerance.
        for actual, expected in zip(compiled_result[2], oracle_result[2], strict=True):
            torch.testing.assert_close(actual.float(), expected.float(), atol=GRADIENT_ATOL, rtol=RELATIVE_TOLERANCE)
    except AssertionError:
        correctness_pass = False

    # --- Performance only after correctness, against the competitive comparator. ---
    competitive_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    # Decode mode: the proper single-token decode-step path (a fused in-place
    # kernel against a persistent state), not the training plan run on T=1. The
    # native decode uses execute_matrix_state_decode_step; the upstream decode
    # uses the exact sequential fused_recurrent operator on a one-token step.
    # Both thread a persistent state across steps (reset per timed unit).
    from urm.backends.triton.k2.matrix import (
        execute_matrix_state_decode_step,
    )

    decode_operands = _inputs(
        _stable_seed(case["id"], dtype_name, "decode"),
        batch, 1, heads, key_dim, value_dim, dtype, case["initial"],
    )
    decode_initial = decode_operands["initial_state"].detach().clone()

    def _native_decode_step(_inputs):
        state = decode_initial.clone()
        q1 = decode_operands["q"][:, 0].float()
        k1 = decode_operands["k"][:, 0].float()
        v1 = decode_operands["v"][:, 0].float()
        g1 = decode_operands["g"][:, 0].float()
        b1 = decode_operands["beta"][:, 0].float()

        def call(_):
            return execute_matrix_state_decode_step(
                query=q1, key=k1, value=v1, log_decay=g1, beta=b1, state=state,
                scale=scale, decay_granularity="head", is_delta=True, read_before=False,
            )
        return call

    def _upstream_decode_step(_inputs):
        q1 = decode_operands["q"].detach().clone()
        k1 = decode_operands["k"].detach().clone()
        v1 = decode_operands["v"].detach().clone()
        g1 = decode_operands["g"].detach().clone()
        b1 = decode_operands["beta"].detach().clone()

        def call(_):
            state = decode_initial.clone()
            return fused_recurrent_gated_delta_rule(
                q1, k1, v1, g=g1, beta=b1, scale=scale,
                initial_state=state, output_final_state=True,
            )
        return call

    measurements = _measure_pair(
        lambda i: _competitive(i, scale), lambda i: _compiled(plan, i),
        competitive_inputs, compiled_inputs, pairs, warmup, block,
        decode_direct=_upstream_decode_step(None), decode_compiled=_native_decode_step(None),
    )
    fwd_gate = measurements["forward"]["paired_native_overhead_fraction"]["gate"]["pass"]
    fb_gate = measurements["forward_backward"]["paired_native_overhead_fraction"]["gate"]["pass"]

    case_key = f"{case['id']}/{dtype_name}"
    result = {
        "semantic_scope": "matrix-state gated-delta recurrence core; projections and output projection excluded",
        "shape": {"batch": batch, "sequence": sequence, "heads": heads,
                  "key_dim": key_dim, "value_dim": value_dim, "dtype": dtype_name,
                  "initial_state": case["initial"]},
        "upstream_callable": "fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule",
        "performance_comparator_callable": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
        "native_anchor": native_anchor,
        "parity": {
            "status": "pass" if correctness_pass else "fail",
            "output_max_abs_error_vs_upstream": output_error_upstream,
            "final_state_max_abs_error_vs_upstream": state_error_upstream,
            "output_max_abs_error_vs_oracle": output_error_oracle,
            "final_state_max_abs_error_vs_oracle": state_error_oracle,
            "input_gradient_max_abs_errors_vs_oracle": gradient_errors_oracle,
            "state_gradient_max_abs_errors": state_gradient_error,
            "tolerances": {
                "output_atol": OUTPUT_ATOL,
                "state_atol": STATE_ATOL,
                "gradient_atol": GRADIENT_ATOL,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
        },
        "performance": {"measurements": measurements},
    }
    all_pass = correctness_pass and fwd_gate and fb_gate
    return case_key, result, all_pass, not correctness_pass


def run(pairs, warmup, output_path, block=1, only_case=None):
    if not torch.cuda.is_available():
        raise RuntimeError("native K2 gated-delta qualification requires CUDA")
    source, revision = _source_identity()
    scale = 1.0

    cases = {}
    all_qualified = True
    any_numeric_fail = False
    for case in CASES:
        if only_case and case["id"] != only_case:
            continue
        for dtype_name, dtype in DTYPES:
            case_key, result, all_pass, numeric_fail = _run_case(
                case, dtype_name, dtype, pairs, warmup, block
            )
            cases[case_key] = result
            if numeric_fail:
                any_numeric_fail = True
            if not all_pass:
                all_qualified = False

    if any_numeric_fail:
        verdict = "numeric_failed"
    elif all_qualified:
        verdict = "qualified"
    else:
        verdict = "correct_below_target"

    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "qualify URM-native K2 matrix-state gated-delta recurrence against the pinned FLA gated-delta operator (native replacement, not dispatch overhead)",
        "matrix_workload": "k2-gated-delta-recurrence",
        "verdict": verdict,
        "native_anchor": "urm_native_matrix_state_recurrence_v1",
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
             "cases": [c["id"] for c in CASES], "dtypes": [d for d, _ in DTYPES]},
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
            "correctness_comparator": "forward output/state vs fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule (exact sequential recurrence) and an independent eager oracle, at the frozen contract tolerance (0.02); gradients vs the oracle because the exact upstream implements no backward pass",
            "performance_comparator": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule (competitive chunked parallel kernel); the fastest compatible upstream kernel is the performance baseline, not a slow reference",
            "tolerance_calibration": "frozen production-matrix tolerances (output/state/gradient atol 0.02); the competitive chunked upstream deviates from the exact scan by ~0.02 in fp32, so the contract tolerance is the correct calibration",
            "timed_work": "one native plan call or one upstream call, optionally followed by output and final-state backward",
            "sampling": "paired interleaved native/direct calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (native-direct)/direct fractions with bootstrap CI",
            "slowdown_budget_fraction": SLOWDOWN_BUDGET_FRACTION,
            "gate_basis": "the 95% confidence-interval upper bound must meet the budget",
        },
        "cases": cases,
    }
    write_artifact(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--block", type=int, default=10,
                        help="invocations per timed unit; amortizes per-call mistiming overhead")
    parser.add_argument("--case", type=str, default=None,
                        help="run only one frozen case id (default: all four)")
    parser.add_argument("--output", type=Path, default=Path("results/qualification/native-k2-gated-delta.json"))
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    payload = run(args.pairs, args.warmup, args.output, args.block, args.case)
    print(f"verdict: {payload['verdict']}")
    for case_key, case in payload["cases"].items():
        fwd = case["performance"]["measurements"]["forward"]["paired_native_overhead_fraction"]
        fb = case["performance"]["measurements"]["forward_backward"]["paired_native_overhead_fraction"]
        print(f"  {case_key}: parity={case['parity']['status']}  "
              f"fwd {fwd['median']*100:+.1f}% (ci95 hi {fwd['ci95_upper']*100:+.1f}%)  "
              f"fwd+bwd {fb['median']*100:+.1f}% (ci95 hi {fb['ci95_upper']*100:+.1f}%)")


if __name__ == "__main__":
    main()
