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
# Frozen production-matrix correctness tolerances (benchmarks/production-matrix.json).
OUTPUT_ATOL = 0.02
STATE_ATOL = 0.02
GRADIENT_ATOL = 0.02
RELATIVE_TOLERANCE = 1e-4

# The four frozen cases for k2-diagonal-recurrence. HGRN is a per-channel scalar
# diagonal recurrence (state [B, C] with unit state width), so the matrix's
# (heads, key_dim, value_dim) map onto channels = heads * key_dim (diagonal:
# key_dim == value_dim per head).
CASES = (
    {"id": "latency_short", "batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32, "initial": "zero"},
    {"id": "throughput_medium", "batch": 8, "sequence": 1024, "heads": 8, "key_dim": 64, "value_dim": 64, "initial": "zero"},
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


def _inputs(seed: int, batch: int, sequence: int, channels: int, dtype: torch.dtype, initial: str) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((batch, sequence, channels), device="cuda", generator=generator) * 0.2).to(dtype)
    log_decay = (
        -torch.rand((batch, sequence, channels), device="cuda", generator=generator) * 0.05
    ).to(dtype)
    # The recurrent state is fp32 (the accumulator), regardless of the input dtype.
    if initial == "zero":
        initial_state = torch.zeros((batch, channels), device="cuda", dtype=torch.float32)
    else:
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


def _time_one(call, inputs, *, backward: bool, block: int = 1, no_grad: bool = False):
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
    (the proper fused in-place decode path, not the training plan on T=1). Each
    closes over its own persistent state and operands and ignores the inputs
    passed by ``_time_one``. When omitted, the decode mode falls back to the
    training callables under no_grad.
    """
    cold_direct = _time_one(direct, direct_inputs, backward=False, block=block)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False, block=block)
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


def _run_case(case, dtype_name, dtype, pairs, warmup, block):
    """Run one frozen case in one dtype; return (case_key, case_result, all_pass, numeric_fail)."""
    batch, sequence = case["batch"], case["sequence"]
    heads, key_dim, value_dim = case["heads"], case["key_dim"], case["value_dim"]
    # HGRN is a per-channel scalar diagonal recurrence: channels = heads * key_dim.
    channels = heads * key_dim
    operands = _inputs(
        hash((case["id"], dtype_name)) % (2**31), batch, sequence, channels, dtype, case["initial"],
    )
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe("hgrn_ssm_core"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype=dtype_name,
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    native_anchor = plan.anchor

    # --- Correctness first: native vs upstream AND native vs independent oracle.
    direct_result = _forward_backward(_direct, direct_inputs)
    compiled_result = _forward_backward(lambda i: _compiled(plan, i), compiled_inputs)
    oracle_output, oracle_state = _oracle_hgrn(
        operands["x"], operands["log_decay"], operands["initial_state"]
    )

    output_error_upstream = (direct_result[0].float() - compiled_result[0].float()).abs().max().item()
    state_error_upstream = (direct_result[1].float() - compiled_result[1].float()).abs().max().item()
    output_error_oracle = (oracle_output - compiled_result[0].float()).abs().max().item()
    state_error_oracle = (oracle_state - compiled_result[1].float()).abs().max().item()
    gradient_errors_upstream = {
        name: (left.float() - right.float()).abs().max().item()
        for name, left, right in zip(operands, direct_result[2], compiled_result[2], strict=True)
    }
    # State-gradient component (matrix: "state_gradients"): the gradient flowing
    # into the initial state, verified against the upstream backward.
    state_gradient_error = {"initial_state": gradient_errors_upstream["initial_state"]}

    # The oracle is fp32; for low-precision dtypes allow one ulp of the output
    # dtype at the observed magnitude on top of the contract tolerance.
    output_oracle_atol = OUTPUT_ATOL
    state_oracle_atol = STATE_ATOL
    if dtype in (torch.bfloat16, torch.float16):
        output_oracle_atol = OUTPUT_ATOL + float(torch.finfo(dtype).eps) * float(oracle_output.abs().max())
        state_oracle_atol = STATE_ATOL + float(torch.finfo(dtype).eps) * float(oracle_state.abs().max())
    correctness_pass = True
    try:
        torch.testing.assert_close(compiled_result[0].float(), direct_result[0].float(), atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1].float(), direct_result[1].float(), atol=STATE_ATOL, rtol=RELATIVE_TOLERANCE)
        # Native vs independent oracle (dtype-aware tolerance).
        torch.testing.assert_close(compiled_result[0].float(), oracle_output, atol=output_oracle_atol, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_result[1].float(), oracle_state, atol=state_oracle_atol, rtol=RELATIVE_TOLERANCE)
        for actual, expected in zip(compiled_result[2], direct_result[2], strict=True):
            torch.testing.assert_close(actual.float(), expected.float(), atol=GRADIENT_ATOL, rtol=RELATIVE_TOLERANCE)
    except AssertionError:
        correctness_pass = False

    # --- Performance only after correctness, against the competitive comparator.
    competitive_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    # Decode mode: the proper single-token decode-step path (a fused in-place
    # kernel against a persistent state), not the training plan run on T=1. The
    # native decode uses execute_diagonal_decode_step on a persistent [B,C,1]
    # fp32 state; the upstream decode uses the exact sequential fused_recurrent
    # operator on a one-token step (T=1). Both thread a persistent state across
    # steps (reset per timed unit).
    from urm.backends.triton.recurrence.diagonal_recurrence import (
        execute_diagonal_decode_step,
    )

    decode_operands = _inputs(
        hash((case["id"], dtype_name, "decode")) % (2**31), batch, 1, channels, dtype, case["initial"],
    )
    decode_x = decode_operands["x"][:, 0].detach().float()  # [B, C] single token
    decode_log_decay = decode_operands["log_decay"][:, 0].detach().float()  # [B, C]
    # The persistent native state is [B, C, N=1] fp32 (HGRN is a per-channel
    # scalar recurrence, so the state width N is one).
    decode_native_initial = (
        decode_operands["initial_state"].detach().float().unsqueeze(-1).contiguous()
    )
    decode_upstream_initial = decode_operands["initial_state"].detach().float().contiguous()

    def _native_decode_step(_inputs):
        state = decode_native_initial.clone()

        def call(_):
            return execute_diagonal_decode_step(
                x=decode_x, log_decay=decode_log_decay,
                input_gate=None, read_gate=None, state=state, read_before=False,
            )
        return call

    def _upstream_decode_step(_inputs):
        x1 = decode_operands["x"].detach().clone()  # [B, 1, C]
        ld1 = decode_operands["log_decay"].detach().clone()  # [B, 1, C]

        def call(_):
            state = decode_upstream_initial.clone()
            return fused_recurrent_hgrn(
                x1, ld1, initial_state=state, output_final_state=True,
            )
        return call

    performance = _measure_pair(
        _competitive, lambda i: _compiled(plan, i), competitive_inputs, compiled_inputs, pairs, warmup, block,
        decode_direct=_upstream_decode_step(None), decode_compiled=_native_decode_step(None),
    )
    fwd_gate = performance["measurements"]["forward"]["paired_native_overhead_fraction"]["gate"]["pass"]
    fb_gate = performance["measurements"]["forward_backward"]["paired_native_overhead_fraction"]["gate"]["pass"]

    case_key = f"{case['id']}/{dtype_name}"
    result = {
        "semantic_scope": "diagonal gated recurrence core; projections and output projection excluded",
        "shape": {"batch": batch, "sequence": sequence, "heads": heads,
                  "key_dim": key_dim, "value_dim": value_dim, "dtype": dtype_name,
                  "initial_state": case["initial"]},
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
            "state_gradient_max_abs_errors": state_gradient_error,
            "tolerances": {
                "output_atol": OUTPUT_ATOL,
                "state_atol": STATE_ATOL,
                "gradient_atol": GRADIENT_ATOL,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
        },
        "performance": performance,
    }
    all_pass = correctness_pass and fwd_gate and fb_gate
    return case_key, result, all_pass, not correctness_pass


def run(pairs: int, warmup: int, output_path: Path, block: int = 1, only_case=None) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("native K2 HGRN qualification requires CUDA")
    source, revision = _source_identity()

    cases = {}
    all_qualified = True
    any_numeric_fail = False
    native_anchor = None
    for case in CASES:
        if only_case and case["id"] != only_case:
            continue
        for dtype_name, dtype in DTYPES:
            case_key, result, all_pass, numeric_fail = _run_case(
                case, dtype_name, dtype, pairs, warmup, block
            )
            cases[case_key] = result
            native_anchor = result["native_anchor"]
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
            "comparison": "URM-native diagonal recurrence kernel vs pinned FLA HGRN; the two share no kernel",
            "correctness_comparator": "fla.ops.hgrn.fused_recurrent_hgrn (exact sequential recurrence) plus an independent eager oracle; output/final-state/input-gradients/state-gradients verified",
            "performance_comparator": "fla.ops.hgrn.chunk_hgrn (competitive chunked parallel kernel); the fastest compatible upstream kernel is the performance baseline, not a slow reference",
            "timed_work": "one native plan call or one upstream call, in each matrix mode (training forward, training forward+backward, inference prefill, single-token decode); the decode mode runs the fused single-token decode-step path (native execute_diagonal_decode_step on a persistent [B,C,1] fp32 state vs upstream fused_recurrent_hgrn on a one-token step), not the training plan on T=1",
            "sampling": "paired interleaved native/direct calls, order alternates, synchronized wall and CUDA event timing; each timed unit is a block of invocations to amortize per-call mistiming overhead",
            "warmup": warmup,
            "pairs": pairs,
            "block": block,
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
    parser.add_argument("--output", type=Path, default=Path("results/qualification/native-k2-hgrn.json"))
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
