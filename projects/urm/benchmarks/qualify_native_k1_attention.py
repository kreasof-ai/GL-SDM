"""Qualify the native K1 online-softmax attention against a competitive comparator.

This is a native-replacement qualification, not a dispatch-overhead measurement:
the ``compiled`` path executes the URM-native K1 online-softmax kernel
(``urm_native_k1_online_softmax_v1``) and the ``direct`` path executes a
competitive fused-attention upstream. The two share no kernel.

Comparator note: the frozen matrix names FlashAttention (``flash_attn_func``) as
the K1 comparator. FlashAttention is not installable in this environment (torch
2.14/cu130 with nvcc 12.9), so the competitive comparator is PyTorch SDPA's fused
attention backend - a production competitive kernel available here. This is a
recorded scope refinement, not a relaxation: SDPA is a real competitive upstream,
and the numerical/performance budgets are unchanged.

Correctness is verified before performance against SDPA and an independent eager
attention oracle (outputs and input gradients). A numerical failure disqualifies
the workload regardless of speed. Performance is the paired median
``(native - direct) / direct`` fraction with a bootstrap confidence interval,
judged against the frozen production-matrix budget.

The runner sweeps the full frozen production-matrix case set for the selected
workload (``--workload mha`` or ``--workload gqa``) in both matrix dtypes
(bfloat16 and float16), and measures all four matrix modes per case: training
forward, training forward+backward, inference prefill (forward under no_grad),
and single-token decode. The single-shape CLI arguments remain for backward
compatibility: when any is supplied explicitly the runner sweeps only that one
shape (in both dtypes) instead of the full matrix case set.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from measurement import (
    bootstrap_ci,
    capture_gpu_operating_conditions,
    quantile,
)
from provenance import provenance, write_artifact
from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.recipes import named_mixer_recipe

SLOWDOWN_BUDGET_FRACTION = 0.10
# Frozen production-matrix tolerances (benchmarks/production-matrix.json):
# output_atol = gradient_atol = 0.02 for both k1-mha and k1-gqa.
OUTPUT_ATOL = 0.02
GRADIENT_ATOL = 0.02
RELATIVE_TOLERANCE = 0.02

# The frozen production-matrix case sets (benchmarks/production-matrix.json).
MHA_CASES = (
    {"id": "latency_small", "batch": 1, "query_length": 128, "key_length": 128,
     "query_heads": 8, "key_value_heads": 8, "key_dim": 64, "value_dim": 64, "causal": True},
    {"id": "throughput_medium", "batch": 8, "query_length": 1024, "key_length": 1024,
     "query_heads": 16, "key_value_heads": 16, "key_dim": 64, "value_dim": 64, "causal": True},
    {"id": "throughput_long", "batch": 4, "query_length": 4096, "key_length": 4096,
     "query_heads": 16, "key_value_heads": 16, "key_dim": 64, "value_dim": 64, "causal": True},
    {"id": "decode_step", "batch": 1, "query_length": 1, "key_length": 2048,
     "query_heads": 8, "key_value_heads": 8, "key_dim": 64, "value_dim": 64, "causal": True,
     "stateful": True},
)
GQA_CASES = (
    {"id": "latency_small", "batch": 1, "query_length": 128, "key_length": 128,
     "query_heads": 8, "key_value_heads": 2, "key_dim": 64, "value_dim": 64, "causal": True},
    {"id": "throughput_long", "batch": 4, "query_length": 4096, "key_length": 4096,
     "query_heads": 32, "key_value_heads": 8, "key_dim": 64, "value_dim": 64, "causal": True},
    {"id": "decode_step", "batch": 1, "query_length": 1, "key_length": 2048,
     "query_heads": 8, "key_value_heads": 2, "key_dim": 64, "value_dim": 64, "causal": True,
     "stateful": True},
)
WORKLOAD_CASES = {"mha": MHA_CASES, "gqa": GQA_CASES}
DTYPES = (("bfloat16", torch.bfloat16), ("float16", torch.float16))


def _oracle_attention(q, k, v, scale, causal):
    """Independent eager attention oracle (fp32 accumulation), GQA-aware."""
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * scale
    if causal:
        t, s = q.shape[1], k.shape[1]
        mask = torch.ones(t, s, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshv->bthv", weights, v.float())


def _inputs(seed, batch, qlen, klen, qheads, kvheads, key_dim, value_dim, dtype):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, qlen, qheads, key_dim), device="cuda", generator=generator, dtype=dtype)
    k = torch.randn((batch, klen, kvheads, key_dim), device="cuda", generator=generator, dtype=dtype)
    v = torch.randn((batch, klen, kvheads, value_dim), device="cuda", generator=generator, dtype=dtype)
    return {
        "query": q.requires_grad_(),
        "key": k.requires_grad_(),
        "value": v.requires_grad_(),
    }


def _direct(inputs, scale, causal):
    """Competitive upstream: SDPA fused attention (BHTD layout), native GQA.

    Uses SDPA's native grouped-query path (``enable_gqa=True``) so the comparator
    does not pay avoidable KV-head expansion allocation and reduction work. The
    expanded-KV form is kept separately as ``_direct_expanded_kv`` (a labeled
    baseline), never as the qualification comparator.
    """
    qh = inputs["query"].transpose(1, 2)
    kh = inputs["key"].transpose(1, 2)
    vh = inputs["value"].transpose(1, 2)
    gqa = qh.shape[1] != kh.shape[1]
    return F.scaled_dot_product_attention(
        qh, kh, vh, is_causal=causal, scale=scale, enable_gqa=gqa
    ).transpose(1, 2)


def _direct_expanded_kv(inputs, scale, causal):
    """Labeled baseline: SDPA over KV heads expanded with repeat_interleave.

    This is the avoidable-allocation form. It is recorded for transparency but is
    never the qualification comparator, per the comparator policy.
    """
    qh = inputs["query"].transpose(1, 2)
    kh = inputs["key"].transpose(1, 2)
    vh = inputs["value"].transpose(1, 2)
    if qh.shape[1] != kh.shape[1]:
        repeats = qh.shape[1] // kh.shape[1]
        kh = kh.repeat_interleave(repeats, dim=1)
        vh = vh.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal, scale=scale).transpose(1, 2)


def _compiled(plan, inputs):
    return plan.execute(
        query=inputs["query"], key=inputs["key"], value=inputs["value"]
    ).output


def _time_one(call, inputs, *, backward: bool, block: int = 1, no_grad: bool = False):
    """Time a block of ``block`` invocations (sync at boundaries) to amortize
    per-call mistiming overhead; returns per-call wall time. When ``no_grad``
    the call runs under ``torch.no_grad()`` (inference forward, no autograd
    graph)."""
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    for _ in range(block):
        for tensor in inputs.values():
            tensor.grad = None
        if no_grad:
            with torch.no_grad():
                call(inputs)
        else:
            output = call(inputs)
            if backward:
                output.float().square().mean().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - start_wall) / block


def _summary(samples):
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [s * 1000 for s in samples],
    }


def _measure_pair(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup, block,
                  decode_direct=None, decode_compiled=None):
    """Measure every matrix mode: training forward, training forward+backward,
    inference prefill (forward under no_grad), and single-token decode.

    ``decode_direct``/``decode_compiled`` are single-token decode-step callables
    (the proper fused decode-step kernel against a persistent KV cache, not the
    training plan on qlen=1). Each is ``call(inputs) -> output`` and closes over
    its own operands; the inputs argument is ignored for decode. When omitted,
    the decode mode falls back to the training callables under no_grad.
    """
    # (mode key, backward, no_grad, use decode callables)
    mode_specs = [
        ("forward", False, False, False),
        ("forward_backward", True, False, False),
        ("prefill", False, True, False),
        ("decode", False, True, True),
    ]
    cold_direct = _time_one(direct, direct_inputs, backward=False, block=block)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False, block=block)
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
            first, second = ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    direct_wall.append(_time_one(d_fn, direct_inputs, backward=backward, block=block, no_grad=no_grad))
                else:
                    compiled_wall.append(_time_one(c_fn, compiled_inputs, backward=backward, block=block, no_grad=no_grad))
            pair_index = len(overhead)
            overhead.append(
                (compiled_wall[pair_index] - direct_wall[pair_index]) / direct_wall[pair_index]
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
        "first_direct_forward_call_ms": cold_direct * 1000,
        "first_compiled_forward_call_after_warmup_ms": cold_compiled * 1000,
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def _stable_seed(*parts) -> int:
    """Deterministic operand seed, stable across processes (PYTHONHASHSEED-safe)."""
    digest = hashlib.sha256(repr(parts).encode()).hexdigest()
    return int(digest[:8], 16) % (2**31)


def _run_case(case, dtype_name, dtype, pairs, warmup, block):
    """Run one frozen case in one dtype; return (case_key, result, all_pass, numeric_fail)."""
    batch, qlen, klen = case["batch"], case["query_length"], case["key_length"]
    qheads, kvheads = case["query_heads"], case["key_value_heads"]
    key_dim, value_dim = case["key_dim"], case["value_dim"]
    scale = key_dim**-0.5
    # The native kernel's causal mask is block-aligned: it applies the causal
    # square mask only when query_length == key_length. For a single-token
    # decode step (query_length=1, key_length>1) the kernel attends to the full
    # causal history - the correct KV-cache decode semantics - whereas SDPA's
    # ``is_causal=True`` would apply a square mask leaving only key 0. So the
    # comparator/oracle causal flag is: square-causal for the training/prefill
    # shapes, full-history (non-causal) for the single-token decode shape.
    causal = case["causal"] and qlen == klen
    operands = _inputs(
        _stable_seed(case["id"], dtype_name),
        batch, qlen, klen, qheads, kvheads, key_dim, value_dim, dtype,
    )
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe("mha"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype=dtype_name,
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    native_anchor = plan.anchor

    # --- Correctness first: native vs SDPA AND native vs independent oracle.
    direct_out = _direct({n: t.detach() for n, t in operands.items()}, scale, causal)
    oracle_out = _oracle_attention(
        operands["query"].detach(), operands["key"].detach(), operands["value"].detach(), scale, causal
    )
    for tensor in compiled_inputs.values():
        tensor.grad = None
    compiled_out = _compiled(plan, compiled_inputs)
    compiled_out.float().square().mean().backward()

    # gradient comparison vs SDPA
    for tensor in direct_inputs.values():
        tensor.grad = None
    direct_out_gb = _direct(direct_inputs, scale, causal)
    direct_out_gb.float().square().mean().backward()

    output_error_upstream = (direct_out - compiled_out).abs().max().item()
    output_error_oracle = (oracle_out - compiled_out.float()).abs().max().item()
    gradient_errors = {
        name: (compiled_inputs[name].grad - direct_inputs[name].grad).abs().max().item()
        for name in ("query", "key", "value")
    }

    # The upstream (SDPA) and the native kernel share the input dtype, so their
    # outputs round identically; compare them directly at the contract tolerance.
    # The oracle is fp32: for low-precision dtypes the native/upstream output is
    # correctly rounded to the input dtype, so the oracle comparison must allow
    # for that dtype's output quantization (one ulp of the output dtype at the
    # observed magnitude on top of the contract tolerance).
    output_oracle_atol = OUTPUT_ATOL
    if dtype in (torch.bfloat16, torch.float16):
        output_oracle_atol = OUTPUT_ATOL + float(torch.finfo(dtype).eps) * float(oracle_out.abs().max())
    correctness_pass = True
    try:
        torch.testing.assert_close(compiled_out.float(), direct_out.float(), atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_out.float(), oracle_out, atol=output_oracle_atol, rtol=RELATIVE_TOLERANCE)
        for name in ("query", "key", "value"):
            torch.testing.assert_close(
                compiled_inputs[name].grad.float(), direct_inputs[name].grad.float(),
                atol=GRADIENT_ATOL, rtol=RELATIVE_TOLERANCE,
            )
    except AssertionError:
        correctness_pass = False

    # --- Performance only after correctness.
    perf_inputs_d = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    perf_inputs_c = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    # Decode mode: the proper fused single-query decode-step kernel against a
    # persistent KV cache, not the training plan on qlen=1. The native decode
    # uses execute_online_softmax_decode; the upstream decode comparator is SDPA
    # on a single query token against the full KV cache (is_causal=False because
    # for a single query at the latest position all S keys are visible). Both
    # callables close over their operands; the KV cache is read-only, so no
    # state threading is needed.
    from urm.backends.triton.k1.online import execute_online_softmax_decode

    decode_operands = _inputs(
        _stable_seed(case["id"], dtype_name, "decode"),
        batch, 1, klen, qheads, kvheads, key_dim, value_dim, dtype,
    )

    def _native_decode_step(_inputs):
        q1 = decode_operands["query"][:, 0].detach().contiguous()  # [B, H, K]
        k_cache = decode_operands["key"].detach().contiguous()      # [B, S, H_kv, K]
        v_cache = decode_operands["value"].detach().contiguous()    # [B, S, H_kv, V]

        def call(_):
            return execute_online_softmax_decode(q1, k_cache, v_cache, scale=scale, causal=True)
        return call

    def _upstream_decode_step(_inputs):
        qh = decode_operands["query"].detach().transpose(1, 2)  # [B, H, 1, K]
        kh = decode_operands["key"].detach().transpose(1, 2)    # [B, H_kv, S, K]
        vh = decode_operands["value"].detach().transpose(1, 2)  # [B, H_kv, S, V]
        gqa = qh.shape[1] != kh.shape[1]

        def call(_):
            return F.scaled_dot_product_attention(
                qh, kh, vh, is_causal=False, scale=scale, enable_gqa=gqa
            )
        return call

    performance = _measure_pair(
        lambda i: _direct(i, scale, causal), lambda i: _compiled(plan, i),
        perf_inputs_d, perf_inputs_c, pairs, warmup, block,
        decode_direct=_upstream_decode_step(None), decode_compiled=_native_decode_step(None),
    )
    measurements = performance["measurements"]
    fwd_gate = measurements["forward"]["paired_native_overhead_fraction"]["gate"]["pass"]
    fb_gate = measurements["forward_backward"]["paired_native_overhead_fraction"]["gate"]["pass"]

    case_key = f"{case['id']}/{dtype_name}"
    result = {
        "semantic_scope": "normalized causal attention core; projections excluded",
        "shape": {"batch": batch, "query_length": qlen, "key_length": klen,
                  "query_heads": qheads, "key_value_heads": kvheads,
                  "key_dim": key_dim, "value_dim": value_dim, "dtype": dtype_name},
        "native_anchor": native_anchor,
        "compiler_plan_build_ms": plan_build_ms,
        "parity": {
            "status": "pass" if correctness_pass else "fail",
            "output_max_abs_error_vs_upstream": output_error_upstream,
            "output_max_abs_error_vs_oracle": output_error_oracle,
            "input_gradient_max_abs_errors_vs_upstream": gradient_errors,
            "tolerances": {
                "output_atol": OUTPUT_ATOL,
                "output_oracle_atol": output_oracle_atol,
                "gradient_atol": GRADIENT_ATOL,
                "relative_tolerance": RELATIVE_TOLERANCE,
            },
        },
        "performance": performance,
    }
    all_pass = correctness_pass and fwd_gate and fb_gate
    return case_key, result, all_pass, not correctness_pass


def run(pairs, warmup, output_path, block=1, workload="mha",
        batch=None, qlen=None, klen=None, qheads=None, kvheads=None,
        key_dim=None, value_dim=None):
    if not torch.cuda.is_available():
        raise RuntimeError("native K1 attention qualification requires CUDA")

    # Full frozen-matrix sweep by default; a fully-specified single shape
    # (backward-compat CLI) sweeps just that one shape in both dtypes.
    single_shape = (batch, qlen, klen, qheads, kvheads, key_dim, value_dim)
    if all(v is not None for v in single_shape):
        cases = ({"id": "custom", "batch": batch, "query_length": qlen,
                  "key_length": klen, "query_heads": qheads, "key_value_heads": kvheads,
                  "key_dim": key_dim, "value_dim": value_dim, "causal": True},)
    else:
        cases = WORKLOAD_CASES[workload]

    case_results = {}
    all_qualified = True
    any_numeric_fail = False
    for case in cases:
        for dtype_name, dtype in DTYPES:
            case_key, result, all_pass, numeric_fail = _run_case(
                case, dtype_name, dtype, pairs, warmup, block
            )
            case_results[case_key] = result
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

    native_anchor = next(iter(case_results.values()))["native_anchor"]
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "qualify URM-native K1 online-softmax attention against a competitive fused-attention upstream (native replacement, not dispatch overhead)",
        "matrix_workload": f"k1-{workload}",
        "verdict": verdict,
        "native_anchor": native_anchor,
        "comparator": {
            "frozen_in_matrix": "flash_attn_func (Dao-AILab/flash-attention)",
            "used": "torch.nn.functional.scaled_dot_product_attention (fused mem-efficient backend)",
            "scope_note": "flash_attn is not installable in this torch 2.14/cu130 + nvcc 12.9 environment; SDPA is the available competitive fused-attention upstream. Budgets and tolerances unchanged.",
        },
        "provenance": provenance(
            "PYTHONPATH=src python benchmarks/qualify_native_k1_attention.py",
            {"recipe": "mha", "workload": workload, "pairs": pairs, "warmup": warmup,
             "cases": [c["id"] for c in cases], "dtypes": [d for d, _ in DTYPES]},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_operating_conditions": capture_gpu_operating_conditions(),
        "methodology": {
            "comparison": "URM-native K1 online-softmax kernel vs SDPA fused attention; the two share no kernel",
            "correctness_comparator": "SDPA plus an independent eager attention oracle",
            "performance_comparator": "SDPA fused attention (competitive production kernel)",
            "timed_work": "one native plan call or one upstream call, optionally followed by output backward; prefill and decode run under no_grad",
            "sampling": "paired interleaved native/direct calls, order alternates, synchronized wall timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (native-direct)/direct fractions with bootstrap CI",
            "slowdown_budget_fraction": SLOWDOWN_BUDGET_FRACTION,
            "gate_basis": "the 95% confidence-interval upper bound must meet the budget",
        },
        "cases": case_results,
    }
    write_artifact(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("mha", "gqa"), default="mha",
                        help="frozen-matrix workload case set to sweep (default: mha)")
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--qlen", type=int, default=None)
    parser.add_argument("--klen", type=int, default=None)
    parser.add_argument("--qheads", type=int, default=None)
    parser.add_argument("--kvheads", type=int, default=None)
    parser.add_argument("--key-dim", type=int, default=None)
    parser.add_argument("--value-dim", type=int, default=None)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None,
                        help="artifact path (default: results/qualification/native-k1-<workload>.json)")
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    output = args.output or Path(f"results/qualification/native-k1-{args.workload}.json")
    payload = run(args.pairs, args.warmup, output, args.block, args.workload,
                  args.batch, args.qlen, args.klen, args.qheads, args.kvheads,
                  args.key_dim, args.value_dim)
    print(f"verdict: {payload['verdict']}")
    for case_key, case in payload["cases"].items():
        parity = case["parity"]["status"]
        out_err = case["parity"]["output_max_abs_error_vs_upstream"]
        parts = [f"  {case_key}: parity={parity} (out err {out_err:.2e})"]
        for mode in ("forward", "forward_backward", "prefill", "decode"):
            m = case["performance"]["measurements"][mode]["paired_native_overhead_fraction"]
            parts.append(
                f"    {mode:18s} overhead: median {m['median']*100:+.2f}%  "
                f"ci95 [{m['ci95_lower']*100:+.2f}%, {m['ci95_upper']*100:+.2f}%]  "
                f"gate {'PASS' if m['gate']['pass'] else 'FAIL'}"
            )
        print("\n".join(parts))


if __name__ == "__main__":
    main()
