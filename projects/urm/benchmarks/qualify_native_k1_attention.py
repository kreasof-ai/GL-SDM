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
"""

from __future__ import annotations

import argparse
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
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe

SLOWDOWN_BUDGET_FRACTION = 0.10
# bf16 attention tolerances (matrix: output_atol 0.02, gradient_atol 0.02).
OUTPUT_ATOL = 0.02
GRADIENT_ATOL = 0.02
RELATIVE_TOLERANCE = 0.02


def _oracle_attention(q, k, v, scale, causal):
    """Independent eager attention oracle (fp32 accumulation)."""
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
    """Competitive upstream: SDPA fused attention (BHTD layout)."""
    qh = inputs["query"].transpose(1, 2)
    kh = inputs["key"].transpose(1, 2)
    vh = inputs["value"].transpose(1, 2)
    return F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal, scale=scale).transpose(1, 2)


def _compiled(plan, inputs):
    return plan.execute(
        query=inputs["query"], key=inputs["key"], value=inputs["value"]
    ).output


def _time_one(call, inputs, *, backward: bool, block: int = 1):
    """Time a block of ``block`` invocations (sync at boundaries) to amortize
    per-call mistiming overhead; returns per-call wall time."""
    torch.cuda.synchronize()
    start_wall = time.perf_counter()
    for _ in range(block):
        for tensor in inputs.values():
            tensor.grad = None
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
            first, second = ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    direct_wall.append(_time_one(direct, direct_inputs, backward=backward, block=block))
                else:
                    compiled_wall.append(_time_one(compiled, compiled_inputs, backward=backward, block=block))
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


def run(pairs, warmup, batch, qlen, klen, qheads, kvheads, key_dim, value_dim, dtype, output_path, block=1):
    if not torch.cuda.is_available():
        raise RuntimeError("native K1 attention qualification requires CUDA")
    scale = key_dim**-0.5
    causal = True
    operands = _inputs(77441, batch, qlen, klen, qheads, kvheads, key_dim, value_dim, dtype)
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe("mha"),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype=str(dtype).removeprefix("torch."),
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

    correctness_pass = True
    try:
        torch.testing.assert_close(compiled_out.float(), direct_out.float(), atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
        torch.testing.assert_close(compiled_out.float(), oracle_out, atol=OUTPUT_ATOL, rtol=RELATIVE_TOLERANCE)
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
    performance = _measure_pair(
        lambda i: _direct(i, scale, causal), lambda i: _compiled(plan, i),
        perf_inputs_d, perf_inputs_c, pairs, warmup, block,
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
        "purpose": "qualify URM-native K1 online-softmax attention against a competitive fused-attention upstream (native replacement, not dispatch overhead)",
        "matrix_workload": "k1-mha",
        "verdict": verdict,
        "native_anchor": native_anchor,
        "comparator": {
            "frozen_in_matrix": "flash_attn_func (Dao-AILab/flash-attention)",
            "used": "torch.nn.functional.scaled_dot_product_attention (fused mem-efficient backend)",
            "scope_note": "flash_attn is not installable in this torch 2.14/cu130 + nvcc 12.9 environment; SDPA is the available competitive fused-attention upstream. Budgets and tolerances unchanged.",
        },
        "provenance": provenance(
            "PYTHONPATH=src python benchmarks/qualify_native_k1_attention.py",
            {"recipe": "mha", "pairs": pairs, "warmup": warmup,
             "shape": [batch, qlen, klen, qheads, kvheads, key_dim, value_dim], "dtype": str(dtype).removeprefix("torch.")},
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
            "timed_work": "one native plan call or one upstream call, optionally followed by output backward",
            "sampling": "paired interleaved native/direct calls, order alternates, synchronized wall timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (native-direct)/direct fractions with bootstrap CI",
            "slowdown_budget_fraction": SLOWDOWN_BUDGET_FRACTION,
            "gate_basis": "the 95% confidence-interval upper bound must meet the budget",
        },
        "cases": {
            "mha": {
                "semantic_scope": "normalized causal attention core; projections excluded",
                "shape": {"batch": batch, "query_length": qlen, "key_length": klen,
                          "query_heads": qheads, "kv_heads": kvheads,
                          "key_dim": key_dim, "value_dim": value_dim, "dtype": str(dtype).removeprefix("torch.")},
                "native_anchor": native_anchor,
                "compiler_plan_build_ms": plan_build_ms,
                "parity": {
                    "status": "pass" if correctness_pass else "fail",
                    "output_max_abs_error_vs_upstream": output_error_upstream,
                    "output_max_abs_error_vs_oracle": output_error_oracle,
                    "input_gradient_max_abs_errors_vs_upstream": gradient_errors,
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
    parser.add_argument("--qlen", type=int, default=1024)
    parser.add_argument("--klen", type=int, default=1024)
    parser.add_argument("--qheads", type=int, default=16)
    parser.add_argument("--kvheads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/qualification/native-k1-mha.json"))
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    payload = run(args.pairs, args.warmup, args.batch, args.qlen, args.klen,
                  args.qheads, args.kvheads, args.key_dim, args.value_dim, dtype, args.output, args.block)
    case = payload["cases"]["mha"]
    fwd = case["performance"]["measurements"]["forward"]["paired_native_overhead_fraction"]
    fb = case["performance"]["measurements"]["forward_backward"]["paired_native_overhead_fraction"]
    print(f"verdict: {payload['verdict']}")
    print(f"parity: {case['parity']['status']}  (output err vs upstream {case['parity']['output_max_abs_error_vs_upstream']:.2e})")
    print(f"forward  overhead: median {fwd['median']*100:+.2f}%  ci95 [{fwd['ci95_lower']*100:+.2f}%, {fwd['ci95_upper']*100:+.2f}%]  gate {'PASS' if fwd['gate']['pass'] else 'FAIL'}")
    print(f"fwd+bwd  overhead: median {fb['median']*100:+.2f}%  ci95 [{fb['ci95_lower']*100:+.2f}%, {fb['ci95_upper']*100:+.2f}%]  gate {'PASS' if fb['gate']['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
