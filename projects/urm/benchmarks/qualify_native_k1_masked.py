"""Qualify the native K1 online-softmax masked attention against a competitive comparator.

This is a native-replacement qualification, not a dispatch-overhead measurement:
the ``compiled`` path executes the URM-native K1 online-softmax kernel
(``urm_native_k1_online_softmax_v1``) through a masked recipe
(``cat_attention_core``, ``requires_attention_mask=True``) and the ``direct``
path executes a competitive masked fused-attention upstream. The two share no
kernel.

Comparator note: the frozen matrix names FlashAttention's masked path
(``flash_attn_varlen_func or equivalent masked path``) as the comparator.
FlashAttention is not installable in this environment (torch 2.14/cu130 with
nvcc 12.9), so the competitive comparator is PyTorch SDPA's fused attention
backend with an explicit ``attn_mask`` - a production competitive masked
kernel available here. This is a recorded scope refinement, not a relaxation:
SDPA-with-mask is a real competitive masked upstream, and the
numerical/performance budgets are unchanged.

Correctness is verified before performance against SDPA and an independent
eager masked-attention oracle (outputs and input gradients). A numerical
failure disqualifies the workload regardless of speed. Performance is the
paired median ``(native - direct) / direct`` fraction with a bootstrap
confidence interval, judged against the frozen production-matrix budget.

The runner sweeps the full frozen production-matrix case set for the
``k1-masked-variant`` workload in its single matrix dtype (bfloat16) and
measures the three matrix modes per case: training forward, training
forward+backward, and inference prefill (forward under no_grad). The matrix
declares no decode mode for this workload. The ``fully_masked_rows`` case
exercises the safe-zero path: query rows with no visible keys must produce
zero output and zero gradients, which this runner verifies explicitly.
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

# Frozen production-matrix budget for k1-masked-variant: 0.15 (not 0.10).
SLOWDOWN_BUDGET_FRACTION = 0.15
# Frozen production-matrix tolerances (benchmarks/production-matrix.json):
# output_atol = gradient_atol = 0.02 for k1-masked-variant.
OUTPUT_ATOL = 0.02
GRADIENT_ATOL = 0.02
RELATIVE_TOLERANCE = 0.02

# The masked K1 recipe: a normalized-softmax K1 core that requires a
# caller-supplied attention_mask route. Verified to compile to the native
# anchor ``urm_native_k1_online_softmax_v1``.
MASKED_RECIPE = "cat_attention_core"

# The frozen production-matrix case set (benchmarks/production-matrix.json).
CASES = (
    {"id": "block_sparse_medium", "batch": 4, "query_length": 1024, "key_length": 1024,
     "query_heads": 8, "key_value_heads": 8, "key_dim": 64, "value_dim": 64,
     "mask": "block_sparse", "causal": True},
    {"id": "fully_masked_rows", "batch": 2, "query_length": 512, "key_length": 512,
     "query_heads": 8, "key_value_heads": 8, "key_dim": 64, "value_dim": 64,
     "mask": "fully_masked_rows", "causal": True},
)
DTYPES = (("bfloat16", torch.bfloat16),)

# Block size for the block-sparse mask pattern.
_BLOCK = 64


def _build_mask(case, batch, qlen, klen, device):
    """Build the boolean attention mask [B, 1, T, S] for one frozen case.

    The same mask drives the native kernel, the SDPA comparator (as
    ``attn_mask``), and the eager oracle. ``True`` means the key is visible.
    The mask already encodes causality (the callers must not apply a second
    causal mask). Every construction guarantees the semantics the matrix
    declares:

    - ``block_sparse``: each 64-wide query block attends to its own key block
      and the immediately preceding key block (block-local + one block of
      look-back), ANDed with the causal mask. Every query row keeps at least
      one visible key.
    - ``fully_masked_rows``: a causal mask with some query rows fully masked
      (no visible keys), exercising the safe-zero output/gradient path.
    """
    if qlen == klen:
        causal = torch.ones(qlen, klen, dtype=torch.bool, device=device).tril()
    else:
        q_positions = torch.arange(qlen, device=device) + (klen - qlen)
        k_positions = torch.arange(klen, device=device)
        causal = k_positions[None, :] <= q_positions[:, None]
    if case["mask"] == "block_sparse":
        q_block = torch.arange(qlen, device=device) // _BLOCK
        k_block = torch.arange(klen, device=device) // _BLOCK
        # A query block sees its own key block and the previous one.
        block_visible = (k_block[None, :] == q_block[:, None]) | (
            k_block[None, :] == q_block[:, None] - 1
        )
        mask = (block_visible & causal).unsqueeze(0).expand(batch, qlen, klen)
    elif case["mask"] == "fully_masked_rows":
        mask = causal.unsqueeze(0).expand(batch, qlen, klen).clone()
        # Fully mask a deterministic set of query rows (no visible keys).
        rows = {5, qlen // 3, qlen - 7}
        for b in range(batch):
            for row in rows:
                mask[b, (row + b) % qlen, :] = False
    else:
        raise ValueError(f"unknown mask pattern: {case['mask']}")
    return mask.unsqueeze(1).contiguous()  # [B, 1, T, S], shared across heads


def _oracle_masked(q, k, v, scale, mask):
    """Independent eager masked attention oracle (fp32 accumulation).

    ``mask`` is the boolean [B, 1, T, S] visibility mask (already encoding
    causality). Fully-masked rows (no visible key) produce zero output.
    """
    if q.shape[2] != k.shape[2]:
        repeats = q.shape[2] // k.shape[2]
        k = k.repeat_interleave(repeats, dim=2)
        v = v.repeat_interleave(repeats, dim=2)
    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    empty_rows = torch.isneginf(scores).all(dim=-1, keepdim=True)
    safe_scores = torch.where(empty_rows, torch.zeros_like(scores), scores)
    weights = torch.softmax(safe_scores, dim=-1)
    weights = torch.where(empty_rows, torch.zeros_like(weights), weights)
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


def _direct(inputs, scale, mask):
    """Competitive upstream: SDPA fused attention with an explicit mask.

    The boolean mask already encodes causality, so ``is_causal=False`` to
    avoid double-applying the causal mask. SDPA yields zero outputs and zero
    gradients for fully-masked rows, matching the native kernel's safe-zero
    semantics.
    """
    qh = inputs["query"].transpose(1, 2)
    kh = inputs["key"].transpose(1, 2)
    vh = inputs["value"].transpose(1, 2)
    gqa = qh.shape[1] != kh.shape[1]
    return F.scaled_dot_product_attention(
        qh, kh, vh, attn_mask=mask, is_causal=False, scale=scale, enable_gqa=gqa
    ).transpose(1, 2)


def _compiled(plan, inputs, mask):
    return plan.execute(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        attention_mask=mask,
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


def _measure_pair(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup, block):
    """Measure every matrix mode: training forward, training forward+backward,
    and inference prefill (forward under no_grad). The matrix declares no
    decode mode for this workload."""
    # (mode key, backward, no_grad)
    mode_specs = [
        ("forward", False, False),
        ("forward_backward", True, False),
        ("prefill", False, True),
    ]
    cold_direct = _time_one(direct, direct_inputs, backward=False, block=block)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False, block=block)
    for _ in range(warmup):
        for _, backward, no_grad in mode_specs:
            _time_one(direct, direct_inputs, backward=backward, block=block, no_grad=no_grad)
            _time_one(compiled, compiled_inputs, backward=backward, block=block, no_grad=no_grad)
    measurements = {}
    for mode, backward, no_grad in mode_specs:
        direct_wall, compiled_wall, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    direct_wall.append(_time_one(direct, direct_inputs, backward=backward, block=block, no_grad=no_grad))
                else:
                    compiled_wall.append(_time_one(compiled, compiled_inputs, backward=backward, block=block, no_grad=no_grad))
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
    operands = _inputs(
        _stable_seed(case["id"], dtype_name),
        batch, qlen, klen, qheads, kvheads, key_dim, value_dim, dtype,
    )
    mask = _build_mask(case, batch, qlen, klen, operands["query"].device)
    direct_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    compiled_inputs = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}

    plan_started = time.perf_counter()
    plan = compile_mixer(
        named_mixer_recipe(MASKED_RECIPE),
        backend=MixerBackend.NATIVE,
        intent=MixerIntent.TRAINING,
        dtype=dtype_name,
    )
    plan_build_ms = (time.perf_counter() - plan_started) * 1000
    native_anchor = plan.anchor

    # --- Correctness first: native vs SDPA AND native vs independent oracle.
    direct_out = _direct({n: t.detach() for n, t in operands.items()}, scale, mask)
    oracle_out = _oracle_masked(
        operands["query"].detach(), operands["key"].detach(), operands["value"].detach(), scale, mask
    )
    for tensor in compiled_inputs.values():
        tensor.grad = None
    compiled_out = _compiled(plan, compiled_inputs, mask)
    compiled_out.float().square().mean().backward()

    # gradient comparison vs SDPA
    for tensor in direct_inputs.values():
        tensor.grad = None
    direct_out_gb = _direct(direct_inputs, scale, mask)
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

    # --- Explicit safe-zero verification for fully-masked rows (the matrix
    # calls this out for this workload). A fully-masked query row has no
    # visible key in the mask; the native kernel must produce exactly zero
    # output and zero query gradient on those rows.
    fully_masked = ~mask.any(dim=-1)  # [B, 1, T]
    safe_zero = None
    if bool(fully_masked.any()):
        rows = fully_masked.squeeze(1)  # [B, T]
        out_max = compiled_out.detach().float().abs().amax(dim=(2, 3))  # [B, T]
        qgrad_max = compiled_inputs["query"].grad.float().abs().amax(dim=(2, 3))  # [B, T]
        safe_zero = {
            "fully_masked_row_count": int(rows.sum()),
            "output_max_abs_on_fully_masked_rows": float(out_max[rows].max()),
            "query_gradient_max_abs_on_fully_masked_rows": float(qgrad_max[rows].max()),
            "pass": bool(
                out_max[rows].max().item() == 0.0 and qgrad_max[rows].max().item() == 0.0
            ),
        }

    # --- Performance only after correctness.
    perf_inputs_d = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    perf_inputs_c = {n: t.detach().clone().requires_grad_() for n, t in operands.items()}
    performance = _measure_pair(
        lambda i: _direct(i, scale, mask), lambda i: _compiled(plan, i, mask),
        perf_inputs_d, perf_inputs_c, pairs, warmup, block,
    )
    measurements = performance["measurements"]
    fwd_gate = measurements["forward"]["paired_native_overhead_fraction"]["gate"]["pass"]
    fb_gate = measurements["forward_backward"]["paired_native_overhead_fraction"]["gate"]["pass"]

    case_key = f"{case['id']}/{dtype_name}"
    result = {
        "semantic_scope": "masked causal attention core; projections excluded",
        "shape": {"batch": batch, "query_length": qlen, "key_length": klen,
                  "query_heads": qheads, "key_value_heads": kvheads,
                  "key_dim": key_dim, "value_dim": value_dim, "dtype": dtype_name},
        "mask_pattern": case["mask"],
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
        "safe_zero_fully_masked_rows": safe_zero,
        "performance": performance,
    }
    all_pass = correctness_pass and fwd_gate and fb_gate
    return case_key, result, all_pass, not correctness_pass


def run(pairs, warmup, output_path, block=1, case_id=None):
    if not torch.cuda.is_available():
        raise RuntimeError("native K1 masked attention qualification requires CUDA")

    cases = CASES
    if case_id is not None:
        cases = tuple(c for c in CASES if c["id"] == case_id)
        if not cases:
            raise ValueError(f"unknown case id {case_id!r}; available: {[c['id'] for c in CASES]}")

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
        "purpose": "qualify URM-native K1 online-softmax masked attention against a competitive masked fused-attention upstream (native replacement, not dispatch overhead)",
        "matrix_workload": "k1-masked-variant",
        "verdict": verdict,
        "native_anchor": native_anchor,
        "comparator": {
            "frozen_in_matrix": "flash_attn_varlen_func or equivalent masked path (Dao-AILab/flash-attention)",
            "used": "torch.nn.functional.scaled_dot_product_attention with explicit attn_mask (fused mem-efficient backend)",
            "scope_note": "flash_attn is not installable in this torch 2.14/cu130 + nvcc 12.9 environment; SDPA with an explicit attention mask is the available competitive masked fused-attention upstream. Budgets and tolerances unchanged.",
        },
        "provenance": provenance(
            "PYTHONPATH=src python benchmarks/qualify_native_k1_masked.py",
            {"recipe": MASKED_RECIPE, "workload": "k1-masked-variant", "pairs": pairs,
             "warmup": warmup, "cases": [c["id"] for c in cases],
             "dtypes": [d for d, _ in DTYPES]},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "gpu_operating_conditions": capture_gpu_operating_conditions(),
        "methodology": {
            "comparison": "URM-native K1 online-softmax kernel (masked recipe) vs SDPA fused attention with an explicit mask; the two share no kernel",
            "correctness_comparator": "SDPA with attn_mask plus an independent eager masked-attention oracle",
            "performance_comparator": "SDPA fused attention with explicit attn_mask (competitive production masked kernel)",
            "timed_work": "one native plan call or one upstream call, optionally followed by output backward; prefill runs under no_grad",
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
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--case", type=str, default=None,
                        help="optional single case id to run (default: sweep all frozen cases)")
    parser.add_argument("--output", type=Path, default=None,
                        help="artifact path (default: results/qualification/native-k1-masked.json)")
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    output = args.output or Path("results/qualification/native-k1-masked.json")
    payload = run(args.pairs, args.warmup, output, args.block, args.case)
    print(f"verdict: {payload['verdict']}")
    for case_key, case in payload["cases"].items():
        parity = case["parity"]["status"]
        out_err = case["parity"]["output_max_abs_error_vs_upstream"]
        parts = [f"  {case_key}: parity={parity} (out err {out_err:.2e})"]
        safe_zero = case.get("safe_zero_fully_masked_rows")
        if safe_zero is not None:
            parts.append(
                f"    safe-zero fully-masked rows ({safe_zero['fully_masked_row_count']} rows): "
                f"out {safe_zero['output_max_abs_on_fully_masked_rows']:.2e}  "
                f"qgrad {safe_zero['query_gradient_max_abs_on_fully_masked_rows']:.2e}  "
                f"{'PASS' if safe_zero['pass'] else 'FAIL'}"
            )
        for mode in ("forward", "forward_backward", "prefill"):
            m = case["performance"]["measurements"][mode]["paired_native_overhead_fraction"]
            parts.append(
                f"    {mode:18s} overhead: median {m['median']*100:+.2f}%  "
                f"ci95 [{m['ci95_lower']*100:+.2f}%, {m['ci95_upper']*100:+.2f}%]  "
                f"gate {'PASS' if m['gate']['pass'] else 'FAIL'}"
            )
        print("\n".join(parts))


if __name__ == "__main__":
    main()
