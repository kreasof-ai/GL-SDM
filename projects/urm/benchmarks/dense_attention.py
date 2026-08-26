"""Dense causal attention comparator: oracle / SDPA-math / upstream / URM.

Production-comparison slice for URM (docs/baselines.md level model):

1. ``oracle``     - explicit fp32 softmax-reduce (correctness only; skipped
                    with a memory-based ``not_applicable`` reason when the
                    S x S matrix would not fit);
2. ``sdpa_math``  - PyTorch SDPA restricted to its math backend;
3. ``flash_attn`` - pinned FlashAttention upstream called directly;
4. ``urm_flash_attn``   - the SAME upstream call behind the URM adapter;
5. ``sdpa_flash``       - PyTorch SDPA restricted to its flash backend;
6. ``urm_sdpa_flash``   - that same call behind the URM adapter.

Measurement methodology (v2):

- Q/K/V and the fixed output gradient are preallocated leaf tensors; input
  cloning, random generation, and unrelated allocation never occur inside a
  timed region. Input preparation is timed separately.
- The timed region is exactly one forward call, or one forward call plus its
  ``backward()`` on a freshly built graph with a fixed preallocated output
  gradient. Leaf gradients are cleared outside the timed region.
- Cold first calls are recorded separately from warm steady state.
- Every sample records both an end-to-end wall-clock latency (host dispatch
  included) and a CUDA-event device-span latency (kernel-side), kept as
  distinct fields.
- Direct-versus-adapter samples are taken as paired, interleaved A/B/B/A
  measurements on one clock so co-tenant drift cancels; the same paired
  samples produce the per-implementation statistics. Adapter overhead is
  reported as the distribution of per-pair fractions (median, p95,
  dispersion, bootstrap CI), not only as a ratio of two independent medians.

All levels share identical semantics: causal alignment, BHSD layout at the
boundary, GQA grouping, scale 1/sqrt(head_dim), dropout disabled, fp16/bf16.
Unsupported configurations are recorded as ``not_applicable`` with a reason,
never as zero performance.

Usage (from projects/urm):
    PYTHONPATH=src python benchmarks/dense_attention.py [--seq 32768 ...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from urm.adapters import UrmDenseCausalAttentionAdapter
from urm.adapters.dense_attention import flash_attn_version

# Oracle and math-backend attention materialize an S x S score matrix in fp32.
ORACLE_MEMORY_BUDGET_BYTES = 8 * 1024**3

STEADY_STATE_MIN_SEQ = 2048
OVERHEAD_GATE_FRACTION = 0.05


def oracle_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Explicit fp32 reference: O = softmax(Q K^T * scale + causal_mask) V."""
    _b, _h, s_q, d = q.shape
    s_k = k.shape[2]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if s_q == s_k:
        mask = torch.ones(s_q, s_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    else:
        # Decode-style rectangular causal mask (query i attends keys <= offset).
        offsets = (s_k - s_q) + torch.arange(s_q, device=q.device)
        limits = offsets.view(1, 1, s_q, 1)
        key_positions = torch.arange(s_k, device=q.device).view(1, 1, 1, s_k)
        scores = scores.masked_fill(key_positions > limits, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    del scores
    out = torch.matmul(probs.to(v.dtype), v)
    del probs
    return out.to(q.dtype)


def sdpa_math(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with sdpa_kernel([SDPBackend.MATH]):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=q.shape[2] == k.shape[2], enable_gqa=True
        )


def sdpa_flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Restricting the dispatcher to FLASH_ATTENTION makes a silent fallback to
    # math/mem-efficient impossible: unsupported shapes raise instead.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=q.shape[2] == k.shape[2], enable_gqa=True
        )


def flash_attn_direct(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    from flash_attn import flash_attn_func

    scale = 1.0 / math.sqrt(q.shape[-1])
    out = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=q.shape[2] == k.shape[2],
        softmax_scale=scale,
    )
    return out.transpose(1, 2)


class TimedImplementation:
    """Preallocated-input callable bundle for one implementation and mode."""

    def __init__(self, name, function, inputs, output_gradient):
        self.name = name
        self.function = function
        self.inputs = inputs
        self.output_gradient = output_gradient

    def invoke(self):
        out = self.function(*self.inputs)
        if self.output_gradient is not None:
            out.backward(self.output_gradient)
        return out

    def clear_gradients(self):
        if self.output_gradient is not None:
            for tensor in self.inputs:
                tensor.grad = None


def _time_one(call):
    """One synchronized measurement of a single invocation.

    Returns (wall_ms, device_span_ms): the wall clock covers host dispatch
    plus kernel execution end-to-end; the CUDA-event span covers only the
    device-side interval between the surrounding event records.
    """
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    call()
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return wall_ms, start_event.elapsed_time(end_event)


def distribution_stats(samples):
    ordered = sorted(samples)
    n = len(ordered)
    median = statistics.median(ordered)

    def percentile(fraction):
        return ordered[max(0, math.ceil(fraction * n) - 1)]

    return {
        "n": n,
        "median": median,
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "iqr": percentile(0.75) - percentile(0.25),
        "median_abs_deviation": statistics.median([abs(x - median) for x in ordered]),
        "stdev": statistics.stdev(ordered) if n > 1 else 0.0,
    }


def bootstrap_median_ci(samples, *, resamples=2000, level=0.95, seed=20260826):
    """Percentile bootstrap confidence interval for the sample median."""
    rng = random.Random(seed)
    n = len(samples)
    medians = sorted(
        statistics.median([samples[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    )
    alpha = (1.0 - level) / 2.0
    low_index = max(0, math.ceil(alpha * resamples) - 1)
    high_index = min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)
    return {
        "level": level,
        "resamples": resamples,
        "lower": medians[low_index],
        "upper": medians[high_index],
    }


def paired_overhead_stats(
    direct_samples, adapted_samples, direct_device, adapted_device
):
    """Distributions of paired (adapted - direct) deltas over interleaved pairs."""
    fractions = [(a - d) / d for d, a in zip(direct_samples, adapted_samples)]
    device_fractions = [(a - d) / d for d, a in zip(direct_device, adapted_device)]
    absolute_us = [(a - d) * 1000.0 for d, a in zip(direct_samples, adapted_samples)]
    ratio_of_medians = (
        statistics.median(adapted_samples) / statistics.median(direct_samples) - 1.0
    )
    return {
        "pairs": len(fractions),
        "wall_fraction": {
            **distribution_stats(fractions),
            "bootstrap_ci95_median": bootstrap_median_ci(fractions),
        },
        "device_span_fraction": distribution_stats(device_fractions),
        "absolute_delta_us": distribution_stats(absolute_us),
        "ratio_of_independent_medians_minus_one": ratio_of_medians,
    }


def round_stats(stats):
    rounded = {}
    for key, value in stats.items():
        rounded[key] = round(value, 6) if isinstance(value, float) else value
    ci = rounded.get("bootstrap_ci95_median")
    if isinstance(ci, dict):
        ci["lower"] = round(ci["lower"], 6)
        ci["upper"] = round(ci["upper"], 6)
    return rounded


def useful_flops(b: int, h: int, s_q: int, s_k: int, d: int, mode: str) -> float:
    """Causal-attention algorithmic FLOPs; documented model, not instructions.

    Forward: two S-length GEMMs per query position; causal halves the work:
    2 (GEMMs) * 2 (mul-add) * B*H*S*min(S,S)*D * 0.5 = 2*B*H*S*S*D.
    Backward: dQ, dK, dV GEMMs plus recompute ~ 2.5x forward (FlashAttention
    convention): 5*B*H*S*S*D.
    """
    pairs = s_q * min(s_k, s_q if s_q == s_k else s_k) if s_q == s_k else s_q * s_k
    causal_factor = 0.5 if s_q == s_k else min(1.0, (s_q / max(s_k, 1)) / 2 + 0.5)
    base = 4.0 * b * h * pairs * d * causal_factor
    return {"forward": base, "backward": 2.5 * base}[mode]


def backend_selection_evidence(name: str) -> dict:
    evidence = {"implementation": name}
    flags = torch.backends.cuda
    if name == "flash_attn":
        evidence["kernel"] = "flash_attn.flash_attn_func (direct FA2 CUDA kernels)"
        evidence["silent_fallback_possible"] = False
    elif name == "urm_flash_attn":
        evidence["kernel"] = (
            "flash_attn.flash_attn_func behind UrmDenseCausalAttentionAdapter"
        )
        evidence["silent_fallback_possible"] = False
    elif name == "sdpa_flash":
        evidence["kernel"] = "torch SDPA restricted to SDPBackend.FLASH_ATTENTION"
        evidence["dispatcher_forced_to"] = ["FLASH_ATTENTION"]
        evidence["silent_fallback_possible"] = False
        evidence["flash_sdp_enabled_flag"] = flags.flash_sdp_enabled()
    elif name == "urm_sdpa_flash":
        evidence["kernel"] = (
            "torch SDPA restricted to SDPBackend.FLASH_ATTENTION behind "
            "UrmDenseCausalAttentionAdapter"
        )
        evidence["dispatcher_forced_to"] = ["FLASH_ATTENTION"]
        evidence["silent_fallback_possible"] = False
    elif name == "sdpa_math":
        evidence["kernel"] = "torch SDPA restricted to SDPBackend.MATH"
        evidence["dispatcher_forced_to"] = ["MATH"]
        evidence["silent_fallback_possible"] = False
    return evidence


def _cold_first_call_ms(impl) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter()
    impl.invoke()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    impl.clear_gradients()
    return elapsed_ms


def collect_solo_samples(impl, warmup, pairs):
    """Warm up, then take `pairs` synchronized single-call samples."""
    for _ in range(warmup):
        impl.invoke()
        impl.clear_gradients()
    torch.cuda.synchronize()
    wall_samples = []
    device_samples = []
    for _ in range(pairs):
        wall, device_span = _time_one(impl.invoke)
        impl.clear_gradients()
        wall_samples.append(wall)
        device_samples.append(device_span)
    return wall_samples, device_samples


def collect_paired_samples(direct_impl, adapted_impl, warmup, pairs):
    """Interleaved A/B, B/A sampling; within-pair order alternates per pair."""
    for impl in (direct_impl, adapted_impl):
        for _ in range(warmup):
            impl.invoke()
            impl.clear_gradients()
    torch.cuda.synchronize()
    direct_wall = []
    adapted_wall = []
    direct_device = []
    adapted_device = []
    for pair_index in range(pairs):
        first, second = (
            (direct_impl, adapted_impl)
            if pair_index % 2 == 0
            else (adapted_impl, direct_impl)
        )
        for impl in (first, second):
            wall, device_span = _time_one(impl.invoke)
            impl.clear_gradients()
            if impl is direct_impl:
                direct_wall.append(wall)
                direct_device.append(device_span)
            else:
                adapted_wall.append(wall)
                adapted_device.append(device_span)
    return direct_wall, adapted_wall, direct_device, adapted_device


def build_metrics(
    name,
    mode,
    shape,
    cold_ms,
    wall_samples,
    device_samples,
):
    wall_stats = distribution_stats(wall_samples)
    device_stats = distribution_stats(device_samples)
    metrics = {
        "mode": mode,
        "cold_first_call_ms": round(cold_ms, 3),
        "backend_selection": backend_selection_evidence(name),
        "wall_ms": round_stats(wall_stats),
        "device_span_ms": round_stats(device_stats),
        "median_ms": wall_stats["median"],
        "p95_ms": wall_stats["p95"],
    }
    flops = useful_flops(shape[0], shape[1], shape[2], shape[2], shape[3], mode)
    metrics["useful_tflops"] = round(flops / (metrics["median_ms"] / 1e3) / 1e12, 3)
    batch, _heads, sequence, _dim = shape
    metrics["tokens_per_second"] = round(
        batch * sequence / (metrics["median_ms"] / 1e3), 1
    )
    return metrics


def measure_peak_bytes(invoke_and_clear) -> int:
    torch.cuda.reset_peak_memory_stats()
    invoke_and_clear()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def run_case(case: dict, args, bf16_peak) -> dict:
    b, h, kv, s, d = (
        case["batch"],
        case["heads"],
        case["kv_heads"],
        case["sequence"],
        case["head_dim"],
    )
    dtype = getattr(torch, args.dtype)

    # ---- Input preparation: timed separately, never inside a timed region.
    prep_started = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(case.get("seed", 7))
    q_data = torch.randn((b, h, s, d), device="cuda", dtype=dtype, generator=generator)
    k_data = torch.randn((b, kv, s, d), device="cuda", dtype=dtype, generator=generator)
    v_data = torch.randn((b, kv, s, d), device="cuda", dtype=dtype, generator=generator)
    gradient_generator = torch.Generator(device="cuda").manual_seed(
        int(case.get("seed", 7)) + 99
    )
    grad_data = torch.randn(
        (b, h, s, d), device="cuda", dtype=dtype, generator=gradient_generator
    )
    torch.cuda.synchronize()
    input_prep_ms = (time.perf_counter() - prep_started) * 1000.0

    oracle_feasible = b * h * s * s * 4 <= ORACLE_MEMORY_BUDGET_BYTES
    math_feasible = oracle_feasible

    reference = None
    correctness = {}
    if oracle_feasible and not args.skip_oracle:
        reference = oracle_attention(q_data, k_data, v_data)

    direct_functions = {}
    if math_feasible:
        direct_functions["sdpa_math"] = sdpa_math
    fa_identity = flash_attn_version()
    if fa_identity.get("package") == "flash-attn":
        direct_functions["flash_attn"] = flash_attn_direct
    else:
        print(f"  flash_attn: not_applicable ({fa_identity})")
    try:
        sdpa_flash(q_data[:, :h], k_data[:, :h], v_data[:, :h])
        direct_functions["sdpa_flash"] = sdpa_flash
    except (RuntimeError, ValueError) as error:
        print(f"  sdpa_flash: not_applicable ({error})")

    modes = ("forward", "backward") if not args.forward_only else ("forward",)
    shape = (b, h, s, d)
    implementations: dict[str, dict] = {}
    overhead: dict[str, dict] = {}

    for mode in modes:
        needs_grad = mode == "backward"
        inputs = tuple(
            tensor.detach().clone().requires_grad_(needs_grad)
            for tensor in (q_data, k_data, v_data)
        )
        output_gradient = grad_data if needs_grad else None

        timed = {}
        for name, function in direct_functions.items():
            timed[name] = TimedImplementation(name, function, inputs, output_gradient)
        for name in ("flash_attn", "sdpa_flash"):
            if name in direct_functions:
                adapter = UrmDenseCausalAttentionAdapter(name)
                timed[f"urm_{name}"] = TimedImplementation(
                    f"urm_{name}", adapter.execute, inputs, output_gradient
                )

        # Correctness on one untimed forward call per implementation.
        if reference is not None and mode == "forward":
            for name, impl in timed.items():
                out = impl.function(*inputs)
                difference = (out.float() - reference.float()).abs()
                correctness[name] = {
                    "max_abs_error_vs_oracle": round(difference.max().item(), 6),
                    "mean_abs_error_vs_oracle": round(difference.mean().item(), 8),
                }
                del out, difference

        samples: dict[str, tuple[list, list]] = {}
        cold_times = {}
        for name, impl in timed.items():
            cold_times[name] = _cold_first_call_ms(impl)

        # Paired interleaved sampling for each direct/adapter couple; solo
        # sampling for implementations without an adapter counterpart. The
        # paired samples are also the source of per-implementation stats.
        paired_names = [name for name in ("flash_attn", "sdpa_flash") if name in timed]
        consumed = set()
        for name in paired_names:
            adapter_name = f"urm_{name}"
            dw, aw, dd, ad = collect_paired_samples(
                timed[name], timed[adapter_name], args.warmup, args.pairs
            )
            samples[name] = (dw, dd)
            samples[adapter_name] = (aw, ad)
            consumed.update((name, adapter_name))
            dispatch_bound = s < STEADY_STATE_MIN_SEQ
            stats_pair = paired_overhead_stats(dw, aw, dd, ad)
            median_fraction = stats_pair["wall_fraction"]["median"]
            gate_pass = median_fraction <= OVERHEAD_GATE_FRACTION or dispatch_bound
            overhead.setdefault(name, {})[mode] = {
                "interleaved": True,
                **{
                    key: (round_stats(value) if isinstance(value, dict) else value)
                    for key, value in stats_pair.items()
                },
                "gate": {
                    "limit_fraction": OVERHEAD_GATE_FRACTION,
                    "steady_state": not dispatch_bound,
                    "dispatch_bound_shape": dispatch_bound,
                    "pass": gate_pass,
                },
            }
        for name, impl in timed.items():
            if name in consumed:
                continue
            wall, device = collect_solo_samples(impl, args.warmup, args.pairs)
            samples[name] = (wall, device)

        for name, impl in timed.items():
            wall, device = samples[name]
            entry = build_metrics(name, mode, shape, cold_times[name], wall, device)
            entry["peak_allocated_bytes"] = measure_peak_bytes(
                lambda impl=impl: (impl.invoke(), impl.clear_gradients())
            )
            if bf16_peak:
                entry["mfu_fraction_of_measured_bf16_peak"] = round(
                    entry["useful_tflops"] / bf16_peak, 4
                )
            implementations.setdefault(name, {})[mode] = entry

    return {
        "case": {**case, "dtype": args.dtype},
        "input_preparation_ms": round(input_prep_ms, 3),
        "correctness": correctness,
        "implementations": implementations,
        "adapter_overhead": overhead,
        "oracle_evaluated": reference is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=None, help="defaults to heads")
    parser.add_argument("--head-dim", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 2048, 8192, 32768])
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/attention/dense-causal.json")
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    limits_path = Path("results/device-limits.json")
    bf16_peak = None
    if limits_path.exists():
        bf16_peak = json.loads(limits_path.read_text())["bf16_tensor_core"][
            "bf16_tensor_core_tfps_measured"
        ]

    properties = torch.cuda.get_device_properties(0)
    document = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "flash_attention_upstream": flash_attn_version(),
        "mfu_denominator": {
            "bf16_tensor_core_tfps_measured": bf16_peak,
            "source": "results/device-limits.json (measured BF16 GEMM)",
            "note": "Tensor-core peak is appropriate here because dense "
            "attention kernels execute MMA instructions, unlike the "
            "FP32 CUDA-core routed-reduction kernels.",
        },
        "methodology": {
            "timed_region": (
                "one forward call, or forward+backward on a fresh graph built "
                "from preallocated leaf inputs with a fixed preallocated "
                "output gradient; gradients cleared outside the timed region"
            ),
            "input_handling": (
                "Q/K/V and the output gradient are preallocated before "
                "timing; no cloning, generation, or allocation happens "
                "inside timed regions; input preparation is reported "
                "separately per case"
            ),
            "latency_fields": {
                "wall_ms": "end-to-end latency per call, host dispatch included",
                "device_span_ms": "CUDA-event device-side span per call",
            },
            "sampling": (
                f"paired interleaved direct/adapter single calls, "
                f"{args.pairs} pairs after {args.warmup} warmup rounds per "
                f"mode; within-pair order alternates A/B, B/A so position "
                f"bias cancels"
            ),
            "overhead_reporting": (
                "distribution of per-pair (adapted-direct)/direct fractions "
                "with a percentile bootstrap CI for the median; ratio of "
                "independent medians recorded only for continuity"
            ),
            "cold_vs_warm": (
                "first-call compile/autotune cost is recorded separately "
                "from warm steady-state samples"
            ),
            "steady_state_min_seq": STEADY_STATE_MIN_SEQ,
            "overhead_gate_fraction": OVERHEAD_GATE_FRACTION,
        },
        "semantics": {
            "causal": True,
            "layout": "BHSD",
            "scale": "1/sqrt(head_dim)",
            "dropout": 0.0,
            "gqa": "heads % kv_heads == 0",
        },
        "cases": {},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batch:
        for head_dim in args.head_dim:
            for seq in args.seq:
                kv_heads = args.kv_heads or args.heads
                case = {
                    "name": f"b{batch}_h{args.heads}_d{head_dim}_s{seq}",
                    "batch": batch,
                    "heads": args.heads,
                    "kv_heads": kv_heads,
                    "sequence": seq,
                    "head_dim": head_dim,
                    "seed": 7,
                }
                print(f"== {case['name']}")
                result = run_case(case, args, bf16_peak)
                document["cases"][case["name"]] = result
                # Incremental write so partial runs are usable artifacts.
                args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
