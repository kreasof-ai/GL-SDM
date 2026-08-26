"""Dense causal attention comparator: oracle / SDPA-math / upstream / URM.

First production-comparison slice for URM (docs/baselines.md level model):

1. ``oracle``     - explicit fp32 softmax-reduce (correctness only; skipped
                    with a memory-based ``not_applicable`` reason when the
                    S x S matrix would not fit);
2. ``sdpa_math``  - PyTorch SDPA restricted to its math backend;
3. ``flash_attn`` - pinned FlashAttention upstream called directly;
4. ``urm_flash_attn``   - the SAME upstream call behind the URM adapter;
5. ``sdpa_flash``       - PyTorch SDPA restricted to its flash backend;
6. ``urm_sdpa_flash``   - that same call behind the URM adapter.

All levels share identical semantics: causal alignment, BHSD layout at the
boundary, GQA grouping, scale 1/sqrt(head_dim), dropout disabled, fp16/bf16,
warm steady state with cold compile reported separately. Unsupported
configurations are recorded as ``not_applicable`` with a reason, never as
zero performance.

Usage (from projects/urm):
    PYTHONPATH=src python benchmarks/dense_attention.py [--seq 32768 ...]
"""

from __future__ import annotations

import argparse
import json
import math
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


def benchmark_callable(function, warmup: int, samples: int, inner: int):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    function()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - started) * 1000.0
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / inner)
    return {
        "cold_first_call_ms": round(cold_ms, 3),
        "median_ms": statistics.median(timings),
        "p95_ms": sorted(timings)[max(0, math.ceil(0.95 * len(timings)) - 1)],
        "samples": samples,
        "inner_iterations": inner,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


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


def run_case(case: dict, args) -> dict[str, object]:
    b, h, kv, s, d = (
        case["batch"],
        case["heads"],
        case["kv_heads"],
        case["sequence"],
        case["head_dim"],
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cuda").manual_seed(case.get("seed", 7))
    q = torch.randn((b, h, s, d), device="cuda", dtype=dtype, generator=generator)
    k = torch.randn((b, kv, s, d), device="cuda", dtype=dtype, generator=generator)
    v = torch.randn((b, kv, s, d), device="cuda", dtype=dtype, generator=generator)

    oracle_feasible = b * h * s * s * 4 <= ORACLE_MEMORY_BUDGET_BYTES
    math_feasible = oracle_feasible

    reference = None
    correctness: dict[str, object] = {}
    if oracle_feasible and not args.skip_oracle:
        reference = oracle_attention(q, k, v)

    implementations: dict[str, callable] = {}
    if math_feasible:
        implementations["sdpa_math"] = sdpa_math
    fa_identity = flash_attn_version()
    if fa_identity.get("package") == "flash-attn":
        implementations["flash_attn"] = flash_attn_direct
        implementations["urm_flash_attn"] = UrmDenseCausalAttentionAdapter(
            "flash_attn"
        ).execute
    else:
        print(f"  flash_attn: not_applicable ({fa_identity})")
    try:
        sdpa_flash(q[:, :h], k[:, :h], v[:, :h])
        implementations["sdpa_flash"] = sdpa_flash
        implementations["urm_sdpa_flash"] = UrmDenseCausalAttentionAdapter(
            "sdpa_flash"
        ).execute
    except (RuntimeError, ValueError) as error:
        print(f"  sdpa_flash: not_applicable ({error})")

    results: dict[str, object] = {}
    modes = ("forward", "backward") if not args.forward_only else ("forward",)

    def make_invoke(
        fn: callable,
        needs_grad: bool,
        seed: int,
        q_src: torch.Tensor,
        k_src: torch.Tensor,
        v_src: torch.Tensor,
        grad: torch.Tensor | None,
    ) -> callable:
        """Bind all loop variables explicitly (no late-binding hazards)."""

        def invoke() -> torch.Tensor:
            q_r = q_src.detach().clone().requires_grad_(needs_grad)
            k_r = k_src.detach().clone().requires_grad_(needs_grad)
            v_r = v_src.detach().clone().requires_grad_(needs_grad)
            out = fn(q_r, k_r, v_r)
            if grad is not None:
                out.backward(grad)
            return out, (q_r, k_r, v_r)

        return invoke

    for name, function in implementations.items():
        entry: dict[str, object] = {}
        for mode in modes:
            needs_grad = mode == "backward"
            seed = int(case.get("seed", 7))
            grad = None
            if needs_grad:
                generator = torch.Generator(device="cuda").manual_seed(seed + 99)
                grad = torch.randn(
                    (b, h, s, d),
                    device="cuda",
                    dtype=dtype,
                    generator=generator,
                )

            invoke = make_invoke(function, needs_grad, seed, q, k, v, grad)

            first_out, first_grads = invoke()
            q_grad, _k_grad, _v_grad = first_grads
            if reference is not None and not needs_grad:
                difference = (first_out.float() - reference.float()).abs()
                correctness[name] = {
                    "max_abs_error_vs_oracle": round(difference.max().item(), 6),
                    "mean_abs_error_vs_oracle": round(difference.mean().item(), 8),
                }
            if needs_grad:
                entry[f"{mode}_gradients_present"] = q_grad.grad is not None
            del first_out, first_grads

            metrics = benchmark_callable(invoke, args.warmup, args.samples, args.inner)
            flops = useful_flops(b, h, s, s, d, mode)
            metrics["useful_tflops"] = round(
                flops / (metrics["median_ms"] / 1e3) / 1e12, 3
            )
            metrics["tokens_per_second"] = round(
                b * s / (metrics["median_ms"] / 1e3), 1
            )
            metrics["mode"] = mode
            entry[mode] = metrics
        results[name] = entry

    # Adapter overhead against the same implementation called directly.
    overhead: dict[str, object] = {}
    for direct_key, adapter_key in (
        ("flash_attn", "urm_flash_attn"),
        ("sdpa_flash", "urm_sdpa_flash"),
    ):
        if direct_key in results and adapter_key in results:
            fractions = {}
            for mode in modes:
                direct = results[direct_key][mode]["median_ms"]
                adapted = results[adapter_key][mode]["median_ms"]
                fractions[mode] = round((adapted - direct) / direct, 4)
            overhead[direct_key] = {"median_overhead_fraction": fractions}

    return {
        "case": {**case, "dtype": args.dtype},
        "correctness": correctness,
        "implementations": results,
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
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--inner", type=int, default=5)
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
    document: dict[str, object] = {
        "schema_version": 1,
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
                result = run_case(case, args)
                if bf16_peak:
                    for modes in result["implementations"].values():
                        for m in modes.values():
                            if isinstance(m, dict) and "useful_tflops" in m:
                                m["mfu_fraction_of_measured_bf16_peak"] = round(
                                    m["useful_tflops"] / bf16_peak, 4
                                )
                document["cases"][case["name"]] = result
                # Incremental write so partial runs are usable artifacts.
                args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
