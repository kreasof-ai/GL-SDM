"""Direct Empirical Comparison: Pure Triton Dual-Form SDM vs Upstream Facebook SDM.

Directly imports and profiles `lingua.sparse_delta_memory.memory_ops.GatedSparseMemoryWriteRead`
from the official repository (https://github.com/facebookresearch/sparse-delta-memory)
against URM's Triton Dual-Form SDM kernel on identical inputs and random seeds.

Covers:
1. Absolute numerical and gradient alignment (cosine similarity, max abs diff, MSE) across inputs.
2. Fair performance comparison on NVIDIA A10G on frozen Phase 3 shapes:
   (P=12, T=1024, S=4096, D=64, W=64, R=64, dtype=bfloat16).
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

# Ensure upstream sparse-delta-memory is importable
upstream_root = Path("/tmp/opencode/sparse-delta-memory")
if str(upstream_root) not in sys.path:
    sys.path.insert(0, str(upstream_root))

from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead
from urm.triton_kernels.dual_form_sdm import triton_dual_form_sdm
from urm.backends.dual_form_sdm import dual_form_sdm


def benchmark_timing_and_memory(fn, iters=20, warmup=10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.reset_peak_memory_stats()
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    peak_mem_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return elapsed_ms, peak_mem_mib


def run_alignment_audit(seeds=(42, 1701, 2026)):
    print("=" * 95)
    print("1. ABSOLUTE ALIGNMENT AUDIT: TRITON DUAL-FORM SDM vs UPSTREAM GatedSparseMemoryWriteRead")
    print("=" * 95)

    device = "cuda"
    dtype = torch.bfloat16
    P, T, S, D, W, R = 12, 1024, 4096, 64, 64, 64

    for seed in seeds:
        torch.manual_seed(seed)
        mem_base = torch.randn(P, S, D, device=device, dtype=dtype) * 0.1
        wi_base = torch.stack([torch.randperm(S, device=device)[:W].sort().values for _ in range(P * T)]).view(P, T, W)
        ri_base = torch.stack([torch.randperm(S, device=device)[:R].sort().values for _ in range(P * T)]).view(P, T, R)
        w_base = torch.rand(P, T, W, device=device, dtype=dtype)
        w_base = w_base / w_base.sum(dim=-1, keepdim=True)
        q_base = torch.rand(P, T, R, device=device, dtype=dtype)
        q_base = q_base / q_base.sum(dim=-1, keepdim=True)
        v_base = torch.randn(P, T, D, device=device, dtype=dtype) * 0.1
        beta_base = torch.rand(P, T, 1, device=device, dtype=dtype) * 0.5
        g_base = -torch.rand(P, T, 1, device=device, dtype=dtype) * 0.05

        # 1. Upstream Setup (globalized slot offsets across heads)
        offsets = torch.arange(P, device=device, dtype=torch.int64).view(P, 1, 1) * S
        wi_up = (wi_base.long() + offsets).view(P * T, W)
        ri_up = (ri_base.long() + offsets).view(P * T, R)
        mem_up = mem_base.clone().reshape(P * S, D).requires_grad_(True)
        w_up = w_base.clone().reshape(P * T, W).requires_grad_(True)
        q_up = q_base.clone().reshape(P * T, R).requires_grad_(True)
        v_up = v_base.clone().reshape(P * T, D).requires_grad_(True)
        beta_up = beta_base.clone().reshape(P * T, 1).requires_grad_(True)
        g_up = g_base.clone().reshape(P * T, 1).requires_grad_(True)

        out_up, _ = GatedSparseMemoryWriteRead.apply(
            mem_up, wi_up, w_up, v_up, beta_up, g_up, ri_up, q_up, T, True, S, P, False, "none", None
        )
        (out_up.float() ** 2).sum().backward()

        # 2. Triton Dual-Form Setup
        mem_tr = mem_base.clone().requires_grad_(True)
        w_tr = w_base.clone().requires_grad_(True)
        q_tr = q_base.clone().requires_grad_(True)
        v_tr = v_base.clone().requires_grad_(True)
        beta_tr = beta_base.clone().requires_grad_(True)
        g_tr = g_base.clone().requires_grad_(True)

        out_tr, _ = triton_dual_form_sdm(
            mem_tr, ri_base, q_tr, write_indices=wi_base, write_weights=w_tr,
            values=v_tr, beta=beta_tr, log_decay=g_tr, block_t=T
        )
        (out_tr.float() ** 2).sum().backward()

        comparisons = [
            ("Forward Readings (Y)", out_up.view(P, T, D), out_tr),
            ("grad(Values)", v_up.grad.view(P, T, D), v_tr.grad),
            ("grad(Beta)", beta_up.grad.view(P, T, 1), beta_tr.grad),
            ("grad(Initial Memory)", mem_up.grad.view(P, S, D), mem_tr.grad),
            ("grad(Write Weights)", w_up.grad.view(P, T, W), w_tr.grad),
            ("grad(Read Weights)", q_up.grad.view(P, T, R), q_tr.grad),
        ]

        print(f"\nAudit for Seed={seed} (Shapes: P={P}, T={T}, S={S}, D={D}, W={W}, R={R} | Dtype: {dtype}):")
        print(f"{'Quantity / Gradient':<24} | {'Cosine Sim':<12} | {'Max Abs Diff':<14} | {'MSE Error':<14} | {'Status'}")
        print("-" * 85)

        for name, u_t, tr_t in comparisons:
            u_f = u_t.float().flatten().unsqueeze(0)
            tr_f = tr_t.float().flatten().unsqueeze(0)
            cos_sim = F.cosine_similarity(u_f, tr_f).item()
            abs_diff = (u_f - tr_f).abs().max().item()
            mse = F.mse_loss(u_f, tr_f).item()
            passed = cos_sim > 0.995 and abs_diff < 0.05
            print(f"{name:<24} | {cos_sim:<12.8f} | {abs_diff:<14.4e} | {mse:<14.4e} | {'MATCHED' if passed else 'DIVERGED'}")

    print("=" * 95)


def run_performance_comparison():
    print("\n" + "=" * 95)
    print("2. FAIR PERFORMANCE & MFU COMPARISON ON NVIDIA A10G (PHASE 3 SHAPES)")
    print("=" * 95)

    device = "cuda"
    dtype = torch.bfloat16
    P, T, S, D, W, R = 12, 1024, 4096, 64, 64, 64
    a10g_peak_tflops = 66.166
    gemm_fwd_flops = 209.4  # GFLOPs
    gemm_step_flops = 522.0  # GFLOPs

    # Generate synthetic frozen Phase 3 inputs
    torch.manual_seed(2026)
    mem_base = torch.randn(P, S, D, device=device, dtype=dtype) * 0.1
    wi_base = torch.stack([torch.randperm(S, device=device)[:W].sort().values for _ in range(P * T)]).view(P, T, W)
    ri_base = torch.stack([torch.randperm(S, device=device)[:R].sort().values for _ in range(P * T)]).view(P, T, R)
    w_base = torch.rand(P, T, W, device=device, dtype=dtype)
    w_base = w_base / w_base.sum(dim=-1, keepdim=True)
    q_base = torch.rand(P, T, R, device=device, dtype=dtype)
    q_base = q_base / q_base.sum(dim=-1, keepdim=True)
    v_base = torch.randn(P, T, D, device=device, dtype=dtype) * 0.1
    beta_base = torch.rand(P, T, 1, device=device, dtype=dtype) * 0.5
    g_base = -torch.rand(P, T, 1, device=device, dtype=dtype) * 0.05

    offsets = torch.arange(P, device=device, dtype=torch.int64).view(P, 1, 1) * S
    wi_up = (wi_base.long() + offsets).view(P * T, W)
    ri_up = (ri_base.long() + offsets).view(P * T, R)
    mem_up = mem_base.clone().reshape(P * S, D)
    w_up = w_base.clone().reshape(P * T, W)
    q_up = q_base.clone().reshape(P * T, R)
    v_up = v_base.clone().reshape(P * T, D)
    beta_up = beta_base.clone().reshape(P * T, 1)
    g_up = g_base.clone().reshape(P * T, 1)

    # Benchmark Configurations:
    # 1. Upstream GatedSparseMemoryWriteRead (chunk_size = 256, production default)
    def upstream_chunk256_fwd():
        m = mem_up.clone()
        return GatedSparseMemoryWriteRead.apply(
            m, wi_up, w_up, v_up, beta_up, g_up, ri_up, q_up, 256, True, S, P, False, "none", None
        )[0]

    def upstream_chunk256_step():
        m = mem_up.clone().requires_grad_(True)
        w = w_up.clone().requires_grad_(True)
        v = v_up.clone().requires_grad_(True)
        b = beta_up.clone().requires_grad_(True)
        out, _ = GatedSparseMemoryWriteRead.apply(
            m, wi_up, w, v, b, g_up, ri_up, q_up, 256, True, S, P, False, "none", None
        )
        (out.float() ** 2).sum().backward()

    # 2. Upstream GatedSparseMemoryWriteRead (chunk_size = 1024, full WY chunk)
    def upstream_chunk1024_fwd():
        m = mem_up.clone()
        return GatedSparseMemoryWriteRead.apply(
            m, wi_up, w_up, v_up, beta_up, g_up, ri_up, q_up, 1024, True, S, P, False, "none", None
        )[0]

    def upstream_chunk1024_step():
        m = mem_up.clone().requires_grad_(True)
        w = w_up.clone().requires_grad_(True)
        v = v_up.clone().requires_grad_(True)
        b = beta_up.clone().requires_grad_(True)
        out, _ = GatedSparseMemoryWriteRead.apply(
            m, wi_up, w, v, b, g_up, ri_up, q_up, 1024, True, S, P, False, "none", None
        )
        (out.float() ** 2).sum().backward()

    # 3. PyTorch Tensor Core Dual-Form SDM
    def pytorch_gemm_dual_form_fwd():
        return dual_form_sdm(
            mem_base, ri_base, q_base,
            write_indices=wi_base, write_weights=w_base,
            values=v_base, beta=beta_base, log_decay=g_base
        )[0]

    def pytorch_gemm_dual_form_step():
        m = mem_base.clone().requires_grad_(True)
        w = w_base.clone().requires_grad_(True)
        v = v_base.clone().requires_grad_(True)
        b = beta_base.clone().requires_grad_(True)
        out, _ = dual_form_sdm(
            m, ri_base, q_base,
            write_indices=wi_base, write_weights=w,
            values=v, beta=b, log_decay=g_base
        )
        (out.float() ** 2).sum().backward()

    # 4. Triton Dual-Form SDM (block_t = 64)
    def triton_dual_form_fwd():
        return triton_dual_form_sdm(
            mem_base, ri_base, q_base, write_indices=wi_base, write_weights=w_base,
            values=v_base, beta=beta_base, log_decay=g_base, block_t=64
        )[0]

    def triton_dual_form_step():
        m = mem_base.clone().requires_grad_(True)
        w = w_base.clone().requires_grad_(True)
        v = v_base.clone().requires_grad_(True)
        b = beta_base.clone().requires_grad_(True)
        out, _ = triton_dual_form_sdm(
            m, ri_base, q_base, write_indices=wi_base, write_weights=w,
            values=v, beta=b, log_decay=g_base, block_t=64
        )
        (out.float() ** 2).sum().backward()

    runners = [
        ("Upstream SDM (C=256, Default)", upstream_chunk256_fwd, upstream_chunk256_step),
        ("Upstream SDM (C=1024, Full WY)", upstream_chunk1024_fwd, upstream_chunk1024_step),
        ("PyTorch Tensor Core SDM (Ours)", pytorch_gemm_dual_form_fwd, pytorch_gemm_dual_form_step),
        ("Triton Dual-Form SDM (Ours)", triton_dual_form_fwd, triton_dual_form_step),
    ]

    results = []
    for name, fwd_fn, step_fn in runners:
        print(f"Benchmarking: {name}...")
        fwd_ms, fwd_mem = benchmark_timing_and_memory(fwd_fn, iters=15, warmup=5)
        step_ms, step_mem = benchmark_timing_and_memory(step_fn, iters=15, warmup=5)
        bwd_ms = max(0.0, step_ms - fwd_ms)
        tflops = gemm_step_flops / step_ms
        mfu = (tflops / a10g_peak_tflops) * 100.0
        results.append((name, fwd_ms, bwd_ms, step_ms, step_mem, tflops, mfu))

    print("\n" + "=" * 115)
    print(f"{'Implementation':<32} | {'Fwd (ms)':<9} | {'Bwd (ms)':<9} | {'Step (ms)':<10} | {'Peak Mem':<10} | {'Throughput':<12} | {'MFU'}")
    print("-" * 115)
    for name, fwd_ms, bwd_ms, step_ms, step_mem, tflops, mfu in results:
        print(f"{name:<32} | {fwd_ms:<9.2f} | {bwd_ms:<9.2f} | {step_ms:<10.2f} | {step_mem:<7.1f} MiB | {tflops:<6.2f} TFLOP/s | {mfu:.1f}%")
    print("=" * 115)


if __name__ == "__main__":
    run_alignment_audit(seeds=(42, 1701, 2026))
    run_performance_comparison()
