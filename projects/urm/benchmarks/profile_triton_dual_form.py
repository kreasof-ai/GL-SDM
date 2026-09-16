"""Comprehensive Profiling and Alignment of the Triton Dual-Form SDM Kernel.

Measures:
1. Exact gradient and forward alignment with upstream SDM reference under matched random seeds.
2. Latency, throughput, and MFU breakdown on NVIDIA A10G for frozen Phase 3 shapes.
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F
import triton

from urm.triton_kernels.dual_form_sdm import (
    TritonDualFormSDMFunction,
    _triton_dual_form_bwd_kernel,
    _triton_dual_form_fwd_kernel,
    triton_dual_form_sdm,
)


def sdm_upstream_reference(memory, k_idx, k_val, v, beta, g, q_idx, q_val):
    """Independent token-by-token recurrence matching upstream Facebook SDM."""
    B_T = k_idx.shape[0]
    mem = memory
    readings = []
    for t in range(B_T):
        mem_at_k = mem[k_idx[t]]
        decay = torch.exp(g[t]).unsqueeze(-1)
        mem_read = mem_at_k * decay
        retrieved = (k_val[t].unsqueeze(-1) * mem_read).sum(0)
        delta_v = beta[t] * (v[t] - retrieved)
        write_vals = k_val[t].unsqueeze(-1) * delta_v.unsqueeze(0)
        delta_at_k = mem_read - mem_at_k + write_vals
        update = torch.zeros_like(mem)
        idx_expanded = k_idx[t].unsqueeze(-1).expand_as(delta_at_k)
        update.scatter_add_(0, idx_expanded, delta_at_k)
        mem = mem + update
        mem_at_q = mem[q_idx[t]]
        reading = (q_val[t].unsqueeze(-1) * mem_at_q).sum(0)
        readings.append(reading)
    return torch.stack(readings, dim=0)


def profile_alignment(seeds=(42, 1701)):
    device = "cuda"
    dtype = torch.float32
    T, S, D, W, R = 32, 128, 16, 8, 8

    print("=========================================================================================")
    print("1. TRITON KERNEL GRADIENT & FORWARD ALIGNMENT vs UPSTREAM SDM")
    print(f"Shapes: T={T}, S={S}, D={D}, W={W}, R={R} | Dtype: {dtype}")
    print("=========================================================================================")

    for seed in seeds:
        torch.manual_seed(seed)
        memory = torch.randn(S, D, device=device, dtype=dtype) * 0.1
        k_idx = torch.stack([torch.randperm(S, device=device)[:W].sort().values for _ in range(T)])
        q_idx = torch.stack([torch.randperm(S, device=device)[:R].sort().values for _ in range(T)])

        kw = torch.rand(T, W, device=device, dtype=dtype)
        kw = kw / kw.sum(dim=-1, keepdim=True)
        qw = torch.rand(T, R, device=device, dtype=dtype)
        qw = qw / qw.sum(dim=-1, keepdim=True)

        v = torch.randn(T, D, device=device, dtype=dtype) * 0.1
        beta = torch.rand(T, 1, device=device, dtype=dtype) * 0.5
        g = -torch.rand(T, 1, device=device, dtype=dtype) * 0.05

        # 1. Reference Run
        mem_r = memory.clone().requires_grad_(True)
        v_r = v.clone().requires_grad_(True)
        b_r = beta.clone().requires_grad_(True)
        g_r = g.clone().requires_grad_(True)
        kw_r = kw.clone().requires_grad_(True)
        qw_r = qw.clone().requires_grad_(True)

        out_r = sdm_upstream_reference(mem_r, k_idx, kw_r, v_r, b_r, g_r, q_idx, qw_r)
        (out_r ** 2).sum().backward()

        # 2. Triton Kernel Run
        mem_t = memory.clone().unsqueeze(0).requires_grad_(True)
        v_t = v.clone().unsqueeze(0).requires_grad_(True)
        b_t = beta.clone().unsqueeze(0).requires_grad_(True)
        g_t = g.clone().unsqueeze(0).requires_grad_(True)
        kw_t = kw.clone().unsqueeze(0).requires_grad_(True)
        qw_t = qw.clone().unsqueeze(0).requires_grad_(True)

        out_t, _ = triton_dual_form_sdm(
            mem_t, q_idx.unsqueeze(0), qw_t,
            write_indices=k_idx.unsqueeze(0), write_weights=kw_t,
            values=v_t, beta=b_t, log_decay=g_t,
            block_t=32,
        )
        (out_t.squeeze(0) ** 2).sum().backward()

        print(f"\nAudit for Seed={seed}:")
        print(f"{'Quantity / Gradient':<22} | {'Cosine Sim':<12} | {'Max Abs Diff':<14} | {'Gate Status'}")
        print("-" * 75)

        comparisons = [
            ("Forward Readings (Y)", out_r, out_t.squeeze(0)),
            ("grad(Values)", v_r.grad, v_t.grad.squeeze(0)),
            ("grad(Beta)", b_r.grad, b_t.grad.squeeze(0)),
            ("grad(Initial Memory)", mem_r.grad, mem_t.grad.squeeze(0)),
            ("grad(Write Weights)", kw_r.grad, kw_t.grad.squeeze(0)),
            ("grad(Read Weights)", qw_r.grad, qw_t.grad.squeeze(0)),
        ]

        for name, r_t, tr_t in comparisons:
            u_f = r_t.reshape(-1).float()
            d_f = tr_t.reshape(-1).float()
            cos_sim = F.cosine_similarity(u_f.unsqueeze(0), d_f.unsqueeze(0)).item()
            abs_diff = (r_t - tr_t).abs().max().item()
            passed = cos_sim > 0.9999 and abs_diff < 1e-4
            print(f"{name:<22} | {cos_sim:<12.8f} | {abs_diff:<14.4e} | {'PASSED' if passed else 'CHECK'}")


def profile_performance():
    device = "cuda"
    dtype = torch.bfloat16
    P, T, S, D, W, R = 12, 1024, 4096, 64, 64, 64
    block_t = 64

    print("\n=========================================================================================")
    print("2. TRITON KERNEL PERFORMANCE PROFILING ON NVIDIA A10G (PHASE 3 SHAPES)")
    print(f"Shapes: P={P}, T={T}, S={S}, D={D}, W={W}, R={R} (12 Heads, Context 1024, Slots 4096)")
    print("=========================================================================================")

    memory = torch.randn(P, S, D, device=device, dtype=dtype, requires_grad=True)
    wi = torch.zeros(P, T, W, device=device, dtype=torch.int32)
    ri = torch.zeros(P, T, R, device=device, dtype=torch.int32)
    for p in range(P):
        for t in range(T):
            wi[p, t] = torch.randperm(S, device=device)[:W]
            ri[p, t] = torch.randperm(S, device=device)[:R]

    w = torch.rand(P, T, W, device=device, dtype=dtype, requires_grad=True)
    q = torch.rand(P, T, R, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(P, T, D, device=device, dtype=dtype, requires_grad=True)
    beta = torch.rand(P, T, 1, device=device, dtype=dtype, requires_grad=True)
    log_decay = -torch.rand(P, T, 1, device=device, dtype=dtype) * 0.05

    # Measure raw Triton forward kernel (_triton_dual_form_fwd_kernel)
    slot_log_decay = torch.zeros(P, T, S, device=device, dtype=torch.float32)
    slot_log_decay.scatter_add_(2, wi.long(), log_decay.float().expand(-1, -1, W))
    slot_cum = torch.zeros(P, T + 1, S, device=device, dtype=torch.float32)
    torch.cumsum(slot_log_decay, dim=1, out=slot_cum[:, 1:])

    V0 = torch.zeros(P, T, D, device=device, dtype=torch.float32)
    Y0 = torch.zeros(P, T, D, device=device, dtype=torch.float32)
    out = torch.empty((P, T, D), device=device, dtype=dtype)
    delta = torch.empty((P, T, D), device=device, dtype=torch.float32)
    A = torch.zeros((P, T, T), device=device, dtype=torch.float32)
    Omega = torch.zeros((P, T, T), device=device, dtype=torch.float32)

    grid = (P, triton.cdiv(T, block_t))

    # Warmup
    _triton_dual_form_fwd_kernel[grid](
        wi, w, ri, q, v, beta.float(), slot_cum, V0, Y0, out, delta, A, Omega,
        P=P, T=T, SLOTS=S, W=W, R=R, D=D, BLOCK_T=block_t,
    )
    torch.cuda.synchronize()

    # Profile Triton Forward
    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        _triton_dual_form_fwd_kernel[grid](
            wi, w, ri, q, v, beta.float(), slot_cum, V0, Y0, out, delta, A, Omega,
            P=P, T=T, SLOTS=S, W=W, R=R, D=D, BLOCK_T=block_t,
        )
    torch.cuda.synchronize()
    raw_fwd_ms = (time.perf_counter() - t0) / iters * 1000.0

    # Profile Full triton_dual_form_sdm (Forward + Autograd Backward)
    y, m = triton_dual_form_sdm(memory, ri, q, write_indices=wi, write_weights=w, values=v, beta=beta, log_decay=log_decay, block_t=block_t)
    loss = (y.float() ** 2).sum() + (m.float() ** 2).sum()
    loss.backward()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y, m = triton_dual_form_sdm(memory, ri, q, write_indices=wi, write_weights=w, values=v, beta=beta, log_decay=log_decay, block_t=block_t)
        loss = (y.float() ** 2).sum() + (m.float() ** 2).sum()
        loss.backward()
    torch.cuda.synchronize()
    total_step_ms = (time.perf_counter() - t0) / iters * 1000.0

    # FLOPs: GEMMs (forward + backward) = ~522 GFLOPs
    gemm_flops = 522.0
    tflops_fwd = (209.4 / raw_fwd_ms)
    tflops_total = (gemm_flops / total_step_ms)
    a10g_peak = 66.166

    print(f"  Raw Triton Forward Kernel Latency: {raw_fwd_ms:.2f} ms")
    print(f"  Forward Tensor Core Throughput:   {tflops_fwd:.2f} TFLOP/s ({tflops_fwd / a10g_peak * 100:.1f}% MFU)")
    print(f"  Total Mixer Step (Fwd + Bwd):      {total_step_ms:.2f} ms")
    print(f"  Overall Throughput (Fwd + Bwd):    {tflops_total:.2f} TFLOP/s ({tflops_total / a10g_peak * 100:.1f}% MFU)")
    print(f"  Speedup vs Native v0 Fwd (8.37 ms):{8.37 / raw_fwd_ms:.2f}x")
    print(f"  Speedup vs Native v0 Step (~87 ms):{87.0 / total_step_ms:.2f}x")
    print("=========================================================================================")


if __name__ == "__main__":
    profile_alignment(seeds=(42, 1701))
    profile_performance()
