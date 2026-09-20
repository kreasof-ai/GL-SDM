"""Historical sparse routed delta prototype; NOT a certified compiler backend.

Known defects include omitted chunked decay, missing decay gradients and unstable
clipped decay factors. Claims below describe the original experiment, not current
acceptance. See docs/kernels/sparse-delta.md for the corrected contract.

Demonstrates high-MFU prefill and training execution by reparameterizing
the serial slot recurrence into:
  1. Address-Collision-Coupled Lower-Triangular Delta Solve
  2. Sparse Causal Cross-Attention GEMM on Tensor Cores
  3. Boundary State Folding into persistent memory M_T
  4. O(1) constant-time recurrent decode step

This respects the origin intention of URM as a unified sequence mixer kernel
without architecture-specific semantics: the sparse indexer (e.g. Product-Key
or 16D Foveal Indexer) is decoupled from the execution kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from urm.compiler.semantic import SparseReadTiming


class SparseDeltaFunction(torch.autograd.Function):
    """Experimental sparse-delta autograd function; backward is not yet complete."""

    @staticmethod
    def forward(
        ctx,
        memory: torch.Tensor,         # [P, S, D]
        read_indices: torch.Tensor,   # [P, T, R]
        read_weights: torch.Tensor,   # [P, T, R]
        write_indices: torch.Tensor,  # [P, T, W]
        write_weights: torch.Tensor,  # [P, T, W]
        values: torch.Tensor,         # [P, T, D]
        beta: torch.Tensor,           # [P, T, 1]
        log_decay: torch.Tensor,      # [P, T, 1]
        read_timing: SparseReadTiming = SparseReadTiming.AFTER_UPDATE,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = memory.device
        dtype = memory.dtype
        P, T, D = values.shape
        slots = memory.shape[1]

        # Accumulation in float32 for training stability and Tensor Core precision
        m_fp32 = memory.float()
        ww_fp32 = write_weights.float()
        rw_fp32 = read_weights.float()
        v_fp32 = values.float()
        b_fp32 = beta.float()
        g_fp32 = log_decay.float()

        # 1. Cumulative slot log decay
        slot_log_decay = torch.zeros(P, T, slots, device=device, dtype=torch.float32)
        slot_log_decay.scatter_add_(2, write_indices, g_fp32.expand(-1, -1, write_indices.shape[-1]))
        slot_cum = torch.zeros(P, T + 1, slots, device=device, dtype=torch.float32)
        torch.cumsum(slot_log_decay, dim=1, out=slot_cum[:, 1:])

        # 2. Initial state projections V0 and Y0
        dec_w0 = torch.exp(slot_cum[:, 1:].gather(2, write_indices) - slot_cum[:, :1].expand(-1, T, -1).gather(2, write_indices))
        dec_r0 = torch.exp(slot_cum[:, 1:].gather(2, read_indices) - slot_cum[:, :1].expand(-1, T, -1).gather(2, read_indices))

        m0_w = torch.take_along_dim(m_fp32, write_indices.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, -1, D)
        V0 = (ww_fp32.unsqueeze(-1) * dec_w0.unsqueeze(-1) * m0_w).sum(dim=2)

        m0_r = torch.take_along_dim(m_fp32, read_indices.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, -1, D)
        Y0 = (rw_fp32.unsqueeze(-1) * dec_r0.unsqueeze(-1) * m0_r).sum(dim=2)

        # 3. Collision and cross-attention GEMMs on Tensor Cores
        exp_w_curr = torch.exp(slot_cum[:, 1:].gather(2, write_indices))
        exp_w_prev = torch.exp((-slot_cum[:, 1:].gather(2, write_indices)).clamp(max=15.0))
        exp_r_curr = torch.exp(slot_cum[:, 1:].gather(2, read_indices))

        # Scatter weights into slot vectors in bfloat16 for Tensor Core MMA
        W_curr = torch.zeros(P, T, slots, device=device, dtype=torch.bfloat16).scatter_(
            -1, write_indices, (ww_fp32 * exp_w_curr).to(torch.bfloat16)
        )
        W_prev = torch.zeros(P, T, slots, device=device, dtype=torch.bfloat16).scatter_(
            -1, write_indices, (ww_fp32 * exp_w_prev).to(torch.bfloat16)
        )
        Q_curr = torch.zeros(P, T, slots, device=device, dtype=torch.bfloat16).scatter_add_(
            -1, read_indices, (rw_fp32 * exp_r_curr).to(torch.bfloat16)
        )

        tril_strict = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), -1)
        tril_causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), 0)

        # Tensor Core GEMMs: (P, T, S) x (P, S, T) -> (P, T, T)
        A = torch.bmm(W_curr, W_prev.transpose(1, 2)).float() * tril_strict
        Omega_read = torch.bmm(Q_curr, W_prev.transpose(1, 2)).float() * tril_causal

        # 4. Triangular delta solve on Tensor Cores
        eye = torch.eye(T, device=device, dtype=torch.float32).unsqueeze(0)
        sys = eye + b_fp32 * A
        RHS = b_fp32 * (v_fp32 - V0)
        Delta = torch.linalg.solve_triangular(sys, RHS, upper=False, unitriangular=True)

        # 5. Output readings via GEMM on Tensor Cores
        # Omega_read @ Delta: (P, T, T) x (P, T, D) -> (P, T, D)
        Y = Y0 + torch.bmm(Omega_read.to(torch.bfloat16), Delta.to(torch.bfloat16)).float()

        # 6. Final State Folding at sequence boundary
        dec_final = torch.exp(slot_cum[:, T:] - slot_cum[:, :1])  # [P, 1, slots]
        final_memory = dec_final.squeeze(1).unsqueeze(-1) * m_fp32

        dec_delta = torch.exp(
            slot_cum[:, T:].expand(-1, T, -1).gather(2, write_indices)
            - slot_cum[:, 1:].gather(2, write_indices)
        )
        delta_scaled = (ww_fp32 * dec_delta).unsqueeze(-1) * Delta.unsqueeze(2)  # [P, T, W, D]

        # Vectorized scatter-add into final_memory
        for p in range(P):
            final_memory[p].index_add_(0, write_indices[p].reshape(-1), delta_scaled[p].reshape(-1, D))

        # Save minimal tensors (Delta is only 1.57 MiB for 12 heads, context 1024, D=64)
        ctx.save_for_backward(
            Delta, A, Omega_read, b_fp32, slot_cum,
            write_indices, read_indices, ww_fp32, rw_fp32,
            m_fp32, v_fp32, V0, Y0,
            W_curr, W_prev, Q_curr, exp_w_curr, exp_w_prev, exp_r_curr
        )
        ctx.T, ctx.P, ctx.D, ctx.slots = T, P, D, slots
        ctx.read_timing = read_timing

        return Y.to(dtype), final_memory.to(dtype)

    @staticmethod
    def backward(ctx, dY: torch.Tensor, dFinalMemory: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (
            Delta, A, Omega_read, b_fp32, slot_cum,
            write_indices, read_indices, ww_fp32, rw_fp32,
            m_fp32, v_fp32, V0, Y0,
            W_curr, W_prev, Q_curr, exp_w_curr, exp_w_prev, exp_r_curr
        ) = ctx.saved_tensors

        T, P, D, slots = ctx.T, ctx.P, ctx.D, ctx.slots
        device = dY.device

        dY_fp32 = dY.float()
        dM_fp32 = dFinalMemory.float()

        # Step 1: dDelta = Omega_read^T @ dY + dDelta_state (GEMM)
        dDelta_read = torch.bmm(Omega_read.transpose(1, 2).to(torch.bfloat16), dY_fp32.to(torch.bfloat16)).float()

        dec_delta = torch.exp(
            slot_cum[:, T:].expand(-1, T, -1).gather(2, write_indices)
            - slot_cum[:, 1:].gather(2, write_indices)
        )
        dM_gathered = torch.take_along_dim(
            dM_fp32, write_indices.reshape(P, -1, 1).expand(-1, -1, D), dim=1
        ).view(P, T, -1, D)
        dDelta_state = (ww_fp32.unsqueeze(-1) * dec_delta.unsqueeze(-1) * dM_gathered).sum(dim=2)
        dDelta = dDelta_read + dDelta_state

        # Step 2: Adjoint solve on Lambda: (I + A^T D_beta) Lambda = dDelta
        eye = torch.eye(T, device=device, dtype=torch.float32).unsqueeze(0)
        sys_adj = eye + torch.bmm(A.transpose(1, 2), torch.diag_embed(b_fp32.squeeze(-1)))
        Lambda = torch.linalg.solve_triangular(sys_adj, dDelta, upper=True, unitriangular=True)

        # Step 3: Cotangents
        dV = b_fp32 * Lambda
        dV0 = -dV
        dY0 = dY_fp32

        # dbeta = sum_d Lambda * (V - V0 - A @ Delta)
        A_Delta = torch.bmm(A, Delta)
        dbeta = (Lambda * (v_fp32 - V0 - A_Delta)).sum(dim=-1, keepdim=True)

        # dA and dOmega (GEMMs on Tensor Cores)
        tril_strict = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), -1)
        tril_causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), 0)
        dA = -torch.bmm((b_fp32 * Lambda).to(torch.bfloat16), Delta.transpose(1, 2).to(torch.bfloat16)).float() * tril_strict
        dOmega = torch.bmm(dY_fp32.to(torch.bfloat16), Delta.transpose(1, 2).to(torch.bfloat16)).float() * tril_causal

        # Step 4: dMemory
        dec_final = torch.exp(slot_cum[:, T:] - slot_cum[:, :1])
        dMemory = dec_final.squeeze(1).unsqueeze(-1) * dM_fp32

        dec_w0 = torch.exp(slot_cum[:, 1:].gather(2, write_indices) - slot_cum[:, :1].expand(-1, T, -1).gather(2, write_indices))
        dec_r0 = torch.exp(slot_cum[:, 1:].gather(2, read_indices) - slot_cum[:, :1].expand(-1, T, -1).gather(2, read_indices))

        term_v0 = (ww_fp32.unsqueeze(-1) * dec_w0.unsqueeze(-1)) * dV0.unsqueeze(2)
        term_y0 = (rw_fp32.unsqueeze(-1) * dec_r0.unsqueeze(-1)) * dY0.unsqueeze(2)

        for p in range(P):
            dMemory[p].index_add_(0, write_indices[p].reshape(-1), term_v0[p].reshape(-1, D))
            dMemory[p].index_add_(0, read_indices[p].reshape(-1), term_y0[p].reshape(-1, D))

        # Step 5: dw and dq via Batched GEMMs on Tensor Cores
        dW_curr = torch.bmm(dA.to(torch.bfloat16), W_prev).float()
        dW_prev = torch.bmm(dA.transpose(1, 2).to(torch.bfloat16), W_curr).float() + torch.bmm(
            dOmega.transpose(1, 2).to(torch.bfloat16), Q_curr
        ).float()
        dQ_curr = torch.bmm(dOmega.to(torch.bfloat16), W_prev).float()

        m0_w = torch.take_along_dim(m_fp32, write_indices.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, -1, D)
        m0_r = torch.take_along_dim(m_fp32, read_indices.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, -1, D)

        dw_v0 = (dec_w0.unsqueeze(-1) * m0_w * dV0.unsqueeze(2)).sum(dim=-1)
        dw_final = (dec_delta.unsqueeze(-1) * dM_gathered * Delta.unsqueeze(2)).sum(dim=-1)
        dw_gemm = (dW_curr.gather(2, write_indices) * exp_w_curr) + (dW_prev.gather(2, write_indices) * exp_w_prev)
        dWrite_weights = dw_v0 + dw_final + dw_gemm

        dq_y0 = (dec_r0.unsqueeze(-1) * m0_r * dY0.unsqueeze(2)).sum(dim=-1)
        dq_gemm = dQ_curr.gather(2, read_indices) * exp_r_curr
        dRead_weights = dq_y0 + dq_gemm

        dLogDecay = torch.zeros_like(b_fp32)

        return (
            dMemory.to(dFinalMemory.dtype),
            None,  # read_indices
            dRead_weights.to(dY.dtype),
            None,  # write_indices
            dWrite_weights.to(dY.dtype),
            dV.to(dY.dtype),
            dbeta.to(dY.dtype),
            dLogDecay.to(dY.dtype),
            None,  # read_timing
        )


def sparse_delta(
    memory: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_timing: SparseReadTiming = SparseReadTiming.AFTER_UPDATE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute sparse routed delta sequence mixer using Tensor Core acceleration."""
    return SparseDeltaFunction.apply(
        memory,
        read_indices,
        read_weights,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_timing,
    )


def chunked_sparse_delta(
    memory: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor | None = None,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute Chunked sparse routed delta with batched intra-chunk GEMMs and inter-chunk state folding.

    When chunk_size >= T, executes the full-sequence single-chunk solve.
    When chunk_size < T, partitions into Nc = T // chunk_size chunks, batching
    intra-chunk collision GEMMs and triangular solves across all (P * Nc) chunks,
    achieving up to ~50% Tensor Core MFU on NVIDIA A10G.
    """
    P, T, D = values.shape
    if chunk_size >= T:
        if log_decay is None:
            log_decay = torch.zeros(P, T, 1, device=values.device, dtype=values.dtype)
        return sparse_delta(
            memory, read_indices, read_weights,
            write_indices=write_indices, write_weights=write_weights,
            values=values, beta=beta, log_decay=log_decay,
        )

    C = chunk_size
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size C={C}"
    Nc = T // C
    S = memory.shape[1]
    W = write_indices.shape[-1]
    R = read_indices.shape[-1]
    device = values.device
    dtype = values.dtype

    # Reshape inputs to batched chunks: [P * Nc, C, ...]
    wi_ch = write_indices.view(P * Nc, C, W)
    w_ch = write_weights.view(P * Nc, C, W)
    ri_ch = read_indices.view(P * Nc, C, R)
    q_ch = read_weights.view(P * Nc, C, R)
    v_ch = values.view(P * Nc, C, D)
    beta_ch = beta.view(P * Nc, C, 1)

    # 1. Expand sparse representations in batched chunks
    W_raw = torch.zeros(P * Nc, C, S, device=device, dtype=dtype).scatter_(-1, wi_ch.long(), w_ch)
    Q_raw = torch.zeros(P * Nc, C, S, device=device, dtype=dtype).scatter_add_(-1, ri_ch.long(), q_ch)

    # 2. Batched Intra-Chunk Collision & Read Attention GEMMs: [P * Nc, C, C]
    tril_strict = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool), -1)
    tril_causal = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool), 0)

    A_ch = torch.bmm(W_raw, W_raw.transpose(1, 2)) * tril_strict
    Omega_ch = torch.bmm(Q_raw, W_raw.transpose(1, 2)) * tril_causal

    # 3. Batched Triangular Inversion: [P * Nc, C, C]
    eye = torch.eye(C, device=device, dtype=dtype).unsqueeze(0).expand(P * Nc, -1, -1)
    M_sys = eye + beta_ch * A_ch
    M_inv = torch.linalg.solve_triangular(M_sys.float(), eye.float(), upper=False).to(dtype)

    # 4. Inter-Chunk State Folding Loop
    readings = []
    curr_mem = memory.clone()

    for c in range(Nc):
        idx_p = torch.arange(P, device=device) * Nc + c
        wi_c = write_indices[:, c * C : (c + 1) * C]
        w_c = write_weights[:, c * C : (c + 1) * C]
        ri_c = read_indices[:, c * C : (c + 1) * C]
        q_c = read_weights[:, c * C : (c + 1) * C]
        v_c = values[:, c * C : (c + 1) * C]
        b_c = beta[:, c * C : (c + 1) * C]

        # Gather V0 and Y0 from current chunk memory state
        mem_gathered_w = torch.take_along_dim(
            curr_mem, wi_c.long().reshape(P, -1, 1).expand(-1, -1, D), dim=1
        ).view(P, C, W, D)
        V0_c = (w_c.unsqueeze(-1) * mem_gathered_w).sum(dim=2)

        mem_gathered_r = torch.take_along_dim(
            curr_mem, ri_c.long().reshape(P, -1, 1).expand(-1, -1, D), dim=1
        ).view(P, C, R, D)
        Y0_c = (q_c.unsqueeze(-1) * mem_gathered_r).sum(dim=2)

        # Solve for delta_c within chunk
        rhs_c = b_c * (v_c - V0_c)
        delta_c = torch.bmm(M_inv[idx_p], rhs_c)

        # Read cross-attention output for chunk c
        Y_c = Y0_c + torch.bmm(Omega_ch[idx_p], delta_c)
        readings.append(Y_c)

        # Memory boundary update for next chunk
        delta_scatter = (w_c.unsqueeze(-1) * delta_c.unsqueeze(2)).reshape(P, -1, D)
        wi_flat = wi_c.long().reshape(P, -1)
        update = torch.zeros_like(curr_mem)
        for p in range(P):
            update[p].index_add_(0, wi_flat[p], delta_scatter[p])
        curr_mem = curr_mem + update

    return torch.cat(readings, dim=1), curr_mem


__all__ = ["SparseDeltaFunction", "chunked_sparse_delta", "sparse_delta"]
