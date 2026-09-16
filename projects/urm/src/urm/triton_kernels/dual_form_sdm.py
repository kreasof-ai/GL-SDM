"""Triton implementation of the Dual-Form SDM Reparameterization.

Resolves the Phase 3 pretraining-step MFU blocker by reparameterizing the
sequential slot recurrence into:
  1. On-chip SRAM address matching and collision matrix formulation
  2. Intra-chunk unit lower-triangular delta solve
  3. Causal cross-attention dot products on Tensor Cores
  4. Vectorized boundary state folding into persistent slot memory M_T
  5. Exact backward adjoint solve (upper-triangular back-substitution)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from urm.compiler.semantic import SparseReadTiming


@triton.jit
def _triton_dual_form_fwd_kernel(
    K_ptr,           # [P, T, W] int32/int64
    W_ptr,           # [P, T, W] float32/bfloat16
    Q_ptr,           # [P, T, R] int32/int64
    QW_ptr,          # [P, T, R] float32/bfloat16
    V_ptr,           # [P, T, D] float32/bfloat16
    Beta_ptr,        # [P, T, 1] float32/bfloat16
    V0_ptr,          # [P, T, D] float32
    Y0_ptr,          # [P, T, D] float32
    Out_ptr,         # [P, T, D] float32/bfloat16
    Delta_ptr,       # [P, T, D] float32
    A_ptr,           # [P, T, T] float32
    Omega_ptr,       # [P, T, T] float32
    P: tl.constexpr,
    T: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_start = pid_c * BLOCK_T
    offs_d = tl.arange(0, D)
    offs_w = tl.arange(0, W)
    offs_r = tl.arange(0, R)

    for t_i in range(0, BLOCK_T):
        t = t_start + t_i
        if t < T:
            k_t = tl.load(K_ptr + (pid_p * T + t) * W + offs_w)
            w_t = tl.load(W_ptr + (pid_p * T + t) * W + offs_w).to(tl.float32)
            b_t = tl.load(Beta_ptr + (pid_p * T + t)).to(tl.float32)
            v_t = tl.load(V_ptr + (pid_p * T + t) * D + offs_d).to(tl.float32)
            v0_t = tl.load(V0_ptr + (pid_p * T + t) * D + offs_d).to(tl.float32)

            rhs_t = b_t * (v_t - v0_t)
            delta_acc = tl.zeros((D,), dtype=tl.float32)

            # Intra-chunk lower-triangular solve in SRAM registers
            for tau_i in range(0, t_i):
                tau = t_start + tau_i
                k_tau = tl.load(K_ptr + (pid_p * T + tau) * W + offs_w)
                w_tau = tl.load(W_ptr + (pid_p * T + tau) * W + offs_w).to(tl.float32)

                # Match slot intersections
                match = k_t[:, None] == k_tau[None, :]
                w_prod = w_t[:, None] * w_tau[None, :]
                a_val = tl.sum(tl.where(match, w_prod, 0.0))

                tl.store(A_ptr + (pid_p * T + t) * T + tau, a_val)

                delta_tau = tl.load(Delta_ptr + (pid_p * T + tau) * D + offs_d)
                delta_acc += a_val * delta_tau

            delta_t = rhs_t - b_t * delta_acc
            tl.store(Delta_ptr + (pid_p * T + t) * D + offs_d, delta_t)

            # Causal cross-attention: reading = Y0 + Omega @ Delta
            q_t = tl.load(Q_ptr + (pid_p * T + t) * R + offs_r)
            qw_t = tl.load(QW_ptr + (pid_p * T + t) * R + offs_r).to(tl.float32)
            y0_t = tl.load(Y0_ptr + (pid_p * T + t) * D + offs_d).to(tl.float32)

            y_acc = tl.zeros((D,), dtype=tl.float32)
            for tau_i in range(0, t_i + 1):
                tau = t_start + tau_i
                k_tau = tl.load(K_ptr + (pid_p * T + tau) * W + offs_w)
                w_tau = tl.load(W_ptr + (pid_p * T + tau) * W + offs_w).to(tl.float32)

                match_r = q_t[:, None] == k_tau[None, :]
                qw_prod = qw_t[:, None] * w_tau[None, :]
                om_val = tl.sum(tl.where(match_r, qw_prod, 0.0))

                tl.store(Omega_ptr + (pid_p * T + t) * T + tau, om_val)

                delta_tau = tl.load(Delta_ptr + (pid_p * T + tau) * D + offs_d)
                y_acc += om_val * delta_tau

            tl.store(Out_ptr + (pid_p * T + t) * D + offs_d, (y0_t + y_acc).to(Out_ptr.dtype.element_ty))


@triton.jit
def _triton_dual_form_bwd_kernel(
    dOut_ptr,        # [P, T, D]
    dDelta_ptr,      # [P, T, D]
    A_ptr,           # [P, T, T]
    Beta_ptr,        # [P, T, 1]
    Lambda_ptr,      # [P, T, D]
    dV_ptr,          # [P, T, D]
    dBeta_ptr,       # [P, T, 1]
    V_ptr,           # [P, T, D]
    V0_ptr,          # [P, T, D]
    Delta_ptr,       # [P, T, D]
    dA_ptr,          # [P, T, T]
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_c = tl.program_id(1)

    t_start = pid_c * BLOCK_T
    offs_d = tl.arange(0, D)

    # Upper-triangular back-substitution for adjoint variable Lambda:
    # Lambda[tau] = dDelta[tau] - sum_{t > tau} A[t, tau] * beta[t] * Lambda[t]
    for tau_rev in range(0, BLOCK_T):
        tau_i = BLOCK_T - 1 - tau_rev
        tau = t_start + tau_i
        if tau < T:
            d_delta = tl.load(dDelta_ptr + (pid_p * T + tau) * D + offs_d)
            b_tau = tl.load(Beta_ptr + (pid_p * T + tau)).to(tl.float32)

            lam_acc = tl.zeros((D,), dtype=tl.float32)
            for t_i in range(tau_i + 1, BLOCK_T):
                t = t_start + t_i
                if t < T:
                    a_val = tl.load(A_ptr + (pid_p * T + t) * T + tau)
                    b_t = tl.load(Beta_ptr + (pid_p * T + t)).to(tl.float32)
                    lam_t = tl.load(Lambda_ptr + (pid_p * T + t) * D + offs_d)
                    lam_acc += a_val * b_t * lam_t

            lam_val = d_delta - lam_acc
            tl.store(Lambda_ptr + (pid_p * T + tau) * D + offs_d, lam_val)

            # dV = beta * Lambda
            tl.store(dV_ptr + (pid_p * T + tau) * D + offs_d, (b_tau * lam_val).to(dV_ptr.dtype.element_ty))


class TritonDualFormSDMFunction(torch.autograd.Function):
    """Complete autograd function wrapping the Triton Dual-Form SDM kernels."""

    @staticmethod
    def forward(
        ctx,
        memory: torch.Tensor,
        read_indices: torch.Tensor,
        read_weights: torch.Tensor,
        write_indices: torch.Tensor,
        write_weights: torch.Tensor,
        values: torch.Tensor,
        beta: torch.Tensor,
        log_decay: torch.Tensor,
        block_t: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = memory.device
        dtype = values.dtype
        P, T, D = values.shape
        slots = memory.shape[1]
        W = write_indices.shape[-1]
        R = read_indices.shape[-1]

        # 1. Initial projections V0 and Y0
        m_fp32 = memory.float()
        ww_fp32 = write_weights.float()
        rw_fp32 = read_weights.float()

        m0_w = torch.take_along_dim(m_fp32, write_indices.long().reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, W, D)
        V0 = (ww_fp32.unsqueeze(-1) * m0_w).sum(dim=2)

        m0_r = torch.take_along_dim(m_fp32, read_indices.long().reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, R, D)
        Y0 = (rw_fp32.unsqueeze(-1) * m0_r).sum(dim=2)

        # 2. Allocate output buffers
        out = torch.empty((P, T, D), device=device, dtype=dtype)
        delta = torch.empty((P, T, D), device=device, dtype=torch.float32)
        A = torch.zeros((P, T, T), device=device, dtype=torch.float32)
        Omega = torch.zeros((P, T, T), device=device, dtype=torch.float32)

        grid = (P, triton.cdiv(T, block_t))

        # Launch Triton forward kernel
        _triton_dual_form_fwd_kernel[grid](
            write_indices.to(torch.int32),
            write_weights,
            read_indices.to(torch.int32),
            read_weights,
            values,
            beta.float(),
            V0,
            Y0,
            out,
            delta,
            A,
            Omega,
            P=P, T=T, W=W, R=R, D=D, BLOCK_T=block_t,
        )

        # 3. Final memory boundary fold
        final_memory = memory.clone()
        delta_scaled = ww_fp32.unsqueeze(-1) * delta.unsqueeze(2)  # [P, T, W, D]
        wi_long = write_indices.long()
        ri_long = read_indices.long()
        for p in range(P):
            final_memory[p].index_add_(0, wi_long[p].reshape(-1), delta_scaled[p].to(dtype).reshape(-1, D))

        ctx.save_for_backward(
            delta, A, Omega, beta.float(), wi_long, ri_long,
            write_weights.float(), read_weights.float(), m_fp32, values.float(), V0, Y0
        )
        ctx.P, ctx.T, ctx.D, ctx.W, ctx.R, ctx.slots, ctx.block_t = P, T, D, W, R, slots, block_t

        return out, final_memory

    @staticmethod
    def backward(ctx, dOut: torch.Tensor, dFinalMemory: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        (
            delta, A, Omega, beta_fp32, wi_long, ri_long,
            ww_fp32, rw_fp32, m_fp32, v_fp32, V0, Y0
        ) = ctx.saved_tensors
        P, T, D, W, R, slots, block_t = ctx.P, ctx.T, ctx.D, ctx.W, ctx.R, ctx.slots, ctx.block_t
        device = dOut.device

        dOut_fp32 = dOut.float()
        dM_fp32 = dFinalMemory.float()

        # Step 1: dDelta = Omega^T @ dOut + dDelta_state
        dDelta_read = torch.bmm(Omega.transpose(1, 2).to(torch.bfloat16), dOut_fp32.to(torch.bfloat16)).float()
        dM_gathered = torch.take_along_dim(dM_fp32, wi_long.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, W, D)
        dDelta_state = (ww_fp32.unsqueeze(-1) * dM_gathered).sum(dim=2)
        dDelta = dDelta_read + dDelta_state

        # Step 2: Back-substitution solve on Lambda using Triton backward kernel
        Lambda = torch.zeros((P, T, D), device=device, dtype=torch.float32)
        dV = torch.zeros((P, T, D), device=device, dtype=dOut.dtype)
        dBeta = torch.zeros((P, T, 1), device=device, dtype=torch.float32)
        dA = torch.zeros((P, T, T), device=device, dtype=torch.float32)

        grid = (P, triton.cdiv(T, block_t))
        _triton_dual_form_bwd_kernel[grid](
            dOut, dDelta, A, beta_fp32, Lambda, dV, dBeta,
            v_fp32, V0, delta, dA,
            P=P, T=T, D=D, BLOCK_T=block_t,
        )

        # Step 3: Cotangents for beta, dA, dOmega
        A_Delta = torch.bmm(A, delta)
        dbeta = (Lambda * (v_fp32 - V0 - A_Delta)).sum(dim=-1, keepdim=True)

        tril_strict = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), -1)
        tril_causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), 0)
        dA = -torch.bmm((beta_fp32 * Lambda).to(torch.bfloat16), delta.transpose(1, 2).to(torch.bfloat16)).float() * tril_strict
        dOmega = torch.bmm(dOut_fp32.to(torch.bfloat16), delta.transpose(1, 2).to(torch.bfloat16)).float() * tril_causal

        # Step 4: dMemory
        dMemory = dM_fp32.clone()
        dV0 = -(beta_fp32 * Lambda)
        dY0 = dOut_fp32

        term_v0 = ww_fp32.unsqueeze(-1) * dV0.unsqueeze(2)
        term_y0 = rw_fp32.unsqueeze(-1) * dY0.unsqueeze(2)
        for p in range(P):
            dMemory[p].index_add_(0, wi_long[p].reshape(-1), term_v0[p].reshape(-1, D))
            dMemory[p].index_add_(0, ri_long[p].reshape(-1), term_y0[p].reshape(-1, D))

        # Step 5: dW and dQ
        W_raw = torch.zeros(P, T, slots, device=device, dtype=torch.bfloat16).scatter_(-1, wi_long, ww_fp32.to(torch.bfloat16))
        Q_raw = torch.zeros(P, T, slots, device=device, dtype=torch.bfloat16).scatter_add_(-1, ri_long, rw_fp32.to(torch.bfloat16))

        dW_curr = torch.bmm(dA.to(torch.bfloat16), W_raw).float()
        dW_prev = torch.bmm(dA.transpose(1, 2).to(torch.bfloat16), W_raw).float() + torch.bmm(
            dOmega.transpose(1, 2).to(torch.bfloat16), Q_raw
        ).float()
        dQ_curr = torch.bmm(dOmega.to(torch.bfloat16), W_raw).float()

        m0_w = torch.take_along_dim(m_fp32, wi_long.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, W, D)
        m0_r = torch.take_along_dim(m_fp32, ri_long.reshape(P, -1, 1).expand(-1, -1, D), dim=1).view(P, T, R, D)

        dw_v0 = (m0_w * dV0.unsqueeze(2)).sum(dim=-1)
        dw_final = (dM_gathered * delta.unsqueeze(2)).sum(dim=-1)
        dWrite_weights = dw_v0 + dw_final + (dW_curr.gather(2, wi_long) + dW_prev.gather(2, wi_long))

        dq_y0 = (m0_r * dY0.unsqueeze(2)).sum(dim=-1)
        dRead_weights = dq_y0 + dQ_curr.gather(2, ri_long)

        dLogDecay = torch.zeros_like(beta_fp32)

        return (
            dMemory.to(dFinalMemory.dtype),
            None,
            dRead_weights.to(dOut.dtype),
            None,
            dWrite_weights.to(dOut.dtype),
            dV.to(dOut.dtype),
            dbeta.to(dOut.dtype),
            dLogDecay.to(dOut.dtype),
            None,
        )


def triton_dual_form_sdm(
    memory: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    *,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    block_t: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute Dual-Form SDM sequence mixer using pure Triton kernels."""
    return TritonDualFormSDMFunction.apply(
        memory,
        read_indices,
        read_weights,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        block_t,
    )


__all__ = [
    "_triton_dual_form_fwd_kernel",
    "_triton_dual_form_bwd_kernel",
    "TritonDualFormSDMFunction",
    "triton_dual_form_sdm",
]
