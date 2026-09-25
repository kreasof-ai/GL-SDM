"""Native Triton provider for the typed TriangularSolve ordinary operator.

The TriangularSolve op (generality axis UT) computes the strict-causal operand
correction ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` by forward substitution:
``u_t = v_t − β_t·Σ_{j<t} P_tj·u_j``. The sequential dependency runs over the
token axis (row ``t`` reads only ``u_{<t}``), so one program owns a
(batch·head, value-dim block) fragment and walks the token axis in order —
the ordered dependency is structural, never atomic.

The backward is the adjoint of forward substitution — a backward substitution
over the cotangent ``g``: ``g_t = ∂L/∂u_t − Σ_{s>t} β_s·P_s,t·g_s`` gives the
value cotangent, ``∂L/∂P_tj = −β_t·(u_j·g_t)`` (j<t) the probability cotangent
and ``∂L/∂β_t = −(g_t·Σ_{j<t} P_tj·u_j)`` the diagonal cotangent (the gradient
flows through the correction row, not through ``u_t`` itself). Accumulation is
fp32 in both directions; cross-D-block partial sums of the scalar/row
cotangents use relaxed atomics (the same policy as the K3 native backward).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...ir.program import TriangularSolve  # noqa: F401  (descriptor type)

NATIVE_TRIANGULAR_SOLVE_NAME = "urm_native_triangular_solve_v1"


@triton.jit
def _triangular_solve_forward_kernel(
    probs,
    beta,
    value,
    out,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)  # one per (batch · head)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = dims < D
    value_base = program * T * D
    probs_base = program * T * T
    for t in range(0, T):
        rhs = tl.load(
            value + value_base + t * D + dims, mask=d_mask, other=0.0
        ).to(tl.float32)
        correction = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, t, BLOCK_T):
            cols = start + tl.arange(0, BLOCK_T)
            c_mask = cols < t
            weights = tl.load(
                probs + probs_base + t * T + cols, mask=c_mask, other=0.0
            ).to(tl.float32)
            previous = tl.load(
                out + value_base + cols[:, None] * D + dims[None, :],
                mask=c_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            correction += tl.sum(weights[:, None] * previous, axis=0)
        diagonal = tl.load(beta + program * T + t).to(tl.float32)
        tl.store(
            out + value_base + t * D + dims,
            rhs - diagonal * correction,
            mask=d_mask,
        )


@triton.jit
def _triangular_solve_backward_kernel(
    probs,
    beta,
    out,
    grad_out,
    grad_probs,
    grad_beta,
    grad_value,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    program = tl.program_id(0).to(tl.int64)  # one per (batch · head)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = dims < D
    value_base = program * T * D
    probs_base = program * T * T
    for reverse in range(0, T):
        t = T - 1 - reverse
        cotangent = tl.load(
            grad_out + value_base + t * D + dims, mask=d_mask, other=0.0
        ).to(tl.float32)
        for start in range(t + 1, T, BLOCK_T):
            rows = start + tl.arange(0, BLOCK_T)
            r_mask = rows < T
            column = tl.load(
                probs + probs_base + rows * T + t, mask=r_mask, other=0.0
            ).to(tl.float32)
            diagonals = tl.load(beta + program * T + rows, mask=r_mask, other=0.0).to(
                tl.float32
            )
            later = tl.load(
                grad_value + value_base + rows[:, None] * D + dims[None, :],
                mask=r_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            cotangent -= tl.sum((column * diagonals)[:, None] * later, axis=0)
        tl.store(grad_value + value_base + t * D + dims, cotangent, mask=d_mask)
        diagonal = tl.load(beta + program * T + t).to(tl.float32)
        correction = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, t, BLOCK_T):
            cols = start + tl.arange(0, BLOCK_T)
            c_mask = cols < t
            weights = tl.load(
                probs + probs_base + t * T + cols, mask=c_mask, other=0.0
            ).to(tl.float32)
            previous = tl.load(
                out + value_base + cols[:, None] * D + dims[None, :],
                mask=c_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            correction += tl.sum(weights[:, None] * previous, axis=0)
            grad_row = -diagonal * tl.sum(previous * cotangent[None, :], axis=1)
            tl.atomic_add(grad_probs + probs_base + t * T + cols, grad_row, mask=c_mask)
        # dβ_t = −(g_t · Σ_{j<t} P_tj·u_j) — the correction row, not u_t.
        tl.atomic_add(
            grad_beta + program * T + t,
            -tl.sum(correction * cotangent, axis=0),
            sem="relaxed",
        )


def _launch_parameters(tokens: int, dim: int) -> tuple[int, int, int, int]:
    """(BLOCK_T, BLOCK_D, num_warps, num_stages); fp32 tiles, D-block parallel programs.

    The inner gather tiles are [BLOCK_T, BLOCK_D] fp32; with Triton's default pipelining
    (num_stages=3) the SMEM buffers multiply, and 256×64 fp32 × 3 stages = 192KB — over
    the A10G's 101KB limit. The token loop is serial (row t reads u_{<t}), so pipelining
    buys nothing: cap BLOCK_T at 64 and run num_stages=1, keeping SMEM ≈ 64×64×4 = 16KB
    per tile, comfortably under the limit at any T.
    """
    block_t = min(max(16, triton.next_power_of_2(tokens)), 64)
    block_d = max(16, min(64, triton.next_power_of_2(dim)))
    return block_t, block_d, 4 if block_d >= 64 else 2, 1


def _triangular_solve_forward(
    probs: torch.Tensor,
    beta: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """fp32 forward substitution; returns ``u`` in float32."""
    B, H, T, D = value.shape
    out = torch.empty((B, H, T, D), device=value.device, dtype=torch.float32)
    block_t, block_d, warps, stages = _launch_parameters(T, D)
    grid = (B * H, triton.cdiv(D, block_d))
    _triangular_solve_forward_kernel[grid](
        probs,
        beta,
        value,
        out,
        T=T,
        D=D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _triangular_solve_backward(
    probs: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    grad_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjoint backward substitution; cotangents in the operand dtypes."""
    B, H, T, D = out.shape
    grad_probs = torch.zeros_like(probs, dtype=torch.float32)
    grad_beta = torch.zeros_like(beta, dtype=torch.float32)
    grad_value = torch.empty_like(out, dtype=torch.float32)
    block_t, block_d, warps, stages = _launch_parameters(T, D)
    grid = (B * H, triton.cdiv(D, block_d))
    _triangular_solve_backward_kernel[grid](
        probs,
        beta,
        out,
        grad_out.contiguous(),
        grad_probs,
        grad_beta,
        grad_value,
        T=T,
        D=D,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=warps,
        num_stages=stages,
    )
    return (
        grad_probs.to(probs.dtype),
        grad_beta.to(beta.dtype),
        grad_value,
    )


class _TriangularSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs, beta, value):
        out = _triangular_solve_forward(
            probs.contiguous(), beta.contiguous(), value.contiguous()
        )
        ctx.save_for_backward(probs.contiguous(), beta.contiguous(), out)
        ctx.value_dtype = value.dtype
        return out.to(value.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        probs, beta, out = ctx.saved_tensors
        if grad_out is None:
            grad_out = torch.zeros_like(out)
        grad_probs, grad_beta, grad_value = _triangular_solve_backward(
            probs, beta, out, grad_out
        )
        return grad_probs, grad_beta, grad_value.to(ctx.value_dtype)


def triangular_solve(
    probs: torch.Tensor,
    beta: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Solve ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` on the native tier.

    ``probs`` is the strict-lower probability operand ``[B, H, T, T]`` (only the
    strict lower triangle is read), ``beta`` the per-token diagonal ``[B, H, T]``
    and ``value`` the right-hand side ``[B, H, T, D]``. fp32 accumulation.
    """
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (probs, beta, value)
    ):
        return _TriangularSolve.apply(probs, beta, value)
    out = _triangular_solve_forward(
        probs.contiguous(), beta.contiguous(), value.contiguous()
    )
    return out.to(value.dtype)


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class TriangularSolveNativeTritonProvider:
    name = NATIVE_TRIANGULAR_SOLVE_NAME
    family = "triangular_solve"
    tier = "native"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, TriangularSolve):
            return "triangular_solve providers require a TriangularSolve op descriptor"
        if request.accumulation_dtype != "float32":
            return "triangular_solve v1 requires float32 accumulation"
        if not torch.cuda.is_available():
            return "native triangular_solve requires CUDA"
        return None

    def execute(self, request, operands):
        out = triangular_solve(
            operands["probs"], operands["beta"], operands["value"]
        )
        return {"output": out}


PROVIDERS = (TriangularSolveNativeTritonProvider(),)

__all__ = [
    "NATIVE_TRIANGULAR_SOLVE_NAME",
    "PROVIDERS",
    "TriangularSolveNativeTritonProvider",
    "triangular_solve",
]
