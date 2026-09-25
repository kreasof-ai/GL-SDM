"""Torch reference provider for the typed TriangularSolve ordinary operator.

The TriangularSolve op (generality axis UT) computes the strict-causal operand
correction ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` by forward substitution:
``u_t = v_t − β_t·Σ_{j<t} P_tj·u_j``. This is a data-dependent, read-dependent
correction the canonical K2 rank-1 law cannot express; DeltaFormer's stage-1
value correction is the client. It is an ordinary typed operator — unfused by
default, with its own cost and effects, never a kernel branch over the mixer
families.
"""

from __future__ import annotations

from typing import Any

from ....ir.program import TriangularSolve  # noqa: F401  (descriptor type)


def triangular_solve_forward(probs: Any, beta: Any, value: Any) -> Any:
    """Solve ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` by forward substitution.

    ``probs`` is the strict-lower probability operand ``[B, H, T, T]`` (only the
    strict lower triangle is read; the diagonal and above are ignored). ``beta``
    is the per-token diagonal ``[B, H, T]``. ``value`` is the right-hand side
    ``[B, H, T, D]``. Returns ``u`` ``[B, H, T, D]``. Runs in float32.

    The strict-lower structure means row ``t`` reads only ``u_{<t}``, so the
    forward substitution is exact (no iterative solve needed).
    """
    torch = __import__("torch")
    P = probs.to(torch.float32)
    b = beta.to(torch.float32)
    v = value.to(torch.float32)
    B, H, T, D = v.shape
    us: list[Any] = []
    for t in range(T):
        if t == 0:
            us.append(v[:, :, 0])
            continue
        w = P[:, :, t, :t]                                # [B,H,t]
        u_prev = torch.stack(us, dim=-2)                  # [B,H,t,D]
        us.append(v[:, :, t] - b[:, :, t].unsqueeze(-1) * (w.unsqueeze(-1) * u_prev).sum(-2))
    return torch.stack(us, dim=2)                          # [B,H,T,D]


class TriangularSolveTorchReferenceProvider:
    name = "urm.unified.triangular_solve.reference.v1"
    family = "triangular_solve"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, TriangularSolve):
            return "triangular_solve providers require a TriangularSolve op descriptor"
        if request.accumulation_dtype != "float32":
            return "triangular_solve v1 requires float32 accumulation"
        return None

    def execute(self, request, operands):
        out = triangular_solve_forward(
            operands["probs"], operands["beta"], operands["value"]
        )
        return {"output": out}


PROVIDERS = (TriangularSolveTorchReferenceProvider(),)

__all__ = ["TriangularSolveTorchReferenceProvider", "triangular_solve_forward"]
