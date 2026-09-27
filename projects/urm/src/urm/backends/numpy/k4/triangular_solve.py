"""Float64 canonical K4 exact feedback substitution over the emitted stream.

The TriangularSolve op (generality axis UT) computes the strict-causal operand
correction ``u = (I + diag(β)·strict_tril(P))^{-1}·v``. Because ``P`` is read
only on its strict lower triangle, row ``t`` of the unknown reads only the
already-emitted rows ``u_{<t}``, so the system is lower-triangular with a unit
diagonal and is solved exactly by forward substitution::

    u_t = v_t − β_t · Σ_{j<t} P_tj · u_j

This module is the independent high-precision oracle for that equation: it is
written from the equation (the forward-substitution recurrence above), in pure
NumPy float64, with no Torch dependency and no gradient path. It is an oracle
for representation and composition checks, never a performance backend.
"""

from __future__ import annotations

import numpy as np

from ....ir.program import TriangularSolve  # noqa: F401  (descriptor type)


def _inputs(probs, beta, value):
    p, b, v = (np.asarray(x, dtype=np.float64) for x in (probs, beta, value))
    if v.ndim != 4:
        raise ValueError("value must be a [B, H, T, D] tensor")
    bsz, heads, t, dim = v.shape
    if p.shape != (bsz, heads, t, t):
        raise ValueError("probs must be a [B, H, T, T] tensor matching value")
    if b.shape != (bsz, heads, t):
        raise ValueError("beta must be a [B, H, T] tensor matching value")
    if any(not np.isfinite(x).all() for x in (p, b, v)):
        raise ValueError("inputs must be finite")
    return p, b, v


def triangular_solve_forward(probs, beta, value):
    """Solve ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` by forward substitution.

    ``probs`` is the strict-lower probability operand ``[B, H, T, T]`` (only the
    strict lower triangle is read; the diagonal and above are ignored). ``beta``
    is the per-token diagonal ``[B, H, T]``. ``value`` is the right-hand side
    ``[B, H, T, D]``. Returns ``u`` ``[B, H, T, D]``. Runs in float64 (the
    oracle tier).

    The unit-diagonal strict-lower structure means row ``t`` reads only
    ``u_{<t}``, so the forward substitution is exact (no iterative solve).
    """
    p, b, v = _inputs(probs, beta, value)
    t = v.shape[2]
    u = np.empty_like(v)
    for i in range(t):
        if i == 0:
            u[:, :, 0] = v[:, :, 0]
            continue
        # Correction Σ_{j<i} P_ij · u_j over the emitted prefix, then the
        # diagonal-scaled subtraction β_i · correction from the right-hand side.
        correction = np.einsum("bhj,bhjd->bhd", p[:, :, i, :i], u[:, :, :i])
        u[:, :, i] = v[:, :, i] - b[:, :, i, None] * correction
    return u


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


class K4NumpyProvider:
    name = "urm.reference.numpy.k4.triangular_solve.v1"
    family = "triangular_solve"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, TriangularSolve):
            return "K4 NumPy provider requires a TriangularSolve op descriptor"
        return None

    def execute(self, request, operands):
        out = triangular_solve_forward(
            operands["probs"], operands["beta"], operands["value"]
        )
        return {"output": out}


PROVIDERS = (K4NumpyProvider(),)

__all__ = ["K4NumpyProvider", "triangular_solve_forward"]
