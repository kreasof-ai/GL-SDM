"""K2 linear-delta state providers: Torch reference, native Triton, and the
independent NumPy oracle, all behind the one Provider contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import ProviderFamily, ProviderRequest  # noqa: F401  (re-exported)
from ....ir.program import LinearDeltaSpec


class _Base:
    name: str = ""
    family: str = ""
    tier: str = "reference"

    def decline(self, request: ProviderRequest) -> str | None:
        return None


class _K1Provider(_Base):
    family = ProviderFamily.K1

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        return None


class _NumpyBase:
    tier = "reference"
    family = ""

    def decline(self, request: ProviderRequest) -> str | None:
        return None


class K2TorchReferenceProvider(_Base):
    name = "urm.unified.k2.state_reference.v1"
    family = ProviderFamily.K2
    tier = "reference"

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, LinearDeltaSpec):
            return "K2 providers require a closed LinearDeltaSpec"
        if request.accumulation_dtype != "float32":
            return "K2 v1 requires float32 accumulation"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .torch import linear_delta_state

        scale_op = operands.get("scale")
        result = linear_delta_state(
            operands["initial_state"],
            operands["key"],
            operands["query"],
            operands["value"],
            operands["beta"],
            operands["log_decay"],
            spec=request.descriptor,
            scale=None if scale_op is None else float(scale_op),
        )
        if request.descriptor.normalized:
            out, (final_state, denominator) = result
            return {"output": out, "final_state": final_state, "final_denominator": denominator}
        out, final_state = result
        return {"output": out, "final_state": final_state}


class K2NativeTritonProvider(K2TorchReferenceProvider):
    """Native K2 anchors share the typed request/result ABI. Until a qualified
    Triton K2 schedule is admitted (Gate 2), the native tier executes the same
    reference recurrence so the public path stays fail-closed and honest about
    its tier."""

    tier = "native"


class K2NativeDiagonalProvider(K2NativeTritonProvider):
    name = "urm_native_diagonal_recurrence_v1"


class K2NativeMatrixProvider(K2NativeTritonProvider):
    name = "urm_native_matrix_state_recurrence_v1"


class K2NumpyProvider(_NumpyBase):
    name = "urm.reference.numpy.k2.linear_delta_state.v1"
    family = ProviderFamily.K2

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, LinearDeltaSpec):
            return "K2 NumPy provider requires a closed LinearDeltaSpec"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .numpy import recurrent

        spec = request.descriptor
        scale_op = operands.get("scale")
        if spec.scale_rule.value == "explicit_operand":
            if scale_op is None:
                raise ValueError("K2 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        elif spec.scale_rule.value == "key_dim_rsqrt":
            scale = float(np.asarray(operands["key"]).shape[-1]) ** -0.5
        else:
            scale = 1.0
        # The oracle consumes one [K, V] partition with [T, ...] sequences; the
        # role operands arrive [B, H, ...]. Run per (batch, head) partition.
        k = np.asarray(operands["key"], dtype=np.float64)
        q = np.asarray(operands["query"], dtype=np.float64)
        v = np.asarray(operands["value"], dtype=np.float64)
        b = np.asarray(operands["beta"], dtype=np.float64)
        g = np.asarray(operands["log_decay"], dtype=np.float64)
        m0 = np.asarray(operands["initial_state"], dtype=np.float64)
        B, H = k.shape[0], k.shape[1]
        outs = np.empty((B, H, v.shape[2], v.shape[3]), dtype=np.float64)
        finals = np.empty_like(m0)
        for bi in range(B):
            for hi in range(H):
                decay = None if spec.gate_scope.value == "none" else g[bi, hi]
                out, m = recurrent(
                    m0[bi, hi], k[bi, hi], q[bi, hi], v[bi, hi], b[bi, hi],
                    np.zeros(k.shape[2]) if decay is None else decay,
                    scale=scale,
                    is_delta=spec.delta,
                    read_before_update=spec.read_timing.value == "before_update",
                    normalizer=spec.normalized,
                    epsilon=spec.epsilon,
                )
                outs[bi, hi] = out[0] if spec.normalized else out
                finals[bi, hi] = m[0] if spec.normalized else m
        return {"output": outs, "final_state": finals}

__all__ = [
    "K2NativeDiagonalProvider",
    "K2NativeMatrixProvider",
    "K2NumpyProvider",
    "K2TorchReferenceProvider",
]
