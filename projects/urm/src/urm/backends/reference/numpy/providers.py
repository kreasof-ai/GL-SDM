"""NumPy reference providers: the independent high-precision oracle tier.

These adapt the float64 NumPy oracles to the same :class:`Provider` contract
as the Torch reference and native Triton tiers, so a backend is one uniform
thing across every tier. The NumPy tier is the independent equation oracle
(fp64, transparent loops); it is never a performance backend and is not
selected by the compiler's anchor tiers — it exists so tests and the evidence
protocol can check any tier against an independent equation.

Operand forms are unified across tiers: K1/K2 use the same role vocabulary as
Torch, and K3 takes the same address-index form as the Torch reference and the
native launcher (the dense slot-vector oracle in :mod:`.k3` is adapted here).
"""

from __future__ import annotations

from typing import Any

from urm.backends.provider import ProviderFamily, ProviderRequest
from urm.ir.program import (
    K1Descriptor,
    K1ScaleRule,
    LinearDeltaSpec,
    SparseStateMixerSpec,
)

import numpy as np


class _NumpyBase:
    tier = "reference"
    family = ""

    def decline(self, request: ProviderRequest) -> str | None:
        return None


class K1NumpyProvider(_NumpyBase):
    name = "urm.reference.numpy.k1.softmax_attention.v1"
    family = ProviderFamily.K1

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 NumPy provider requires a closed K1Descriptor"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .k1_attention import attention

        descriptor = request.descriptor
        query, key, value = operands["query"], operands["key"], operands["value"]
        scale_op = operands.get("scale")
        if descriptor.scale_rule is K1ScaleRule.EXPLICIT_OPERAND:
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = None  # the oracle applies key_dim**-0.5 itself
        q = np.asarray(query, dtype=np.float64)
        k = np.asarray(key, dtype=np.float64)
        v = np.asarray(value, dtype=np.float64)
        # The oracle consumes [B, H, T, D]; the role operands arrive [B, T, H, D].
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        bias = operands.get("score_bias")
        mask = operands.get("attention_mask")
        out = np.stack(
            [
                attention(
                    q[b],
                    k[b],
                    v[b],
                    scale=scale,
                    causal=descriptor.causal,
                    score_bias=None if bias is None else np.asarray(bias, dtype=np.float64),
                    attention_mask=None if mask is None else np.asarray(mask),
                )
                for b in range(q.shape[0])
            ]
        )
        return {"output": np.transpose(out, (0, 2, 1, 3))}


class K2NumpyProvider(_NumpyBase):
    name = "urm.reference.numpy.k2.linear_delta_state.v1"
    family = ProviderFamily.K2

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, LinearDeltaSpec):
            return "K2 NumPy provider requires a closed LinearDeltaSpec"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .k2 import recurrent

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


class K3NumpyProvider(_NumpyBase):
    name = "urm.reference.numpy.k3.sparse_delta_state.v1"
    family = ProviderFamily.K3

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, SparseStateMixerSpec):
            return "K3 NumPy provider requires a closed SparseStateMixerSpec"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        """K3 oracle in the unified address-index operand form.

        The torch/native tiers take read/write address indices; the NumPy
        oracle's dense slot-vector form is adapted to that same form here, so
        all three tiers take identical operands.
        """
        from .k3 import recurrent

        memory = np.asarray(operands["memory"], dtype=np.float64)
        read_idx = np.asarray(operands["read_addresses"], dtype=np.int64)
        read_w = np.asarray(operands["read_weights"], dtype=np.float64)
        write_idx = np.asarray(operands["write_addresses"], dtype=np.int64)
        write_w = np.asarray(operands["write_weights"], dtype=np.float64)
        values = np.asarray(operands["values"], dtype=np.float64)
        beta = np.asarray(operands["beta"], dtype=np.float64)
        log_decay = np.asarray(operands["log_decay"], dtype=np.float64)

        # Scatter the address-index operands into the oracle's dense slot-vector
        # form. Contract: after-update read, within-token unique write slots.
        # The oracle consumes one partition as 2-D [T, S] arrays (no batch dim).
        if request.descriptor.read_timing.value != "after_update":
            raise ValueError("the K3 NumPy oracle covers the after-update read")
        parallel, sequence, reads = read_idx.shape
        slots = memory.shape[1]
        outs = np.empty((parallel, sequence, memory.shape[2]), dtype=np.float64)
        finals = np.empty_like(memory)
        for p in range(parallel):
            writes = np.zeros((sequence, slots))
            reads_v = np.zeros((sequence, slots))
            sel = np.zeros((sequence, slots), dtype=bool)
            for t in range(sequence):
                w_addr, w_w = write_idx[p, t], write_w[p, t]
                r_addr, r_w = read_idx[p, t], read_w[p, t]
                writes[t, w_addr] = w_w
                reads_v[t, r_addr] = r_w
                sel[t, w_addr] = True
            out_p, m = recurrent(
                memory[p], writes, reads_v, values[p],
                beta[p, :, 0], log_decay[p, :, 0], sel,
            )
            outs[p] = out_p
            finals[p] = m
        return {"readings": outs, "updated_memory": finals}


__all__ = [
    "K1NumpyProvider",
    "K2NumpyProvider",
    "K3NumpyProvider",
]
