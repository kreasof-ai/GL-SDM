"""Pinned external execution fallback for URM's certified-route state algebra.

This translation exists only in the execution layer.  The semantic operation
remains the URM-owned :class:`SparseStateMixerSpec`; no upstream layout or flag
is admitted into the compiler IR.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from urm.adapters.sparse_delta_memory import (
    MODE_INFERENCE,
    MODE_READ_ONLY,
    MODE_TRAINING,
    UrmSparseDeltaMemoryAdapter,
)
from urm.backends.sparse_state_mixer import (
    CertifiedSparseStateRoutes,
    SparseState,
)
from urm.compiler.semantic import (
    DType,
    SparseReadTiming,
    SparseStateExecutionMode,
    SparseStateMixerSpec,
    SparseStateOperation,
)


@dataclass(frozen=True, slots=True)
class PreparedExternalSparseStateOperands:
    spec: SparseStateMixerSpec
    routes: CertifiedSparseStateRoutes
    global_read_indices: torch.Tensor
    global_write_indices: torch.Tensor | None
    values: torch.Tensor | None
    beta: torch.Tensor | None
    log_decay: torch.Tensor | None


class SDMSparseStateMixerFallback:
    """Translate certified logical routes into the pinned comparator layout."""

    name = "facebook_sparse_delta_memory_183e7df_precomputed_route_adapter"

    def __init__(self, spec: SparseStateMixerSpec) -> None:
        root = int(spec.slots_per_partition**0.5)
        if (
            spec.slots_per_partition < 8
            or spec.slots_per_partition % 8
            or root * root != spec.slots_per_partition
        ):
            raise ValueError("pinned fallback requires square slots divisible by eight")
        if spec.reads > 128 or spec.writes > 128:
            raise ValueError("pinned fallback route widths must not exceed 128")
        if (
            spec.operation is SparseStateOperation.UPDATE
            and spec.read_timing is not SparseReadTiming.AFTER_UPDATE
        ):
            raise ValueError("pinned fallback supports only post-update reads")
        if spec.mode is SparseStateExecutionMode.TRAINING and spec.sequence < 16:
            raise ValueError("pinned fallback training requires sequence >= 16")
        dtype = {DType.FLOAT32: torch.float32, DType.BFLOAT16: torch.bfloat16}[
            spec.dtype
        ]
        mode = (
            MODE_READ_ONLY
            if spec.operation is SparseStateOperation.READ_ONLY
            else MODE_TRAINING
            if spec.mode is SparseStateExecutionMode.TRAINING
            else MODE_INFERENCE
        )
        self.spec = spec
        self.adapter = UrmSparseDeltaMemoryAdapter(
            slots_per_partition=spec.slots_per_partition,
            value_dim=spec.value_dim,
            num_writes=max(1, spec.writes),
            num_reads=spec.reads,
            mode=mode,
            dtype=dtype,
        )

    @staticmethod
    def _globalize(indices: torch.Tensor, slots: int) -> torch.Tensor:
        offsets = torch.arange(
            indices.shape[0], device=indices.device, dtype=torch.int64
        ).view(-1, 1, 1)
        return (indices.to(torch.int64) + offsets * slots).contiguous()

    def prepare(
        self,
        routes: CertifiedSparseStateRoutes,
        *,
        values: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        log_decay: torch.Tensor | None = None,
    ) -> PreparedExternalSparseStateOperands:
        if routes.spec != self.spec:
            raise ValueError("certified routes do not match fallback semantics")
        routes.require_intact()
        supplied = (values, beta, log_decay)
        if self.spec.operation is SparseStateOperation.READ_ONLY:
            if any(item is not None for item in supplied):
                raise ValueError("read-only fallback does not accept update operands")
        else:
            expected_value = (
                self.spec.parallel,
                self.spec.sequence,
                self.spec.value_dim,
            )
            expected_scalar = (self.spec.parallel, self.spec.sequence, 1)
            if values is None or tuple(values.shape) != expected_value:
                raise ValueError(f"values must have shape {expected_value}")
            if beta is None or tuple(beta.shape) != expected_scalar:
                raise ValueError(f"beta must have shape {expected_scalar}")
            if log_decay is None or tuple(log_decay.shape) != expected_scalar:
                raise ValueError(f"log_decay must have shape {expected_scalar}")
            expected_dtype = self.adapter.config.dtype
            route_device = routes.read_indices.device
            for tensor in supplied:
                if (
                    tensor.dtype != expected_dtype
                    or tensor.device != route_device
                    or not tensor.is_contiguous()
                    or not bool(torch.isfinite(tensor).all().item())
                ):
                    raise ValueError(
                        "update operands must be finite contiguous route-device tensors"
                    )
        return PreparedExternalSparseStateOperands(
            self.spec,
            routes,
            self._globalize(routes.read_indices, self.spec.slots_per_partition),
            (
                self._globalize(routes.write_indices, self.spec.slots_per_partition)
                if routes.write_indices is not None
                else None
            ),
            values,
            beta,
            log_decay,
        )

    def execute(
        self,
        state: SparseState,
        prepared: PreparedExternalSparseStateOperands,
        *,
        grad_final_memory: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SparseState]:
        if out is not None:
            raise ValueError("pinned external fallback has no preallocated-output API")
        if prepared.spec != self.spec:
            raise ValueError("prepared operands do not match fallback semantics")
        prepared.routes.require_intact()
        expected = (
            self.spec.parallel,
            self.spec.slots_per_partition,
            self.spec.value_dim,
        )
        if tuple(state.memory.shape) != expected or not state.memory.is_contiguous():
            raise ValueError(f"state memory must be contiguous with shape {expected}")
        flat_memory = state.memory.flatten(0, 1)
        calls = self.adapter.direct_calls
        if self.spec.operation is SparseStateOperation.READ_ONLY:
            readings = calls["read"](
                flat_memory,
                prepared.routes.read_weights,
                prepared.global_read_indices,
            )
            return readings, state
        flat_grad = (
            grad_final_memory.flatten(0, 1).contiguous()
            if grad_final_memory is not None
            else None
        )
        if flat_grad is None:
            readings, final_memory = calls["update"](
                flat_memory,
                prepared.global_write_indices,
                prepared.routes.write_weights,
                prepared.values,
                prepared.beta,
                prepared.log_decay,
                prepared.global_read_indices,
                prepared.routes.read_weights,
            )
        else:
            readings, final_memory = calls["update"](
                flat_memory,
                prepared.global_write_indices,
                prepared.routes.write_weights,
                prepared.values,
                prepared.beta,
                prepared.log_decay,
                prepared.global_read_indices,
                prepared.routes.read_weights,
                grad_final_memory=flat_grad,
            )
        state.memory = final_memory.view(expected)
        state.sequence_length += self.spec.sequence
        return readings, state

    @property
    def bound_callables(self) -> dict[str, object]:
        return self.adapter.direct_calls


__all__ = [
    "PreparedExternalSparseStateOperands",
    "SDMSparseStateMixerFallback",
]
