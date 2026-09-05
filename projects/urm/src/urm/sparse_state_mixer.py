"""Frozen typed contract and NumPy oracle for SparseStateMixer v0.

This module is dependency-light and contains no execution-library bindings.
Certified routes are logical partition-local slot addresses; route production is
outside this operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from urm.compiler.semantic import (
    DType,
    MergePolicy,
    SparseReadTiming,
    SparseStateLayout,
    SparseStateMixerSpec,
    SparseStateOperation,
    SparseStatePolicy,
    SparseUpdateRule,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


NATIVE_SPARSE_STATE_MIXER_NAME = "urm_native_sparse_state_mixer_v0"


@dataclass(frozen=True, slots=True)
class SparseStateCapabilityEnvelope:
    """Pre-tuning limits of the first native A10G-oriented lowering."""

    schema_version: int = 0
    minimum_compute_capability: tuple[int, int] = (8, 0)
    maximum_parallel: int = 16
    maximum_sequence: int = 2048
    maximum_slots_per_partition: int = 1_048_576
    maximum_value_dim: int = 1024
    maximum_route_width: int = 64
    supported_dtypes: tuple[DType, ...] = (DType.FLOAT32, DType.BFLOAT16)
    supported_index_dtypes: tuple[str, ...] = ("int32", "int64")


FROZEN_V0_ENVELOPE = SparseStateCapabilityEnvelope()


@dataclass(frozen=True, slots=True)
class SparseStateSupportStatus:
    supported: bool
    code: str
    reason: str | None = None
    details: Mapping[str, object] | None = None

    @classmethod
    def yes(cls) -> SparseStateSupportStatus:
        return cls(True, "supported")

    @classmethod
    def no(cls, code: str, reason: str, **details: object) -> SparseStateSupportStatus:
        return cls(False, code, reason, details)

    def require(self) -> None:
        if not self.supported:
            raise ValueError(
                f"{NATIVE_SPARSE_STATE_MIXER_NAME} declined [{self.code}]: "
                f"{self.reason}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "code": self.code,
            "reason": self.reason,
            "details": dict(self.details or {}),
        }


def sparse_state_spec_status(
    spec: SparseStateMixerSpec,
    *,
    device_type: str,
    index_dtype: str = "int64",
    contiguous: bool = True,
    compute_capability: tuple[int, int] | None = None,
) -> SparseStateSupportStatus:
    """Return a structured v0 capability decision without importing Torch."""
    envelope = FROZEN_V0_ENVELOPE
    exact_semantics = (
        spec.update_rule is SparseUpdateRule.DECAYED_DELTA
        and spec.collision_policy is MergePolicy.ORDERED
        and spec.within_token_collision_policy is MergePolicy.REJECT
        and spec.state_policy is SparseStatePolicy.PERSISTENT_IN_PLACE
        and spec.accumulation_dtype is DType.FLOAT32
        and spec.state_layout is SparseStateLayout.PARTITION_SLOT_VALUE
        and spec.page_size == 1
    )
    if not exact_semantics:
        return SparseStateSupportStatus.no(
            "unsupported_semantics", "operation is outside the frozen v0 algebra"
        )
    if spec.operation is SparseStateOperation.READ_ONLY:
        if spec.read_timing is not SparseReadTiming.CURRENT_STATE:
            return SparseStateSupportStatus.no(
                "unsupported_semantics", "read-only mode must read current state"
            )
    elif spec.read_timing not in {
        SparseReadTiming.BEFORE_UPDATE,
        SparseReadTiming.AFTER_UPDATE,
    }:
        return SparseStateSupportStatus.no(
            "unsupported_semantics", "update read timing is unsupported"
        )
    if spec.dtype not in envelope.supported_dtypes:
        return SparseStateSupportStatus.no(
            "unsupported_dtype", f"dtype {spec.dtype.value} is outside v0"
        )
    if device_type != "cuda":
        return SparseStateSupportStatus.no(
            "unsupported_device", "v0 native execution requires CUDA"
        )
    if not contiguous:
        return SparseStateSupportStatus.no(
            "unsupported_layout", "v0 requires contiguous logical tensors"
        )
    if index_dtype not in envelope.supported_index_dtypes:
        return SparseStateSupportStatus.no(
            "unsupported_dtype", f"index dtype {index_dtype} is outside v0"
        )
    if compute_capability is not None and compute_capability < (
        envelope.minimum_compute_capability
    ):
        return SparseStateSupportStatus.no(
            "unsupported_hardware",
            "v0 requires SM80 or newer",
            found_compute_capability=compute_capability,
        )
    limits = {
        "parallel": (spec.parallel, envelope.maximum_parallel),
        "sequence": (spec.sequence, envelope.maximum_sequence),
        "slots_per_partition": (
            spec.slots_per_partition,
            envelope.maximum_slots_per_partition,
        ),
        "value_dim": (spec.value_dim, envelope.maximum_value_dim),
        "writes": (spec.writes, envelope.maximum_route_width),
        "reads": (spec.reads, envelope.maximum_route_width),
    }
    exceeded = {
        name: {"requested": requested, "maximum": maximum}
        for name, (requested, maximum) in limits.items()
        if requested > maximum
    }
    if exceeded:
        return SparseStateSupportStatus.no(
            "unsupported_shape",
            "one or more dimensions exceed the predeclared v0 bounds",
            exceeded=exceeded,
        )
    return SparseStateSupportStatus.yes()


def sparse_state_launch_parameters(value_dim: int) -> tuple[int, int]:
    """Shared production schedule; D=64 retains the reviewed D=4 fragment."""
    if value_dim == 64:
        return 4, 2
    block = max(16, 1 << (min(value_dim, 256) - 1).bit_length())
    return block, 8 if block >= 256 else 4 if block >= 64 else 2


def sparse_state_launch_schedule(spec: SparseStateMixerSpec) -> dict[str, str | int]:
    """Serialize the deterministic v0 schedule without importing a GPU runtime."""
    block_d, warps = sparse_state_launch_parameters(spec.value_dim)
    return {
        "schedule_family": "partition_owned_ordered_token_scan",
        "block_d": block_d,
        "num_warps": warps,
        "num_stages": 3,
        "tokens_per_program": -1,
        "read_timing": spec.read_timing.value,
        "state_layout": spec.state_layout.value,
    }


def numpy_sparse_state_mixer(
    memory: npt.ArrayLike,
    read_indices: npt.ArrayLike,
    read_weights: npt.ArrayLike,
    *,
    write_indices: npt.ArrayLike | None = None,
    write_weights: npt.ArrayLike | None = None,
    values: npt.ArrayLike | None = None,
    beta: npt.ArrayLike | None = None,
    log_decay: npt.ArrayLike | None = None,
    read_timing: SparseReadTiming = SparseReadTiming.CURRENT_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Transparent NumPy oracle using fp32 arithmetic and stored-state casts."""
    state = np.array(memory, copy=True)
    read_indices_array = np.asarray(read_indices, dtype=np.int64)
    read_weights_array = np.asarray(read_weights)
    parallel, sequence, _ = read_indices_array.shape
    outputs = np.empty(
        (parallel, sequence, state.shape[-1]), dtype=np.asarray(memory).dtype
    )
    updating = write_indices is not None
    if updating:
        wi = np.asarray(write_indices, dtype=np.int64)
        ww = np.asarray(write_weights)
        value_array = np.asarray(values)
        beta_array = np.asarray(beta)
        decay_array = np.asarray(log_decay)
    for partition in range(parallel):
        for token in range(sequence):
            if read_timing in {
                SparseReadTiming.CURRENT_STATE,
                SparseReadTiming.BEFORE_UPDATE,
            }:
                selected = state[partition, read_indices_array[partition, token]]
                outputs[partition, token] = np.sum(
                    read_weights_array[partition, token, :, None].astype(np.float32)
                    * selected.astype(np.float32),
                    axis=0,
                ).astype(state.dtype)
            if updating:
                addresses = wi[partition, token]
                old = state[partition, addresses].astype(np.float32)
                decayed = old * np.exp(
                    decay_array[partition, token, 0].astype(np.float32)
                )
                weights = ww[partition, token, :, None].astype(np.float32)
                retrieved = np.sum(weights * decayed, axis=0)
                delta = beta_array[partition, token, 0].astype(np.float32) * (
                    value_array[partition, token].astype(np.float32) - retrieved
                )
                state[partition, addresses] = (decayed + weights * delta).astype(
                    state.dtype
                )
            if read_timing is SparseReadTiming.AFTER_UPDATE:
                selected = state[partition, read_indices_array[partition, token]]
                outputs[partition, token] = np.sum(
                    read_weights_array[partition, token, :, None].astype(np.float32)
                    * selected.astype(np.float32),
                    axis=0,
                ).astype(state.dtype)
    return outputs, state


__all__ = [
    "FROZEN_V0_ENVELOPE",
    "NATIVE_SPARSE_STATE_MIXER_NAME",
    "SparseStateCapabilityEnvelope",
    "SparseStateSupportStatus",
    "numpy_sparse_state_mixer",
    "sparse_state_launch_schedule",
    "sparse_state_spec_status",
]
