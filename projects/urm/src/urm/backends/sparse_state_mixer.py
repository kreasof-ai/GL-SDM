"""Capability-checked native backend for the URM SparseStateMixer v0 contract."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from urm.compiler.semantic import (
    DType,
    SparseReadTiming,
    SparseStateMixerSpec,
    SparseStateOperation,
)
from urm.sparse_state_mixer import (
    FROZEN_V0_ENVELOPE,
    NATIVE_SPARSE_STATE_MIXER_NAME,
    SparseStateSupportStatus,
    sparse_state_launch_schedule,
    sparse_state_spec_status,
)

_ROUTE_CERTIFICATE = object()
_OPERAND_CERTIFICATE = object()


def native_dependencies_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("triton") is not None
    )


def _dtype_name(tensor: object) -> str:
    return str(tensor.dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class CertifiedSparseStateRoutes:
    """Validated local addresses and weights independent of route production."""

    spec: SparseStateMixerSpec
    read_indices: object
    read_weights: object
    write_indices: object | None = None
    write_weights: object | None = None
    _versions: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _ROUTE_CERTIFICATE:
            raise ValueError("routes must be created by certify()")

    @classmethod
    def certify(
        cls,
        spec: SparseStateMixerSpec,
        read_indices: object,
        read_weights: object,
        *,
        write_indices: object | None = None,
        write_weights: object | None = None,
    ) -> CertifiedSparseStateRoutes:
        import torch

        expected_read = (spec.parallel, spec.sequence, spec.reads)
        if tuple(read_indices.shape) != expected_read:
            raise ValueError(f"read indices must have shape {expected_read}")
        if tuple(read_weights.shape) != expected_read:
            raise ValueError(f"read weights must have shape {expected_read}")
        if spec.operation is SparseStateOperation.UPDATE:
            expected_write = (spec.parallel, spec.sequence, spec.writes)
            if write_indices is None or tuple(write_indices.shape) != expected_write:
                raise ValueError(f"write indices must have shape {expected_write}")
            if write_weights is None or tuple(write_weights.shape) != expected_write:
                raise ValueError(f"write weights must have shape {expected_write}")
        elif write_indices is not None or write_weights is not None:
            raise ValueError("read-only routes must not contain write tensors")
        indices = tuple(
            item for item in (read_indices, write_indices) if item is not None
        )
        weights = tuple(
            item for item in (read_weights, write_weights) if item is not None
        )
        tensors = (*indices, *weights)
        if not tensors or not all(tensor.is_contiguous() for tensor in tensors):
            raise ValueError("route tensors must be contiguous")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("route tensors must share one device")
        if any(tensor.dtype not in (torch.int32, torch.int64) for tensor in indices):
            raise ValueError("route addresses must use int32 or int64")
        if len({tensor.dtype for tensor in indices}) != 1:
            raise ValueError("read/write address dtypes must match")
        expected_dtype = {
            DType.FLOAT32: torch.float32,
            DType.BFLOAT16: torch.bfloat16,
        }[spec.dtype]
        if any(tensor.dtype != expected_dtype for tensor in weights):
            raise ValueError("route weights must match the semantic state dtype")
        for name, tensor in (
            ("read", read_indices),
            ("write", write_indices),
        ):
            if tensor is None:
                continue
            if bool(((tensor < 0) | (tensor >= spec.slots_per_partition)).any().item()):
                raise ValueError(
                    f"{name} addresses must be partition-local and in bounds"
                )
            if tensor.shape[-1] > 1 and bool(
                (tensor[..., 1:] <= tensor[..., :-1]).any().item()
            ):
                raise ValueError(
                    f"{name} addresses must be strictly increasing and unique"
                )
        # BF16 Softmax outputs can differ from one by one representable quantum
        # after storage (observed worst case 0.00244 for width three); 0.004 is
        # the pre-tuning certification envelope, not a numerical kernel gate.
        normalization_atol = 2e-5 if spec.dtype is DType.FLOAT32 else 4e-3
        for name, tensor in (
            ("read", read_weights),
            ("write", write_weights),
        ):
            if tensor is None:
                continue
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"{name} weights must be finite")
            if bool((tensor < 0).any().item()):
                raise ValueError(f"{name} weights must be nonnegative")
            sums = tensor.float().sum(dim=-1)
            if not torch.allclose(
                sums, torch.ones_like(sums), atol=normalization_atol, rtol=0
            ):
                raise ValueError(f"{name} weights must be normalized")
        return cls(
            spec,
            read_indices,
            read_weights,
            write_indices,
            write_weights,
            tuple(tensor._version for tensor in tensors),
            _ROUTE_CERTIFICATE,
        )

    def require_intact(self) -> None:
        tensors = tuple(
            item
            for item in (
                self.read_indices,
                self.write_indices,
                self.read_weights,
                self.write_weights,
            )
            if item is not None
        )
        if tuple(tensor._version for tensor in tensors) != self._versions:
            raise ValueError("certified route tensors were mutated")


@dataclass(frozen=True, slots=True)
class CertifiedSparseStateOperands:
    spec: SparseStateMixerSpec
    routes: CertifiedSparseStateRoutes
    values: object | None
    beta: object | None
    log_decay: object | None
    _versions: tuple[int, ...] = field(default=(), repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _OPERAND_CERTIFICATE:
            raise ValueError("operands must be created by prepare()")

    def require_intact(self) -> None:
        self.routes.require_intact()
        tensors = tuple(
            item
            for item in (self.values, self.beta, self.log_decay)
            if item is not None
        )
        if tuple(tensor._version for tensor in tensors) != self._versions:
            raise ValueError("certified value/gate tensors were mutated")


@dataclass(slots=True)
class SparseState:
    memory: object
    sequence_length: int = 0


class TritonSparseStateMixerBackend:
    """Native URM dispatch; no comparator package is imported or consulted."""

    name = NATIVE_SPARSE_STATE_MIXER_NAME

    def __init__(self, spec: SparseStateMixerSpec) -> None:
        self.spec = spec
        status = self.support_status(spec)
        status.require()

    @staticmethod
    def support_status(spec: SparseStateMixerSpec) -> SparseStateSupportStatus:
        if not native_dependencies_available():
            return SparseStateSupportStatus.no(
                "missing_dependency", "PyTorch and Triton are required"
            )
        import torch

        if not torch.cuda.is_available():
            return SparseStateSupportStatus.no(
                "unsupported_hardware", "CUDA is unavailable"
            )
        return sparse_state_spec_status(
            spec,
            device_type="cuda",
            compute_capability=torch.cuda.get_device_capability(),
        )

    def prepare(
        self,
        routes: CertifiedSparseStateRoutes,
        *,
        values: object | None = None,
        beta: object | None = None,
        log_decay: object | None = None,
    ) -> CertifiedSparseStateOperands:
        import torch

        if routes.spec != self.spec:
            raise ValueError("certified routes do not match backend semantics")
        routes.require_intact()
        supplied = (values, beta, log_decay)
        if self.spec.operation is SparseStateOperation.READ_ONLY:
            if any(item is not None for item in supplied):
                raise ValueError("read-only operation does not accept update operands")
            tensors = ()
        else:
            expected_value = (
                self.spec.parallel,
                self.spec.sequence,
                self.spec.value_dim,
            )
            if values is None or tuple(values.shape) != expected_value:
                raise ValueError(f"values must have shape {expected_value}")
            expected_scalar = (self.spec.parallel, self.spec.sequence, 1)
            if beta is None or tuple(beta.shape) != expected_scalar:
                raise ValueError(f"beta must have shape {expected_scalar}")
            if log_decay is None or tuple(log_decay.shape) != expected_scalar:
                raise ValueError(f"log_decay must have shape {expected_scalar}")
            tensors = supplied
            route_device = routes.read_indices.device
            if any(
                not tensor.is_contiguous()
                or tensor.device != route_device
                or _dtype_name(tensor) != self.spec.dtype.value
                for tensor in tensors
            ):
                raise ValueError(
                    "update operands must be contiguous and match route device/dtype"
                )
            if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
                raise ValueError("update operands must be finite")
        return CertifiedSparseStateOperands(
            self.spec,
            routes,
            values,
            beta,
            log_decay,
            tuple(tensor._version for tensor in tensors),
            _OPERAND_CERTIFICATE,
        )

    def _validate_state(self, state: SparseState) -> None:
        memory = state.memory
        expected = (
            self.spec.parallel,
            self.spec.slots_per_partition,
            self.spec.value_dim,
        )
        if tuple(memory.shape) != expected:
            raise ValueError(f"state memory must have shape {expected}")
        if not memory.is_cuda or not memory.is_contiguous():
            raise ValueError("state memory must be contiguous CUDA storage")
        if _dtype_name(memory) != self.spec.dtype.value:
            raise ValueError("state memory dtype does not match semantics")
        if not isinstance(state.sequence_length, int) or state.sequence_length < 0:
            raise ValueError("state sequence length must be a non-negative integer")

    def execute(
        self,
        state: SparseState,
        prepared: CertifiedSparseStateOperands,
        *,
        out: object | None = None,
    ) -> tuple[object, SparseState]:
        if prepared.spec != self.spec:
            raise ValueError("prepared operands do not match backend semantics")
        prepared.require_intact()
        self._validate_state(state)
        if state.memory.device != prepared.routes.read_indices.device:
            raise ValueError("state and routes must share one CUDA device")
        from urm.triton_kernels.sparse_state_mixer import (
            sparse_state_read,
            sparse_state_update,
        )

        if self.spec.operation is SparseStateOperation.READ_ONLY:
            readings = sparse_state_read(
                state.memory,
                prepared.routes.read_indices,
                prepared.routes.read_weights,
                out=out,
            )
            return readings, state
        readings, memory = sparse_state_update(
            state.memory,
            prepared.routes.write_indices,
            prepared.routes.write_weights,
            prepared.values,
            prepared.beta,
            prepared.log_decay,
            prepared.routes.read_indices,
            prepared.routes.read_weights,
            read_before_update=self.spec.read_timing is SparseReadTiming.BEFORE_UPDATE,
            out=out,
        )
        state.memory = memory
        state.sequence_length += self.spec.sequence
        return readings, state

    @staticmethod
    def capability() -> dict[str, object]:
        envelope = FROZEN_V0_ENVELOPE
        return {
            "name": NATIVE_SPARSE_STATE_MIXER_NAME,
            "schema_version": envelope.schema_version,
            "native_urm_lowering": True,
            "maximum_parallel": envelope.maximum_parallel,
            "maximum_sequence": envelope.maximum_sequence,
            "maximum_slots_per_partition": envelope.maximum_slots_per_partition,
            "maximum_value_dim": envelope.maximum_value_dim,
            "maximum_route_width": envelope.maximum_route_width,
            "supported_dtypes": [dtype.value for dtype in envelope.supported_dtypes],
            "supported_index_dtypes": list(envelope.supported_index_dtypes),
            "minimum_compute_capability": list(envelope.minimum_compute_capability),
        }

    def launch_schedule(self) -> dict[str, str | int]:
        return sparse_state_launch_schedule(self.spec)


__all__ = [
    "CertifiedSparseStateOperands",
    "CertifiedSparseStateRoutes",
    "SparseState",
    "TritonSparseStateMixerBackend",
    "native_dependencies_available",
]
