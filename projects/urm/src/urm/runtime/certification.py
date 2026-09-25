"""Route and operand certification for the K3 indexed-state family.

This is value/provenance validation — the runtime's job per the runtime
contract ("runtime owns tensor-value validation, state alias/lifetime checks,
provider invocation"). It is deliberately free of kernel math: it certifies
that routes and update operands are well-formed before a provider is invoked,
and it never recomputes the equation. The native Triton schedule it certifies
for lives in :mod:`urm.backends.triton.k3`; the capability decision lives in
:mod:`urm.compiler.select`.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

from urm.ir.program import (
    DType,
    SparseReadTiming,
    SparseStateMixerSpec,
    SparseStateOperation,
)
from urm.ir.k3 import (
    FROZEN_V0_ENVELOPE,
    NATIVE_SPARSE_STATE_MIXER_NAME,
    SparseStateSupportStatus,
    sparse_state_launch_schedule,
    sparse_state_spec_status,
)

_ROUTE_CERTIFICATE = object()
_SCORE_CERTIFICATE = object()
_ROUTE_OUTPUT_CERTIFICATE = object()
NATIVE_SPARSE_ROUTE_NAME = "urm_native_sparse_route_selection_v0"
_OPERAND_CERTIFICATE = object()


def native_dependencies_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("triton") is not None
    )


def _dtype_name(tensor: object) -> str:
    return str(tensor.dtype).removeprefix("torch.")


@dataclass(frozen=True, slots=True)
class SparseRouteSupportStatus:
    supported: bool
    code: str
    reason: str | None = None

    @classmethod
    def yes(cls):
        return cls(True, "supported")

    @classmethod
    def no(cls, code: str, reason: str):
        return cls(False, code, reason)

    def require(self) -> None:
        if not self.supported:
            raise ValueError(
                f"{NATIVE_SPARSE_ROUTE_NAME} declined [{self.code}]: {self.reason}"
            )


@dataclass(frozen=True, slots=True)
class CertifiedSparseRouteScores:
    """Static score contract certified without reading dynamic CUDA values."""

    spec: SparseRouteSelectionSpec
    scores: object
    _version: int = field(default=-1, repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _SCORE_CERTIFICATE:
            raise ValueError("route scores must be created by certify()")

    @classmethod
    def certify(cls, spec: SparseRouteSelectionSpec, scores: object):
        import torch

        expected = (spec.parallel, spec.sequence, spec.score_width)
        if not isinstance(scores, torch.Tensor) or tuple(scores.shape) != expected:
            raise ValueError(f"scores must have shape {expected}")
        expected_dtype = {
            DType.FLOAT32: torch.float32,
            DType.BFLOAT16: torch.bfloat16,
        }[spec.dtype]
        if scores.dtype != expected_dtype:
            raise ValueError("score dtype must match route semantics")
        if not scores.is_cuda or not scores.is_contiguous():
            raise ValueError("scores must be contiguous CUDA storage")
        return cls(spec, scores, scores._version, _SCORE_CERTIFICATE)

    def require_intact(self) -> None:
        import torch

        if torch.compiler.is_compiling():
            return
        if self.scores._version != self._version:
            raise ValueError("certified route scores were mutated")


@dataclass(frozen=True, slots=True)
class NativeSparseRouteOutput:
    """Trusted result produced only by the URM route kernel."""

    spec: SparseRouteSelectionSpec
    addresses: object
    weights: object
    _versions: tuple[int, int] = field(default=(-1, -1), repr=False, compare=False)
    _certificate: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._certificate is not _ROUTE_OUTPUT_CERTIFICATE:
            raise ValueError("native routes must be produced by generate_certified()")

    def require_intact(self) -> None:
        import torch

        if torch.compiler.is_compiling():
            return
        if (self.addresses._version, self.weights._version) != self._versions:
            raise ValueError("native generated routes were mutated")


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

    @classmethod
    def certify_trusted(
        cls,
        spec: SparseStateMixerSpec,
        read_indices: object,
        read_weights: object,
        *,
        write_indices: object | None = None,
        write_weights: object | None = None,
    ) -> CertifiedSparseStateRoutes:
        """Certify routes the caller asserts are well-formed, without GPU value scans.

        This is the trusted-input path for decode sessions and trusted route
        producers: it performs the cheap host-side structural validation (shape,
        dtype, device, contiguity) but NOT the GPU value scans (in-bounds,
        strictly-increasing, normalized) that ``certify`` pays per call. The
        caller MUST guarantee the routes are well-formed - partition-local
        in-bounds addresses, strictly increasing and unique within each token,
        and finite nonnegative normalized weights. Passing malformed routes here
        is a caller bug that produces undefined kernel behavior, not a caught
        error. When in doubt, use ``certify``.
        """
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
        return cls(
            spec,
            read_indices,
            read_weights,
            write_indices,
            write_weights,
            tuple(tensor._version for tensor in tensors),
            _ROUTE_CERTIFICATE,
        )

    @classmethod
    def from_native_generation(
        cls,
        spec: SparseStateMixerSpec,
        read_output: object,
        *,
        write_output: object | None = None,
    ) -> CertifiedSparseStateRoutes:
        """Bridge only trusted URM route-kernel results without GPU value scans."""
        

        if not isinstance(read_output, NativeSparseRouteOutput):
            raise TypeError("read routes are not certified native route output")
        read_output.require_intact()
        expected_read = (
            spec.parallel,
            spec.sequence,
            spec.slots_per_partition,
            spec.reads,
            spec.dtype,
        )
        actual_read = (
            read_output.spec.parallel,
            read_output.spec.sequence,
            read_output.spec.source_extent,
            read_output.spec.route_width,
            read_output.spec.dtype,
        )
        if actual_read != expected_read:
            raise ValueError("native read routes do not match state semantics")
        write_indices = write_weights = None
        if spec.operation is SparseStateOperation.UPDATE:
            if not isinstance(write_output, NativeSparseRouteOutput):
                raise ValueError("update requires certified native write routes")
            write_output.require_intact()
            expected_write = (*expected_read[:3], spec.writes, spec.dtype)
            actual_write = (
                write_output.spec.parallel,
                write_output.spec.sequence,
                write_output.spec.source_extent,
                write_output.spec.route_width,
                write_output.spec.dtype,
            )
            if actual_write != expected_write:
                raise ValueError("native write routes do not match state semantics")
            write_indices = write_output.addresses
            write_weights = write_output.weights
        elif write_output is not None:
            raise ValueError("read-only state must not receive write routes")
        tensors = [
            read_output.addresses,
            write_indices,
            read_output.weights,
            write_weights,
        ]
        concrete = tuple(tensor for tensor in tensors if tensor is not None)
        return cls(
            spec,
            read_output.addresses,
            read_output.weights,
            write_indices,
            write_weights,
            tuple(tensor._version for tensor in concrete),
            _ROUTE_CERTIFICATE,
        )

    def require_intact(self) -> None:
        import torch

        if torch.compiler.is_compiling():
            return
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
        import torch

        if torch.compiler.is_compiling():
            return
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
