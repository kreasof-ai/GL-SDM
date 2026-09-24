"""Frozen tensor contract for the first URM kernel vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class ScalarType(StrEnum):
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class TensorLayout(StrEnum):
    ROW_MAJOR_CONTIGUOUS = "row_major_contiguous"
    STRIDED = "strided"


class TensorLike(Protocol):
    shape: tuple[int, ...]
    stride: object
    dtype: object
    device: object

    def is_contiguous(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class TensorMetadata:
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: ScalarType
    device: DeviceType
    layout: TensorLayout

    @classmethod
    def from_tensor(cls, tensor: TensorLike) -> TensorMetadata:
        shape = tuple(int(size) for size in tensor.shape)
        stride_method = tensor.stride
        raw_strides = stride_method() if callable(stride_method) else stride_method
        strides = tuple(int(stride) for stride in raw_strides)
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        device_name = str(tensor.device).split(":", maxsplit=1)[0]
        try:
            dtype = ScalarType(dtype_name)
        except ValueError as error:
            raise ValueError(f"unsupported tensor dtype: {dtype_name}") from error
        try:
            device = DeviceType(device_name)
        except ValueError as error:
            raise ValueError(f"unsupported tensor device: {device_name}") from error
        layout = (
            TensorLayout.ROW_MAJOR_CONTIGUOUS
            if tensor.is_contiguous()
            else TensorLayout.STRIDED
        )
        return cls(
            shape=shape,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
        )


@dataclass(frozen=True, slots=True)
class RoutedReductionSignature:
    """Static signature for ``O[q, d] = sum_k W[q, k] V[I[q, k], d]``.

    Version 1 deliberately requires rank-2 row-major tensors. Routes are
    precomputed and may repeat both within and across query rows. Floating-point
    products and reductions use fp32 accumulation; the result uses ``value_dtype``.
    """

    queries: int
    sources: int
    route_width: int
    value_dim: int
    index_dtype: ScalarType
    weight_dtype: ScalarType
    value_dtype: ScalarType
    device: DeviceType
    index_layout: TensorLayout = TensorLayout.ROW_MAJOR_CONTIGUOUS
    weight_layout: TensorLayout = TensorLayout.ROW_MAJOR_CONTIGUOUS
    value_layout: TensorLayout = TensorLayout.ROW_MAJOR_CONTIGUOUS
    accumulation_dtype: ScalarType = ScalarType.FLOAT32
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("queries", self.queries),
            ("sources", self.sources),
            ("route_width", self.route_width),
            ("value_dim", self.value_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.route_width > self.sources:
            raise ValueError("route_width must not exceed sources")
        if self.index_dtype not in {ScalarType.INT32, ScalarType.INT64}:
            raise ValueError("indices must use int32 or int64")
        floating = {ScalarType.FLOAT16, ScalarType.BFLOAT16, ScalarType.FLOAT32}
        if self.weight_dtype not in floating or self.value_dtype not in floating:
            raise ValueError("weights and values must use floating-point dtypes")
        if self.accumulation_dtype is not ScalarType.FLOAT32:
            raise ValueError("routed reduction v1 requires fp32 accumulation")

    @classmethod
    def from_tensors(
        cls,
        indices: TensorLike,
        weights: TensorLike,
        values: TensorLike,
    ) -> RoutedReductionSignature:
        index_meta = TensorMetadata.from_tensor(indices)
        weight_meta = TensorMetadata.from_tensor(weights)
        value_meta = TensorMetadata.from_tensor(values)
        if len(index_meta.shape) != 2:
            raise ValueError("indices must have shape [queries, route_width]")
        if weight_meta.shape != index_meta.shape:
            raise ValueError("weights must have the same shape as indices")
        if len(value_meta.shape) != 2:
            raise ValueError("values must have shape [sources, value_dim]")
        if not (index_meta.device is weight_meta.device is value_meta.device):
            raise ValueError("indices, weights, and values must share a device")
        return cls(
            queries=index_meta.shape[0],
            sources=value_meta.shape[0],
            route_width=index_meta.shape[1],
            value_dim=value_meta.shape[1],
            index_dtype=index_meta.dtype,
            weight_dtype=weight_meta.dtype,
            value_dtype=value_meta.dtype,
            device=index_meta.device,
            index_layout=index_meta.layout,
            weight_layout=weight_meta.layout,
            value_layout=value_meta.layout,
        )


@dataclass(frozen=True, slots=True)
class SupportStatus:
    supported: bool
    reason: str | None = None

    @classmethod
    def yes(cls) -> SupportStatus:
        return cls(supported=True)

    @classmethod
    def no(cls, reason: str) -> SupportStatus:
        return cls(supported=False, reason=reason)

    def require(self, backend_name: str) -> None:
        if not self.supported:
            raise ValueError(
                f"backend {backend_name} does not support signature: {self.reason}"
            )


@dataclass(frozen=True, slots=True)
class RoutedReductionResult:
    output: object
    indices: object
    weights: object
    metadata: dict[str, object]


class RoutedReductionBackend(Protocol):
    name: str

    def support_status(self, signature: RoutedReductionSignature) -> SupportStatus: ...

    def execute(
        self,
        indices: object,
        weights: object,
        values: object,
        *,
        validate_indices: bool = True,
    ) -> RoutedReductionResult: ...


class RoutedReductionRegistry:
    def __init__(self, backends: tuple[RoutedReductionBackend, ...] = ()) -> None:
        self._backends: dict[str, RoutedReductionBackend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: RoutedReductionBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"backend already registered: {backend.name}")
        self._backends[backend.name] = backend

    def get(
        self, name: str, signature: RoutedReductionSignature
    ) -> RoutedReductionBackend:
        try:
            backend = self._backends[name]
        except KeyError as error:
            raise KeyError(f"unknown routed-reduction backend: {name}") from error
        backend.support_status(signature).require(name)
        return backend

    def compatible(self, signature: RoutedReductionSignature) -> tuple[str, ...]:
        return tuple(
            name
            for name, backend in self._backends.items()
            if backend.support_status(signature).supported
        )


def require_row_major(signature: RoutedReductionSignature) -> SupportStatus:
    layouts = (
        signature.index_layout,
        signature.weight_layout,
        signature.value_layout,
    )
    if any(layout is not TensorLayout.ROW_MAJOR_CONTIGUOUS for layout in layouts):
        return SupportStatus.no("v1 requires row-major contiguous inputs")
    return SupportStatus.yes()
