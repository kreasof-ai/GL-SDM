from __future__ import annotations

from dataclasses import dataclass

import pytest

from urm.backends.triton_backend import triton_signature_status
from urm.routed_reduction import (
    DeviceType,
    RoutedReductionRegistry,
    RoutedReductionSignature,
    ScalarType,
    SupportStatus,
    TensorLayout,
)


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    _strides: tuple[int, ...]
    dtype: str
    device: str
    contiguous: bool = True

    def stride(self) -> tuple[int, ...]:
        return self._strides

    def is_contiguous(self) -> bool:
        return self.contiguous


def make_signature(**overrides: object) -> RoutedReductionSignature:
    fields: dict[str, object] = {
        "queries": 32,
        "sources": 64,
        "route_width": 8,
        "value_dim": 128,
        "index_dtype": ScalarType.INT32,
        "weight_dtype": ScalarType.FLOAT16,
        "value_dtype": ScalarType.FLOAT16,
        "device": DeviceType.CUDA,
    }
    fields.update(overrides)
    return RoutedReductionSignature(**fields)  # type: ignore[arg-type]


def test_signature_is_derived_from_tensor_metadata() -> None:
    indices = FakeTensor((7, 3), (3, 1), "torch.int32", "cuda:0")
    weights = FakeTensor((7, 3), (3, 1), "torch.float16", "cuda:0")
    values = FakeTensor((11, 13), (13, 1), "torch.bfloat16", "cuda:0")

    signature = RoutedReductionSignature.from_tensors(indices, weights, values)

    assert signature.queries == 7
    assert signature.sources == 11
    assert signature.route_width == 3
    assert signature.value_dim == 13
    assert signature.accumulation_dtype is ScalarType.FLOAT32


def test_signature_rejects_shape_and_device_mismatches() -> None:
    indices = FakeTensor((7, 3), (3, 1), "int32", "cpu")
    weights = FakeTensor((7, 4), (4, 1), "float32", "cpu")
    values = FakeTensor((11, 13), (13, 1), "float32", "cpu")
    with pytest.raises(ValueError, match="same shape"):
        RoutedReductionSignature.from_tensors(indices, weights, values)

    weights = FakeTensor((7, 3), (3, 1), "float32", "cuda:0")
    with pytest.raises(ValueError, match="share a device"):
        RoutedReductionSignature.from_tensors(indices, weights, values)


def test_v1_signature_rejects_invalid_dimensions_and_dtypes() -> None:
    with pytest.raises(ValueError, match="route_width"):
        make_signature(route_width=65, sources=64)
    with pytest.raises(ValueError, match="indices"):
        make_signature(index_dtype=ScalarType.FLOAT32)


def test_triton_static_capability_checks_are_explicit() -> None:
    assert triton_signature_status(make_signature()).supported
    assert not triton_signature_status(make_signature(device=DeviceType.CPU)).supported
    assert not triton_signature_status(make_signature(route_width=65, sources=128)).supported
    status = triton_signature_status(
        make_signature(value_layout=TensorLayout.STRIDED)
    )
    assert status == SupportStatus.no("v1 requires row-major contiguous inputs")


class FakeBackend:
    name = "fake"

    def support_status(self, signature: RoutedReductionSignature) -> SupportStatus:
        del signature
        return SupportStatus.yes()

    def execute(
        self,
        indices: object,
        weights: object,
        values: object,
        *,
        validate_indices: bool = True,
    ) -> object:
        del indices, weights, values, validate_indices
        return object()


def test_registry_has_no_silent_fallback() -> None:
    registry = RoutedReductionRegistry((FakeBackend(),))
    signature = make_signature()

    assert registry.compatible(signature) == ("fake",)
    assert registry.get("fake", signature).name == "fake"
    with pytest.raises(KeyError, match="unknown routed-reduction backend"):
        registry.get("missing", signature)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeBackend())
