"""Capability-checked Triton backend for the routed-reduction v1 contract."""

from __future__ import annotations

import importlib.util

from ..routed_reduction import (
    DeviceType,
    RoutedReductionResult,
    RoutedReductionSignature,
    ScalarType,
    SupportStatus,
    require_row_major,
)


def triton_dependencies_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("triton") is not None
    )


def triton_signature_status(signature: RoutedReductionSignature) -> SupportStatus:
    layout_status = require_row_major(signature)
    if not layout_status.supported:
        return layout_status
    if signature.device is not DeviceType.CUDA:
        return SupportStatus.no("Triton v1 requires CUDA tensors")
    if signature.route_width > 64:
        return SupportStatus.no("Triton v1 supports route_width <= 64")
    supported_values = {
        ScalarType.FLOAT16,
        ScalarType.BFLOAT16,
        ScalarType.FLOAT32,
    }
    if signature.weight_dtype not in supported_values:
        return SupportStatus.no("unsupported weight dtype")
    if signature.value_dtype not in supported_values:
        return SupportStatus.no("unsupported value dtype")
    return SupportStatus.yes()


def _validate_index_bounds(indices: object, sources: int) -> None:
    import torch

    minimum, maximum = torch.aminmax(indices)
    if int(minimum.item()) < 0 or int(maximum.item()) >= sources:
        raise ValueError("indices must be in [0, sources)")


def _layout_key(tensor: object) -> tuple[object, ...]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
        tensor.is_contiguous(),
    )


_SUPPORT_CACHE: dict[tuple[object, ...], tuple[SupportStatus, int, int]] = {}


class TritonRoutedReductionBackend:
    name = "triton_routed_reduction"

    def support_status(self, signature: RoutedReductionSignature) -> SupportStatus:
        signature_status = triton_signature_status(signature)
        if not signature_status.supported:
            return signature_status
        if not triton_dependencies_available():
            return SupportStatus.no("PyTorch and Triton must be installed")
        import torch

        if not torch.cuda.is_available():
            return SupportStatus.no("CUDA is not available to PyTorch")
        return SupportStatus.yes()

    def execute(
        self,
        indices: object,
        weights: object,
        values: object,
        *,
        validate_indices: bool = True,
    ) -> RoutedReductionResult:
        key = _layout_key(indices) + _layout_key(weights) + _layout_key(values)
        cached = _SUPPORT_CACHE.get(key)
        if cached is not None:
            status, route_width, value_dim = cached
            status.require(self.name)
        else:
            signature = RoutedReductionSignature.from_tensors(indices, weights, values)
            status = self.support_status(signature)
            status.require(self.name)
            route_width = signature.route_width
            value_dim = signature.value_dim
            _SUPPORT_CACHE[key] = (status, route_width, value_dim)
        if validate_indices:
            _validate_index_bounds(indices, values.shape[0])

        from ..triton_kernels.routed_reduce import launch_metadata, routed_reduce

        output = routed_reduce(indices, weights, values)
        metadata: dict[str, object] = {
            "backend": self.name,
            "schema_version": 1,
            "accumulation_dtype": "float32",
            "index_validation_sync": validate_indices,
        }
        metadata.update(launch_metadata(route_width, value_dim, indices.shape[0]))
        return RoutedReductionResult(
            output=output,
            indices=indices,
            weights=weights,
            metadata=metadata,
        )
