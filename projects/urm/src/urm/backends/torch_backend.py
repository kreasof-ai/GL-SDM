"""Transparent PyTorch baseline for routed weighted reduction."""

from __future__ import annotations

import importlib.util

from ..routed_reduction import (
    DeviceType,
    RoutedReductionResult,
    RoutedReductionSignature,
    SupportStatus,
)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _validate_index_bounds(indices: object, sources: int) -> None:
    import torch

    minimum, maximum = torch.aminmax(indices)
    if int(minimum.item()) < 0 or int(maximum.item()) >= sources:
        raise ValueError("indices must be in [0, sources)")


class TorchRoutedReductionBackend:
    name = "torch_routed_reduction"

    def support_status(self, signature: RoutedReductionSignature) -> SupportStatus:
        if not torch_available():
            return SupportStatus.no("PyTorch is not installed")
        if signature.device not in {DeviceType.CPU, DeviceType.CUDA}:
            return SupportStatus.no("PyTorch backend requires CPU or CUDA tensors")
        return SupportStatus.yes()

    def execute(
        self,
        indices: object,
        weights: object,
        values: object,
        *,
        validate_indices: bool = True,
    ) -> RoutedReductionResult:
        import torch

        signature = RoutedReductionSignature.from_tensors(indices, weights, values)
        self.support_status(signature).require(self.name)
        if validate_indices:
            _validate_index_bounds(indices, signature.sources)

        gathered = values[indices.to(dtype=torch.int64)]
        output = (
            gathered.to(dtype=torch.float32)
            .mul(weights.to(dtype=torch.float32).unsqueeze(-1))
            .sum(dim=1)
            .to(dtype=values.dtype)
        )
        return RoutedReductionResult(
            output=output,
            indices=indices,
            weights=weights,
            metadata={
                "backend": self.name,
                "schema_version": signature.schema_version,
                "accumulation_dtype": signature.accumulation_dtype.value,
            },
        )
