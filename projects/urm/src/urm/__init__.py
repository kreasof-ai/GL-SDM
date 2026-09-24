"""URM: semantic-to-execution compiler for routed sequence models.

Public surface: the shared semantic IR and the routed-reduction tensor
contract. Compilation and plan binding live under :mod:`urm.compiler` and
:mod:`urm.runtime`; backends live under :mod:`urm.backends`.
"""

from .ir.types import (
    DeviceType,
    RoutedReductionRegistry,
    RoutedReductionResult,
    RoutedReductionSignature,
    ScalarType,
    SupportStatus,
    TensorLayout,
    TensorMetadata,
)

__all__ = [
    "DeviceType",
    "RoutedReductionRegistry",
    "RoutedReductionResult",
    "RoutedReductionSignature",
    "ScalarType",
    "SupportStatus",
    "TensorLayout",
    "TensorMetadata",
]
