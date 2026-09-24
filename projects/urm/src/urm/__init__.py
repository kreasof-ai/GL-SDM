"""URM: semantic-to-execution compiler for routed sequence models.

Public surface: the typed frontend spec, the shared semantic IR, and the
routed-reduction tensor contract. Compilation and plan binding live under
:mod:`urm.compiler` and :mod:`urm.runtime`; backends live under
:mod:`urm.backends`.
"""

from .frontend.spec import (
    BalanceStrategy,
    CapacityPolicy,
    CollisionPolicy,
    DecayGranularity,
    Domain,
    EditGateKind,
    ExpertFunction,
    ExpertRoutingSpec,
    ExpertScoreKind,
    ExpertSelection,
    MixerSpec,
    MutationKind,
    Normalization,
    RecurrentAlgorithm,
    RecurrentSpec,
    Residency,
    RoutingKind,
    ScanMode,
    ScoreActivation,
    SelectionGranularity,
    SelectionScope,
    SparseAttentionSpec,
    SparseIndexerKind,
    StateLayout,
)
from .backends.reference.numpy import ReferenceResult, execute, merge_writes
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
from .runtime import Backend, BackendRegistry, BackendResult

__all__ = [
    "Backend",
    "BackendRegistry",
    "BackendResult",
    "BalanceStrategy",
    "CapacityPolicy",
    "CollisionPolicy",
    "DecayGranularity",
    "DeviceType",
    "Domain",
    "EditGateKind",
    "ExpertFunction",
    "ExpertRoutingSpec",
    "ExpertScoreKind",
    "ExpertSelection",
    "MixerSpec",
    "MutationKind",
    "Normalization",
    "RecurrentAlgorithm",
    "RecurrentSpec",
    "ReferenceResult",
    "Residency",
    "RoutedReductionRegistry",
    "RoutedReductionResult",
    "RoutedReductionSignature",
    "RoutingKind",
    "ScalarType",
    "ScanMode",
    "ScoreActivation",
    "SelectionGranularity",
    "SelectionScope",
    "SparseAttentionSpec",
    "SparseIndexerKind",
    "StateLayout",
    "SupportStatus",
    "TensorLayout",
    "TensorMetadata",
    "execute",
    "merge_writes",
]
