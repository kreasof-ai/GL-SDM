"""Typed operations and contracts for Unified Routed Mixers.

The three semantic families each have a canonical IR home that owns its
contract and validation boundary:

- :mod:`urm.ir.softmax` - K1 normalized routed reduction
- :mod:`urm.ir.recurrence` - K2 structured recurrence
- :mod:`urm.ir.sparse_state` - K3 ordered sparse-state operations

The shared, backend independent :class:`UnifiedMixerSpec` and its enums live in
:mod:`urm.ir.mixer`.
"""

from urm.frontend.spec import (
    BalanceStrategy,
    CapacityPolicy,
    CollisionPolicy,
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
from . import recurrence, softmax, sparse_state
from .mixer import (
    DecayGranularity,
    FeatureMap,
    K1Operation,
    MixerBackend,
    MixerIntent,
    MixerKernelFamily,
    PolynomialBasis,
    ReadTiming,
    RecurrentLayout,
    StateEffect,
    StateNormalizer,
    StateTransition,
    StateUpdateRule,
    UnifiedMixerSpec,
)

__all__ = [
    "BalanceStrategy",
    "CapacityPolicy",
    "CollisionPolicy",
    "DecayGranularity",
    "Domain",
    "EditGateKind",
    "ExpertFunction",
    "ExpertRoutingSpec",
    "ExpertScoreKind",
    "ExpertSelection",
    "FeatureMap",
    "K1Operation",
    "MixerBackend",
    "MixerIntent",
    "MixerKernelFamily",
    "MixerSpec",
    "MutationKind",
    "Normalization",
    "PolynomialBasis",
    "ReadTiming",
    "RecurrentAlgorithm",
    "RecurrentLayout",
    "RecurrentSpec",
    "Residency",
    "RoutingKind",
    "ScanMode",
    "ScoreActivation",
    "SelectionGranularity",
    "SelectionScope",
    "SparseAttentionSpec",
    "SparseIndexerKind",
    "StateEffect",
    "StateLayout",
    "StateNormalizer",
    "StateTransition",
    "StateUpdateRule",
    "UnifiedMixerSpec",
    "recurrence",
    "softmax",
    "sparse_state",
]
