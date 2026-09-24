"""Typed operations and contracts for Unified Routed Mixers.

The three semantic families each have a canonical IR home that owns its
contract and validation boundary:

- :mod:`urm.ir.k1` - K1 normalized routed reduction
- :mod:`urm.ir.k2` - K2 structured recurrence
- :mod:`urm.ir.k3` - K3 ordered sparse-state operations

The shared, backend-independent :class:`UnifiedMixerSpec` and its family enums
live in :mod:`urm.ir.graph`; the explicit effect system lives in
:mod:`urm.ir.effects`. The declarative frontend spec (``MixerSpec`` and its
enums) is re-exported here for convenience.
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
from .graph import (
    DecayGranularity,
    FeatureMap,
    K1Operation,
    MixerBackend,
    MixerIntent,
    MixerKernelFamily,
    PolynomialBasis,
    ReadTiming,
    RecurrenceOperator,
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
    "RecurrenceOperator",
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
    "k1",
    "k2",
    "k3",
]


def __getattr__(name: str):
    # Lazy submodule access breaks the import cycle between the family
    # validation modules (k1/k2/k3) and the program/graph IR they reference.
    if name in {"k1", "k2", "k3"}:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
