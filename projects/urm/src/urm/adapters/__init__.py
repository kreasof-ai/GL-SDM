"""URM adapters for pinned upstream kernel families.

Each adapter wraps one maintained upstream implementation behind a typed,
validated dispatch boundary so that URM dispatch overhead can be measured
against a direct upstream call at identical semantics.
"""

from .dense_attention import (
    DenseAttentionSpec,
    UrmDenseCausalAttentionAdapter,
)
from .gated_delta_rule import (
    GatedDeltaRuleSpec,
    UrmGatedDeltaRuleAdapter,
)
from .sparse_delta_memory import (
    SDMAdapterConfig,
    SDMAdapterMode,
    SDMAddressTrace,
    SDMOperationSpec,
    SDMState,
    SDMTraceOrigin,
    UrmSparseDeltaMemoryAdapter,
)
from .sparse_state_mixer_external import SDMSparseStateMixerFallback

__all__ = [
    "DenseAttentionSpec",
    "GatedDeltaRuleSpec",
    "SDMAdapterConfig",
    "SDMAdapterMode",
    "SDMAddressTrace",
    "SDMOperationSpec",
    "SDMSparseStateMixerFallback",
    "SDMState",
    "SDMTraceOrigin",
    "UrmDenseCausalAttentionAdapter",
    "UrmGatedDeltaRuleAdapter",
    "UrmSparseDeltaMemoryAdapter",
]
