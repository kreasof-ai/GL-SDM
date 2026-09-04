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
    SDMAddressTrace,
    SDMOperationSpec,
    SDMState,
    UrmSparseDeltaMemoryAdapter,
)

__all__ = [
    "DenseAttentionSpec",
    "GatedDeltaRuleSpec",
    "SDMAddressTrace",
    "SDMOperationSpec",
    "SDMState",
    "UrmDenseCausalAttentionAdapter",
    "UrmGatedDeltaRuleAdapter",
    "UrmSparseDeltaMemoryAdapter",
]
