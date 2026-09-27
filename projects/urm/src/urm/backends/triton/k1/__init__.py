"""K1 family — native Triton ops over the full source history (score/select/reduce).

One file per op. The family directory hosts the ops; this module only re-exports the
combined provider tuple so the registry's per-family import finds them.
"""

from .channel_decay import K1NativeChannelDecayProvider
from .indexed import K1NativeIndexedTritonProvider
from .map_normalize import K1NativeMapNormalizeProvider
from .online_softmax import K1NativeTritonProvider
from .squared_sum import K1NativeSquaredSumProvider
from .threshold_relu import K1NativeThresholdReluProvider

PROVIDERS = (K1NativeTritonProvider(), K1NativeIndexedTritonProvider(),
             K1NativeSquaredSumProvider(), K1NativeThresholdReluProvider(),
             K1NativeChannelDecayProvider(), K1NativeMapNormalizeProvider())

__all__ = ["PROVIDERS", "K1NativeTritonProvider", "K1NativeIndexedTritonProvider",
           "K1NativeSquaredSumProvider", "K1NativeThresholdReluProvider",
           "K1NativeChannelDecayProvider", "K1NativeMapNormalizeProvider"]
