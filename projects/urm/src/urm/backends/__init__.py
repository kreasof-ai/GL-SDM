from .numpy_backend import NumpyBackend
from .sparse_memory import TritonSparseMemoryBackend
from .sparse_route import TritonSparseRouteBackend
from .sparse_state_mixer import TritonSparseStateMixerBackend
from .torch_backend import TorchRoutedReductionBackend
from .triton_backend import TritonRoutedReductionBackend

__all__ = [
    "NumpyBackend",
    "TorchRoutedReductionBackend",
    "TritonRoutedReductionBackend",
    "TritonSparseMemoryBackend",
    "TritonSparseRouteBackend",
    "TritonSparseStateMixerBackend",
]


def __getattr__(name):
    legacy = {
        "DualFormSDMFunction": "SparseDeltaFunction",
        "dual_form_sdm": "sparse_delta",
    }
    if name in legacy:
        from importlib import import_module

        return getattr(import_module("urm.experimental.sparse_delta"), legacy[name])
    raise AttributeError(name)
