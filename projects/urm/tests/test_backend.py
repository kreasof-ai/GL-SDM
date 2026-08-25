import pytest

from urm.backend import BackendRegistry
from urm.backends import NumpyBackend
from urm.presets import (
    DEEPSEEK_V3_MOE,
    DENSE_ATTENTION,
    LINEAR_RECURRENT_MIXER,
    ROUTING_FREE_MOE,
    SPARSE_DELTA_MEMORY,
)


def test_registry_returns_only_compatible_backend() -> None:
    registry = BackendRegistry([NumpyBackend()])

    assert registry.compatible(DENSE_ATTENTION) == ("numpy_reference",)
    assert registry.compatible(LINEAR_RECURRENT_MIXER) == ()
    assert registry.compatible(SPARSE_DELTA_MEMORY) == ()
    assert registry.compatible(DEEPSEEK_V3_MOE) == ()
    assert registry.compatible(ROUTING_FREE_MOE) == ()
    assert registry.get("numpy_reference", DENSE_ATTENTION).name == "numpy_reference"


def test_registry_has_no_silent_semantic_fallback() -> None:
    registry = BackendRegistry([NumpyBackend()])

    with pytest.raises(ValueError, match="does not support"):
        registry.get("numpy_reference", LINEAR_RECURRENT_MIXER)


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="already registered"):
        BackendRegistry([NumpyBackend(), NumpyBackend()])
