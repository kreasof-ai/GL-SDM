from dataclasses import replace
import pytest

from urm.backend import BackendRegistry
from urm.backends import NumpyBackend
from urm.backends.interface import BackendCapability, BackendRequest
from urm.backends.registry import BackendDeclined, BackendRegistry as CapabilityRegistry
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


def test_capability_registry_matches_semantics_device_dtype_layout_and_mode() -> None:
    class StubBackend:
        name = "triton"

        capability = BackendCapability(
            operations=frozenset({"softmax"}),
            semantic_contracts=frozenset({"causal_masked_v1"}),
            devices=frozenset({"cuda"}),
            dtypes=frozenset({"float32"}),
            layouts=frozenset({"BTHD"}),
            modes=frozenset({"training"}),
        )

        def execute(self, request, **operands):
            return operands

    request = BackendRequest(
        operation="softmax",
        semantic_contract="causal_masked_v1",
        device="cuda",
        dtype="float32",
        layout="BTHD",
        mode="training",
    )
    implementation = StubBackend()
    registry = CapabilityRegistry([implementation])
    selected, plan = registry.select(request, backend="triton")
    assert selected is implementation
    assert plan.selected_backend == "triton"
    assert not plan.fallback_used

    for field, value in (
        ("semantic_contract", "dense_softmax_v1"),
        ("device", "cpu"),
        ("dtype", "bfloat16"),
        ("layout", "BHDT"),
        ("mode", "inference"),
    ):
        with pytest.raises(BackendDeclined):
            registry.select(replace(request, **{field: value}), backend="triton")


def test_automatic_backend_fallback_is_visible_in_execution_plan() -> None:
    class StubBackend:
        def __init__(self, name, capability):
            self.name = name
            self.capability = capability

        def execute(self, request, **operands):
            return self.name

    request = BackendRequest("softmax", "dense_v1", "cpu", "float32", "BTHD", "inference")
    unsupported = BackendCapability(
        operations=frozenset({"softmax"}),
        semantic_contracts=frozenset({"dense_v1"}),
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32"}),
        layouts=frozenset({"BTHD"}),
        modes=frozenset({"inference"}),
    )
    supported = replace(unsupported, devices=frozenset({"cpu"}))
    registry = CapabilityRegistry(
        [StubBackend("triton", unsupported), StubBackend("numpy", supported)]
    )
    selected, plan = registry.select(request, allow_fallback=True)
    assert selected.name == "numpy"
    assert plan.selected_backend == "numpy"
    assert plan.attempted_backends == ("triton", "numpy")
    assert plan.fallback_used


def test_explicit_backend_never_silently_falls_back() -> None:
    class StubBackend:
        name = "triton"
        capability = BackendCapability(
            frozenset({"softmax"}), frozenset({"dense_v1"}), frozenset({"cuda"}),
            frozenset({"float32"}), frozenset({"BTHD"}), frozenset({"inference"}),
        )

        def execute(self, request, **operands):
            raise AssertionError("declined backend must not execute")

    registry = CapabilityRegistry([StubBackend()])
    request = BackendRequest("softmax", "dense_v1", "cpu", "float32", "BTHD", "inference")
    with pytest.raises(BackendDeclined, match="unsupported device"):
        registry.select(request, backend="triton", allow_fallback=True)
