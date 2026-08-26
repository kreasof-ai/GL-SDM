"""CPU-only contract tests for upstream version enforcement.

The FLA gated delta-rule comparator is pinned to flash-linear-attention 0.5.2.
The expected pin and the installed version must be recorded separately, and an
incompatible installation must be REJECTED rather than silently labeled with
the expected pin. These tests import torch but never require CUDA, Triton,
FlashAttention, or FLA itself.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from urm.adapters.gated_delta_rule import (
    EXPECTED_FLA_VERSION,
    GatedDeltaRuleSpec,
    UrmGatedDeltaRuleAdapter,
    _version_compatible,
    fla_version,
)


def test_expected_pin_is_the_frozen_contract_version() -> None:
    assert EXPECTED_FLA_VERSION == "0.5.2"


def test_version_compatibility_is_exact_match() -> None:
    assert _version_compatible("0.5.2") is True
    for incompatible in ("0.5.1", "0.6.0", "1.0.0", "0.5.2.post1", ""):
        assert _version_compatible(incompatible) is False, incompatible


def test_identity_records_expected_and_installed_separately() -> None:
    identity = fla_version()
    if identity.get("status") == "not_applicable":
        pytest.skip("flash-linear-attention not installed")
    assert identity["expected_version"] == EXPECTED_FLA_VERSION
    assert identity["installed_version"]
    assert identity["version_compatible"] is (
        identity["installed_version"] == EXPECTED_FLA_VERSION
    )


def test_incompatible_installed_version_is_rejected_not_relabeled() -> None:
    adapter = UrmGatedDeltaRuleAdapter("prefill")

    class _FakeSpec:
        mode = "prefill"
        sequence = 8

    original = fla_version
    try:
        fla_version_cache = {
            "package": "flash-linear-attention",
            "expected_version": EXPECTED_FLA_VERSION,
            "installed_version": "0.9.9",
            "version_compatible": False,
        }
        import urm.adapters.gated_delta_rule as module

        module.fla_version = lambda: fla_version_cache  # type: ignore[assignment]
        reason = adapter.support_status(_FakeSpec())  # type: ignore[arg-type]
    finally:
        import urm.adapters.gated_delta_rule as module

        module.fla_version = original  # type: ignore[assignment]

    assert reason is not None
    assert "0.9.9" in reason and EXPECTED_FLA_VERSION in reason
    assert "reinstall" in reason


def test_spec_and_support_status_stay_importable_without_cuda() -> None:
    # Importing the adapter must not require a GPU; only execution does.
    assert GatedDeltaRuleSpec is not None
    assert EXPECTED_FLA_VERSION
