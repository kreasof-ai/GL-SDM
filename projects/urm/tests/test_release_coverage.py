"""Release-gate accounting must derive verdicts from frozen-matrix coverage.

The gate never trusts an artifact's self-declared ``qualified`` string
(acceptance-contract section 10). These tests pin the derivation: a workload is
``qualified`` only when every mandatory case, dtype, mode, and correctness
component is covered with passing parity and performance gates; a self-declared
``qualified`` on a partial slice is overridden to ``partial_coverage``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import release_coverage  # noqa: E402


def _matrix_workload():
    return {
        "id": "k2-gated-delta-recurrence",
        "family": "K2",
        "dtypes": ["bfloat16", "float32"],
        "modes": ["training_forward", "training_forward_backward", "prefill", "decode"],
        "correctness": {"components": ["output", "final_state", "input_gradients", "state_gradients"]},
        "cases": [
            {"id": "latency_short", "batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32},
            {"id": "throughput_medium", "batch": 8, "sequence": 1024, "heads": 8, "key_dim": 64, "value_dim": 64},
        ],
    }


def _case(case_id, shape, dtype, *, parity="pass", gate=True, modes=("forward", "forward_backward")):
    return case_id, {
        "shape": {**shape, "dtype": dtype},
        "parity": {
            "status": parity,
            "output_max_abs_error_vs_oracle": 1e-6,
            "final_state_max_abs_error_vs_oracle": 1e-6,
            "input_gradient_max_abs_errors_vs_oracle": {"query": 1e-6},
        },
        "performance": {
            "measurements": {
                m: {"paired_native_overhead_fraction": {"median": 0.01, "ci95_upper": 0.02, "gate": {"pass": gate}}}
                for m in modes
            }
        },
    }


def test_qualified_requires_complete_matrix_coverage():
    workload = _matrix_workload()
    cases = {}
    for cid, shape in (
        ("latency_short", {"batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32}),
        ("throughput_medium", {"batch": 8, "sequence": 1024, "heads": 8, "key_dim": 64, "value_dim": 64}),
    ):
        for dtype in ("float32", "bfloat16"):
            key, payload = _case(
                f"{cid}/{dtype}", shape, dtype,
                modes=("forward", "forward_backward", "prefill", "decode"),
            )
            # Add the state-gradient component so coverage is complete.
            payload["parity"]["state_gradient_max_abs_errors"] = {"state": 1e-6}
            cases[key] = payload
    artifact = {"verdict": "qualified", "cases": cases}
    derived = release_coverage.derive_workload_coverage(workload, artifact)
    assert derived["verdict"] == "qualified"
    assert derived["complete"]


def test_self_declared_qualified_on_partial_slice_is_overridden():
    workload = _matrix_workload()
    # Only one case, one dtype, two modes: a passing benchmark slice.
    key, payload = _case(
        "latency_short/float32",
        {"batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32},
        "float32",
    )
    artifact = {"verdict": "qualified", "cases": {key: payload}}
    derived = release_coverage.derive_workload_coverage(workload, artifact)
    assert derived["verdict"] == "partial_coverage"
    assert not derived["complete"]
    assert any("throughput_medium" in m for m in derived["missing"])
    assert any("prefill" in m or "decode" in m for m in derived["missing"])


def test_parity_failure_yields_numeric_failed_not_qualified():
    workload = _matrix_workload()
    key, payload = _case(
        "latency_short/float32",
        {"batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32},
        "float32",
        parity="fail",
    )
    artifact = {"verdict": "qualified", "cases": {key: payload}}
    derived = release_coverage.derive_workload_coverage(workload, artifact)
    assert derived["verdict"] == "numeric_failed"


def test_state_gradient_component_is_mandatory_for_stateful_workloads():
    workload = _matrix_workload()
    cases = {}
    for cid, shape in (
        ("latency_short", {"batch": 1, "sequence": 64, "heads": 4, "key_dim": 32, "value_dim": 32}),
        ("throughput_medium", {"batch": 8, "sequence": 1024, "heads": 8, "key_dim": 64, "value_dim": 64}),
    ):
        for dtype in ("float32", "bfloat16"):
            key, payload = _case(
                f"{cid}/{dtype}", shape, dtype,
                modes=("forward", "forward_backward", "prefill", "decode"),
            )
            cases[key] = payload  # no state_gradient field -> coverage gap
    artifact = {"verdict": "qualified", "cases": cases}
    derived = release_coverage.derive_workload_coverage(workload, artifact)
    assert derived["verdict"] == "partial_coverage"
    assert any("state_gradients" in g for g in derived["correctness_gaps"])


def test_no_artifact_is_not_run():
    derived = release_coverage.derive_workload_coverage(_matrix_workload(), None)
    assert derived["verdict"] == "not_run"
    assert not derived["complete"]
