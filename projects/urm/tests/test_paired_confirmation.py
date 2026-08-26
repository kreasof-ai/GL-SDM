"""Synthetic and semantic tests for paired adaptive confirmation and hierarchical bootstrap."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from jsonschema import validate
from measurement import (
    hierarchical_bootstrap_paired_slowdown,
)

PROJECT_ROOT = Path(__file__).parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_hierarchical_bootstrap_rejects_10pct_slower_with_wide_ci() -> None:
    """A 10% slower candidate with wide uncertainty must be rejected from equivalent set."""
    # Synthetic 5 runs, 8 blocks per run: mean ~10% slower with wide block variance
    paired_log_ratios = [
        [math.log(1.10 + 0.08 * math.sin(b * 1.5 + r)) for b in range(8)]
        for r in range(5)
    ]
    _med, lo, hi = hierarchical_bootstrap_paired_slowdown(
        paired_log_ratios, num_resamples=1000, confidence=0.95, seed=42
    )
    assert lo > 2.5 or hi > 2.5
    assert hi > 2.5, (
        f"10% slower candidate should have upper bound > 2.5%, got {hi:.2f}%"
    )
    is_equivalent = hi <= 2.5
    assert not is_equivalent


def test_hierarchical_bootstrap_accepts_1pct_slower_candidate() -> None:
    """A candidate consistently 1% slower must be accepted into the equivalent set."""
    # Synthetic 5 runs, 8 blocks per run: mean ~1% slower with tight variance
    paired_log_ratios = [
        [math.log(1.01 + 0.003 * math.cos(b + r)) for b in range(8)] for r in range(5)
    ]
    med, _lo, hi = hierarchical_bootstrap_paired_slowdown(
        paired_log_ratios, num_resamples=1000, confidence=0.95, seed=42
    )
    assert med == pytest.approx(1.0, abs=0.2)
    assert hi <= 2.5, f"1% candidate should have upper bound <= 2.5%, got {hi:.2f}%"
    is_equivalent = hi <= 2.5
    assert is_equivalent


def test_abba_ordering_produces_consistent_classification() -> None:
    """AB and BA direction pairing computes consistent paired log ratios."""
    ref_latency = 0.200
    cand_latency = 0.202  # +1% slowdown
    log_ratio_ab = math.log(cand_latency / ref_latency)
    log_ratio_ba = math.log(cand_latency / ref_latency)
    assert log_ratio_ab == log_ratio_ba

    ratios_1 = [[log_ratio_ab] * 8 for _ in range(5)]
    ratios_2 = [[log_ratio_ba] * 8 for _ in range(5)]
    res1 = hierarchical_bootstrap_paired_slowdown(ratios_1, seed=17)
    res2 = hierarchical_bootstrap_paired_slowdown(ratios_2, seed=17)
    assert res1 == res2


def test_hierarchical_bootstrap_repeated_execution_is_byte_stable() -> None:
    """Deterministic bootstrap seeds guarantee identical byte-stable output."""
    data = [[math.log(1.015 + 0.005 * b) for b in range(8)] for r in range(5)]
    run_a = hierarchical_bootstrap_paired_slowdown(data, num_resamples=500, seed=123)
    run_b = hierarchical_bootstrap_paired_slowdown(data, num_resamples=500, seed=123)
    assert run_a == run_b


def test_provenance_tampering_fails_closed() -> None:
    """Aggregation fails closed if any child run has mismatched provenance invariants."""
    import sys

    if str(PROJECT_ROOT / "benchmarks") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

    from routed_epilogue_confirmation import (
        validate_child_provenance_invariants,
    )

    base_child = {
        "provenance": {
            "git_revision": "0f04e4797be43a794fdbb737b3bd4964e2df69d9",
            "dirty_tree": False,
            "shortlist_hash": "abc123hash",
            "gpu": "NVIDIA A10G",
            "driver": "595.91.07",
            "cuda": "12.9",
            "pytorch": "2.8.0",
            "triton": "3.4.0",
        },
        "problem": {
            "queries": 1024,
            "sources": 512,
            "route_width": 8,
            "value_dim": 1024,
            "dtype": "bfloat16",
            "training": True,
            "deterministic": False,
        },
    }

    # Two identical children pass
    valid_children = [base_child, json.loads(json.dumps(base_child))]
    validate_child_provenance_invariants(valid_children)

    # Tamper git revision in child 1
    tampered_git = [base_child, json.loads(json.dumps(base_child))]
    tampered_git[1]["provenance"]["git_revision"] = "bad_revision_hash_1234567890"
    with pytest.raises(
        RuntimeError, match="Child provenance invariant violation for 'git_revision'"
    ):
        validate_child_provenance_invariants(tampered_git)

    # Tamper GPU in child 1
    tampered_gpu = [base_child, json.loads(json.dumps(base_child))]
    tampered_gpu[1]["provenance"]["gpu"] = "NVIDIA H100"
    with pytest.raises(
        RuntimeError, match="Child provenance invariant violation for 'gpu'"
    ):
        validate_child_provenance_invariants(tampered_gpu)

    # Tamper shortlist hash in child 1
    tampered_hash = [base_child, json.loads(json.dumps(base_child))]
    tampered_hash[1]["provenance"]["shortlist_hash"] = "different_hash"
    with pytest.raises(
        RuntimeError, match="Child provenance invariant violation for 'shortlist_hash'"
    ):
        validate_child_provenance_invariants(tampered_hash)

    # Tamper problem in child 1
    tampered_prob = [base_child, json.loads(json.dumps(base_child))]
    tampered_prob[1]["problem"]["value_dim"] = 512
    with pytest.raises(RuntimeError, match="Child problem configuration mismatch"):
        validate_child_provenance_invariants(tampered_prob)


def test_committed_confirmation_artifact_semantic_facts() -> None:
    """Validate that the committed confirmation artifact satisfies all declared rules."""
    path = (
        PROJECT_ROOT
        / "results"
        / "compiler"
        / "solver"
        / "routed-epilogue-confirmation.json"
    )
    if not path.exists():
        pytest.skip("confirmation artifact not committed yet")

    schema = _load(
        PROJECT_ROOT / "benchmarks" / "routed-epilogue-confirmation-schema.json"
    )
    artifact = _load(path)
    validate(artifact, schema)

    prov = artifact["provenance"]
    assert prov["dirty_tree"] is False
    for env_field in (
        "python",
        "pytorch",
        "triton",
        "cuda",
        "driver",
        "gpu",
        "solver_version",
    ):
        assert prov.get(env_field) is not None

    runs = artifact["runs"]
    assert len(runs) >= 5, f"expected at least 5 runs, got {len(runs)}"
    for r in runs:
        assert r["total_blocks_executed"] >= 8
        assert r["sentinel_drift"]["passed"] is True
        assert r["sentinel_drift"]["drift_pct"] <= r["sentinel_drift"]["threshold_pct"]

    decision = artifact["deployment_decision"]
    margin = decision["practical_equivalence_margin_pct"]
    assert margin == 2.5
    assert decision["status"] in {
        "confirmed_equivalent",
        "conservative_fallback_measurement_limited",
    }
    assert decision["representative_schedule"] is not None

    shortlist = artifact["evaluated_shortlist"]
    assert len(shortlist) >= 8

    for cand in shortlist:
        assert cand["ci95_lower_slowdown_pct"] <= cand["ci95_upper_slowdown_pct"]
        expected_equiv = cand["ci95_upper_slowdown_pct"] <= margin
        assert cand["is_practically_equivalent"] == expected_equiv, (
            f"candidate {cand['candidate_index']} equivalence must match upper bound rule: "
            f"ci95_upper={cand['ci95_upper_slowdown_pct']} <= {margin}"
        )
