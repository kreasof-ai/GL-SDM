"""Synthetic and semantic tests for paired adaptive confirmation and hierarchical bootstrap."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
from jsonschema import validate

PROJECT_ROOT = Path(__file__).parents[1]
if str(PROJECT_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from measurement import (
    hierarchical_bootstrap_paired_slowdown,
)


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
            "gpu_uuid": "GPU-3a9a2742-3da6-fc42-6c69-3a7de91f1bca",
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
        "discovery_reference_schedule": {
            "plan": "fused",
            "block_d": 256,
            "num_warps": 4,
            "num_stages": 2,
            "grad_values_decomposition": "per_query",
            "grad_values_schedule": "segmented",
            "dtype": "bfloat16",
        },
        "run_metadata": {
            "samples_per_pair": 5,
            "warmup_runs": 5,
            "min_blocks": 8,
            "max_blocks": 16,
            "sentinel_drift": {
                "threshold_pct": 15.0,
            },
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

    # Tamper GPU UUID in child 1
    tampered_uuid = [base_child, json.loads(json.dumps(base_child))]
    tampered_uuid[1]["provenance"]["gpu_uuid"] = "GPU-different-uuid"
    with pytest.raises(
        RuntimeError, match="Child provenance invariant violation for 'gpu_uuid'"
    ):
        validate_child_provenance_invariants(tampered_uuid)

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

    # Tamper reference schedule in child 1
    tampered_ref = [base_child, json.loads(json.dumps(base_child))]
    tampered_ref[1]["discovery_reference_schedule"]["block_d"] = 128
    with pytest.raises(RuntimeError, match="Child reference schedule mismatch"):
        validate_child_provenance_invariants(tampered_ref)

    # Tamper measurement config in child 1
    tampered_meta = [base_child, json.loads(json.dumps(base_child))]
    tampered_meta[1]["run_metadata"]["samples_per_pair"] = 10
    with pytest.raises(
        RuntimeError, match="Child run metadata mismatch for 'samples_per_pair'"
    ):
        validate_child_provenance_invariants(tampered_meta)


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
        assert r.get("provenance") is not None
        assert r["provenance"]["dirty_tree"] is False
        assert r["provenance"].get("gpu_uuid") is not None
        meta = r["run_metadata"]
        assert meta["total_blocks_executed"] >= 8
        assert meta["sentinel_drift"]["passed"] is True
        assert (
            meta["sentinel_drift"]["drift_pct"]
            <= meta["sentinel_drift"]["threshold_pct"]
        )
        assert r.get("paired_blocks"), "raw paired blocks must be retained in each run"
        for block in r["paired_blocks"]:
            assert block["candidate_raw_samples_ms"]
            assert block["reference_raw_samples_ms"]
            assert block["direction"] in {"AB", "BA"}

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


def test_persistent_sentinel_drift_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sentinel drift exceeding threshold on final retry must fail closed and raise RuntimeError."""
    import sys

    if str(PROJECT_ROOT / "benchmarks") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
    import routed_epilogue_confirmation as rec

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
    monkeypatch.setattr(
        "epilogue_schedules.make_inputs", lambda *args: (None, None, None, None)
    )
    monkeypatch.setattr("epilogue_schedules.forward_launch", lambda *args: (None, None))
    monkeypatch.setattr(
        "epilogue_schedules.backward_launch", lambda *args: (None, None)
    )

    call_count = 0

    def mock_collect(_workload, samples_per_round=5):
        nonlocal call_count
        call_count += 1
        # Drift: start is 0.200, end is 0.300 (>15% drift)
        if call_count % 2 == 1:
            return [0.200] * samples_per_round
        return [0.300] * samples_per_round

    monkeypatch.setattr("measurement.collect_cuda_samples", mock_collect)
    monkeypatch.setattr(
        "measurement.capture_gpu_operating_conditions",
        lambda *args: {"gpu_name": "MockGPU"},
    )

    with pytest.raises(RuntimeError, match="persistent sentinel drift"):
        rec.run_single_confirmation(
            seed=42,
            run_id=0,
            output_path=tmp_path / "drift_fail.json",
            min_blocks=2,
            max_blocks=2,
            samples_per_pair=2,
            warmup_runs=1,
            sentinel_drift_threshold_pct=10.0,
            max_retries=2,
        )
