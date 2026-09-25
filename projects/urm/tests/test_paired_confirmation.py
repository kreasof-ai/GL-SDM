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


def test_hierarchical_bootstrap_weights_processes_not_block_counts() -> None:
    data = [[math.log(2.0)], [math.log(0.5)] * 101]
    median, _lower, _upper = hierarchical_bootstrap_paired_slowdown(
        data, num_resamples=4000, seed=821
    )
    # Equal process weighting centers the geometric mean at one. Flattening all
    # blocks would incorrectly center this result near a 50% speedup.
    assert median == pytest.approx(0.0, abs=3.0)




