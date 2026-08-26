"""CPU tests for benchmark measurement primitives (no GPU required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from measurement import (
    dedupe,
    interleave_round_order,
    quantile,
    summarize_samples,
)


def test_quantile_matches_known_values_on_synthetic_samples() -> None:
    # 1..100: type-7 p95 = .95*99 + 1 = 95.05; median = 50.5; min = 1.
    samples = [float(n) for n in range(1, 101)]
    assert quantile(samples, 0.95) == pytest.approx(95.05)
    assert quantile(samples, 0.5) == pytest.approx(50.5)
    assert quantile(samples, 0.0) == pytest.approx(1.0)
    assert quantile(samples, 1.0) == pytest.approx(100.0)


def test_quantile_interpolates_linearly() -> None:
    # rank = 0.9 * 4 = 3.6 -> s[3] + 0.6*(s[4]-s[3]) = 8 + 2.5 = 10.5
    assert quantile([2.0, 4.0, 6.0, 8.0, 11.0], 0.9) == pytest.approx(8.0 + 0.6 * 3.0)


def test_median_via_quantile_equals_statistics_median() -> None:
    import random
    import statistics

    generator = random.Random(7)
    for _ in range(50):
        samples = [
            generator.uniform(0.0, 100.0) for _ in range(generator.randint(1, 40))
        ]
        assert quantile(samples, 0.5) == pytest.approx(statistics.median(samples))


def test_quantile_rejects_empty_and_out_of_range() -> None:
    with pytest.raises(ValueError):
        quantile([], 0.5)
    with pytest.raises(ValueError):
        quantile([1.0], 1.5)


def test_summarize_reports_genuine_p95_not_the_median() -> None:
    # A skewed synthetic set where median and p95 must differ.
    samples = [10.0] * 19 + [50.0]
    stats = summarize_samples(samples)
    assert stats["sample_count"] == 20
    assert stats["median_ms"] == pytest.approx(10.0)
    assert stats["p95_ms"] > stats["median_ms"]
    assert stats["min_ms"] == pytest.approx(10.0)


def test_dedupe_keeps_first_occurrence_order() -> None:
    class Point:
        def __init__(self, key: str) -> None:
            self.key = key

    items = [Point("a"), Point("b"), Point("a"), Point("c"), Point("b")]
    unique = dedupe(items, key=lambda p: p.key)
    assert [p.key for p in unique] == ["a", "b", "c"]


def test_round_orders_are_seeded_and_shuffle_later_rounds() -> None:
    keys = ("a", "b", "c", "d", "e")
    first = interleave_round_order(keys, seed=11, rounds=4)
    again = interleave_round_order(keys, seed=11, rounds=4)
    other_seed = interleave_round_order(keys, seed=12, rounds=4)
    assert first == again  # deterministic given the seed
    assert first[0] == list(keys)  # round 0 keeps caller order
    assert any(first[r] != list(keys) for r in range(1, len(first)))
    assert sorted(first[1]) == sorted(keys)  # shuffles preserve membership
    assert other_seed[1] != first[1] or len(keys) < 2
