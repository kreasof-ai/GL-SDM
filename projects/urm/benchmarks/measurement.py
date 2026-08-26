"""Honest GPU timing primitives: raw samples, explicit percentiles, dedup.

Replaces the previous ``cuda_median_ms`` helper that returned a median
masquerading as p95. Every statistic here is derived from RAW samples with a
single documented quantile definition (type 7 / linear interpolation, the
numpy default), and schedules are measured in seeded, shuffled interleaved
rounds so thermal/temporal drift cannot systematically favor one point.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def quantile(samples: Sequence[float], q: float) -> float:
    """Type-7 (linear interpolation) quantile - the numpy default.

    For sorted samples ``s[0..n-1]``: ``rank = q * (n - 1)``; the value is
    linearly interpolated between ``s[floor(rank)]`` and ``s[ceil(rank)]``.
    Requires 0 <= q <= 1 and at least one sample. ``quantile(x, 0.5)`` equals
    :func:`statistics.median` for every n.
    """
    if not samples:
        raise ValueError("quantile requires at least one sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile probability must be in [0, 1], got {q}")
    ordered = sorted(float(s) for s in samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    lower = int(rank // 1)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_samples(samples: Sequence[float]) -> dict[str, float | int]:
    """Sample count plus median/p95/min from one raw sample list."""
    if not samples:
        raise ValueError("summarize_samples requires at least one sample")
    return {
        "sample_count": len(samples),
        "median_ms": quantile(samples, 0.5),
        "p95_ms": quantile(samples, 0.95),
        "min_ms": min(float(s) for s in samples),
    }


def collect_cuda_samples(
    fn: Callable[[], object],
    *,
    samples_per_round: int = 5,
) -> list[float]:
    """Raw CUDA-event timings for one round of ``samples_per_round`` runs.

    Each sample records one ``fn()`` invocation between CUDA events and a
    synchronize. The caller interleaves schedules by invoking this once per
    round per schedule (see :func:`measure_schedules_interleaved`).
    """
    samples: list[float] = []
    for _ in range(samples_per_round):
        start = torch_event(True)
        end = torch_event(True)
        start.record()
        fn()
        end.record()
        torch_sync()
        samples.append(start.elapsed_time(end))
    return samples


def torch_event(enable_timing: bool):
    import torch

    return torch.cuda.Event(enable_timing=enable_timing)


def torch_sync() -> None:
    import torch

    torch.cuda.synchronize()


def interleave_round_order(
    keys: Sequence[str], *, seed: int, rounds: int
) -> list[list[str]]:
    """Per-round randomized orderings (seeded; deterministic given seed).

    Round 0 keeps caller order (stable priming reference); each later round
    is independently shuffled so no schedule owns a fixed slot in time.
    """
    generator = random.Random(seed)
    orders: list[list[str]] = [list(keys)]
    for _ in range(max(0, rounds - 1)):
        shuffled = list(keys)
        generator.shuffle(shuffled)
        orders.append(shuffled)
    return orders


def measure_schedules_interleaved(
    workloads: dict[str, Callable[[], object]],
    *,
    warmup_runs: int = 5,
    rounds: int = 4,
    samples_per_round: int = 5,
    seed: int = 17,
) -> dict[str, list[float]]:
    """Measure every workload in seeded, shuffled interleaved rounds.

    Each workload gets ``warmup_runs`` untimed priming calls first, then
    ``rounds x samples_per_round`` timed samples gathered round-robin across
    workloads within each round, so systematic thermal or clock drift spreads
    evenly instead of privileging whichever schedule ran first.

    Returns raw samples per workload key; derive medians/percentiles through
    :func:`summarize_samples`.
    """
    if not workloads:
        raise ValueError("measure_schedules_interleaved requires workloads")
    keys = list(workloads)
    for key in keys:
        for _ in range(warmup_runs):
            workloads[key]()
    torch_sync()
    samples: dict[str, list[float]] = {key: [] for key in keys}
    for order in interleave_round_order(tuple(keys), seed=seed, rounds=rounds):
        for key in order:
            samples[key].extend(
                collect_cuda_samples(
                    workloads[key], samples_per_round=samples_per_round
                )
            )
    return samples


def dedupe(items: Sequence[T], key: Callable[[T], str]) -> list[T]:
    """Stable de-duplication by ``key`` (first occurrence wins)."""
    seen: set[str] = set()
    out: list[T] = []
    for item in items:
        identity = key(item)
        if identity in seen:
            continue
        seen.add(identity)
        out.append(item)
    return out


__all__ = [
    "collect_cuda_samples",
    "dedupe",
    "interleave_round_order",
    "measure_schedules_interleaved",
    "quantile",
    "summarize_samples",
]
