"""Honest GPU timing primitives: raw samples, explicit percentiles, dedup.

Replaces the previous ``cuda_median_ms`` helper that returned a median
masquerading as p95. Every statistic here is derived from RAW samples with a
single documented quantile definition (type 7 / linear interpolation, the
numpy default), and schedules are measured in seeded, shuffled interleaved
rounds so thermal/temporal drift cannot systematically favor one point.
"""

from __future__ import annotations

import math
import random
import subprocess
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


def compute_ranks(seq: Sequence[float]) -> list[float]:
    """Compute 1-based ranks with average ties."""
    if not seq:
        return []
    indexed = sorted(enumerate(seq), key=lambda x: x[1])
    ranks = [0.0] * len(seq)
    i = 0
    n = len(seq)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 1.0 + (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation between two vectors."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    ranks_x = compute_ranks(x)
    ranks_y = compute_ranks(y)
    mean_x = sum(ranks_x) / len(ranks_x)
    mean_y = sum(ranks_y) / len(ranks_y)
    num = sum(
        (a - mean_x) * (b - mean_y) for a, b in zip(ranks_x, ranks_y, strict=True)
    )
    den_x = sum((a - mean_x) ** 2 for a in ranks_x)
    den_y = sum((b - mean_y) ** 2 for b in ranks_y)
    if den_x <= 0 or den_y <= 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] | None = None,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Deterministic bootstrap confidence interval."""
    if not values:
        raise ValueError("bootstrap_ci requires values")
    if statistic is None:
        statistic = lambda s: quantile(s, 0.5)
    if len(values) == 1:
        val = float(values[0])
        return val, val
    rng = random.Random(seed)
    n = len(values)
    boot_stats = []
    for _ in range(num_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        boot_stats.append(statistic(resample))
    boot_stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lower = quantile(boot_stats, alpha)
    upper = quantile(boot_stats, 1.0 - alpha)
    return lower, upper


def hierarchical_bootstrap_paired_slowdown(
    paired_log_ratios_by_run: Sequence[Sequence[float]],
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Two-level hierarchical bootstrap over runs and paired blocks.

    Level 1: resample runs with replacement.
    Level 2: resample paired blocks within each selected run with replacement.
    Returns (median_slowdown_pct, ci95_lower_pct, ci95_upper_pct).
    """
    num_runs = len(paired_log_ratios_by_run)
    if num_runs == 0:
        raise ValueError("hierarchical_bootstrap requires at least one run")
    rng = random.Random(seed)
    boot_slowdowns: list[float] = []
    for _ in range(num_resamples):
        resampled_runs = [
            paired_log_ratios_by_run[rng.randrange(num_runs)] for _ in range(num_runs)
        ]
        resampled_log_ratios: list[float] = []
        for run_blocks in resampled_runs:
            num_blocks = len(run_blocks)
            if num_blocks > 0:
                resampled_log_ratios.extend(
                    run_blocks[rng.randrange(num_blocks)] for _ in range(num_blocks)
                )
        if not resampled_log_ratios:
            continue
        mean_log = sum(resampled_log_ratios) / len(resampled_log_ratios)
        slowdown_pct = (math.exp(mean_log) - 1.0) * 100.0
        boot_slowdowns.append(slowdown_pct)
    if not boot_slowdowns:
        return 0.0, 0.0, 0.0
    boot_slowdowns.sort()
    alpha = (1.0 - confidence) / 2.0
    med = quantile(boot_slowdowns, 0.5)
    lo = quantile(boot_slowdowns, alpha)
    hi = quantile(boot_slowdowns, 1.0 - alpha)
    return med, lo, hi


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


def capture_gpu_operating_conditions(
    own_pids: Sequence[int] | None = None,
) -> dict[str, object]:
    """Capture read-only operating conditions of the GPU via nvidia-smi."""
    import os

    query_cmd = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,clocks.current.sm,clocks.current.memory,power.draw,power.limit,temperature.gpu,utilization.gpu,utilization.memory,persistence_mode,compute_mode",
        "--format=csv,noheader,nounits",
    ]
    try:
        res = subprocess.run(query_cmd, capture_output=True, text=True, check=True)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        if len(parts) >= 12:
            (
                name,
                uuid,
                driver,
                sm_clk,
                mem_clk,
                pwr,
                pwr_lim,
                temp,
                util_gpu,
                util_mem,
                persist,
                comp_mode,
            ) = parts[:12]

            proc_res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            all_compute_apps: list[dict[str, object]] = []
            external_compute_apps: list[dict[str, object]] = []
            known_pids = {os.getpid(), os.getppid()}
            if own_pids:
                known_pids.update(own_pids)

            for line in proc_res.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                app_parts = [p.strip() for p in line.split(",")]
                pid = int(app_parts[0]) if app_parts[0].isdigit() else -1
                pname = app_parts[1] if len(app_parts) > 1 else "unknown"
                pmem = app_parts[2] if len(app_parts) > 2 else "unknown"
                entry = {"pid": pid, "process_name": pname, "used_memory": pmem}
                all_compute_apps.append(entry)
                if pid not in known_pids:
                    external_compute_apps.append(entry)

            # Check application clocks
            app_clock_query = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.applications.graphics,clocks.applications.memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            app_clocks_supported = (
                "deprecated" not in app_clock_query.stdout.lower()
                and app_clock_query.returncode == 0
            )

            return {
                "gpu_name": name,
                "gpu_uuid": uuid,
                "driver_version": driver,
                "sm_clock_mhz": int(sm_clk) if sm_clk.isdigit() else None,
                "memory_clock_mhz": int(mem_clk) if mem_clk.isdigit() else None,
                "power_draw_w": float(pwr) if _is_number(pwr) else None,
                "power_limit_w": float(pwr_lim) if _is_number(pwr_lim) else None,
                "temperature_c": int(temp) if temp.isdigit() else None,
                "gpu_utilization_pct": (int(util_gpu) if util_gpu.isdigit() else None),
                "memory_utilization_pct": (
                    int(util_mem) if util_mem.isdigit() else None
                ),
                "persistence_mode": persist,
                "compute_mode": comp_mode,
                "application_clocks_fixed": None,
                "application_clocks_status": (
                    "supported" if app_clocks_supported else "deprecated_or_unsupported"
                ),
                "clock_locking": {
                    "requested": False,
                    "applied": False,
                    "verified": False,
                    "restored": False,
                },
                "exclusive_process_mode": comp_mode.lower() == "exclusive_process",
                "all_compute_processes": all_compute_apps,
                "external_compute_processes": external_compute_apps,
                "unavailable_reason": None,
            }
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "driver_version": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "temperature_c": None,
            "gpu_utilization_pct": None,
            "memory_utilization_pct": None,
            "persistence_mode": None,
            "compute_mode": None,
            "application_clocks_fixed": None,
            "clock_locking": {
                "requested": False,
                "applied": False,
                "verified": False,
                "restored": False,
            },
            "exclusive_process_mode": None,
            "all_compute_processes": None,
            "external_compute_processes": None,
            "unavailable_reason": f"unexpected query format: {res.stdout[:100]}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "gpu_name": None,
            "gpu_uuid": None,
            "driver_version": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "temperature_c": None,
            "gpu_utilization_pct": None,
            "memory_utilization_pct": None,
            "persistence_mode": None,
            "compute_mode": None,
            "application_clocks_fixed": None,
            "clock_locking": {
                "requested": False,
                "applied": False,
                "verified": False,
                "restored": False,
            },
            "exclusive_process_mode": None,
            "all_compute_processes": None,
            "external_compute_processes": None,
            "unavailable_reason": str(exc),
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
    "bootstrap_ci",
    "capture_gpu_operating_conditions",
    "collect_cuda_samples",
    "compute_ranks",
    "dedupe",
    "hierarchical_bootstrap_paired_slowdown",
    "interleave_round_order",
    "measure_schedules_interleaved",
    "quantile",
    "spearman_rank_correlation",
    "summarize_samples",
]
