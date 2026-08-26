"""Analytical cost model for plans and candidates.

Estimates are *explicitly analytical*: they are kept separate from measured
counters and never presented as measurements. Where the committed device-limit
artifact (``results/device-limits.json``) is available, its measured
denominators (sustainable bandwidth, FP32 CUDA-core peak) are used; otherwise
conservative defaults are marked ``measured=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_HBM_GBPS = 100.0  # deliberately conservative when unmeasured


@dataclass(frozen=True, slots=True)
class DeviceLimits:
    """Denominators for analytical estimates."""

    hbm_gbps: float = DEFAULT_HBM_GBPS
    fp32_tfps: float = 1.0
    measured: bool = False
    source: str = "defaults"

    @classmethod
    def load(cls, path: Path | None) -> DeviceLimits:
        if path is None or not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            bandwidth = float(data["bandwidth"]["sustainable_gbps"])
            compute = float(data["fp32_cuda_core"]["fp32_cuda_core_tfps_measured"])
        except (KeyError, TypeError, ValueError):
            return cls()
        return cls(
            hbm_gbps=bandwidth,
            fp32_tfps=compute,
            measured=True,
            source=str(path),
        )


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Analytical features of one compiled candidate.

    All fields are estimates. Nothing here is a measured counter; benchmarked
    artifacts live under results/ with their own schemas.
    """

    useful_flops: int
    logical_bytes: int
    physical_bytes_estimate: int
    launch_count: int
    temporary_bytes: int
    route_imbalance: float = 0.0
    atomic_contention_indicator: float = 0.0
    communication_bytes: int = 0
    collective_startup_us: float = 0.0
    critical_path_us: float = 0.0
    recompute_bytes: int = 0
    saved_state_bytes: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "useful_flops": self.useful_flops,
            "logical_bytes": self.logical_bytes,
            "physical_bytes_estimate": self.physical_bytes_estimate,
            "launch_count": self.launch_count,
            "temporary_bytes": self.temporary_bytes,
            "route_imbalance": self.route_imbalance,
            "atomic_contention_indicator": self.atomic_contention_indicator,
            "communication_bytes": self.communication_bytes,
            "collective_startup_us": self.collective_startup_us,
            "critical_path_us": self.critical_path_us,
            "recompute_bytes": self.recompute_bytes,
            "saved_state_bytes": self.saved_state_bytes,
            "provenance": "analytical_estimate",
            "notes": list(self.notes),
        }


def routed_reduction_cost(
    *,
    queries: int,
    sources: int,
    route_width: int,
    value_dim: int,
    dtype_bytes: int = 2,
) -> CostEstimate:
    """Static analytic model for routed weighted reduction.

    Logical traffic: read indices+weights once, read gathered rows (upper
    bound - duplicates may hit cache), write output once. FLOPs follow the
    documented useful-FLOP model: 2*Q*K*D multiply-adds per direction.
    """

    index_bytes = queries * route_width * 8
    weight_bytes = queries * route_width * dtype_bytes
    gather_upper = queries * route_width * value_dim * dtype_bytes
    out_bytes = queries * value_dim * dtype_bytes
    logical = index_bytes + weight_bytes + gather_upper + out_bytes
    flops = 2 * queries * route_width * value_dim
    imbalance = 1.0  # uniform placeholder; traces refine per-route histograms
    return CostEstimate(
        useful_flops=flops,
        logical_bytes=logical,
        physical_bytes_estimate=logical,
        launch_count=1,
        temporary_bytes=out_bytes,
        route_imbalance=imbalance,
        atomic_contention_indicator=0.0,
        notes=("gather upper bound assumes no L2 reuse of duplicate rows",),
    )


def row_scale_transform_cost(
    *,
    queries: int,
    value_dim: int,
    dtype_bytes: int = 2,
) -> CostEstimate:
    """Cost of a materialized standalone row-scale pass."""
    io = queries * value_dim * dtype_bytes  # read + write
    scale_bytes = queries * dtype_bytes
    return CostEstimate(
        useful_flops=queries * value_dim,
        logical_bytes=io + scale_bytes,
        physical_bytes_estimate=io + scale_bytes,
        launch_count=1,
        temporary_bytes=0,
        notes=("materialized transform reads and rewrites the full tensor",),
    )


def exchange_cost(
    *,
    payloads: int,
    payload_bytes: int,
    hop_count: int = 1,
    collective_startup_us: float = 5.0,
) -> CostEstimate:
    """Point-to-point / grouped exchange estimate over a simulated mesh."""
    wire = payloads * payload_bytes * 2  # send + receive accounting
    flops = 0
    return CostEstimate(
        useful_flops=flops,
        logical_bytes=wire,
        physical_bytes_estimate=wire,
        launch_count=max(1, hop_count),
        temporary_bytes=payloads * payload_bytes,
        communication_bytes=wire,
        collective_startup_us=collective_startup_us if hop_count else 0.0,
        critical_path_us=(
            wire * hop_count / (DEFAULT_HBM_GBPS * 1e9) * 1e6
            + collective_startup_us * hop_count
        ),
        notes=("simulated mesh: bytes are analytic, not NVLink-measured",),
    )


def combine(*parts: CostEstimate) -> CostEstimate:
    """Sum independent estimates into one program-level estimate."""
    if not parts:
        raise ValueError("combine requires at least one estimate")
    total_flops = sum(p.useful_flops for p in parts)
    total_logical = sum(p.logical_bytes for p in parts)
    total_physical = sum(p.physical_bytes_estimate for p in parts)
    total_launches = sum(p.launch_count for p in parts)
    total_temp = sum(p.temporary_bytes for p in parts)
    total_comm = sum(p.communication_bytes for p in parts)
    max_path = max(p.critical_path_us for p in parts)
    recompute = sum(p.recompute_bytes for p in parts)
    saved = sum(p.saved_state_bytes for p in parts)
    return CostEstimate(
        useful_flops=total_flops,
        logical_bytes=total_logical,
        physical_bytes_estimate=total_physical,
        launch_count=total_launches,
        temporary_bytes=total_temp,
        communication_bytes=total_comm,
        critical_path_us=max_path,
        recompute_bytes=recompute,
        saved_state_bytes=saved,
        notes=sum((p.notes for p in parts), start=()),
    )
