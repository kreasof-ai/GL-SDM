"""Device-limit profile loading for the analytical cost model.

Where the committed device-limit artifact (``results/device-limits.json``) is
available, its measured denominators (sustainable bandwidth, FP32 CUDA-core
peak) are used; otherwise conservative defaults are marked ``measured=False``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HBM_GBPS = 100.0


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


__all__ = ["DEFAULT_HBM_GBPS", "DeviceLimits"]
