"""Optional binding to ATMA's pinned slot-table gated-delta decode kernel."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

ATMA_GATED_DELTA_REVISION = "28bb3de8afbe7c0b00115e0fbff36afc9ad49c11"


@lru_cache(maxsize=1)
def atma_gated_delta_decode_step():
    """Load ATMA's exact pinned Triton decode step, when the checkout is installed."""
    try:
        from kernel import gated_delta_triton
    except ImportError as error:
        raise RuntimeError(
            "ATMA gated-delta decode requires the ATMA checkout on PYTHONPATH"
        ) from error

    module_path = Path(gated_delta_triton.__file__).resolve()
    repository = module_path.parents[1]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot establish ATMA source revision at {repository}"
        ) from error
    if revision != ATMA_GATED_DELTA_REVISION:
        raise RuntimeError(
            "ATMA gated-delta decode requires source revision "
            f"{ATMA_GATED_DELTA_REVISION}, got {revision}"
        )
    if not gated_delta_triton.HAS_TRITON:
        raise RuntimeError("ATMA gated-delta decode requires a CUDA Triton runtime")
    return gated_delta_triton.gated_delta_decode_step


class AtmaGatedDeltaDecodeAdapter:
    """Call ATMA's CUDA graph capturable in-place decode kernel."""

    name = "atma_gated_delta_decode_adapter"

    def __call__(
        self,
        query: Any,
        key: Any,
        value: Any,
        gamma: Any,
        beta: Any,
        state_table: Any,
        slots: Any,
    ):
        return atma_gated_delta_decode_step()(
            query, key, value, gamma, beta, state_table, slots
        )


__all__ = [
    "ATMA_GATED_DELTA_REVISION",
    "AtmaGatedDeltaDecodeAdapter",
    "atma_gated_delta_decode_step",
]
