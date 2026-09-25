"""Pinned fla K2 (linear-delta / linear-attention) ops adapter.

Loads the pinned fla package (fla @ 864a87f6) with ``sys.path`` priority so the
recurrent ops resolve to the pinned source, not an installed distribution.
Each accessor returns the pinned recurrent/parallel op callable the sweep used
to verify the composition-k2-gated rows (015–052). All ops are Triton-backed
and require CUDA.

The adapter is intentionally thin: it verifies the pin (revision + clean tree)
and returns the pinned callable. Parity tests bind the URM K2 graph against
these ops on identical operands.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
PINS_DIR = Path("/tmp/urm-comparator-pins")


@lru_cache(maxsize=1)
def fla_k2_source_identity() -> dict[str, str]:
    repository = PINS_DIR / "fla"
    if not repository.exists():
        raise RuntimeError(
            f"pinned fla checkout missing at {repository}; run "
            "benchmarks/provision_comparators.py fla"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_FLA_REVISION or dirty:
        raise RuntimeError(
            "fla K2 ops require the clean pinned revision "
            f"{EXPECTED_FLA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    return {
        "repository": "https://github.com/fla-org/flash-linear-attention",
        "revision": revision,
        "source_path": str(repository / "fla"),
    }


def _load(name: str) -> Any:
    """Import a pinned fla op callable with the pin root taking priority."""
    fla_k2_source_identity()
    pin_root = str(PINS_DIR / "fla")
    if sys.path[0] != pin_root:
        if pin_root in sys.path:
            sys.path.remove(pin_root)
        sys.path.insert(0, pin_root)
    import importlib

    module_path, _, attr = name.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def fla_op(name: str) -> Any:
    """Return a pinned fla op by dotted path, e.g. ``fla.ops.gla.fused_recurrent_gla``."""
    return _load(name)


__all__ = ["EXPECTED_FLA_REVISION", "fla_k2_source_identity", "fla_op"]
