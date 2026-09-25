"""Pinned H3 adapter.

Imports the pinned H3 package (src/models/ssm/h3.py @ 5c4d06b5) with the pin
root on ``sys.path`` — the same source the sweep used to verify arch-076. The
pinned ``H3`` module runs its non-fast path (use_fast_fftconv=False), whose
causal FFT convolutions are plain ``torch.fft`` — the authority for the mixer
equation the URM external module transcribes.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_H3_REVISION = "5c4d06b5795405170387c80998b58d76179a8a1a"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "src/models/ssm/h3.py"


@lru_cache(maxsize=1)
def h3_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "h3" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned h3 checkout missing at {source}; run "
            "benchmarks/provision_comparators.py h3"
        )
    repository = PINS_DIR / "h3"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_H3_REVISION or dirty:
        raise RuntimeError(
            "h3 requires the clean pinned revision "
            f"{EXPECTED_H3_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/HazyResearch/H3",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_h3_class():
    identity = h3_source_identity()
    pin_root = str(PINS_DIR / "h3")
    # Other pinned checkouts (e.g. tucker) also ship a top-level `src` package;
    # evict any pre-imported one and import with the h3 root first.
    for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        sys.modules.pop(name, None)
    if pin_root in sys.path:
        sys.path.remove(pin_root)
    sys.path.insert(0, pin_root)
    import src  # noqa: PLC0415 — rebind a possibly-namespace `src` package

    if getattr(src, "__file__", None) is None:  # namespace package (no __init__)
        src.__path__ = [str(PINS_DIR / "h3" / "src")]
    import src.models.ssm.h3 as pinned  # noqa: PLC0415

    if Path(pinned.__file__).resolve() != Path(identity["source_path"]):
        raise RuntimeError(
            f"src.models.ssm.h3 resolved to {pinned.__file__}, "
            f"not the pinned {identity['source_path']}"
        )
    return pinned.H3


def h3_adapter(*, d_model: int, head_dim: int, d_state: int = 4, l_max: int | None = None):
    """Construct the pinned H3 module (non-fast path); return (module, identity)."""
    identity = h3_source_identity()
    cls = _pinned_h3_class()
    module = cls(
        d_model=d_model, head_dim=head_dim, d_state=d_state,
        l_max=l_max, use_fast_fftconv=False,
    )
    module.eval()
    return module, identity


__all__ = ["EXPECTED_H3_REVISION", "h3_adapter", "h3_source_identity"]
