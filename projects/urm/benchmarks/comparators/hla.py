"""Pinned HLA (higher-order linear attention) adapter.

The pinned checkout (yifanzhang-pro/HLA @ 484fef2b) ships the paper (HLA.pdf)
and README — no executable reference implementation. The sweep verified
arch-074 against the paper's Algorithm 1 and the closed-form masked identity.
This adapter anchors to the pinned source by revision + clean-tree + source
hash; the executable oracle is the independent brute-force transcription in
``tests/test_architectures_hla.py``.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

EXPECTED_HLA_REVISION = "484fef2bb40d"  # short form; full hash checked loosely
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "HLA.pdf"


@lru_cache(maxsize=1)
def hla_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "hla_higher_order" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned hla checkout missing at {source}; run "
            "benchmarks/provision_comparators.py hla_higher_order"
        )
    repository = PINS_DIR / "hla_higher_order"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if not revision.startswith(EXPECTED_HLA_REVISION) or dirty:
        raise RuntimeError(
            f"hla requires the clean pinned revision {EXPECTED_HLA_REVISION}*; "
            f"got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/yifanzhang-pro/HLA",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


__all__ = ["EXPECTED_HLA_REVISION", "hla_source_identity"]
