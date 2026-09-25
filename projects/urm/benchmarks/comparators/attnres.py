"""Pinned fla AttnRes adapter.

Loads ``naive_attnres`` from the pinned fla checkout (fla/ops/attnres/naive.py
@ 864a87f6) — the same source the sweep used to verify arch-054. The pinned
reference runs in fp32 on CPU, so it is directly loadable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "fla/ops/attnres/naive.py"


@lru_cache(maxsize=1)
def attnres_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "fla" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned fla checkout missing at {source}; run "
            "benchmarks/provision_comparators.py fla"
        )
    repository = PINS_DIR / "fla"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_FLA_REVISION or dirty:
        raise RuntimeError(
            "attnres requires the clean pinned revision "
            f"{EXPECTED_FLA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/fla-org/flash-linear-attention",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _naive_attnres():
    identity = attnres_source_identity()
    spec = importlib.util.spec_from_file_location(
        "fla_attnres_naive_pinned", identity["source_path"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned attnres source at {identity['source_path']}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_attnres


def attnres_adapter(
    query: Any,
    residuals: list[Any],
    rms_weight: Any,
    output_rms_weight: Any | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
) -> tuple[Any, dict[str, str]]:
    """Run pinned ``naive_attnres`` and return (output, identity)."""
    identity = attnres_source_identity()
    out = _naive_attnres()(
        query, residuals, rms_weight,
        output_rms_weight=output_rms_weight, rms_eps=rms_eps, scale=scale,
    )
    return out, identity


__all__ = ["EXPECTED_FLA_REVISION", "attnres_adapter", "attnres_source_identity"]
