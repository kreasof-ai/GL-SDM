"""Pinned fla naive attention reference adapter.

Loads ``naive_parallel_attn`` from the pinned fla checkout
(fla/ops/attn/naive.py @ 864a87f6) — the same file the direction sweep used to
verify the MHA/MQA/GQA mixer equations (rows 001/002/003). The pinned source
computes grouped-head softmax attention by reshaping ``[B,T,HQ,D]`` to
``[B,T,H,G,D]`` and sharing K/V across the group, which is exactly the K1
head-map equation.

Pin verification follows the same contract as the other comparators: clean
checkout at the expected revision plus the source-file SHA.
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
SOURCE_RELATIVE = "fla/ops/attn/naive.py"


@lru_cache(maxsize=1)
def fla_naive_source_identity() -> dict[str, str]:
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
            "fla naive attention requires the clean pinned revision "
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
def _naive_parallel_attn():
    identity = fla_naive_source_identity()
    spec = importlib.util.spec_from_file_location(
        "fla_naive_attention_pinned", identity["source_path"]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned fla naive source at {identity['source_path']}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_parallel_attn


def fla_naive_attention_adapter(
    query: Any,
    key: Any,
    value: Any,
    *,
    causal: bool = True,
    scale: float | None = None,
) -> tuple[Any, dict[str, str]]:
    """Run pinned fla ``naive_parallel_attn`` and return (output, identity).

    Layout contract is the pinned source's: q ``[B, T, HQ, D]``, k/v
    ``[B, T, H, D]`` with ``HQ % H == 0``; the group dim is internal to the
    pinned equation. Only the plain path is exposed — the optional cumulative
    gate bias, sink bias and sliding window of the pinned source are
    out of contract for the MHA/MQA/GQA rows (sweep row-001 open question)
    and are never passed by this adapter.
    """
    identity = fla_naive_source_identity()
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError("fla naive attention expects q [B,T,HQ,D], k/v [B,T,H,D]")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must share batch and head_dim")
    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("query heads must be divisible by kv heads")
    output, _max_logits = _naive_parallel_attn()(query, key, value, scale=scale, causal=causal)
    return output, identity


__all__ = ["EXPECTED_FLA_REVISION", "fla_naive_attention_adapter", "fla_naive_source_identity"]
