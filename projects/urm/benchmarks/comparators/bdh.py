"""Pinned adapter for the source BDH strict-past attention operator."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


EXPECTED_BDH_REVISION = "2b0d7a45b058d4309c84a10e0768d541fe18bdc2"


@lru_cache(maxsize=1)
def bdh_source_identity() -> dict[str, str]:
    module = importlib.import_module("bdh")
    source = Path(module.__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify the BDH source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if revision != EXPECTED_BDH_REVISION or dirty:
        raise RuntimeError(
            "BDH requires the clean pinned source revision "
            f"{EXPECTED_BDH_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    return {
        "repository": "https://github.com/pathwaycom/bdh",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


@lru_cache(maxsize=16)
def _attention_module(device: str, heads: int, dim: int):
    module = importlib.import_module("bdh")
    config = module.BDHConfig(
        n_embd=heads * dim,
        n_head=heads,
        mlp_internal_dim_multiplier=1,
        dropout=0.0,
    )
    return module.Attention(config).to(device=device).eval()


def bdh_attention_adapter(query: Any, value: Any) -> tuple[Any, dict[str, str]]:
    """Call the pinned ``bdh.Attention`` with the source's ``K is Q`` contract.

    URM uses BTHD at its API. The upstream module takes BHTD and constructs its
    rotary phases internally, so these are layout views around the source call.
    """
    identity = bdh_source_identity()
    if query.ndim != 4 or value.ndim != 4:
        raise ValueError("BDH attention query/value must use rank-four BTHD layout")
    batch, sequence, heads, dim = query.shape
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("BDH values must match query batch, sequence and head axes")
    if dim % 2:
        raise ValueError("BDH rotary head dimension must be even")
    if query.dtype != value.dtype or query.device != value.device:
        raise ValueError("BDH query/value must share dtype and device")
    if query.dtype != torch.float32:
        raise ValueError("the pinned BDH attention source adapter is qualified for float32")
    attention = _attention_module(str(query.device), heads, dim)
    query_bhtd = query.transpose(1, 2)
    value_bhtd = value.transpose(1, 2)
    output = attention(query_bhtd, query_bhtd, value_bhtd).transpose(1, 2)
    return output, identity


__all__ = ["EXPECTED_BDH_REVISION", "bdh_attention_adapter", "bdh_source_identity"]
