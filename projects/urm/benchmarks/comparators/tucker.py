"""Pinned Tucker Attention Triton adapter."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

EXPECTED_TUCKER_REVISION = "c3e3d3cec991f4303b824c7fb7cbb95e3748d5c7"
PINS_TUCKER_VIT = Path("/tmp/urm-comparator-pins/tucker/ViT")
EXPECTED_TUCKER_SOURCE_SHA256 = (
    "09ce13096d191d3171b94dad76a8eef3646861674bc7dba9daa88a07f9eadc8b"
)


@lru_cache(maxsize=1)
def _tucker_operator():
    import sys

    identity = tucker_source_identity()
    vit_root = str(Path(identity["source_path"]).resolve().parents[2])
    if sys.path[0] != vit_root:
        if vit_root in sys.path:
            sys.path.remove(vit_root)
        sys.path.insert(0, vit_root)
    source = importlib.import_module("src.attn.triton.tucker_attn")
    if str(Path(source.__file__).resolve()) != identity["source_path"]:
        raise RuntimeError(
            f"src.attn.triton.tucker_attn resolved to {source.__file__}, "
            f"not the pinned {identity['source_path']}"
        )
    return source.FlashAttentionTucker(causal=False, attn_autotune=False)


@lru_cache(maxsize=1)
def tucker_source_identity() -> dict[str, str]:
    import sys

    pin_vit = str(PINS_TUCKER_VIT)
    if pin_vit not in sys.path:
        sys.path.insert(0, pin_vit)
    module = importlib.import_module("src.attn.triton.tucker_attn")
    source = Path(module.__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify Tucker source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
        revision != EXPECTED_TUCKER_REVISION
        or dirty
        or digest != EXPECTED_TUCKER_SOURCE_SHA256
    ):
        raise RuntimeError(
            "Tucker requires the clean pinned source revision and source hash "
            f"{EXPECTED_TUCKER_REVISION}/{EXPECTED_TUCKER_SOURCE_SHA256}; got "
            f"{revision}/{digest} with dirty={bool(dirty)}"
        )
    return {
        "repository": "https://github.com/ScSteffen/Tucker-Attention",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


def tucker_attention_adapter(
    query: Any, key: Any, value: Any, B_pre: Any
) -> tuple[Any, dict[str, str]]:
    """Run pinned Tucker's fused K1 attention and return URM BTHD layout."""
    identity = tucker_source_identity()
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or B_pre.ndim != 3:
        raise ValueError("Tucker Q/K/V/B_pre operands must use BTR/HDR rank-three layouts")
    if query.shape[0] != key.shape[0] or query.shape[:2] != value.shape[:2]:
        raise ValueError("Tucker Q/K/V batch and sequence dimensions must match")
    if query.shape[:2] != key.shape[:2]:
        raise ValueError("Tucker query and key batch/sequence dimensions must match")
    if B_pre.shape[1:] != (query.shape[-1], key.shape[-1]):
        raise ValueError("Tucker B_pre dimensions must map query rank to key rank")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("pinned Tucker Triton attention requires float16 or bfloat16")
    if any(t.dtype != query.dtype for t in (key, value, B_pre)):
        raise ValueError("Tucker Q/K/V/B_pre operands must share a dtype")
    if any(t.device != query.device for t in (key, value, B_pre)):
        raise ValueError("Tucker Q/K/V/B_pre operands must share a device")
    if any(not t.is_contiguous() for t in (query, key, value, B_pre)):
        raise ValueError("Tucker Q/K/V/B_pre operands must be contiguous")
    output = _tucker_operator()(
        query, key, value, B_pre, sm_scale=key.shape[-1] ** -0.5
    )
    return output.transpose(1, 2), identity


__all__ = [
    "EXPECTED_TUCKER_REVISION",
    "EXPECTED_TUCKER_SOURCE_SHA256",
    "tucker_attention_adapter",
    "tucker_source_identity",
]
