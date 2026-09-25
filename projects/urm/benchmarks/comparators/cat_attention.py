"""Pinned CAT (fla) adapter.

The sweep verified arch-066 against fla/models/cat/modeling_cat.py @ 864a87f6.
The pinned decoder uses ``flex_attention_compiled`` over a BlockMask built from
``get_cat_mask_mod`` — GPU/compiled — so this adapter exposes (a) the pinned
mask *function*, transcribed character-for-character from
``get_cat_mask_mod(block_size)``, and (b) an eager oracle that evaluates the
pinned equation directly: per-head softmax attention restricted to the CAT
mask (masked positions −inf pre-softmax, fully masked rows zeroed, as the K1
contract specifies). The eager oracle is the CPU-checkable authority for the
mixer equation; the pinned FlexAttention path is exercised by the native-tier
qualification (Gate 2), not here.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "fla/models/cat/modeling_cat.py"


@lru_cache(maxsize=1)
def cat_source_identity() -> dict[str, str]:
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
            "cat requires the clean pinned revision "
            f"{EXPECTED_FLA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/fla-org/flash-linear-attention",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


def pinned_cat_mask(b: int, h: int, q_idx: int, kv_idx: int, *, block_size: int) -> bool:
    """The pinned get_cat_mask_mod predicate, character-for-character."""
    within_block = (q_idx // block_size) == (kv_idx // block_size)
    compressed_token = (kv_idx % block_size) == 0
    causal = q_idx >= kv_idx
    return (within_block | compressed_token) & causal


def cat_attention_oracle(
    query: Any,
    key: Any,
    value: Any,
    *,
    block_size: int,
) -> tuple[Any, dict[str, str]]:
    """Eager evaluation of the pinned CAT decoder equation.

    q ``[B, T, HQ, D]``, k/v ``[B, T, H, D]`` with ``HQ % H == 0`` (grouped
    head sharing, as in the pinned decoder's ``repeat`` over kv groups);
    per-head softmax attention restricted to the pinned CAT mask, masked rows
    zeroed. Returns (output [B, T, HQ, D], identity).
    """
    import torch

    identity = cat_source_identity()
    B, T, HQ, D = query.shape
    H = key.shape[2]
    group = HQ // H
    scale = D ** -0.5

    mask = torch.zeros(T, T, dtype=torch.bool, device=query.device)
    for q_idx in range(T):
        for kv_idx in range(T):
            mask[q_idx, kv_idx] = pinned_cat_mask(0, 0, q_idx, kv_idx, block_size=block_size)

    q = query.float().transpose(1, 2)                       # [B,HQ,T,D]
    k = key.float().repeat_interleave(group, dim=2).transpose(1, 2)
    v = value.float().repeat_interleave(group, dim=2).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)                # all-masked rows -> zero
    out = torch.matmul(probs, v).transpose(1, 2).to(query.dtype)
    return out, identity


__all__ = ["EXPECTED_FLA_REVISION", "cat_attention_oracle", "cat_source_identity", "pinned_cat_mask"]
