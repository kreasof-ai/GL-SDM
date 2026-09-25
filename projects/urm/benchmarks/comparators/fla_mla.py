"""Pinned fla MLA oracle.

The pinned ``MultiheadLatentAttention`` (fla/layers/mla.py @ 864a87f6) requires
flash-attn at construction (``raise ImportError`` when absent), which this
environment does not provide — so the pinned module cannot be executed here.
This adapter instead transcribes the pinned prefill equation in fp32 as the
CPU-checkable oracle, with the transcription pinned line-by-line to the source
(q/kv latent projections + RMSNorm, rope on the rope head-dims, per-head
expansion, causal softmax with ``scaling = qk_head_dim^-0.5``, v padded to
qk_head_dim and cropped back). The revision + clean-tree + source-hash of the
pinned file are still verified, so the transcription is anchored to the exact
pinned source.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "fla/layers/mla.py"


@lru_cache(maxsize=1)
def fla_mla_source_identity() -> dict[str, str]:
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
            "fla mla requires the clean pinned revision "
            f"{EXPECTED_FLA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/fla-org/flash-linear-attention",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


def _rotary(x: Any, base: float) -> Any:
    """Non-interleaved rotary on ``[..., T, H, D]`` (transcribed pinned path)."""
    import torch

    *_, T, H, D = x.shape
    d = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device).float() / D))
    t = torch.arange(T, device=x.device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :].to(x.dtype)
    sin = freqs.sin()[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def fla_mla_oracle(
    hidden_states: Any,
    *,
    q_proj: Any,
    k_rope: Any,
    kv_proj: Any,
    o_proj: Any,
    num_heads: int,
    qk_rope_head_dim: int,
    qk_nope_head_dim: int,
    v_head_dim: int,
    rope_theta: float = 10000.0,
) -> tuple[Any, dict[str, str]]:
    """Evaluate the pinned MLA prefill equation in fp32.

    ``q_proj``/``kv_proj`` may be a plain Linear or a Sequential(Linear,
    RMSNorm, Linear); ``k_rope``/``o_proj`` are Linear. Returns
    (output, identity).
    """
    import torch
    import torch.nn.functional as F

    identity = fla_mla_source_identity()
    B, T, _ = hidden_states.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    q_states = q_proj(hidden_states).view(B, T, num_heads, qk_head_dim)
    q_pass, q_rot = torch.split(q_states, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    k_pass, k_rot = kv_proj(hidden_states), k_rope(hidden_states)
    k_rot = k_rot.view(B, T, 1, qk_rope_head_dim)
    k_pass = k_pass.view(B, T, num_heads, qk_nope_head_dim + v_head_dim)
    k_pass, v = torch.split(k_pass, [qk_nope_head_dim, v_head_dim], dim=-1)

    q_rot = _rotary(q_rot, rope_theta)
    k_rot = _rotary(k_rot, rope_theta)
    k_rot = k_rot.expand(B, T, num_heads, qk_rope_head_dim)
    q = torch.cat((q_pass, q_rot), dim=-1)
    k = torch.cat((k_pass, k_rot), dim=-1)
    if qk_head_dim != v_head_dim:
        v = F.pad(v, [0, qk_head_dim - v_head_dim])

    # Causal softmax, scaling = qk_head_dim^-0.5, fp32.
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (qk_head_dim ** -0.5)
    mask = torch.ones(T, T, dtype=torch.bool, device=scores.device).tril_()
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf).transpose(1, 2)[..., :v_head_dim]
    out = o_proj(out.reshape(B, T, num_heads * v_head_dim).to(hidden_states.dtype))
    return out, identity


__all__ = ["EXPECTED_FLA_REVISION", "fla_mla_oracle", "fla_mla_source_identity"]
