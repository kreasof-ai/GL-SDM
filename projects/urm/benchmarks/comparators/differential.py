"""Pinned Diff-Transformer V1 adapter.

Extracts ``MultiheadDiffAttn`` from the pinned unilm Diff-Transformer checkout
(multihead_diffattn.py @ 50224e38) by AST — the same source the sweep used to
verify arch-067 — without importing the apex/rms_norm package. The class is the
authority for the Diff equation: one softmax over the concatenated 2H map,
split, ``attn1 − λ_full·attn2``, subln, (1−λ_init) rescale.

The pinned forward applies rotary and the causal mask internally; this adapter
drives it with rotary rel_pos supplied by the caller (fp32 cos/sin) and no
attn_mask (the causal mask is constructed inside the pinned forward).
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_DIFFERENTIAL_REVISION = "50224e387211f15ac6a3b2685730b9a0c850f145"  # unilm
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "Diff-Transformer/multihead_diffattn.py"


@lru_cache(maxsize=1)
def differential_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "differential" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned differential checkout missing at {source}; run "
            "benchmarks/provision_comparators.py differential"
        )
    repository = PINS_DIR / "differential"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_DIFFERENTIAL_REVISION or dirty:
        raise RuntimeError(
            "differential requires the clean pinned revision "
            f"{EXPECTED_DIFFERENTIAL_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/microsoft/unilm",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_diffattn_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import repeat as _repeat  # noqa: F401

    identity = differential_source_identity()
    source_path = Path(identity["source_path"])
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    # Pinned helpers the class body references.
    def lambda_init_fn(depth):
        return 0.8 - 0.6 * (2.718281828459045 ** (-0.3 * depth))

    def repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        bsz, n_kv_heads, slen, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bsz, n_kv_heads * n_rep, slen, head_dim)
        )

    # The pinned module calls kernel.rotary.apply_rotary_emb (a CUDA Triton
    # kernel, GPT-J interleaved). Load the real pinned kernel from the checkout
    # so the oracle is the pinned rotary, not a transcription.
    import importlib.util

    rotary_path = source_path.with_name("kernel") / "rotary.py"
    spec = importlib.util.spec_from_file_location("urm_diff_rotary_pinned", rotary_path)
    rotary_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rotary_mod)
    apply_rotary_emb = rotary_mod.apply_rotary_emb

    namespace: dict[str, Any] = {
        "torch": torch, "nn": nn, "F": F,
        "lambda_init_fn": lambda_init_fn,
        "repeat_kv": repeat_kv,
        "apply_rotary_emb": apply_rotary_emb,
        "RMSNorm": torch.nn.RMSNorm,
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MultiheadDiffAttn":
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"),
                namespace,
            )
    if "MultiheadDiffAttn" not in namespace:
        raise RuntimeError("pinned diffattn source did not define MultiheadDiffAttn")
    return namespace["MultiheadDiffAttn"]


def differential_attention_adapter(x, rel_pos, *, seed: int, device: str = "cpu", **config):
    """Run the pinned MultiheadDiffAttn; return (output, module, identity).

    ``rel_pos`` is the (cos, sin) rotary pair the pinned forward applies; the
    pinned rotary is a CUDA Triton kernel, so ``device="cuda"`` moves the module
    and inputs there.
    """
    import torch

    identity = differential_source_identity()
    cls = _pinned_diffattn_class()
    torch.manual_seed(seed)
    module = cls(**config).to(device)
    module.eval()
    out = module(x, rel_pos)
    return out, module, identity


__all__ = [
    "EXPECTED_DIFFERENTIAL_REVISION",
    "differential_attention_adapter",
    "differential_source_identity",
]
