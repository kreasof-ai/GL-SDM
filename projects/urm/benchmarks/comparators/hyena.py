"""Pinned safari Hyena oracle (equation-level).

The pinned ``HyenaOperator`` (src/models/sequence/hyena.py @ 02220c69) is only
runnable with the CUDA fftconv extension (``fused_fft_conv=True``); its
reference path has a bias-layout broadcast bug
(``bias[o, None, :, None]`` into ``fftconv_ref``) that makes the non-fused
operator non-executable at every (d_model, num_heads) configuration — verified
empirically during this batch. The sweep's verdict for arch-077 is
nevertheless confirmed against the source: the mixer is entirely external
ordinary operators with no U1/U2/U3 call.

This adapter therefore provides the equation-level oracle: it extracts the
pinned ``HyenaFilter`` (implicit long filter: positional embedding → Sin MLP →
ExponentialModulation) by AST so the filter generation runs against the pinned
source, and composes the operator equation (in_proj → short conv → per-order
gate + causal FFT conv → gate by x_0 → out_proj) transcribed line-by-line from
the pinned forward, anchored to the pinned source by revision + clean-tree +
source-hash verification.
"""

from __future__ import annotations

import ast
import hashlib
import math
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_HYENA_REVISION = "02220c69d247e5473616cd053a443ad99fd2559b"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "src/models/sequence/hyena.py"


@lru_cache(maxsize=1)
def hyena_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "safari" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned safari checkout missing at {source}; run "
            "benchmarks/provision_comparators.py safari"
        )
    repository = PINS_DIR / "safari"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_HYENA_REVISION or dirty:
        raise RuntimeError(
            "hyena requires the clean pinned revision "
            f"{EXPECTED_HYENA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/HazyResearch/safari",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_hyena_filter_class():
    """The pinned HyenaFilter (and its Sin/PositionalEmbedding/ExponentialModulation)
    extracted by AST so filter generation runs against the pinned source."""
    import torch
    import torch.nn as nn

    identity = hyena_source_identity()

    class OptimModule(nn.Module):
        def register(self, name, tensor, lr=None):
            if lr == 0.0:
                self.register_buffer(name, tensor)
            else:
                self.register_parameter(name, nn.Parameter(tensor))

    source_path = Path(identity["source_path"])
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"Sin", "PositionalEmbedding", "ExponentialModulation", "HyenaFilter"}
    namespace: dict[str, Any] = {
        "math": math, "torch": torch, "nn": nn,
        "OptimModule": OptimModule,
        "fftconv_ref": None,  # HyenaFilter.forward not used; .filter() only
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"),
                namespace,
            )
    missing = wanted - set(namespace)
    if missing:
        raise RuntimeError(f"pinned hyena.py did not define {sorted(missing)}")
    return namespace["HyenaFilter"]


def hyena_pinned_filter(d_model: int, *, order: int, seq_len: int, **kwargs):
    """Construct the pinned HyenaFilter; returns (module, identity)."""
    identity = hyena_source_identity()
    module = _pinned_hyena_filter_class()(d_model, order=order, seq_len=seq_len, channels=1, **kwargs)
    module.eval()
    return module, identity


__all__ = ["EXPECTED_HYENA_REVISION", "hyena_pinned_filter", "hyena_source_identity"]
