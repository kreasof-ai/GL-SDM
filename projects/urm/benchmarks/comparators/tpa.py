"""Pinned TPA (T6) adapter.

Extracts ``CPLinear`` and ``CausalSelfAttention`` from the pinned TPA checkout
(model/T6.py @ c276c80d) by AST — the same source the sweep used to verify
arch-069 — without importing the transformers-dependent package. The pinned
classes are the authority for the CP-factorized QKV production and the plain
causal mixer.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_TPA_REVISION = "c276c80d5ad807881dedb4707d8d3c20b4e97ec6"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "model/T6.py"


@lru_cache(maxsize=1)
def tpa_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "tpa" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned tpa checkout missing at {source}; run "
            "benchmarks/provision_comparators.py tpa"
        )
    repository = PINS_DIR / "tpa"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_TPA_REVISION or dirty:
        raise RuntimeError(
            "tpa requires the clean pinned revision "
            f"{EXPECTED_TPA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/tensorgi/TPA",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_tpa_classes() -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    identity = tpa_source_identity()
    source = Path(identity["source_path"]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"RMSNorm", "Rotary", "CPLinear", "CausalSelfAttention"}
    namespace: dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "apply_rotary_emb":
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), identity["source_path"], "exec"),
                namespace,
            )
        elif isinstance(node, ast.ClassDef) and node.name in wanted:
            exec(
                compile(ast.Module(body=[node], type_ignores=[]), identity["source_path"], "exec"),
                namespace,
            )
    missing = wanted - set(namespace)
    if missing:
        raise RuntimeError(f"pinned T6.py did not define {sorted(missing)}")
    return namespace


def tpa_attention_adapter(x: Any, *, config: Any, seed: int):
    """Run the pinned CausalSelfAttention layer and return (output, module, identity).

    A fresh pinned module is constructed (seeded) per call so tests can copy
    its parameters into the URM module under verification. ``config`` needs
    ``n_head``, ``head_dim``, ``n_embd``, ``rank``, ``q_rank``.
    """
    import torch

    identity = tpa_source_identity()
    classes = _pinned_tpa_classes()
    torch.manual_seed(seed)
    module = classes["CausalSelfAttention"](config)
    module.eval()
    return module(x), module, identity


__all__ = ["EXPECTED_TPA_REVISION", "tpa_attention_adapter", "tpa_source_identity"]
