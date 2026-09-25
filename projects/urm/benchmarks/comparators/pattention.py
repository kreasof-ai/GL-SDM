"""Pinned TokenFormer Pattention adapter.

Extracts the ``Pattention`` class from the pinned TokenFormer checkout
(megatron/model/tokenformer.py @ 4d56c73f) by AST — the same source the sweep
used to verify arch-057 — without importing the wider megatron package. The
class is the authority for the parameter-token attention equation and its
nonlinear parameter-domain normalizer variants.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_PATTENTION_REVISION = "4d56c73f407635e62f6df16b97dc897b4477129e"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "megatron/model/tokenformer.py"


@lru_cache(maxsize=1)
def patention_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "pattention" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned patention checkout missing at {source}; run "
            "benchmarks/provision_comparators.py patention"
        )
    repository = PINS_DIR / "pattention"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_PATTENTION_REVISION or dirty:
        raise RuntimeError(
            "patention requires the clean pinned revision "
            f"{EXPECTED_PATTENTION_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/Haiyang-W/TokenFormer",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_pattention_class():
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    identity = patention_source_identity()
    source = Path(identity["source_path"]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Pattention"
    )
    namespace: dict[str, Any] = {
        "math": math, "torch": torch, "nn": nn, "F": F,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), identity["source_path"], "exec"),
        namespace,
    )
    return namespace["Pattention"]


def patention_adapter(
    inputs: Any,
    *,
    input_channels: int,
    output_channels: int,
    param_token_num: int,
    norm_activation_type: str,
    seed: int,
    scale: float | None = None,
) -> tuple[Any, Any, dict[str, str]]:
    """Run the pinned Pattention layer; return (output, module, identity).

    A fresh pinned module is constructed (seeded) per call so tests can copy
    its parameter tokens into the URM module under verification.
    """
    import torch
    from types import SimpleNamespace

    identity = patention_source_identity()
    cls = _pinned_pattention_class()
    torch.manual_seed(seed)
    module = cls(
        SimpleNamespace(norm_activation_type=norm_activation_type),
        input_channels,
        output_channels,
        param_token_num,
        torch.nn.init.xavier_normal_,
        torch.nn.init.xavier_normal_,
    )
    module.eval()
    out = module(inputs, dropout_p=0.0, scale=scale)
    return out, module, identity


__all__ = [
    "EXPECTED_PATTENTION_REVISION",
    "patention_adapter",
    "patention_source_identity",
]
