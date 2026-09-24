"""Source-pinned access to Samba's dependency-light attention class."""

from __future__ import annotations

import ast
import hashlib
import math
import os
import subprocess
from pathlib import Path
from typing import Any


SAMBA_REVISION = "617c7a0f8c71f1b7cb6180b86f9543d146f5c66f"


def samba_source_root() -> Path | None:
    """Return the local pinned source checkout when present."""
    candidate = Path(
        os.environ.get("URM_SAMBA_SOURCE", "/tmp/urm-comparator-pins/samba")
    ).expanduser()
    return candidate.resolve() if (candidate / ".git").exists() else None


def load_samba_attention(source_root: Path | None = None):
    """Compile the exact CausalSelfAttention class node from pinned source.

    Loading the complete Samba module imports optional xformers, causal-conv,
    and custom rotary extensions. The no-position-embedding attention
    configuration only needs the class body and PyTorch, so this loader
    executes that unchanged AST node and leaves all called methods intact.
    """
    import torch

    repository = source_root or samba_source_root()
    if repository is None:
        raise FileNotFoundError("pinned Samba checkout is not available")
    repository = repository.resolve()
    source_path = repository / "lit_gpt/model.py"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != SAMBA_REVISION or dirty:
        raise RuntimeError(f"expected clean Samba revision {SAMBA_REVISION}, got {revision}")
    source = source_path.read_text(encoding="utf-8")
    parsed = ast.parse(source, filename=str(source_path))
    node = next(
        item
        for item in parsed.body
        if isinstance(item, ast.ClassDef) and item.name == "CausalSelfAttention"
    )
    extracted = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {
        "math": math,
        "nn": torch.nn,
        "torch": torch,
        "FlashAttention2Available": False,
    }
    exec(compile(extracted, str(source_path), "exec"), namespace)
    identity = {
        "repository": str(repository),
        "revision": revision,
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "callable_source": "CausalSelfAttention class AST from the pinned lit_gpt/model.py",
    }
    return namespace["CausalSelfAttention"], identity


__all__ = ["SAMBA_REVISION", "load_samba_attention", "samba_source_root"]
