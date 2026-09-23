"""Pinned Longformer sliding-chunks attention adapter (local, no global tokens)."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

EXPECTED_LONGFORMER_REVISION = "caefee668e39cacdece7dd603a0bebf24df6d8ca"
EXPECTED_LONGFORMER_HASHES = {
    "sliding_chunks.py": "9cf13638a7b0eabd3e12ea502132342e12d590ff8097d5d21836aeeaab45cd87",
    "diagonaled_mm_tvm.py": "70c3008f0d39624daf9fb85b29e67519f5ceaab492627882b7e863f510b2b130",
}


@lru_cache(maxsize=1)
def _load_sliding_chunks():
    package_path = next(
        (
            Path(entry).resolve() / "longformer"
            for entry in sys.path
            if (Path(entry).resolve() / "longformer" / "sliding_chunks.py").is_file()
        ),
        None,
    )
    if package_path is None:
        raise ModuleNotFoundError(
            "pinned Longformer source requires its checkout root on PYTHONPATH"
        )
    loaded = sys.modules.get("longformer")
    if loaded is None:
        loaded = types.ModuleType("longformer")
        loaded.__path__ = [str(package_path)]
        loaded.__package__ = "longformer"
        sys.modules["longformer"] = loaded
    elif str(package_path) not in getattr(loaded, "__path__", ()):
        raise RuntimeError("a different longformer package is already imported")
    return importlib.import_module("longformer.sliding_chunks")


@lru_cache(maxsize=1)
def longformer_source_identity() -> dict[str, Any]:
    module = _load_sliding_chunks()
    source = Path(module.__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify Longformer source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    paths = (
        source,
        source.parent / "diagonaled_mm_tvm.py",
    )
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if (
        revision != EXPECTED_LONGFORMER_REVISION
        or dirty
        or hashes != EXPECTED_LONGFORMER_HASHES
    ):
        raise RuntimeError(
            "Longformer requires the clean pinned source revision and exact files; "
            f"got revision={revision}, dirty={bool(dirty)}, hashes={hashes}"
        )
    return {
        "repository": "https://github.com/allenai/longformer",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": hashes,
    }


def longformer_attention_adapter(
    query: Any, key: Any, value: Any, attention_window: int
) -> tuple[Any, dict[str, Any]]:
    """Run Longformer's upstream banded QK/PV operations for local-only rows."""
    identity = longformer_source_identity()
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("Longformer Q/K/V operands must use BTHD layout")
    if query.shape != key.shape or query.shape[:3] != value.shape[:3]:
        raise ValueError("Longformer local attention requires matching Q/K and B/T/H")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Longformer attention requires float32, float16, or bfloat16")
    if any(tensor.device != query.device for tensor in (key, value)):
        raise ValueError("Longformer Q/K/V operands must share a device")
    if not isinstance(attention_window, int) or attention_window <= 0:
        raise ValueError("attention_window must be a positive one-sided window size")
    if query.shape[1] % (2 * attention_window):
        raise ValueError("Longformer sequence length must be divisible by 2*attention_window")
    source = _load_sliding_chunks()
    scaled_query = query * (query.shape[-1] ** -0.5)
    scores = source.sliding_chunks_matmul_qk(
        scaled_query, key, attention_window, padding_value=0.0
    )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
    output = source.sliding_chunks_matmul_pv(probabilities, value, attention_window)
    return output, identity


__all__ = [
    "EXPECTED_LONGFORMER_HASHES",
    "EXPECTED_LONGFORMER_REVISION",
    "longformer_attention_adapter",
    "longformer_source_identity",
]
