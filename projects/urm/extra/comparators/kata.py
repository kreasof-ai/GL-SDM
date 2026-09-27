"""Pinned Kernelized Linear Attention (KATA) Triton adapter."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

EXPECTED_KATA_REVISION = "f93fe75750be6400a0068749794985d70666926d"
EXPECTED_KATA_HASHES = {
    "parallel_kata_attn.py": "d395a2c6ca8f9c882f68447aa1202695135278040ddd2764d11487d84fd8a7d6",
    "parallel_kata.py": "05cdc8f16c87bef677e84ad7a1ce79446cbed82cdc7786bfe5724d4e48edd484",
}


@lru_cache(maxsize=1)
def kata_source_identity() -> dict[str, Any]:
    module = importlib.import_module("kata.parallel_kata_attn")
    source = Path(module.__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify KATA source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    kernel = source.parent / "kernels" / "parallel_kata.py"
    hashes = {
        source.name: hashlib.sha256(source.read_bytes()).hexdigest(),
        kernel.name: hashlib.sha256(kernel.read_bytes()).hexdigest(),
    }
    if (
        revision != EXPECTED_KATA_REVISION
        or dirty
        or hashes != EXPECTED_KATA_HASHES
    ):
        raise RuntimeError(
            "KATA requires the clean pinned source revision and exact kernel files; "
            f"got revision={revision}, dirty={bool(dirty)}, hashes={hashes}"
        )
    return {
        "repository": "https://github.com/ayghri/KATA",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": hashes,
    }


def kata_attention_adapter(
    query: Any, key: Any, value: Any, num_groups: int
) -> tuple[Any, dict[str, Any]]:
    """Run source's causal SPD-normalized K1 attention for MHA inputs."""
    identity = kata_source_identity()
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("KATA Q/K/V operands must use BTHD layout")
    if query.shape != key.shape or query.shape[:3] != value.shape[:3]:
        raise ValueError("the KATA profile requires matching MHA Q/K/V layouts")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("pinned KATA Triton requires float16 or bfloat16")
    if any(tensor.dtype != query.dtype for tensor in (key, value)):
        raise ValueError("KATA Q/K/V operands must share a dtype")
    if any(tensor.device != query.device for tensor in (key, value)):
        raise ValueError("KATA Q/K/V operands must share a device")
    if not isinstance(num_groups, int) or num_groups not in (1, 2, 4):
        raise ValueError("KATA Triton supports num_groups in {1, 2, 4}")
    if query.shape[-1] % num_groups:
        raise ValueError("KATA head dimension must be divisible by num_groups")
    source = importlib.import_module("kata.parallel_kata_attn")
    output = source.parallel_kata_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        num_groups=num_groups,
        scale=None,
        use_triton_bwd=True,
    )
    return output, identity


__all__ = [
    "EXPECTED_KATA_HASHES",
    "EXPECTED_KATA_REVISION",
    "kata_attention_adapter",
    "kata_source_identity",
]
