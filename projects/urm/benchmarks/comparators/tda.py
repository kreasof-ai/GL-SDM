"""Adapter for the pinned Threshold Differential Attention Triton source."""

from __future__ import annotations

import hashlib
import importlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

EXPECTED_TDA_REVISION = "cd8ddc9d5b43a1dcf86f9cfda302edb5cc108da2"


@lru_cache(maxsize=1)
def tda_source_identity() -> dict[str, str]:
    module = importlib.import_module("triton_threshold_attention")
    source = Path(module.__file__).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not identify TDA source checkout for {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if revision != EXPECTED_TDA_REVISION or dirty:
        raise RuntimeError(
            "TDA requires the clean pinned source revision "
            f"{EXPECTED_TDA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    return {
        "repository": "https://github.com/snap-research/TDA",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def tda_attention_adapter(
    query_a: Any,
    query_b: Any,
    key_a: Any,
    key_b: Any,
    value: Any,
    beta: Any,
    lambda_weight: Any,
) -> tuple[Any, dict[str, str]]:
    """Run the source's two threshold-rectified Triton reductions.

    URM uses BTHD operands; TDA's entry point takes BHTD. The source currently
    implements the normalized, power-two differential operator in float32.
    """
    identity = tda_source_identity()
    tensors = (query_a, query_b, key_a, key_b, value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("TDA Q/K/V operands must use rank-four BTHD layout")
    if any(tensor.shape != query_a.shape for tensor in tensors[1:]):
        raise ValueError("pinned TDA requires Q/K/V operands with matching shapes")
    if query_a.dtype is not torch.float32:
        raise ValueError("the pinned TDA Triton adapter is qualified for float32")
    if any(tensor.device != query_a.device for tensor in (*tensors, beta, lambda_weight)):
        raise ValueError("TDA operands must share a device")
    if beta.numel() != 1 or lambda_weight.numel() != 1:
        raise ValueError("the pinned TDA adapter requires scalar beta and lambda")
    source = importlib.import_module("triton_threshold_attention")
    output = source.differential_threshold_rela_triton(
        *(tensor.transpose(1, 2).contiguous() for tensor in (query_a, query_b, key_a, key_b, value)),
        beta,
        lambda_weight,
        relu_power=2.0,
        normalize=True,
    ).transpose(1, 2)
    return output, identity


__all__ = ["EXPECTED_TDA_REVISION", "tda_attention_adapter", "tda_source_identity"]
