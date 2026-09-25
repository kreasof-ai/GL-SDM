"""Pinned espnet Conformer rel-pos attention adapter.

Extracts ``RelPositionMultiHeadedAttention`` from the pinned espnet checkout
(espnet2/asr_transducer/encoder/modules/attention.py @ 2950325e) by AST — the
same source the sweep used to verify arch-075 — without importing the wider
espnet package. The class is the authority for the relative-position score
construction and the noncausal masked-softmax mixer.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_CONFORMER_REVISION = "2950325ea62c8052f448aaf11affdabe169ec8ab"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "espnet2/asr_transducer/encoder/modules/attention.py"


@lru_cache(maxsize=1)
def conformer_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "conformer" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned espnet checkout missing at {source}; run "
            "benchmarks/provision_comparators.py conformer"
        )
    repository = PINS_DIR / "conformer"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_CONFORMER_REVISION or dirty:
        raise RuntimeError(
            "conformer requires the clean pinned revision "
            f"{EXPECTED_CONFORMER_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/espnet/espnet",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_rel_pos_attention_class():
    import math
    from typing import Optional, Tuple

    import torch

    identity = conformer_source_identity()
    source = Path(identity["source_path"]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "RelPositionMultiHeadedAttention"
    )
    namespace: dict[str, Any] = {
        "math": math, "torch": torch, "Optional": Optional, "Tuple": Tuple,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), identity["source_path"], "exec"),
        namespace,
    )
    return namespace["RelPositionMultiHeadedAttention"]


def conformer_rel_pos_attention_adapter(
    query: Any,
    key: Any,
    value: Any,
    pos_enc: Any,
    mask: Any,
    *,
    num_heads: int,
    embed_size: int,
    seed: int,
) -> tuple[Any, Any, dict[str, str]]:
    """Run the pinned module and return (output, module, identity).

    A fresh pinned module is constructed (seeded) per call so tests can copy
    its parameters into the URM module under verification. Inputs follow the
    pinned forward signature: q/k/v ``[B, T, E]``, pos_enc ``[B, 2T-1, E]``,
    mask ``[B, T2]`` (True = masked out).
    """
    import torch

    identity = conformer_source_identity()
    cls = _pinned_rel_pos_attention_class()
    torch.manual_seed(seed)
    module = cls(num_heads=num_heads, embed_size=embed_size, dropout_rate=0.0)
    module.eval()
    out = module(query, key, value, pos_enc, mask)
    return out, module, identity


__all__ = [
    "EXPECTED_CONFORMER_REVISION",
    "conformer_rel_pos_attention_adapter",
    "conformer_source_identity",
]
