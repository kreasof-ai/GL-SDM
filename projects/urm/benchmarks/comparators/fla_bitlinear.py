"""Pinned fla BitLinear/FusedBitLinear adapter.

Loads ``LayerNormLinearQuantFn`` (the fused Triton kernel behind
``FusedBitLinear``) from the pinned fla checkout
(fla/modules/fused_bitlinear.py @ 864a87f6) — the same source the sweep used
to verify arch-014. The pinned fused kernel is the authority for the
quantization *gradient policy*; its forward values are the closed form
transcribed into ``architectures.bit_attention`` (RMSNorm → per-token 8-bit
activation quant → per-tensor 1.58-bit weight quant → linear).

The pinned module imports the wider ``fla`` package (Triton RMSNorm kernels),
so this adapter adds the pinned checkout to ``sys.path`` and imports it as a
package — GPU-only, since the pinned kernels are Triton.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "fla/modules/fused_bitlinear.py"


@lru_cache(maxsize=1)
def fla_bitlinear_source_identity() -> dict[str, str]:
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
            "fla fused_bitlinear requires the clean pinned revision "
            f"{EXPECTED_FLA_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/fla-org/flash-linear-attention",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_fused_bitlinear():
    identity = fla_bitlinear_source_identity()
    pin_root = str(PINS_DIR / "fla")
    # Another consumer may already have imported an installed `fla` distribution
    # (e.g. the K2 comparator). This adapter is the parity authority for
    # BitLinear: evict any pre-imported fla modules and re-import them from the
    # pinned checkout, with the pin path first. The identity check below proves
    # the module actually resolved to the pinned file.
    sys.modules.pop("fla.modules.fused_bitlinear", None)
    for name in [m for m in sys.modules if m == "fla" or m.startswith("fla.")]:
        sys.modules.pop(name, None)
    if sys.path[0] != pin_root:
        if pin_root in sys.path:
            sys.path.remove(pin_root)
        sys.path.insert(0, pin_root)
    import fla.modules.fused_bitlinear as pinned  # noqa: PLC0415

    if Path(pinned.__file__).resolve() != Path(identity["source_path"]):
        raise RuntimeError(
            f"fla.modules.fused_bitlinear resolved to {pinned.__file__}, "
            f"not the pinned {identity['source_path']}"
        )
    return pinned


def fla_fused_bitlinear_adapter(
    x: Any,
    norm_weight: Any,
    norm_bias: Any,
    linear_weight: Any,
    linear_bias: Any | None = None,
) -> tuple[Any, dict[str, str]]:
    """Run the pinned fused BitLinear kernel (RMS norm + quantized linear).

    ``x`` is ``[..., in_features]``; the pinned Triton kernels require CUDA.
    Returns ``(output, identity)``. The output carries the pinned custom
    autograd (STE quantization policy), so callers may backpropagate through it.
    """
    identity = fla_bitlinear_source_identity()
    pinned = _pinned_fused_bitlinear()
    if not x.is_cuda:
        raise ValueError("the pinned fla fused BitLinear kernels require CUDA tensors")
    out = pinned.layer_norm_linear_quant_fn(
        x, norm_weight, norm_bias, linear_weight, linear_bias, is_rms_norm=True
    )
    return out, identity


__all__ = [
    "EXPECTED_FLA_REVISION",
    "fla_bitlinear_source_identity",
    "fla_fused_bitlinear_adapter",
]
