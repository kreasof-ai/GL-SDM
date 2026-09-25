"""Pinned Hopfield layers adapter.

Loads the pinned ``Hopfield`` association module from the hopfield-layers
checkout (hflayers/__init__.py @ f56f929c) — the same source the sweep used to
verify arch-078. Instantiated with pattern norms, projections and output
projection disabled, it is the pure iterated association: per-head scaled
softmax association with the query-refresh loop and per-head stopping.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPECTED_HOPFIELD_REVISION = "f56f929c95b77a070ae675ea4f56b6d54d36e730"
PINS_DIR = Path("/tmp/urm-comparator-pins")
SOURCE_RELATIVE = "hflayers/__init__.py"


@lru_cache(maxsize=1)
def hopfield_source_identity() -> dict[str, str]:
    source = (PINS_DIR / "hopfield" / SOURCE_RELATIVE).resolve()
    if not source.exists():
        raise RuntimeError(
            f"pinned hopfield checkout missing at {source}; run "
            "benchmarks/provision_comparators.py hopfield"
        )
    repository = PINS_DIR / "hopfield"
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if revision != EXPECTED_HOPFIELD_REVISION or dirty:
        raise RuntimeError(
            "hopfield requires the clean pinned revision "
            f"{EXPECTED_HOPFIELD_REVISION}; got {revision} with dirty={bool(dirty)}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "repository": "https://github.com/ml-jku/hopfield-layers",
        "revision": revision,
        "source_path": str(source),
        "source_sha256": digest,
    }


@lru_cache(maxsize=1)
def _pinned_hopfield_module():
    identity = hopfield_source_identity()
    package_init = Path(identity["source_path"])
    # hflayers/__init__.py does `from .functional import ...`; load the pinned
    # functional first and register a synthetic pinned package.
    functional = package_init.with_name("functional.py")
    spec_f = importlib.util.spec_from_file_location("hflayers_pinned.functional", functional)
    module_f = importlib.util.module_from_spec(spec_f)
    sys.modules["hflayers_pinned.functional"] = module_f
    spec_f.loader.exec_module(module_f)

    spec = importlib.util.spec_from_file_location(
        "hflayers_pinned", package_init, submodule_search_locations=[str(package_init.parent)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["hflayers_pinned"] = module
    spec.loader.exec_module(module)
    return module


def hopfield_association_adapter(
    state_patterns: Any,
    stored_patterns: Any,
    *,
    num_heads: int,
    scaling: Any,
    update_steps_max: int = 0,
    update_steps_eps: float = 1e-4,
) -> tuple[Any, dict[str, str]]:
    """Run the pinned iterated association and return (readout, identity).

    ``state_patterns`` is ``[B, T, D_total]`` and ``stored_patterns`` is
    ``[B, S, D_total]`` with ``D_total = num_heads * head_dim``; the pinned
    module is configured with no norms, no projections and no output
    projection, so the returned tensor is the raw association readout ξ·Y.
    ``scaling`` is the per-head tensor form (the pinned learned-scaling path).
    """
    identity = hopfield_source_identity()
    pinned = _pinned_hopfield_module()
    d_total = state_patterns.shape[-1]
    head_dim = d_total // num_heads
    module = pinned.Hopfield(
        input_size=d_total,
        hidden_size=head_dim,
        num_heads=num_heads,
        scaling=scaling,
        update_steps_max=update_steps_max,
        update_steps_eps=update_steps_eps,
        normalize_stored_pattern=False,
        normalize_stored_pattern_affine=False,
        normalize_state_pattern=False,
        normalize_state_pattern_affine=False,
        normalize_pattern_projection=False,
        normalize_pattern_projection_affine=False,
        input_bias=False,
        disable_out_projection=True,
        batch_first=True,
    )
    out = module((stored_patterns, state_patterns, stored_patterns))
    return out, identity


__all__ = [
    "EXPECTED_HOPFIELD_REVISION",
    "hopfield_association_adapter",
    "hopfield_source_identity",
]
