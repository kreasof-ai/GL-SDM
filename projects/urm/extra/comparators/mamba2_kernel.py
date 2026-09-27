"""Load the pinned SSD Triton package without its optional CUDA extension.

Only namespace package initializers are bypassed; all kernel source files are
imported unmodified from the revision-verified checkout.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import importlib
import subprocess
import sys
import types

EXPECTED_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


@lru_cache(maxsize=1)
def ssd_kernel():
    root = Path("/tmp/urm-comparator-pins/mamba")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != EXPECTED_REVISION or dirty:
        raise RuntimeError("SSD requires the clean pinned Mamba checkout")
    for package in ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.triton", "mamba_ssm.utils"):
        if package not in sys.modules:
            namespace = types.ModuleType(package)
            namespace.__path__ = [str(root.joinpath(*package.split(".")))]
            namespace.__package__ = package
            sys.modules[package] = namespace
    return importlib.import_module("mamba_ssm.ops.triton.ssd_combined").mamba_chunk_scan_combined
