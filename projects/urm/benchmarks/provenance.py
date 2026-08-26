"""Shared provenance capture for every committed benchmark artifact.

Artifacts must record: the exact code revision, whether the tree was dirty,
the full benchmark command, a hash of the benchmark configuration, the
installed solver version, and a hash of the constraint-model summary. A
dirty tree is *recorded*, never hidden.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tree_is_dirty() -> bool:
    """True when tracked CODE differs from HEAD.

    Changes under any ``results/`` directory are ignored on purpose: those
    are the artifact outputs this protocol produces and commits afterwards;
    they cannot change the identity of the code being benchmarked. Untracked
    files are likewise ignored.
    """
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return True  # cannot verify cleanliness; assume dirt, record it
    code_lines = [
        line
        for line in status.splitlines()
        if "/results/" not in line and not line.endswith("/results")
    ]
    return bool(code_lines)


def config_hash(configuration: object) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def solver_version() -> str | None:
    try:
        import z3

        return z3.get_version_string()
    except ImportError:
        return None


def pytorch_version() -> str | None:
    try:
        import torch

        return str(torch.__version__)
    except ImportError:
        return None


def triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except ImportError:
        return None


def cuda_version() -> str | None:
    try:
        import torch

        return str(torch.version.cuda) if torch.cuda.is_available() else None
    except ImportError:
        return None


def driver_version() -> str | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = res.stdout.strip().splitlines()
        return lines[0] if lines else None
    except Exception:  # noqa: BLE001
        return None


def gpu_name() -> str | None:
    try:
        import torch

        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        return None


def gpu_uuid() -> str | None:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = res.stdout.strip().splitlines()
        return lines[0] if lines else None
    except Exception:  # noqa: BLE001
        return None


def provenance(
    command: str, configuration: object, *, include_gpu: bool = True
) -> dict[str, object]:
    return {
        "git_revision": git_revision(),
        "dirty_tree": tree_is_dirty(),
        "benchmark_command": command,
        "config_hash": config_hash(configuration),
        "solver_version": solver_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": pytorch_version() if include_gpu else None,
        "triton": triton_version() if include_gpu else None,
        "cuda": cuda_version() if include_gpu else None,
        "driver": driver_version() if include_gpu else None,
        "gpu": gpu_name() if include_gpu else None,
        "gpu_uuid": gpu_uuid() if include_gpu else None,
    }


def write_artifact(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "config_hash",
    "git_revision",
    "provenance",
    "solver_version",
    "tree_is_dirty",
    "utc_now",
    "write_artifact",
]
