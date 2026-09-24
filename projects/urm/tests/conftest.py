"""Shared test fixtures and environment setup.

The benchmark harnesses spawn ``python benchmarks/...`` subprocesses that import
``urm`` (from ``src/``) and project-level packages such as ``tests.fixtures`` and
``benchmarks.comparators`` (from the project root). When the package is not
installed (e.g. a bare source checkout), those subprocesses need both ``src/``
and the project root on ``PYTHONPATH``; the pytest process itself gets ``src/``
from ``pyproject.toml``'s ``pythonpath`` but does not export it to children.
Setting both here keeps the subprocess-based tests hermetic regardless of how
the suite was invoked.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
_SRC = str(Path(__file__).resolve().parents[1] / "src")
_add = [_SRC, _ROOT]
_existing = os.environ.get("PYTHONPATH")
_parts = _existing.split(os.pathsep) if _existing else []
for _p in _add:
    if _p not in _parts:
        _parts.insert(0, _p)
os.environ["PYTHONPATH"] = os.pathsep.join(_parts)
