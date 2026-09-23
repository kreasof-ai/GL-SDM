"""Shared test fixtures and environment setup.

The benchmark harnesses spawn ``python benchmarks/...`` subprocesses that import
``urm``. When the package is not installed (e.g. a bare source checkout), those
subprocesses need ``src/`` on ``PYTHONPATH``; the pytest process itself gets it
from ``pyproject.toml``'s ``pythonpath`` but does not export it to children.
Setting it here keeps the subprocess-based tests hermetic regardless of how the
suite was invoked.
"""

from __future__ import annotations

import os
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_existing = os.environ.get("PYTHONPATH")
if _existing:
    if _SRC not in _existing.split(os.pathsep):
        os.environ["PYTHONPATH"] = _SRC + os.pathsep + _existing
else:
    os.environ["PYTHONPATH"] = _SRC
