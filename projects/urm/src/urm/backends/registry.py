"""Backend auto-discovery: one directory per backend, one file per family.

The dispatch table is built from the filesystem, not from a central registry
that a new backend must edit. Each backend lives in ``backends/<name>/`` and
implements each family it supports in ``backends/<name>/<family>.py``. A family
module exposes its canonical kernel functions (``forward``/``backward`` and the
family's performance-axis form) and a ``PROVIDERS`` tuple of
:class:`~urm.backends.contract.Provider` instances; the registry imports every
``<name>/<family>.py`` and indexes the declared providers by anchor name.

Adding a backend (e.g. ``tilelang``) is one new directory with ``k1.py`` /
``k2.py`` / ``k3.py`` written to the same canonical signatures — no edit to any
shared file. A backend that does not implement a family simply omits the file;
its providers decline.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from .contract import Provider

# Directories that are not live backends. ``historical`` preserves unadmitted
# variants that are never dispatched; anything else with an ``__init__.py`` is
# a live backend.
_EXCLUDED_DIRS = frozenset({"historical", "__pycache__"})


def _live_backends() -> tuple[str, ...]:
    """Every subdirectory of ``backends/`` with an ``__init__.py`` is live."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent
    return tuple(
        sorted(
            child.name
            for child in here.iterdir()
            if child.is_dir()
            and child.name not in _EXCLUDED_DIRS
            and (child / "__init__.py").exists()
        )
    )


def discover_providers() -> dict[str, Provider]:
    """Import every module in each live backend directory and index its providers.

    Every ``backends/<backend>/*.py`` module is scanned; any module exposing a
    ``PROVIDERS`` tuple contributes its providers. Adding a backend is creating
    its directory — no edit to any shared file. The mapping is anchor name →
    provider; an anchor name collision across backends is an error.
    """
    providers: dict[str, Provider] = {}
    for backend in _live_backends():
        package = f"urm.backends.{backend}"
        package_module = importlib.import_module(package)
        for info in pkgutil.iter_modules(package_module.__path__):
            module = importlib.import_module(f"{package}.{info.name}")
            for provider in getattr(module, "PROVIDERS", ()):
                if provider.name in providers:
                    raise ValueError(
                        f"anchor name {provider.name!r} is provided by two backends"
                    )
                providers[provider.name] = provider
    return providers


__all__ = ["discover_providers"]
