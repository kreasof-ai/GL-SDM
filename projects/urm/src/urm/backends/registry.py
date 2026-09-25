"""Backend auto-discovery: one directory per backend, one per family, one file per op.

The dispatch table is built from the filesystem, not from a central registry
that a new backend must edit. Each backend lives in ``backends/<name>/`` and
implements each family it supports in ``backends/<name>/<family>/`` — a family
directory whose ``__init__.py`` re-exports the combined ``PROVIDERS`` tuple of its
op modules (one op per file). Ordinary (non-recurrence) typed operators live in a
flat ``<name>/<op>.py`` module at the backend root. The registry imports every
``<name>/<family>/`` package and every flat ``<name>/<op>.py`` module and indexes
the declared providers by anchor name.

Adding a backend (e.g. ``tilelang``) is one new directory with the family
subdirectories it supports — no edit to any shared file. Adding an op to a family
is one new file in the family directory plus a line in its ``__init__.py``. A
backend that does not implement a family simply omits the directory; its providers
decline.
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
    """Import every family package and flat op module per backend, and index providers.

    Every ``backends/<backend>/<family>/`` package and every flat
    ``backends/<backend>/<op>.py`` module is scanned; anything exposing a
    ``PROVIDERS`` tuple contributes its providers. The walk is bounded — backend,
    then family, then the family's op modules; no arbitrary-depth recursion. Adding
    a backend is creating its directory; adding an op is adding one file to a family
    directory. The mapping is anchor name → provider; an anchor name collision
    across backends is an error.
    """
    providers: dict[str, Provider] = {}

    def _index(module) -> None:
        for provider in getattr(module, "PROVIDERS", ()):
            if provider.name in providers:
                raise ValueError(
                    f"anchor name {provider.name!r} is provided by two backends"
                )
            providers[provider.name] = provider

    for backend in _live_backends():
        package = f"urm.backends.{backend}"
        package_module = importlib.import_module(package)
        for info in pkgutil.iter_modules(package_module.__path__):
            if info.ispkg:
                # Family directory: the family's __init__ re-exports the combined
                # PROVIDERS of its op modules (one op per file, no deeper nesting).
                _index(importlib.import_module(f"{package}.{info.name}"))
            else:
                # Flat module at the backend root: an ordinary (non-recurrence)
                # typed operator or a not-yet-family-classified provider.
                _index(importlib.import_module(f"{package}.{info.name}"))
    return providers


__all__ = ["discover_providers"]
