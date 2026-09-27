"""Versioned JSON recipe loaders.

The recipe catalog is declarative JSON only (``recipes/kernels/*.json``,
``recipes/architectures/*.json``); this package holds the loading and
validation machinery and nothing else. Compilation entry points live in
:mod:`urm.compiler.pipeline` (:func:`compile_graph`).
"""

from . import recipes

__all__ = ["recipes"]
