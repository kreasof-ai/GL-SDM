"""Typed operations and contracts for the semantic program IR.

The canonical homes:

- :mod:`urm.ir.program` - the typed operation graph (``SemanticProgram`` and
  its closed op vocabulary) plus the K3 sparse-state contract types.
- :mod:`urm.ir.effects` - the explicit effect system.
- :mod:`urm.ir.types` - shared scalar/shape types.
- :mod:`urm.ir.k3` - K3 launch-schedule helpers and the independent NumPy
  sparse-state reference.
"""

from . import effects, program, types

__all__ = [
    "effects",
    "k3",
    "program",
    "types",
]


def __getattr__(name: str):
    # Lazy submodule access keeps ``urm.ir`` importable without pulling the
    # NumPy-dependent K3 helpers into minimal environments.
    if name == "k3":
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
