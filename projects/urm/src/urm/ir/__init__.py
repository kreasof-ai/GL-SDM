"""Typed operations and contracts for the semantic program IR.

The canonical homes:

- :mod:`urm.ir.program` - the typed operation graph (``SemanticProgram`` and
  its closed op vocabulary) plus the family specs (K1 descriptor, K2
  linear-delta spec/state, K3 sparse-state contract types).
- :mod:`urm.ir.effects` - the explicit effect system.
- :mod:`urm.ir.types` - shared scalar/shape types.
"""

from . import effects, program, types

__all__ = [
    "effects",
    "program",
    "types",
]
