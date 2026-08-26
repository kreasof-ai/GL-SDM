"""URM compiler: semantic-to-execution compilation for routed sequence models.

Layering (docs/compiler-charter.md):

    architecture/NAS specification
      -> semantic routing and state IR        (semantic.py)
      -> verified algebraic reparameterization (rewrite.py)
      -> placement, sharding, communication    (placement.py, planner.py)
      -> trusted execution anchors + visitors  (execution.py)
      -> FA / FLA / grouped GEMM / scan / SDM / collectives / generated kernels

The compiler package is typed and declarative. It never accepts arbitrary
tensor callables into the core IR; behavior enters only through registered,
typed rules and anchors.
"""

from urm.compiler.cost import CostEstimate, DeviceLimits
from urm.compiler.diagnostics import CompilerError, Diagnostic, DiagnosticCode

__all__ = [
    "CompilerError",
    "CostEstimate",
    "DeviceLimits",
    "Diagnostic",
    "DiagnosticCode",
    "cost",
    "diagnostics",
    "effects",
    "execution",
    "locality",
    "placement",
    "planner",
    "rewrite",
    "semantic",
]
