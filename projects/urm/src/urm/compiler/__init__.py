"""URM compiler: semantic-to-execution compilation for routed sequence models.

Layering (docs/compiler-charter.md, docs/kernel-generation.md):

    architecture/NAS specification
      -> semantic routing and state IR        (semantic.py)
      -> verified algebraic reparameterization (rewrite.py)
      -> rewrite/lowering candidate enumeration (planner.py)
      -> backend-independent constraint IR      (constraints.py)
      -> optional Z3 feasibility/optimization   (solver.py)
      -> independent imperative verification    (verification.py)
      -> placement, sharding, communication     (placement.py, planner.py,
                                                 route_protocols.py)
      -> trusted execution anchors + visitors   (execution.py)
      -> FA / FLA / grouped GEMM / scan / SDM / collectives / generated kernels

The compiler package is typed and declarative. It never accepts arbitrary
tensor callables into the core IR; behavior enters only through registered,
typed rules and anchors. Solver expressions never leak outside
``compiler/solver.py``.
"""

from urm.compiler.constraints import ConstraintModel
from urm.compiler.cost import CostEstimate, DeviceLimits
from urm.compiler.diagnostics import CompilerError, Diagnostic, DiagnosticCode

__all__ = [
    "CompilerError",
    "ConstraintModel",
    "CostEstimate",
    "DeviceLimits",
    "Diagnostic",
    "DiagnosticCode",
    "constraints",
    "cost",
    "diagnostics",
    "effects",
    "execution",
    "kernel_plan",
    "locality",
    "placement",
    "planner",
    "rewrite",
    "schedule_space",
    "semantic",
    "solver",
    "verification",
]
