"""URM compiler: semantic-to-execution compilation for routed sequence models.

Layering (docs/compiler/compiler-charter.md, docs/planning/roadmap.md):

    architecture/NAS specification
      -> semantic routing and state IR        (ir/program.py, ir/k3.py)
      -> verified algebraic reparameterization (rewrite/)
      -> rewrite/lowering candidate enumeration (pipeline.py)
      -> backend-independent constraint IR      (solve/constraints.py)
      -> optional Z3 feasibility/optimization   (solve/z3.py)
      -> independent imperative verification    (verify/)
      -> placement, sharding, communication     (placement/)
      -> trusted execution anchors + visitors   (select/anchors.py)

The compiler package is typed and declarative. It never accepts arbitrary
tensor callables into the core IR; behavior enters only through registered,
typed rules and anchors. Solver expressions never leak outside
``compiler/solve/z3.py``.

Public names are exposed lazily so that importing a leaf stage (for example
``compiler/common/diagnostics`` from the IR layer) does not eagerly pull in the
full orchestration pipeline.
"""

from __future__ import annotations

__all__ = [
    "CompilerError",
    "ConstraintModel",
    "CostEstimate",
    "DeviceLimits",
    "Diagnostic",
    "DiagnosticCode",
]


def __getattr__(name: str):
    if name in {"CompilerError", "Diagnostic", "DiagnosticCode"}:
        from urm.compiler.common import diagnostics

        return getattr(diagnostics, name)
    if name in {"CostEstimate", "DeviceLimits"}:
        from urm.compiler.cost import model

        return getattr(model, name)
    if name == "ConstraintModel":
        from urm.compiler.solve.constraints import ConstraintModel

        return ConstraintModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
