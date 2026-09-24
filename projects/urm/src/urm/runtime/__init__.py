"""Runtime: plan binding and state sessions.

Exposes the plan-authority binder (:class:`BoundGraphPlan`) and the persistent
state sessions for compiled plans. There is no backend registry here: selection
is a compiler decision (:mod:`urm.compiler.select`), and execution binds the
verified plan.
"""

from __future__ import annotations

__all__ = ["BoundGraphPlan", "PlanBindingError"]


def __getattr__(name: str):
    if name in {"BoundGraphPlan", "PlanBindingError"}:
        from .bind import BoundGraphPlan, PlanBindingError

        return {"BoundGraphPlan": BoundGraphPlan, "PlanBindingError": PlanBindingError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
