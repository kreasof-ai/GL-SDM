"""Lowering to the authoritative serialized plan.

Owns the physical PlanStep and the executable plan serialization that the
runtime binds and invokes. See :mod:`urm.compiler.lower.plan`.
"""

from urm.compiler.lower.plan import ExecutionPlan

__all__ = ["ExecutionPlan"]
