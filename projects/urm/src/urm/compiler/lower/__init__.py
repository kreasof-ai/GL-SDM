"""Lowering to the authoritative serialized plan.

Owns the physical PlanStep and the executable plan serialization that the
runtime binds and invokes. The single provider-selection contract is the
anchor/selector story in :mod:`urm.compiler.select.anchors`; the retired
string-set ``CapabilityRegistry`` was removed in the Batch-0 unification.
"""

__all__: list[str] = []
