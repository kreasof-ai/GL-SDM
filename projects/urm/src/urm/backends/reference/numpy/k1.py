"""Independent NumPy K1 reference equations (routed reduction and ordered writes).

This module re-exports the independent K1 equations. The legacy
``MixerSpec``-dispatched ``NumpyBackend`` adapter was removed in the cutover:
the reference backends are consumed as typed equations through the compiler and
plan binder, not through a spec-dispatch registry.
"""

from __future__ import annotations

from .k1_routed import ReferenceResult, execute, merge_writes

__all__ = ["ReferenceResult", "execute", "merge_writes"]
