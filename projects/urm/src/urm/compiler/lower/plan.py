"""The authoritative executable plan produced by the compiler's lowering.

A lowered plan records which backend capability was selected, which were
attempted, and whether a visible fallback occurred. The runtime binds and
invokes this plan; it does not redispatch from a recipe name.
"""

from __future__ import annotations

from dataclasses import dataclass

from urm.backends.interface import BackendRequest


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    requested_backend: str | None
    selected_backend: str
    request: BackendRequest
    attempted_backends: tuple[str, ...]
    fallback_used: bool


__all__ = ["ExecutionPlan"]
