"""Explicit backend selection with visible declines and fallback decisions."""

from __future__ import annotations

from collections.abc import Iterable

from urm.backends.interface import (
    BackendImplementation,
    BackendRequest,
    ExecutionPlan,
)


class BackendDeclined(ValueError):
    """Raised when an explicit backend cannot honor a request."""


class BackendRegistry:
    def __init__(self, backends: Iterable[BackendImplementation] = ()) -> None:
        self._backends: dict[str, BackendImplementation] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: BackendImplementation) -> None:
        if not backend.name.strip():
            raise ValueError("backend name must not be empty")
        if backend.name in self._backends:
            raise ValueError(f"backend already registered: {backend.name}")
        self._backends[backend.name] = backend

    def select(
        self,
        request: BackendRequest,
        *,
        backend: str | None = None,
        allow_fallback: bool = False,
    ) -> tuple[BackendImplementation, ExecutionPlan]:
        if backend is not None:
            implementation = self._backends.get(backend)
            if implementation is None:
                raise BackendDeclined(f"unknown backend: {backend}")
            reason = implementation.capability.decline_reason(request)
            if reason is not None:
                raise BackendDeclined(f"backend {backend} declined: {reason}")
            return implementation, ExecutionPlan(
                requested_backend=backend,
                selected_backend=backend,
                request=request,
                attempted_backends=(backend,),
                fallback_used=False,
            )

        attempted: list[str] = []
        compatible: list[BackendImplementation] = []
        for implementation in self._backends.values():
            attempted.append(implementation.name)
            if implementation.capability.supports(request):
                compatible.append(implementation)
        if not compatible:
            raise BackendDeclined(
                "no registered backend supports operation, semantics, device, "
                "dtype, layout, and mode"
            )
        if len(compatible) > 1 and not allow_fallback:
            raise BackendDeclined(
                "automatic selection is ambiguous; request a backend explicitly "
                "or enable visible fallback ordering"
            )
        selected = compatible[0]
        attempted = attempted[: attempted.index(selected.name) + 1]
        return selected, ExecutionPlan(
            requested_backend=None,
            selected_backend=selected.name,
            request=request,
            attempted_backends=tuple(attempted),
            fallback_used=len(attempted) > 1,
        )


__all__ = ["BackendDeclined", "BackendRegistry"]
