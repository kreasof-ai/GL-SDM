"""Backend protocol and explicit registry for URM lowerings."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .ir import MixerSpec


class BackendResult(Protocol):
    """Structural result contract; tensor types remain backend-specific."""

    output: object
    indices: object
    weights: object


class Backend(Protocol):
    name: str

    def supports(self, spec: MixerSpec) -> bool: ...

    def execute(
        self,
        scores: object,
        values: object,
        spec: MixerSpec,
        *,
        route_mask: object | None = None,
    ) -> BackendResult: ...


class BackendRegistry:
    """Small registry with no silent fallback between semantic families."""

    def __init__(self, backends: Iterable[Backend] = ()) -> None:
        self._backends: dict[str, Backend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: Backend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"backend already registered: {backend.name}")
        self._backends[backend.name] = backend

    def get(self, name: str, spec: MixerSpec) -> Backend:
        try:
            backend = self._backends[name]
        except KeyError as error:
            raise KeyError(f"unknown backend: {name}") from error
        if not backend.supports(spec):
            raise ValueError(f"backend {name} does not support {spec.name}")
        return backend

    def compatible(self, spec: MixerSpec) -> tuple[str, ...]:
        return tuple(
            name for name, backend in self._backends.items() if backend.supports(spec)
        )
