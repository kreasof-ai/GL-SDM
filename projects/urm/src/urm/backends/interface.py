"""Dependency-light capability contracts shared by backend packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class BackendRequest:
    operation: str
    semantic_contract: str
    device: str
    dtype: str
    layout: str
    mode: str


@dataclass(frozen=True, slots=True)
class BackendCapability:
    operations: frozenset[str]
    semantic_contracts: frozenset[str]
    devices: frozenset[str]
    dtypes: frozenset[str]
    layouts: frozenset[str]
    modes: frozenset[str]

    def decline_reason(self, request: BackendRequest) -> str | None:
        for field, supported, requested in (
            ("operation", self.operations, request.operation),
            ("semantics", self.semantic_contracts, request.semantic_contract),
            ("device", self.devices, request.device),
            ("dtype", self.dtypes, request.dtype),
            ("layout", self.layouts, request.layout),
            ("execution mode", self.modes, request.mode),
        ):
            if requested not in supported:
                return f"unsupported {field}: {requested}"
        return None

    def supports(self, request: BackendRequest) -> bool:
        return self.decline_reason(request) is None


class BackendImplementation(Protocol):
    name: str
    capability: BackendCapability

    def execute(self, request: BackendRequest, **operands: Any) -> Any: ...


__all__ = [
    "BackendCapability",
    "BackendImplementation",
    "BackendRequest",
]
