"""Runtime contracts; executable plans remain owned by the compiler anchors."""

from .registry import Backend, BackendRegistry, BackendResult

__all__ = ["Backend", "BackendRegistry", "BackendResult"]
