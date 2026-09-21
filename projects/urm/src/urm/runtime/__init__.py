"""Runtime contracts and executable bindings for compiler-produced plans."""

from .registry import Backend, BackendRegistry, BackendResult

__all__ = ["Backend", "BackendRegistry", "BackendResult"]
