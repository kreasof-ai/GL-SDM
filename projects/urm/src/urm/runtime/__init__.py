"""Runtime contracts and executable bindings for compiler-produced plans.

Exposes the semantic-family backend registry (used by reference/native backends
for ``MixerSpec``-level selection) and the generic compiled-plan state sessions.
"""

from .registry import Backend, BackendRegistry, BackendResult

__all__ = ["Backend", "BackendRegistry", "BackendResult"]
