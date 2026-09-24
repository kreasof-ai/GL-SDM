"""Backend-independent model specification frontend.

Exposes the declarative spec, the named-recipe catalog, and the versioned JSON
recipe loader for kernel fragments and complete model graphs. The public compile
API is :func:`compile`, delegating to the compiler pipeline.
"""

from .spec import MixerSpec

__all__ = ["MixerSpec", "compile"]


def __getattr__(name: str):
    if name == "compile":
        from urm.compiler.pipeline import compile_mixer

        return compile_mixer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
