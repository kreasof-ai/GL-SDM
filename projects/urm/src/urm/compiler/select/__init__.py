"""Backend/candidate selection.

Owns capability facts, anchor selection, decline/fallback policy, and the
candidate model. Backends publish immutable capability facts; selection is a
compiler decision. See :mod:`urm.compiler.select.anchors` and
:mod:`urm.compiler.select.registry`.
"""
