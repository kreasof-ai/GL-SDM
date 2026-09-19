"""Legacy model-named aliases; use urm.experimental.sparse_delta.

This compatibility module does not define model semantics or certify a backend.
"""

from urm.experimental.sparse_delta import (
    SparseDeltaFunction as DualFormSDMFunction,
)
from urm.experimental.sparse_delta import (
    chunked_sparse_delta as chunked_dual_form_sdm,
)
from urm.experimental.sparse_delta import (
    sparse_delta as dual_form_sdm,
)

__all__ = ["DualFormSDMFunction", "chunked_dual_form_sdm", "dual_form_sdm"]
