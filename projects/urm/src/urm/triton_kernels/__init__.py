"""Triton kernels; imported lazily so CPU-only installations remain usable."""

from __future__ import annotations

try:
    from .dual_form_sdm import (
        TritonDualFormSDMFunction,
        _triton_dual_form_bwd_kernel,
        _triton_dual_form_fwd_kernel,
        triton_dual_form_sdm,
    )
except ImportError:
    pass
