"""Native Triton provider for the typed Merge ordinary operator.

The merge op computes ``out = Σ_i c_i · (s_i · x_i)`` over producer terms. The native
kernel fuses the scale-multiply-accumulate into one pass over the flattened elements —
one program per element block, no tensor materialization of the intermediates.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from ...ir.program import Merge  # noqa: F401  (descriptor type)


@triton.jit
def _merge_kernel(
    OUT, TERMS_PTR, SCALES_PTR, COEFFS_PTR,
    n_elements,
    N_TERMS: tl.constexpr,
    HAS_SCALES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """out[i] = Σ_t coeffs[t] * (scales[t] * terms[t][i]) over the flattened elements."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in tl.static_range(N_TERMS):
        term_ptr = TERMS_PTR + t * n_elements
        x = tl.load(term_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        coeff = tl.load(COEFFS_PTR + t)
        if HAS_SCALES:
            # Per-term scalar scale (broadcast); scales are 0-dim or [1] tensors.
            s = tl.load(SCALES_PTR + t).to(tl.float32)
            x = x * s
        acc = acc + coeff * x
    tl.store(OUT + offs, acc, mask=mask)


def merge_linear_combination_native(
    terms: list[Any],
    coefficients: tuple[float, ...],
    scales: list[Any] | None = None,
) -> Any:
    """Native fused merge: one pass, no intermediate materialization."""
    if not terms:
        raise ValueError("merge requires at least one term")
    shape = terms[0].shape
    if any(t.shape != shape for t in terms):
        raise ValueError("merge terms must share a shape")
    if any(t.device.type != "cuda" for t in terms):
        raise ValueError("native merge requires CUDA tensors")

    n_terms = len(terms)
    n_elements = terms[0].numel()
    dtype = terms[0].dtype

    # Pack terms into a single contiguous buffer: [n_terms, n_elements]
    packed = torch.stack([t.reshape(-1) for t in terms]).contiguous()
    coeffs_t = torch.tensor(
        [coefficients[i] if i < len(coefficients) else 1.0 for i in range(n_terms)],
        dtype=torch.float32, device=terms[0].device,
    )
    has_scales = scales is not None and any(s is not None for s in scales)
    if has_scales:
        scale_vals = [
            float(s) if s is not None else 1.0
            for s in scales
        ]
        scales_t = torch.tensor(scale_vals, dtype=torch.float32, device=terms[0].device)
    else:
        scales_t = torch.empty(0, dtype=torch.float32, device=terms[0].device)

    out = torch.empty(n_elements, dtype=dtype, device=terms[0].device)
    BLOCK = 1024
    grid = (triton.cdiv(n_elements, BLOCK),)
    _merge_kernel[grid](
        out, packed, scales_t, coeffs_t,
        n_elements, N_TERMS=n_terms, HAS_SCALES=has_scales, BLOCK=BLOCK,
    )
    return out.reshape(shape)


class MergeNativeTritonProvider:
    name = "urm_native_merge_linear_combination_v1"
    family = "merge"
    tier = "native"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, Merge):
            return "merge providers require a Merge op descriptor"
        return None

    def execute(self, request, operands):
        terms = [operands[name] for name in request.descriptor.inputs]
        scale_names = request.descriptor.scale_operands or ("",) * len(terms)
        scales = [operands[s] if s else None for s in scale_names]
        out = merge_linear_combination_native(
            terms, request.descriptor.coefficients, scales
        )
        return {"output": out}


PROVIDERS = (MergeNativeTritonProvider(),)

__all__ = ["MergeNativeTritonProvider", "merge_linear_combination_native"]
