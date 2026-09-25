"""Torch reference provider for the typed Merge ordinary operator.

The Merge op (cross-call composition, generality axis A14) computes a typed
linear combination of producer outputs: ``out = Σ_i c_i · x_i``, with optional
runtime scale operands. It is an ordinary typed operator — unfused by default,
with its own cost and effects, never a kernel branch over the mixer families.
"""

from __future__ import annotations

from typing import Any

from ...ir.program import Merge  # noqa: F401  (descriptor type)


def merge_linear_combination(
    terms: list[Any],
    coefficients: tuple[float, ...],
    scales: list[Any] | None = None,
) -> Any:
    """``out = Σ_i c_i · (s_i · x_i)`` over the producer terms.

    ``terms`` are the producer outputs; ``coefficients`` are the closed semantic
    coefficients (default 1.0); ``scales`` are optional per-term runtime scale
    operands (e.g. the Diff λ). All terms must share a shape.
    """
    torch = __import__("torch")
    if not terms:
        raise ValueError("merge requires at least one term")
    shape = terms[0].shape
    if any(t.shape != shape for t in terms):
        raise ValueError("merge terms must share a shape")
    if coefficients and len(coefficients) != len(terms):
        raise ValueError("merge coefficients must match the number of terms")
    if scales is not None and len(scales) != len(terms):
        raise ValueError("merge scale operands must match the number of terms")

    if scales is not None and len(scales) != len(terms):
        raise ValueError("merge scale operands must align one-to-one with the terms")

    out = torch.zeros_like(terms[0])
    for i, term in enumerate(terms):
        c = coefficients[i] if coefficients else 1.0
        value = term
        if scales is not None and scales[i] is not None:
            value = value * scales[i]
        out = out + c * value
    return out


class MergeTorchReferenceProvider:
    name = "torch.merge.linear_combination.v1"
    family = "merge"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, Merge):
            return "merge providers require a Merge op descriptor"
        return None

    def execute(self, request, operands):
        terms = [operands[name] for name in request.descriptor.inputs]
        # Per-term scale operands aligned to inputs; "" means unscaled.
        scale_names = request.descriptor.scale_operands or ("",) * len(terms)
        scales = [operands[s] if s else None for s in scale_names]
        out = merge_linear_combination(
            terms, request.descriptor.coefficients, scales
        )
        return {"output": out}


PROVIDERS = (MergeTorchReferenceProvider(),)

__all__ = ["MergeTorchReferenceProvider", "merge_linear_combination"]
