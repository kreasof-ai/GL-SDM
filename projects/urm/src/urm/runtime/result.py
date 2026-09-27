"""Runtime result type for the routed-reduction execution contract.

This is the result a bound routed-reduction plan step returns. It is the
runtime's record of an executed operation, distinct from the compile-time
signature in :mod:`urm.ir.types`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoutedReductionResult:
    output: object
    indices: object
    weights: object
    metadata: dict[str, object]


__all__ = ["RoutedReductionResult"]
