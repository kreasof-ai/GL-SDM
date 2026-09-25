"""The single provider contract shared by every URM backend tier.

One provider implements one closed family descriptor (K1 softmax attention,
K2 linear-delta state, K3 sparse-delta state, or the pure K3 route operation)
on one tier (reference NumPy, reference Torch, or native Triton). Every
provider exposes the same two-method surface:

- :meth:`decline` — a structured refusal, or ``None`` if the provider accepts
  the whole request. Decline is always before any tensor execution.
- :meth:`execute` — run the equation for the request's role-bound operands and
  return its outputs.

The request carries the closed family descriptor, the compilation intent mode,
and the accumulation policy; operands are bound by *role* (never by global
tensor name). This is the only backend interface the runtime dispatch table
knows — there is no per-family ad-hoc entry point, and no equation logic lives
outside a provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from urm.ir.program import (
    DyadicBankedState,
    K1Descriptor,
    LinearDeltaSpec,
    Merge,
    SparseRouteSelectionSpec,
    SparseStateMixerSpec,
    TriangularSolve,
)


class ProviderFamily:
    """The closed provider families (string constants, not an enum, so the
    dispatch table keys stay stable in serialized plans). The four mixer
    families (K1/K2/K3/K3_ROUTE) carry the routed/state equations; ``MERGE`` is
    the ordinary typed operator for cross-call composition (axis A14)."""

    K1 = "k1"
    K2 = "k2"
    K3 = "k3"
    K3_ROUTE = "k3_route"
    MERGE = "merge"
    TRIANGULAR_SOLVE = "triangular_solve"
    DYADIC_BANKED_STATE = "dyadic_banked_state"


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """One immutable semantic request handed to a provider.

    ``descriptor`` is the closed family spec (K1Descriptor / LinearDeltaSpec /
    SparseStateMixerSpec / SparseRouteSelectionSpec). ``mode`` is the
    compilation intent string (``training`` / ``inference`` /
    ``forward_only_analysis``). ``accumulation_dtype`` pins the numeric policy.
    """

    family: str
    descriptor: (
        K1Descriptor
        | LinearDeltaSpec
        | SparseStateMixerSpec
        | SparseRouteSelectionSpec
        | Merge
        | TriangularSolve
        | DyadicBankedState
    )
    mode: str
    accumulation_dtype: str = "float32"


class Provider(Protocol):
    """The uniform backend surface: structured decline, then execute."""

    name: str
    family: str
    tier: str  # "reference" | "native" | "library"

    def decline(self, request: ProviderRequest) -> str | None:
        """Return a structured decline reason, or None to accept the request."""
        ...

    def execute(
        self, request: ProviderRequest, operands: dict[str, Any]
    ) -> dict[str, Any]:
        """Run the equation on role-bound operands; return named outputs."""
        ...


__all__ = [
    "Provider",
    "ProviderFamily",
    "ProviderRequest",
]
