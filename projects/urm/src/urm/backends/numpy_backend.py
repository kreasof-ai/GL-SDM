"""Adapter exposing the NumPy oracle through the backend protocol."""

from __future__ import annotations

import numpy.typing as npt

from ..ir import MixerSpec, MutationKind, RoutingKind
from ..reference import ReferenceResult, execute


class NumpyBackend:
    name = "numpy_reference"

    def supports(self, spec: MixerSpec) -> bool:
        return (
            spec.routing is not RoutingKind.KERNELIZED_RECURRENCE
            and spec.routing is not RoutingKind.THRESHOLD
            and spec.mutation is not MutationKind.IN_PLACE_RECURRENT
            and spec.expert is None
            and spec.sparse_attention is None
        )

    def execute(
        self,
        scores: npt.ArrayLike,
        values: npt.ArrayLike,
        spec: MixerSpec,
        *,
        route_mask: npt.ArrayLike | None = None,
    ) -> ReferenceResult:
        if not self.supports(spec):
            raise ValueError(f"{self.name} does not support {spec.name}")
        return execute(scores, values, spec, route_mask=route_mask)
