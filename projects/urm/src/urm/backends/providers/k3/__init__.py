"""K3 sparse-delta state and route providers: Torch reference, native Triton,
and the independent NumPy oracle, all behind the one Provider contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import ProviderFamily, ProviderRequest  # noqa: F401  (re-exported)
from ....ir.program import SparseRouteSelectionSpec, SparseStateMixerSpec


def _torch() -> Any:
    import torch

    return torch


class _Base:
    name: str = ""
    family: str = ""
    tier: str = "reference"

    def decline(self, request: ProviderRequest) -> str | None:
        return None


class _K1Provider(_Base):
    family = ProviderFamily.K1

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        return None


class _NumpyBase:
    tier = "reference"
    family = ""

    def decline(self, request: ProviderRequest) -> str | None:
        return None


def _k3_runtime_spec(spec: SparseStateMixerSpec, operands: dict[str, Any]) -> SparseStateMixerSpec:
    """Re-materialize the recipe-time batch dims from the operand shapes.

    The recipe declares the equation (slots, value_dim, widths, read timing);
    ``parallel``/``sequence`` are runtime batch dims carried as 1 in the recipe
    and recovered from the actual operands here.
    """
    from dataclasses import replace as _replace

    values = operands.get("values")
    parallel, sequence = None, None
    if values is not None:
        parallel, sequence = int(values.shape[0]), int(values.shape[1])
    elif operands.get("read_addresses") is not None:
        parallel = int(operands["read_addresses"].shape[0])
        sequence = int(operands["read_addresses"].shape[1])
    return _replace(
        spec,
        parallel=parallel or spec.parallel,
        sequence=sequence or spec.sequence,
    )


class K3TorchReferenceProvider(_Base):
    name = "urm.unified.k3.sparse_delta_reference.v1"
    family = ProviderFamily.K3
    tier = "reference"

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, SparseStateMixerSpec):
            return "K3 providers require a closed SparseStateMixerSpec"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .torch import sparse_delta_state

        spec = _k3_runtime_spec(request.descriptor, operands)
        outputs, state = sparse_delta_state(
            operands["memory"],
            operands["read_addresses"],
            operands["read_weights"],
            write_addresses=operands.get("write_addresses"),
            write_weights=operands.get("write_weights"),
            values=operands.get("values"),
            beta=operands.get("beta"),
            log_decay=operands.get("log_decay"),
            spec=spec,
        )
        return {"readings": outputs, "updated_memory": state}


class K3NativeTritonProvider(K3TorchReferenceProvider):
    name = "urm_native_sparse_state_mixer_v0"
    tier = "native"

    def decline(self, request: ProviderRequest) -> str | None:
        base = super().decline(request)
        if base is not None:
            return base
        torch = _torch()
        if not torch.cuda.is_available():
            return "native K3 requires CUDA"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .triton_state_launcher import (
            CertifiedSparseStateRoutes,
            SparseState,
            TritonSparseStateMixerBackend,
        )

        spec = _k3_runtime_spec(request.descriptor, operands)
        # The routes are produced by this graph's own route nodes (trusted), so
        # bind through the structural certification bridge (fullgraph-safe).
        routes = CertifiedSparseStateRoutes.certify_trusted(
            spec,
            operands["read_addresses"],
            operands["read_weights"],
            write_indices=operands.get("write_addresses"),
            write_weights=operands.get("write_weights"),
        )
        backend = TritonSparseStateMixerBackend(spec)
        prepared = backend._prepare_generated_routes(
            routes,
            values=operands.get("values"),
            beta=operands.get("beta"),
            log_decay=operands.get("log_decay"),
        )
        state = SparseState(memory=operands["memory"], sequence_length=0)
        readings, new_state = backend.execute(state, prepared)
        return {"readings": readings, "updated_memory": new_state.memory}


class K3RouteNativeTritonProvider(_Base):
    name = "urm_native_sparse_route_selection_v0"
    family = ProviderFamily.K3_ROUTE
    tier = "native"

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, SparseRouteSelectionSpec):
            return "K3 route providers require a closed SparseRouteSelectionSpec"
        torch = _torch()
        if not torch.cuda.is_available():
            return "native K3 route selection requires CUDA"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .triton_route import sparse_route_selection

        spec = request.descriptor
        addresses, weights = sparse_route_selection(
            operands["scores"], spec.source_extent, spec.route_width
        )
        return {"addresses": addresses, "weights": weights}


class K3NumpyProvider(_NumpyBase):
    name = "urm.reference.numpy.k3.sparse_delta_state.v1"
    family = ProviderFamily.K3

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, SparseStateMixerSpec):
            return "K3 NumPy provider requires a closed SparseStateMixerSpec"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        """K3 oracle in the unified address-index operand form.

        The torch/native tiers take read/write address indices; the NumPy
        oracle's dense slot-vector form is adapted to that same form here, so
        all three tiers take identical operands.
        """
        from .numpy import sparse_delta_state

        spec = _k3_runtime_spec(request.descriptor, operands)
        outs, finals = sparse_delta_state(
            operands["memory"],
            operands["read_addresses"],
            operands["read_weights"],
            write_addresses=operands.get("write_addresses"),
            write_weights=operands.get("write_weights"),
            values=operands.get("values"),
            beta=operands.get("beta"),
            log_decay=operands.get("log_decay"),
            spec=spec,
        )
        return {"readings": outs, "updated_memory": finals}

__all__ = [
    "K3NativeTritonProvider",
    "K3NumpyProvider",
    "K3RouteNativeTritonProvider",
    "K3TorchReferenceProvider",
]
