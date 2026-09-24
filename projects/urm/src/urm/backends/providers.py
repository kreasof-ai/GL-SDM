"""Provider implementations: the uniform contract bound to each backend tier.

Every provider here adapts one tier's kernel (reference Torch, native Triton,
or the trusted SDPA library call) to the single :class:`Provider` surface.
Structured decline happens in :meth:`decline` before any tensor work; the
runtime dispatch table maps a serialized anchor name to exactly one provider
and calls ``decline`` then ``execute``. No equation logic lives here beyond
invoking the tier's own equation module.
"""

from __future__ import annotations

from typing import Any

from urm.ir.program import (
    K1Descriptor,
    LinearDeltaSpec,
    SparseRouteSelectionSpec,
    SparseStateMixerSpec,
)
from .provider import ProviderFamily, ProviderRequest


def _torch() -> Any:
    import torch

    return torch


class _Base:
    name: str = ""
    family: str = ""
    tier: str = "reference"

    def decline(self, request: ProviderRequest) -> str | None:
        return None


# ---------------------------------------------------------------------------
# K1: softmax attention
# ---------------------------------------------------------------------------


class _K1Provider(_Base):
    family = ProviderFamily.K1

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 providers require a closed K1Descriptor"
        if request.accumulation_dtype != "float32":
            return "K1 v1 requires float32 accumulation"
        return None


class K1TorchReferenceProvider(_K1Provider):
    name = "urm.unified.k1.softmax_reference.v1"
    tier = "reference"

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .reference.torch.k1_attention import torch_k1_softmax_attention

        out = torch_k1_softmax_attention(
            operands["query"],
            operands["key"],
            operands["value"],
            descriptor=request.descriptor,
            score_bias=operands.get("score_bias"),
            attention_mask=operands.get("attention_mask"),
            scale=(
                None
                if operands.get("scale") is None
                else float(operands["scale"])
            ),
        )
        return {"output": out}


class K1SdpaLibraryProvider(_K1Provider):
    name = "torch.nn.functional.scaled_dot_product_attention"
    tier = "library"

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        torch = _torch()
        descriptor = request.descriptor
        scale_op = operands.get("scale")
        scale = None
        if descriptor.scale_rule.value == "explicit_operand":
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = float(operands["query"].shape[-1]) ** -0.5

        q = operands["query"].transpose(1, 2)
        k = operands["key"].transpose(1, 2)
        v = operands["value"].transpose(1, 2)

        def _head_major(mask):
            if mask.dim() == 2:
                return mask.view(1, 1, *mask.shape)
            if mask.dim() == 3:
                return mask.unsqueeze(1)
            return mask

        attn_mask = None
        attention_mask = operands.get("attention_mask")
        score_bias = operands.get("score_bias")
        if attention_mask is not None:
            attn_mask = _head_major(attention_mask)
        if score_bias is not None:
            bias = _head_major(score_bias)
            attn_mask = bias if attn_mask is None else attn_mask & bias
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=descriptor.causal and attn_mask is None,
            scale=scale,
        )
        return {"output": out.transpose(1, 2)}


class K1NativeTritonProvider(_K1Provider):
    name = "urm_native_k1_online_softmax_v1"
    tier = "native"

    def decline(self, request: ProviderRequest) -> str | None:
        base = super().decline(request)
        if base is not None:
            return base
        torch = _torch()
        if not torch.cuda.is_available():
            return "native K1 requires CUDA"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .triton.k1.online import execute_online_softmax

        descriptor = request.descriptor
        scale_op = operands.get("scale")
        if descriptor.scale_rule.value == "explicit_operand":
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = float(operands["query"].shape[-1]) ** -0.5
        out = execute_online_softmax(
            operands["query"],
            operands["key"],
            operands["value"],
            attention_mask=operands.get("attention_mask"),
            score_bias=operands.get("score_bias"),
            causal=descriptor.causal,
            scale=scale,
        )
        return {"output": out}


# ---------------------------------------------------------------------------
# K2: linear-delta state
# ---------------------------------------------------------------------------


class K2TorchReferenceProvider(_Base):
    name = "urm.unified.k2.state_reference.v1"
    family = ProviderFamily.K2
    tier = "reference"

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, LinearDeltaSpec):
            return "K2 providers require a closed LinearDeltaSpec"
        if request.accumulation_dtype != "float32":
            return "K2 v1 requires float32 accumulation"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .reference.torch.k2 import torch_linear_delta_state

        scale_op = operands.get("scale")
        result = torch_linear_delta_state(
            operands["initial_state"],
            operands["key"],
            operands["query"],
            operands["value"],
            operands["beta"],
            operands["log_decay"],
            spec=request.descriptor,
            scale=None if scale_op is None else float(scale_op),
        )
        if request.descriptor.normalized:
            out, (final_state, denominator) = result
            return {"output": out, "final_state": final_state, "final_denominator": denominator}
        out, final_state = result
        return {"output": out, "final_state": final_state}


class K2NativeTritonProvider(K2TorchReferenceProvider):
    """Native K2 anchors share the typed request/result ABI. Until a qualified
    Triton K2 schedule is admitted (Gate 2), the native tier executes the same
    reference recurrence so the public path stays fail-closed and honest about
    its tier."""

    tier = "native"


class K2NativeDiagonalProvider(K2NativeTritonProvider):
    name = "urm_native_diagonal_recurrence_v1"


class K2NativeMatrixProvider(K2NativeTritonProvider):
    name = "urm_native_matrix_state_recurrence_v1"


# ---------------------------------------------------------------------------
# K3: sparse-delta state + the pure route operation
# ---------------------------------------------------------------------------


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
        from .reference.torch.k3 import torch_sparse_state_mixer

        spec = _k3_runtime_spec(request.descriptor, operands)
        outputs, state = torch_sparse_state_mixer(
            operands["memory"],
            operands["read_addresses"],
            operands["read_weights"],
            write_indices=operands.get("write_addresses"),
            write_weights=operands.get("write_weights"),
            values=operands.get("values"),
            beta=operands.get("beta"),
            log_decay=operands.get("log_decay"),
            read_timing=spec.read_timing,
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
        from .triton.k3.state_launcher import (
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
        from .triton.k3.route import sparse_route_selection

        spec = request.descriptor
        addresses, weights = sparse_route_selection(
            operands["scores"], spec.source_extent, spec.route_width
        )
        return {"addresses": addresses, "weights": weights}


__all__ = [
    "K1NativeTritonProvider",
    "K1SdpaLibraryProvider",
    "K1TorchReferenceProvider",
    "K2NativeDiagonalProvider",
    "K2NativeMatrixProvider",
    "K2TorchReferenceProvider",
    "K3NativeTritonProvider",
    "K3RouteNativeTritonProvider",
    "K3TorchReferenceProvider",
]
