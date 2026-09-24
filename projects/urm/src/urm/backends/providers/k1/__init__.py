"""K1 softmax-attention providers: Torch reference, native Triton, SDPA library,
and the independent NumPy oracle, all behind the one Provider contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import ProviderFamily, ProviderRequest  # noqa: F401  (re-exported)
from ....ir.program import K1Descriptor, K1ScaleRule


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


class K1TorchReferenceProvider(_K1Provider):
    name = "urm.unified.k1.softmax_reference.v1"
    tier = "reference"

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .torch import torch_k1_softmax_attention

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
        from .triton import execute_online_softmax

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


class K1NumpyProvider(_NumpyBase):
    name = "urm.reference.numpy.k1.softmax_attention.v1"
    family = ProviderFamily.K1

    def decline(self, request: ProviderRequest) -> str | None:
        if not isinstance(request.descriptor, K1Descriptor):
            return "K1 NumPy provider requires a closed K1Descriptor"
        return None

    def execute(self, request: ProviderRequest, operands: dict[str, Any]) -> dict[str, Any]:
        from .numpy import attention

        descriptor = request.descriptor
        query, key, value = operands["query"], operands["key"], operands["value"]
        scale_op = operands.get("scale")
        if descriptor.scale_rule is K1ScaleRule.EXPLICIT_OPERAND:
            if scale_op is None:
                raise ValueError("K1 explicit_operand scale rule requires a scale value")
            scale = float(scale_op)
        else:
            scale = None  # the oracle applies key_dim**-0.5 itself
        q = np.asarray(query, dtype=np.float64)
        k = np.asarray(key, dtype=np.float64)
        v = np.asarray(value, dtype=np.float64)
        # The oracle consumes [B, H, T, D]; the role operands arrive [B, T, H, D].
        q = np.transpose(q, (0, 2, 1, 3))
        k = np.transpose(k, (0, 2, 1, 3))
        v = np.transpose(v, (0, 2, 1, 3))
        bias = operands.get("score_bias")
        mask = operands.get("attention_mask")
        out = np.stack(
            [
                attention(
                    q[b],
                    k[b],
                    v[b],
                    scale=scale,
                    causal=descriptor.causal,
                    score_bias=None if bias is None else np.asarray(bias, dtype=np.float64),
                    attention_mask=None if mask is None else np.asarray(mask),
                )
                for b in range(q.shape[0])
            ]
        )
        return {"output": np.transpose(out, (0, 2, 1, 3))}

__all__ = [
    "K1NativeTritonProvider",
    "K1NumpyProvider",
    "K1SdpaLibraryProvider",
    "K1TorchReferenceProvider",
]
