"""Capability gate and call boundary for URM's native K1 implementation."""

from __future__ import annotations

from typing import Any

from urm.backends.interface import BackendCapability, BackendRequest
from urm.ir.graph import UnifiedMixerSpec


class TritonOnlineSoftmaxBackend:
    name = "triton_online_softmax"
    capability = BackendCapability(
        operations=frozenset({"K1"}),
        semantic_contracts=frozenset({"normalized_softmax_attention_v1"}),
        devices=frozenset({"cuda"}),
        dtypes=frozenset({"float32", "float16", "bfloat16"}),
        layouts=frozenset({"BTHD"}),
        modes=frozenset({"training", "inference", "forward_only_analysis"}),
    )

    @staticmethod
    def supports_spec(spec: UnifiedMixerSpec) -> bool:
        return spec.is_normalized_softmax_attention()

    @classmethod
    def request(
        cls,
        *,
        device: str,
        dtype: str,
        layout: str,
        mode: str,
    ) -> BackendRequest:
        return BackendRequest(
            operation="K1",
            semantic_contract="normalized_softmax_attention_v1",
            device=device,
            dtype=dtype,
            layout=layout,
            mode=mode,
        )

    def execute(self, spec: UnifiedMixerSpec, **operands: Any) -> Any:
        if not self.supports_spec(spec):
            raise ValueError(
                f"{self.name} declines K1 semantics: {spec.k1_operation.value}"
            )
        query = operands.pop("query")
        key = operands.pop("key")
        value = operands.pop("value")
        attention_mask = operands.pop("attention_mask", None)
        score_bias = operands.pop("score_bias", None)
        if spec.requires_attention_mask and attention_mask is None:
            raise ValueError("this K1 operation requires a precomputed attention_mask route")
        if score_bias is not None and not spec.accepts_score_bias:
            raise ValueError("this K1 operation does not accept score_bias")
        if operands:
            raise TypeError(f"unexpected K1 operands: {', '.join(sorted(operands))}")

        from urm.backends.triton.k1.online import execute_online_softmax

        key_dim = query.shape[-1]
        return execute_online_softmax(
            query,
            key,
            value,
            attention_mask=attention_mask,
            score_bias=score_bias,
            causal=spec.causal,
            scale=spec.attention_scale or key_dim**-0.5,
        )
