"""External model module: Pattention / parameter-token attention (arch-057).

Verified composition-now row as a typed-domain composition (sweep row-057,
against megatron/model/tokenformer.py @ 4d56c73f): the mixer is the K1 reduction
over a fixed parameter-token source domain — ``scores = query @ key_paramᵀ ×
scale``; a closed score-map/normalizer algebra over the parameter tokens; then
``output = norm(scores) @ value_param``.

The score-map/normalizer algebra is the admitted K1 ``MAP_NORMALIZE`` reducer
(axis A13), a reference-tier admission: the elementwise map (``EXP`` / ``GELU`` /
``IDENTITY``), the Lp normalizer degree ``normalizer_p``, and the map↔norm order
(``normalize_before_map``) are closed descriptor fields. The three pinned variants
map onto it exactly:

- ``softmax``       → map=EXP,  p=1, order=map-then-norm   (exp, L1, ×count)
- ``gelu_l2_norm``  → map=GELU, p=2, order=map-then-norm   (gelu, L2, ×√count)
- ``l2_norm_gelu``  → map=GELU, p=2, order=norm-then-map   (L2, ×√count, then gelu)

The layer composes the typed ``weighted_reduce`` (K1) call over the parameter
domain; the parameter tokens are learned ``nn.Parameter`` tensors bound as the
K1 key/value roles, so gradients flow to them through the contraction. Parameter-
token construction/reparameterization, the MoE ``router_index`` selection and the
model replacement (Pattention replacing QKV/output projections and the MLP) are
external; "one attention contraction equals MLP/MoE" is explicitly not claimed
(sweep blocker). The native/SDPA K1 anchors decline the non-softmax reducer; only
the reference tier executes it (a native schedule is residual, pending a second
client per the two-client physical-branch rule).

The independent pinned comparator (the source equation transcription) lives in
the parity-gate test (``tests/test_architectures_pattention.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


# Closed mapping from the pinned normalize_type to the K1 MAP_NORMALIZE fields.
_REDUCER_FIELDS = {
    # score_map, normalizer_p, normalize_before_map
    "softmax": ("exp", 1.0, False),        # exp, then L1 × count
    "gelu_l2_norm": ("gelu", 2.0, False),  # gelu, then L2 × √count
    "l2_norm_gelu": ("gelu", 2.0, True),   # L2 × √count, then gelu
}


class PattentionLayer(torch.nn.Module):
    """One Pattention layer: parameter tokens + a typed K1 MAP_NORMALIZE reduce.

    The query is the input; key/value are learned parameter tokens
    ``[param_token_num, key/value_dim]``. The score stage and the output
    contraction are the single typed K1 call (dot score + the map_normalize
    reducer) over the parameter-block source domain.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        param_token_num: int,
        *,
        norm_activation_type: str = "softmax",
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if norm_activation_type not in _REDUCER_FIELDS:
            raise ValueError(f"unsupported norm_activation_type {norm_activation_type!r}")
        self.param_token_num = param_token_num
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = norm_activation_type

        self.key_param_tokens = torch.nn.Parameter(torch.rand(param_token_num, input_channels))
        self.value_param_tokens = torch.nn.Parameter(torch.rand(param_token_num, output_channels))

        score_map, normalizer_p, nbm = _REDUCER_FIELDS[norm_activation_type]
        # The K1 graph: query [B,L,1,K], key/value parameter tokens [B,P,1,K/V].
        # Single head (the parameter domain has no head axis in the pinned source);
        # non-causal (the parameter domain is a static set, not a sequence).
        document = {
            "schema_version": 2, "name": "pattention_attn", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "L", 1, input_channels]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "P", 1, input_channels]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "P", 1, output_channels]},
                    {"name": "scale", "dtype": "float32", "shape": []},
                ],
                "nodes": [
                    {
                        "id": "attn", "op": "weighted_reduce",
                        "inputs": ["query", "key", "value"], "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "parameter_block",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": False, "head_map": "equal",
                            "scale_rule": "explicit_operand",
                            "reducer_law": "map_normalize",
                            "score_map": score_map, "normalizer_p": normalizer_p,
                            "normalize_before_map": nbm,
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "scale": "scale"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(
        self,
        inputs: torch.Tensor,
        router_index: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """``inputs`` ``[..., L, key_dim]`` → ``[..., L, value_dim]``.

        ``router_index`` selects a subset of parameter tokens (the pinned MoE
        mode); ``scale`` is the pinned ``scale_factor`` (default 1). The K1 graph
        takes [B,L,1,*], so leading batch dims are flattened.
        """
        query = inputs
        if router_index is None:
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]

        scale_factor = 1.0 if scale is None else float(scale)
        # Flatten leading batch dims into B; the K1 graph is [B, L, 1, D].
        lead = query.shape[:-2]
        L, K = query.shape[-2], query.shape[-1]
        q = query.reshape(-1, L, 1, K).float()
        # Broadcast the parameter tokens across the (flattened) batch.
        B = q.shape[0]
        P = key.shape[-2]
        k = key.reshape(1, P, 1, K).float().expand(B, P, 1, K)
        v = value.reshape(1, P, 1, self.param_value_dim).float().expand(B, P, 1, self.param_value_dim)
        out = self._plan.execute(query=q, key=k, value=v, scale=scale_factor)["output"]
        out = out.reshape(*lead, L, self.param_value_dim)
        return out.to(inputs.dtype)


__all__ = ["PattentionLayer"]
