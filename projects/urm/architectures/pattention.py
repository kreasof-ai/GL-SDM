"""External model module: Pattention / parameter-token attention (arch-057).

Verified composition-now row as a typed-domain composition (sweep row-057,
against megatron/model/tokenformer.py @ 4d56c73f): the mixer is the K1 reduction
over a fixed parameter-token source domain — ``scores = query @ key_paramᵀ ×
scale``; a closed score-map/normalizer algebra over the parameter tokens; then
``output = norm(scores) @ value_param``.

The score-map/normalizer algebra is the admitted K1 ``MAP_NORMALIZE`` reducer
(axis A13): the elementwise map (``EXP`` / ``GELU`` /
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
(sweep blocker). Native softmax mode precomputes scores externally and composes the public
softmax reducer with multiplication by the parameter-token count. Other modes
retain the public map/normalizer reducer. Wide values use independent heads or
column tiles; normalization remains over the same parameter-token domain.

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
        self._native = target == "native"

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
                    {"name": "value", "dtype": "float32", "shape": ["B", "P", 1, "V"]},
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
        self._softmax_plan = None
        if self._native and norm_activation_type == "softmax":
            # EXP/L1 times source count is ordinary softmax times source count.
            # Precompute wide scores externally and use the existing production
            # softmax reducer, including its native backward, in 64-wide heads.
            import copy
            softmax_document = copy.deepcopy(document)
            softmax_document["name"] = "pattention_softmax_scores"
            softmax_document["graph"]["inputs"][:3] = [
                {"name": "query", "dtype": "float32", "shape": ["B", "L", "VH", 1]},
                {"name": "key", "dtype": "float32", "shape": ["B", "P", "VH", 1]},
                {"name": "value", "dtype": "float32", "shape": ["B", "P", "VH", 64]},
            ]
            softmax_document["graph"]["inputs"].append({
                "name": "score_bias", "dtype": "float32", "shape": ["B", 1, "L", "P"],
            })
            node = softmax_document["graph"]["nodes"][0]
            for field in ("reducer_law", "score_map", "normalizer_p", "normalize_before_map"):
                node["params"].pop(field)
            node["inputs"].append("score_bias")
            node["params"]["roles"]["score_bias"] = "score_bias"
            softmax_program = normalize_graph_document(load_graph_recipe_document(softmax_document).document)
            self._softmax_plan = compile_graph(softmax_program, target=target, intent=CompilationIntent(intent))

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
        dtype = (torch.get_autocast_dtype(query.device.type)
                 if torch.is_autocast_enabled(query.device.type) else torch.float32)
        q = query.reshape(-1, L, 1, K).to(dtype)
        # Broadcast the parameter tokens across the (flattened) batch. The expand
        # must be materialized: the kernels index (batch*TK + key) flat offsets, so
        # a stride-0 view reads out of bounds for batch > 0 (context-dependent
        # garbage — the resume-gate nondeterminism this surfaced).
        B = q.shape[0]
        P = key.shape[-2]
        k = key.reshape(1, P, 1, K).to(dtype).expand(B, P, 1, K).contiguous()
        v = value.reshape(1, P, 1, self.param_value_dim).to(dtype).expand(B, P, 1, self.param_value_dim).contiguous()
        if self._softmax_plan is not None:
            with torch.autocast(query.device.type, enabled=False):
                scores = q.squeeze(2).float() @ k.squeeze(2).float().transpose(-1, -2)
                scores = (scores * scale_factor).unsqueeze(1)
            heads = (self.param_value_dim + 63) // 64
            queries = torch.zeros(B, L, heads, 1, device=query.device, dtype=dtype)
            keys = torch.zeros(B, P, heads, 1, device=query.device, dtype=dtype)
            values = F.pad(v, (0, heads * 64 - self.param_value_dim)).reshape(B, P, heads, 64)
            out = self._softmax_plan.execute(query=queries, key=keys, value=values,
                                             score_bias=scores, scale=1.0)["output"] * P
            out = out.reshape(B, L, heads * 64)[..., :self.param_value_dim]
        elif self._native and self.param_value_dim > 256:
            # Keep the native reduction's value tile within a practical register
            # budget. Scores and normalization are independent of value columns.
            out = torch.cat([
                self._plan.execute(query=q, key=k, value=v[..., start:start + 256].contiguous(),
                                   scale=scale_factor)["output"]
                for start in range(0, self.param_value_dim, 256)
            ], dim=-1)
        else:
            out = self._plan.execute(query=q, key=k, value=v, scale=scale_factor)["output"]
        out = out.reshape(*lead, L, self.param_value_dim)
        return out.to(dtype).to(inputs.dtype)


class _RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight)


class TokenformerBlock(torch.nn.Module):
    """The full Tokenformer decoder block (granularity="block"), per the pinned
    ``ParallelTokenformerLayer`` + ``ParallelSelfAttention`` (megatron/model/
    tokenformer.py @ 4d56c73f): EVERY linear map is a Pattention — the Q/K/V/output
    projections (each over ``qkv_slot_num`` parameter tokens) and the MLP (over
    ``ffn_slot_num`` parameter tokens). The sequence mixing itself is standard
    multi-head causal attention over the projected heads:

        h = LN1(x); q,k,v = Pattn_q(h), Pattn_k(h), Pattn_v(h)
        x = x + Pattn_o(CausalAttention(q, k, v))
        x = x + Pattn_mlp(LN2(x))

    No dense Linear anywhere; the pinned ``norm_activation_type`` maps onto the K1
    MAP_NORMALIZE reducer fields per the layer-level docstring above.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim: int,
        ffn_slots: int,
        qkv_slots: int,
        *,
        norm_activation_type: str = "softmax",
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm1 = _RMSNorm(model_dim)
        self.norm2 = _RMSNorm(model_dim)
        pattn = lambda out_dim, slots: PattentionLayer(  # noqa: E731 — ctor shorthand
            model_dim, out_dim, param_token_num=slots,
            norm_activation_type=norm_activation_type, target=target, intent=intent)
        self.query = pattn(model_dim, qkv_slots)
        self.key = pattn(model_dim, qkv_slots)
        self.value = pattn(model_dim, qkv_slots)
        self.proj = pattn(model_dim, qkv_slots)
        self.mlp = pattn(model_dim, ffn_slots)
        # The pin overwrites the ``torch.rand`` parameter-token init with neox's
        # small-normal ``init_method`` — required here because FIVE pattentions
        # cascade per block (uniform-positive keys × RMSNorm'd queries overflow the
        # exp score map at depth). Matches the pinned runtime init.
        for module in (self.query, self.key, self.value, self.proj, self.mlp):
            torch.nn.init.normal_(module.key_param_tokens, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.value_param_tokens, mean=0.0, std=0.02)
        # The sequence mixing: standard multi-head causal softmax attention over the
        # pattention-projected heads — the typed K1 call, projections stripped (they
        # are the pattentions above, per the pinned source).
        document = {
            "schema_version": 2, "name": "tokenformer_causal_attn", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                ],
                "nodes": [{
                    "id": "attend", "op": "weighted_reduce",
                    "inputs": ["query", "key", "value"], "outputs": ["output"],
                    "params": {
                        "query_domain": "sequence", "source_domain": "sequence",
                        "selection": "dense", "normalization": "softmax",
                        "capacity_policy": "dropless", "deterministic": True,
                        "causal": True, "head_map": "equal",
                        "roles": {"query": "query", "key": "key", "value": "value"},
                    },
                }],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._attn_plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        h = self.norm1(x)
        q = self.query(h).view(B, T, H, D)
        k = self.key(h).view(B, T, H, D)
        v = self.value(h).view(B, T, H, D)
        if torch.is_autocast_enabled(x.device.type):
            dtype = torch.get_autocast_dtype(x.device.type)
            q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        ctx = self._attn_plan.execute(query=q, key=k, value=v)["output"].reshape(B, T, C)
        x = x + self.proj(ctx)
        x = x + self.mlp(self.norm2(x))
        return x


__all__ = ["PattentionLayer", "TokenformerBlock"]
