"""External model module: dense softmax attention with a typed query→KV head map.

This module composes the closed K1 ``weighted_reduce`` contract with ordinary
external operators to cover three verified composition-now ledger rows:

- arch-001 MHA — identity head map (``head_map="equal"``)
- arch-002 MQA — all query heads share one KV head (``head_map="single"``)
- arch-003 GQA — explicit grouped head map (``head_map="grouped"``, group_size=HQ//H)

The sweep (docs/planning/direction-sweep.md) verified the mixer equation is
exactly U1.S; the Q/K/V projections and output projection are the declared
external stages. The head map is a descriptor field of the K1 node, never a
runtime shape guess. The compiled graph executes through the public path
(frontend → normalize → compile → BoundGraphPlan); this module imports only
public frontend/compiler/runtime APIs.

Parity oracle: pinned fla ``naive_parallel_attn`` (fla/ops/attn/naive.py @
864a87f6), which computes the same grouped-head equation by reshaping
``[B,T,HQ,D] → [B,T,H,G,D]`` and sharing K/V across the group.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class HeadMapAttention(torch.nn.Module):
    """One attention layer: external Q/K/V/O projections + typed K1 mixer.

    Parameters
    ----------
    model_dim:
        Embedding width of the layer (Q projection input, O projection output).
    query_heads:
        Number of query heads (HQ).
    kv_heads:
        Number of key/value heads (H). ``HQ == H`` is MHA, ``H == 1`` is MQA,
        otherwise GQA. Must divide ``query_heads``.
    head_dim:
        Per-head width (D).
    causal:
        Causal masking policy of the K1 node.
    head_map:
        K1 descriptor head-map field: ``"equal"``, ``"single"`` or
        ``"grouped"``. Must agree with the HQ/H ratio — the K1 descriptor
        rejects a grouped map without a matching positive ``group_size``.
    target:
        Graph execution tier (``"reference"`` default; ``"native"`` once the
        native K1 anchor is qualified for the descriptor).
    """

    def __init__(
        self,
        model_dim: int,
        query_heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        causal: bool = True,
        head_map: str,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by kv_heads ({kv_heads})"
            )
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.causal = causal

        # External stages: ordinary dense projections (E[QKV/output]).
        self.q_proj = torch.nn.Linear(model_dim, query_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, kv_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, kv_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(query_heads * head_dim, model_dim, bias=False)

        group_size = query_heads // kv_heads
        params: dict[str, object] = {
            "query_domain": "sequence",
            "source_domain": "sequence",
            "selection": "dense",
            "normalization": "softmax",
            "capacity_policy": "dropless",
            "deterministic": True,
            "causal": causal,
            "head_map": head_map,
        }
        if head_map == "grouped":
            params["group_size"] = group_size

        document = {
            "schema_version": 2,
            "name": f"head_map_attention:{head_map}",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", query_heads, head_dim]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", kv_heads, head_dim]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", kv_heads, head_dim]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value"],
                        "outputs": ["output"],
                        "params": params,
                    }
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(
            program, target=target, intent=CompilationIntent(intent)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the layer on ``[B, T, model_dim]`` hidden states."""
        B, T, _ = hidden.shape
        q = self.q_proj(hidden).view(B, T, self.query_heads, self.head_dim)
        k = self.k_proj(hidden).view(B, T, self.kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(B, T, self.kv_heads, self.head_dim)
        out = self._plan.execute(query=q, key=k, value=v)["output"]
        return self.o_proj(out.reshape(B, T, self.query_heads * self.head_dim))


__all__ = ["HeadMapAttention"]
