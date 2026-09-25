"""Shared external base for the A2 indexed-K1 clients (MoBA, DSA, Sparse Transformer).

Each A2 client is: an external ROUTE producing a per-query source-index set, then
a typed indexed-K1 call (gather K/V at the indices, softmax-attend over the
gathered set). The mixer is shared; only the route differs. This base holds the
typed indexed-K1 graph; subclasses build the route and call ``_attend``.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class IndexedAttentionBase(torch.nn.Module):
    """The typed indexed-K1 gather-attend mixer (shared across A2 clients)."""

    def __init__(self, num_heads: int, head_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        document = {
            "schema_version": 2, "name": "indexed_attention", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "gather_indices", "dtype": "float32", "shape": ["B", num_heads, "T", "W"]},
                ],
                "nodes": [
                    {
                        "id": "attn", "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "gather_indices"], "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "sequence",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": False, "head_map": "equal",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "gather_indices": "gather_indices"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def _attend(self, query, key, value, gather_indices):
        """gather_indices [B,H,T,W] source positions (-1 = padding/masked)."""
        return self._plan.execute(query=query, key=key, value=value,
                                  gather_indices=gather_indices.float())["output"]


__all__ = ["IndexedAttentionBase"]
