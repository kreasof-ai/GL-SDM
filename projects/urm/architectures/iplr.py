"""External model module: Generalized delta IPLR (arch-031).

A8 transition-breadth client — identity-plus-rank-1 (IPLR) low-rank K2
transition with NO diagonal decay. Verified against
fla/ops/generalized_delta_rule/iplr/naive.py @ 864a87f6:

    kv_t = k_t ⊗ v_t + (α_tᵀ S_{t-1}) ⊗ β_t
    S_t = S_{t-1} + kv_t
    o_t = (scale·q_t)ᵀ·S_t

An identity-plus-rank-1 transition (read α off the pre-update state, written
back along β) plus the additive rank-1 write k⊗v — the typed ``LinearDeltaSpec``
with ``low_rank=True`` and no decay (gate_scope=none). IPLR is the no-decay
special case of DPLR (arch-032).
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class IPLRLayer(torch.nn.Module):
    """One IPLR layer: typed K2 low-rank call, no decay."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        document = {
            "schema_version": 2, "name": "iplr", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "alpha", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "low_rank_beta", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
                ],
                "nodes": [
                    {
                        "id": "mixer", "op": "linear_delta_state",
                        "inputs": ["query", "key", "value", "beta", "alpha",
                                   "low_rank_beta", "initial_state"],
                        "outputs": ["output", "final_state"],
                        "params": {
                            "delta": False, "gate_scope": "none",
                            "read_timing": "after_update", "scale_rule": "key_dim_rsqrt",
                            "low_rank": True,
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "beta": "beta", "alpha": "alpha",
                                      "low_rank_beta": "low_rank_beta",
                                      "initial_state": "initial_state"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, q, k, v, alpha, beta):
        """All operands ``[B, H, T, *]`` (head-first, matching the pinned naive)."""
        B, H, T, K = k.shape
        V = v.shape[-1]
        return self._plan.execute(
            query=q, key=k, value=v,
            beta=torch.ones(B, H, T, device=q.device),
            alpha=alpha, low_rank_beta=beta,
            initial_state=torch.zeros(B, H, K, V, device=q.device),
        )["output"]


__all__ = ["IPLRLayer"]
