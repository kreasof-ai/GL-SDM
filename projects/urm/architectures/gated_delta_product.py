"""External model module: Gated DeltaProduct (arch-029).

A8 transition-breadth client — the ordered multi-delta (rank-R) transition.
Verified against fla/ops/gated_delta_product/naive.py @ 864a87f6: per token,
first apply a per-head scalar decay ``h ← h·exp(g_t)``, then R ordered
Householder-like delta factors — for j = 0..R-1 in order:
``h ← h + (v_tj − hᵀk_tj)·β_tj ⊗ k_tj`` — each retrieving from the state the
previous factor produced; then read ``o_t = (scale·q_t)ᵀ·h``.

The within-token transition is the ORDERED PRODUCT of R deltas — the typed
``LinearDeltaSpec`` with ``num_deltas=R``. The k/v/beta operands carry R factors
per token in the time axis (T*R). The factor construction is external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class GatedDeltaProductLayer(torch.nn.Module):
    """One Gated DeltaProduct layer: typed K2 ordered multi-delta call (rank R)."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, num_householder: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.num_householder = num_householder
        document = {
            "schema_version": 2, "name": "gated_delta_product", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    # k/v/beta carry R factors per token (T*R in the time axis).
                    {"name": "key", "dtype": "float32", "shape": ["B", "H", "TR", "K"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "TR", "V"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "TR"]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
                    {"name": "scale", "dtype": "float32", "shape": []},
                ],
                "nodes": [
                    {
                        "id": "mixer", "op": "linear_delta_state",
                        "inputs": ["query", "key", "value", "beta", "log_decay", "initial_state"],
                        "outputs": ["output", "final_state"],
                        "params": {
                            "delta": True, "gate_scope": "head",
                            "read_timing": "after_update", "scale_rule": "explicit_operand",
                            "num_deltas": num_householder,
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "beta": "beta", "log_decay": "log_decay",
                                      "initial_state": "initial_state", "scale": "scale"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, q, k, v, g, beta, scale):
        """q [B,H,T,K]; k/v [B,H,T*R,*]; g/beta per-head [B,H,T] / [B,H,T*R]."""
        B, H, T, K = q.shape
        V = v.shape[-1]
        return self._plan.execute(
            query=q, key=k, value=v, beta=beta, log_decay=g,
            initial_state=torch.zeros(B, H, K, V, device=q.device),
            scale=torch.tensor(scale, dtype=torch.float32, device=q.device),
        )["output"]


__all__ = ["GatedDeltaProductLayer"]
