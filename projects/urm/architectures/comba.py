"""External model module: Comba (arch-037).

A8 transition-breadth client — the dual-key delta: an independent predict/erase
key ``p`` distinct from the write key ``k``. Verified against
fla/ops/comba/naive.py @ 864a87f6:

    S' = exp(g_t)·S_{t-1}              (per-head scalar log decay)
    delta = β_t·(v_t − p_tᵀ·S')        (read off the predict key p)
    S_t = S' + k_t ⊗ delta             (write commits along the write key k)
    o_t = (scale·q_t)ᵀ·S_t             (read after update)

The retrieval/eraser key ``p`` and the write key ``k`` are distinct operands —
the typed ``LinearDeltaSpec`` with the ``predict_key`` role and head-scope decay.
The layer frontend (β=sigmoid, g=logsigmoid(a_proj), optional q/k L2-norm) is
external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class CombaLayer(torch.nn.Module):
    """One Comba layer: typed K2 dual-key delta call (predict_key role)."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        document = {
            "schema_version": 2, "name": "comba", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
                    {"name": "predict_key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
                ],
                "nodes": [
                    {
                        "id": "mixer", "op": "linear_delta_state",
                        "inputs": ["query", "key", "value", "predict_key", "beta",
                                   "log_decay", "initial_state"],
                        "outputs": ["output", "final_state"],
                        "params": {
                            "delta": True, "gate_scope": "head",
                            "read_timing": "after_update", "scale_rule": "key_dim_rsqrt",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "predict_key": "predict_key", "beta": "beta",
                                      "log_decay": "log_decay",
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

    def forward(self, q, k, v, p, beta, g):
        """All operands ``[B, H, T, *]``; beta/g are per-head scalars ``[B,H,T]``."""
        B, H, T, K = k.shape
        V = v.shape[-1]
        return self._plan.execute(
            query=q, key=k, value=v, predict_key=p, beta=beta, log_decay=g,
            initial_state=torch.zeros(B, H, K, V, device=q.device),
        )["output"]


__all__ = ["CombaLayer"]
