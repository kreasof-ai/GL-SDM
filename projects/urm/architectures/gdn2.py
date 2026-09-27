"""External model module: GDN2 / Gated DeltaNet 2 (arch-027).

A8 transition-breadth client — the generalized rank-1 K2 transition with
distinct erase/write factors. Verified against fla/ops/gdn2/naive.py @ 864a87f6:

    h ← h·exp(g_t)                       (channel decay on the KEY axis, g [B,T,H,K])
    erase = (b_t ⊙ k_t)ᵀ·h               (read at the erase-gated key)
    v_new = w_t ⊙ v_t − erase            (write-gated value minus the read-back)
    h ← h + k_t ⊗ v_new
    o_t = (scale·q_t)ᵀ·h                 (read after update)

Distinct K-axis erase gate ``b`` and V-axis write gate ``w`` plus channel decay —
collapsing b=w=scalar β recovers KDA (sweep verdict). This is the typed
``LinearDeltaSpec`` with the ``erase_gate`` and ``write_gate`` roles; the gate
projections are external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class GDN2Layer(torch.nn.Module):
    """One GDN2 layer: typed K2 generalized rank-1 call (erase + write gates)."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        document = {
            "schema_version": 2, "name": "gdn2", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "erase_gate", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "write_gate", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
                    {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
                ],
                "nodes": [
                    {
                        "id": "mixer", "op": "linear_delta_state",
                        "inputs": ["query", "key", "value", "beta", "log_decay",
                                   "erase_gate", "write_gate", "initial_state"],
                        "outputs": ["output", "final_state"],
                        "params": {
                            "delta": True, "gate_scope": "channel",
                            "read_timing": "after_update", "scale_rule": "key_dim_rsqrt",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "beta": "beta", "log_decay": "log_decay",
                                      "erase_gate": "erase_gate", "write_gate": "write_gate",
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

    def forward(self, query, key, value, g, erase_gate, write_gate):
        """All operands ``[B, H, T, *]``; g/erase_gate ``[B,H,T,K]``, write_gate ``[B,H,T,V]``."""
        B, H, T, K = key.shape
        V = value.shape[-1]
        return self._plan.execute(
            query=query, key=key, value=value,
            beta=torch.ones(B, H, T, device=query.device),  # GDN2 has no separate beta (β=1)
            log_decay=g, erase_gate=erase_gate, write_gate=write_gate,
            initial_state=torch.zeros(B, H, K, V, device=query.device),
        )["output"]


__all__ = ["GDN2Layer"]
