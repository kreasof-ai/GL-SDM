"""External model module: RWKV-7 (arch-042).

A8 transition-breadth client — the DPLR (diagonal + rank-1) low-rank K2
transition. Verified against fla/ops/rwkv7/fused_recurrent.py @ 864a87f6, which
maps to fused_recurrent_dplr_delta_rule with gk = w:

    S_t = Diag(exp(w_t))·S_{t-1} + (a_tᵀ S_{t-1})⊗b_t + v_t⊗k_t
    o_t = (scale·r_t)ᵀ·S_t          (read after update)

The transition is diagonal (Diag(exp w)) plus a rank-1 term (a bᵀ) read off the
pre-decay state, plus an additive rank-1 write v⊗k — the typed ``LinearDeltaSpec``
with ``low_rank=True`` (the α/β roles) and channel decay. The input-precomputable
coefficients (the RWKV-7 token-shift / lerp frontend producing r/w/k/v/a/b) are
external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class RWKV7Layer(torch.nn.Module):
    """One RWKV-7 layer: typed K2 DPLR low-rank call."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        document = {
            "schema_version": 2, "name": "rwkv7", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "V"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "alpha", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "low_rank_beta", "dtype": "float32", "shape": ["B", "H", "T", "K"]},
                    {"name": "initial_state", "dtype": "float32", "shape": ["B", "H", "K", "V"]},
                ],
                "nodes": [
                    {
                        "id": "mixer", "op": "linear_delta_state",
                        "inputs": ["query", "key", "value", "beta", "log_decay",
                                   "alpha", "low_rank_beta", "initial_state"],
                        "outputs": ["output", "final_state"],
                        "params": {
                            "delta": False, "gate_scope": "channel",
                            "read_timing": "after_update", "scale_rule": "key_dim_rsqrt",
                            "low_rank": True,
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "beta": "beta", "log_decay": "log_decay",
                                      "alpha": "alpha", "low_rank_beta": "low_rank_beta",
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

    def forward(self, r, w, k, v, a, b):
        """RWKV-7 operands, all ``[B, T, H, *]`` (the fla layout); the graph runs [B,H,T,*].

        r: read key [B,T,H,K]; w: log decay [B,T,H,K]; k: write key [B,T,H,K];
        v: value [B,T,H,V]; a/b: the rank-1 transition factors [B,T,H,K].
        """
        B, T, H, K = k.shape
        V = v.shape[-1]
        out = self._plan.execute(
            query=r.transpose(1, 2), key=k.transpose(1, 2), value=v.transpose(1, 2),
            beta=torch.ones(B, H, T, device=k.device),
            log_decay=w.transpose(1, 2),
            alpha=a.transpose(1, 2), low_rank_beta=b.transpose(1, 2),
            initial_state=torch.zeros(B, H, K, V, device=k.device),
        )["output"]
        return out.transpose(1, 2)


__all__ = ["RWKV7Layer"]
