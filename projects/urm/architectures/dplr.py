"""External model module: DPLR (arch-032) — the diagonal-plus-low-rank delta rule.

A8 transition-breadth client — the decay∘low-rank composition, the second
structurally independent client of the native K2 composed branch (the first is
RWKV-7, arch-042). Verified against
fla/ops/generalized_delta_rule/dplr/naive.py @ 864a87f6:

    lr_read = α_tᵀ·S_{t-1}                                (off the PRE-decay state)
    S_t     = Diag(exp(gk_t))·S_{t-1} + k_t·v_tᵀ + β_t·lr_read
    o_t     = (scale·q_t)ᵀ·S_t          (read after update, scale = K^-0.5)

Unlike RWKV-7 (whose pinned law is the SAME composed kernel with externally
token-shifted operands), DPLR binds the naive's operand set directly: the decay
is the per-channel gk, the rank-1 read factor is α and the rank-1 write factor
is β (the naive's ``alpha``/``beta``), and the forward returns the final state
alongside the output. The typed ``LinearDeltaSpec`` with ``low_rank=True`` (the
α/β roles) and channel decay; the input-precomputable coefficients are external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class DPLRLayer(torch.nn.Module):
    """One DPLR layer: typed K2 decay∘low-rank composed call."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        document = {
            "schema_version": 2, "name": "dplr", "kind": "kernel_fragment",
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

    def forward(self, q, k, v, alpha, beta, gk, initial_state=None, output_final_state=True):
        """The pinned DPLR naive's operand set, all ``[B, H, T, *]`` (head-first, the
        naive's layout).

        q/k: read/write keys [B,H,T,K]; v: value [B,H,T,V]; gk: per-channel log decay
        [B,H,T,K]; alpha: rank-1 read factor [B,H,T,K]; beta: rank-1 write factor
        [B,H,T,K]. Returns ``(output, final_state)`` — the pinned naive's contract.
        """
        B, H, T, K = k.shape
        V = v.shape[-1]
        if initial_state is None:
            initial_state = torch.zeros(B, H, K, V, device=k.device)
        result = self._plan.execute(
            query=q, key=k, value=v,
            beta=torch.ones(B, H, T, device=k.device),
            log_decay=gk,
            alpha=alpha, low_rank_beta=beta,
            initial_state=initial_state,
        )
        out = result["output"]
        if not output_final_state:
            return out, None
        return out, result["final_state"]


__all__ = ["DPLRLayer"]
