"""External model module: FoX / Forgetting Attention (arch-008).

A13 score/reducer client. Verified against fla/ops/forgetting_attn/naive.py @
864a87f6: the K1 score carries a data-dependent cumulative-log-gate additive
bias ``gc_i − gc_j`` with ``gc = cumsum(logsigmoid(f))`` (per head), causal
(optionally windowed) mask, softmax, @v.

The cumulative forget-gate bias is data-dependent, so no softmax-only or
fixed-pairwise-bias materialization claim is possible (sweep verdict). The bias
is computed externally (cumsum of the per-head log-gate) and passed as the K1
``score_bias`` operand — the existing additive-score-bias K1 contract already
covers this score algebra; no new descriptor field is required. The forget-gate
projection and grouped-KV expansion are external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class ForgettingAttentionLayer(torch.nn.Module):
    """One FoX layer: external forget-gate cumsum + typed K1 with score bias.

    q/k/v ``[B, T, H, D]`` (grouped KV supported), g (per-head log forget gate)
    ``[B, T, HQ]``.
    """

    def __init__(self, num_heads: int, head_dim: int, *,
                 window_size: int | None = None,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        document = {
            "schema_version": 2, "name": "forgetting_attention", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "score_bias", "dtype": "float32", "shape": ["B", num_heads, "T", "S"]},
                ],
                "nodes": [
                    {
                        "id": "attn", "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "score_bias"], "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "sequence",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": True, "head_map": "equal",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "score_bias": "score_bias"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                g: torch.Tensor) -> torch.Tensor:
        """``g`` is the per-head log forget gate ``[B, T, HQ]`` (logsigmoid output)."""
        B, T, HQ, _ = query.shape
        # External cumulative forget gate: gc = cumsum(g); bias = gc_i − gc_j.
        gc = g.float().cumsum(1)                                # [B, T, HQ]
        bias = gc.transpose(1, 2).unsqueeze(-1) - gc.transpose(1, 2).unsqueeze(-2)  # [B,HQ,T,T]
        out = self._plan.execute(query=query, key=key, value=value, score_bias=bias)["output"]
        if self.window_size is not None:
            # Windowed variant re-masks outside [i-w+1, i]; handled by an additive mask.
            raise NotImplementedError("windowed FoX is a residual variant")
        return out


__all__ = ["ForgettingAttentionLayer"]
