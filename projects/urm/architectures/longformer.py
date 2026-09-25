"""External model module: Longformer (arch-071).

A2 indexed-K1 client. Verified against the pinned longformer source
(longformer/longformer.py): plain softmax attention restricted to a per-query
visible set — a dilated sliding band of width 2w+1 UNION the global token
columns, softmaxed JOINTLY in fp32 (longformer.py:187-188 — the band and the
extra global columns share one softmax denominator). Non-global query i attends
to band(i) ∪ {global tokens}.

The mixer is the typed indexed-K1 call over the gathered band∪global source set;
the route (the band/global indices) and the separate global-token pass (separate
q_global/k_global/v_global projections whose output overwrites the global rows,
normalized separately, longformer.py:221-251) are external. The dilation,
padding mask and the global-output overwrite are residual, not claimed.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class LongformerLayer(torch.nn.Module):
    """One Longformer mixer: typed indexed-K1 over the band ∪ global source set."""

    def __init__(self, num_heads: int, head_dim: int, window: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window = window
        document = {
            "schema_version": 2, "name": "longformer", "kind": "kernel_fragment",
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

    def build_gather_indices(self, T: int, global_positions: list[int], device=None) -> torch.Tensor:
        """Per-query source set: band(i) ∪ {global}, padding to a fixed width with -1.

        band(i) = [i-w, i+w] (clamped, non-causal symmetric band). Returns
        [1, 1, T, W] (head/batch broadcast). Global positions are included in
        every row's set.
        """
        w = self.window
        sets = []
        for i in range(T):
            band = list(range(max(0, i - w), min(T, i + w + 1)))
            s = sorted(set(band) | set(global_positions))
            sets.append(s)
        W = max(len(s) for s in sets)
        idx = torch.full((T, W), -1, dtype=torch.long)
        for i, s in enumerate(sets):
            idx[i, :len(s)] = torch.tensor(s)
        return idx.view(1, 1, T, W).to(device)

    def forward(self, query, key, value, gather_indices):
        """``gather_indices`` [B,H,T,W] (or the [1,1,T,W] broadcast from build_gather_indices)."""
        B = query.shape[0]
        H = self.num_heads
        if gather_indices.shape[0] == 1 and B > 1:
            gather_indices = gather_indices.expand(B, -1, -1, -1)
        if gather_indices.shape[1] == 1 and H > 1:
            gather_indices = gather_indices.expand(-1, H, -1, -1)
        return self._plan.execute(query=query, key=key, value=value,
                                  gather_indices=gather_indices.float())["output"]


__all__ = ["LongformerLayer"]
