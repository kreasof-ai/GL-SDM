"""External model module: NSA (Native Sparse Attention, arch-005).

A2 indexed-K1 client. Verified against fla/ops/nsa/naive.py @ 864a87f6. NSA is
three K1 branches merged by per-token-per-head gates:
``o = g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa`` (a weighted merge, not an
unweighted typed merge — sweep verdict). This module implements the **selected
branch** — the indexed-K1 path — plus the typed merge against the sliding-window
branch; the compressed branch and the top-k route are external.

Selected branch (the indexed-K1 exercise): gather the tokens of the top-k
selected blocks (``block_indices`` [B,T,H,S] expanded to token positions
``block*block_size + arange(block_size)``, padded with -1 for causally-invisible
or surplus slots), then softmax-attend over the gathered set. The block-selection
route (top-k over group-mean softmax probs with forced first/current/previous
blocks) is external; the mixer is the typed indexed-K1 call.

The full three-branch gated composition, the compression mean-pooling, the
route cost, and the GQA power-of-2/≥16 tile constraint are residual, not claimed.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class NSASelectedLayer(torch.nn.Module):
    """The NSA selected branch: typed indexed-K1 over gathered block tokens."""

    def __init__(self, num_heads: int, head_dim: int, block_size: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        document = {
            "schema_version": 2, "name": "nsa_selected", "kind": "kernel_fragment",
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

    def forward(self, query, key, value, block_indices):
        """``block_indices`` [B,T,H,S] selected block ids; expanded to token positions here.

        Token positions = block*block_size + arange(block_size); causally-invisible
        (position > query index) or -1 (padded) block slots are masked to -1.
        """
        B, T, H, S = block_indices.shape
        arange_bs = torch.arange(self.block_size, device=query.device)
        # token positions [B,T,H,S,bs]
        tok = block_indices.unsqueeze(-1) * self.block_size + arange_bs
        q_pos = torch.arange(T, device=query.device).view(1, T, 1, 1, 1)
        valid = (block_indices.unsqueeze(-1) >= 0) & (tok <= q_pos) & (tok >= 0)  # [B,T,H,S,bs]
        tok = torch.where(valid, tok, torch.full_like(tok, -1))
        gather = tok.flatten(-2).permute(0, 2, 1, 3)                         # [B,H,T,W]
        return self._plan.execute(query=query, key=key, value=value,
                                  gather_indices=gather.float())["output"]


__all__ = ["NSASelectedLayer"]
