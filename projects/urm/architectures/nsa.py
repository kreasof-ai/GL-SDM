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


def _indexed_k1_plan(num_heads: int, head_dim: int, window: int, kv_len: str,
                     target: str, intent: str):
    """One typed indexed-K1 plan: q [B,T,H,D], k/v [B,S,H,D], gather [B,H,T,W]."""
    document = {
        "schema_version": 2, "name": "nsa_indexed_branch", "kind": "kernel_fragment",
        "graph": {
            "inputs": [
                {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                {"name": "key", "dtype": "float32", "shape": ["B", kv_len, num_heads, head_dim]},
                {"name": "value", "dtype": "float32", "shape": ["B", kv_len, num_heads, head_dim]},
                {"name": "gather_indices", "dtype": "float32", "shape": ["B", num_heads, "T", window]},
            ],
            "nodes": [{
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
            }],
            "outputs": ["output"],
        },
    }
    recipe = load_graph_recipe_document(document)
    program = normalize_graph_document(recipe.document)
    return compile_graph(program, target=target, intent=CompilationIntent(intent))


class NSAFullLayer(torch.nn.Module):
    """The full three-branch NSA (arch-005 completed), per fla/ops/nsa/naive.py:

        o = g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa

    - compressed: mean-pooled block keys/values; query t sees block c only when the
      block is fully in the past (``(c+1)*block_size - 1 <= t``);
    - selected: top-k blocks routed by the compression scores (external top-k), the
      indexed-K1 branch (the prior NSASelectedLayer's law);
    - sliding: the trailing ``window_size`` token window.

    All three branches are typed indexed-K1 calls (gather sets encode the branch);
    the projections, mean-pool, top-k route and the per-token-per-head gate merge
    are declared external stages. GQA tile constraints of the pinned triton kernel
    are kernel-side, not the law, and are not imposed.
    """

    def __init__(self, model_dim: int, num_heads: int, head_dim: int, *,
                 block_size: int = 8, topk: int = 2, window_size: int = 16,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.block_size, self.topk, self.window_size = block_size, topk, window_size
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.gate_proj = torch.nn.Linear(model_dim, 3 * num_heads, bias=False)
        # Branch plans: selected (S = topk*block_size tokens), compressed (C blocks),
        # sliding (window_size tokens). Compressed KV length is sequence/block_size —
        # symbolic in the plan; window counts are fixed at construction.
        self._slc_plan = _indexed_k1_plan(num_heads, head_dim, topk * block_size, "S", target, intent)
        self._cmp_plan = _indexed_k1_plan(num_heads, head_dim, "WC", "C", target, intent)
        self._swa_plan = _indexed_k1_plan(num_heads, head_dim, window_size, "S", target, intent)

    @staticmethod
    def _gather(idx: torch.Tensor) -> torch.Tensor:
        """Causal-token gather indices [B,T,H,W] → plan layout [B,H,T,W], float, -1 padded."""
        return idx.permute(0, 2, 1, 3).float()

    def forward(self, hidden: torch.Tensor,
                block_indices: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = hidden.shape
        H, D, BS, W = self.num_heads, self.head_dim, self.block_size, self.window_size
        device = hidden.device
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        gates = torch.sigmoid(self.gate_proj(hidden)).view(B, T, H, 3)  # cmp, slc, swa

        # -- compressed KV: mean-pool over blocks (pad the tail block with its mean) --
        n_blocks = (T + BS - 1) // BS
        pad = n_blocks * BS - T
        k_pad = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v_pad = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
        k_cmp = k_pad.view(B, n_blocks, BS, H, D).mean(dim=2)  # [B, C, H, D]
        v_cmp = v_pad.view(B, n_blocks, BS, H, D).mean(dim=2)
        # visible blocks for query t: c with (c+1)*BS - 1 <= t
        q_pos = torch.arange(T, device=device).view(T, 1)
        blk = torch.arange(n_blocks, device=device).view(1, -1)
        cmp_visible = (blk + 1) * BS - 1 <= q_pos  # [T, C]
        cmp_idx = torch.where(cmp_visible, blk, torch.full_like(blk, -1))
        cmp_idx = cmp_idx.view(1, T, 1, n_blocks).expand(B, T, H, n_blocks)
        o_cmp = self._cmp_plan.execute(query=q, key=k_cmp, value=v_cmp,
                                       gather_indices=self._gather(cmp_idx))["output"]

        # -- selected: top-k blocks by mean compression score (external route) -------
        if block_indices is None:
            with torch.no_grad():
                scores = torch.einsum("bthd,bchd->bhtc", q.float(), k_cmp.float())
                scores = scores.masked_fill(~cmp_visible.view(1, 1, T, n_blocks), float("-inf"))
                top = scores.topk(min(self.topk, n_blocks), dim=-1).indices  # [B,H,T,topk]
            top = top.permute(0, 2, 1, 3)  # [B,T,H,topk]
        else:
            # Pinned override: block_indices [B,T,H,S] bypass the computed route.
            top = block_indices[..., : self.topk].long()
        tok = top.unsqueeze(-1) * BS + torch.arange(BS, device=device)   # [B,T,H,topk,BS]
        valid = (top.unsqueeze(-1) >= 0) & (tok <= q_pos.view(1, T, 1, 1, 1))
        slc_idx = torch.where(valid, tok, torch.full_like(tok, -1)).flatten(-2)
        o_slc = self._slc_plan.execute(query=q, key=k, value=v,
                                       gather_indices=self._gather(slc_idx))["output"]

        # -- sliding window ----------------------------------------------------------
        off = torch.arange(W, device=device).view(1, W)
        swa_idx = q_pos - (W - 1 - off)  # [T, W] trailing window
        swa_idx = torch.where(swa_idx >= 0, swa_idx, torch.full_like(swa_idx, -1))
        swa_idx = swa_idx.view(1, T, 1, W).expand(B, T, H, W)
        o_swa = self._swa_plan.execute(query=q, key=k, value=v,
                                       gather_indices=self._gather(swa_idx))["output"]

        out = (gates[..., 0:1] * o_cmp + gates[..., 1:2] * o_slc + gates[..., 2:3] * o_swa)
        return self.o_proj(out.reshape(B, T, H * D))


__all__ = ["NSASelectedLayer", "NSAFullLayer"]
