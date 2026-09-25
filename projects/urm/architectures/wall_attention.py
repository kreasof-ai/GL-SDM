"""External model module: Wall Attention (arch-011).

A13 score/reducer client — the CHANNEL_DECAY score law. Verified against
fla/ops/wall_attn/naive.py @ 864a87f6: per pair (i,j),
``s_ij = Σ_n q_in k_jn · exp2(P_in − P_jn) · scale · RCP_LN2`` with ``P`` the
prefix cumsum of a per-channel log2 gate ``g [B,T,HQ,K]``; causal mask; softmax
(base-2 equivalent) ; output = weights @ v.

The per-channel decay sits *inside* the channel contraction, so it is NOT
expressible as an additive pairwise logit bias (sweep verdict) — it is the
typed ``K1ScoreLaw.CHANNEL_DECAY`` score law. The per-channel gate projection
and prefix cumsum are external; the mixer is the typed K1 call with the
``channel_gate`` operand. The pinned's base-2/exp2 form equals the natural-exp
form here (``exp2(x·RCP_LN2·...)`` folded into the gate); the optional sink
bias / window / varlen branches are residual.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class WallAttentionLayer(torch.nn.Module):
    """One Wall layer: external per-channel gate + typed K1 channel-decay call.

    q/k/v ``[B, T, H, D]`` (grouped KV supported), g (per-channel log gate)
    ``[B, T, H, D]``.
    """

    def __init__(self, num_heads: int, head_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        # The pinned Wall source works in base-2: score·RCP_LN2 then base-2 softmax.
        # Base-2 softmax(score·RCP_LN2) == base-e softmax(score), so the natural
        # scale (key-dim rule) already matches the pinned softmax base — no fold.
        import math  # noqa: F401  (kept for the module docstring's RCP_LN2 note)
        self.scale = head_dim ** -0.5
        document = {
            "schema_version": 2, "name": "wall_attention", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "channel_gate", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "scale", "dtype": "float32", "shape": []},
                ],
                "nodes": [
                    {
                        "id": "attn", "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "channel_gate", "scale"],
                        "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "sequence",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": True, "head_map": "equal", "score_law": "channel_decay",
                            "scale_rule": "explicit_operand",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "channel_gate": "channel_gate", "scale": "scale"},
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
        """``g`` is the per-channel log gate ``[B, T, H, D]`` (pre-cumsum, natural log)."""
        scale = torch.tensor(self.scale, device=query.device)
        return self._plan.execute(query=query, key=key, value=value, channel_gate=g,
                                  scale=scale)["output"]


__all__ = ["WallAttentionLayer"]
