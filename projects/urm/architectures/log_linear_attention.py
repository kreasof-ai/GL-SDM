"""External model module: Log-linear attention (arch-009) / A4 hierarchical chunks.

A4 hierarchical client. Verified against fla/ops/log_linear_attn/naive.py @
864a87f6: the mixer is a hierarchical K2-family form — a hierarchy of dyadic
level states with per-level scales — evaluated in the full-matrix reference form

    H = Σ_level exp(segsum(g)) ∘ mask_level           (the dyadic level-mask law)
    o = (H ∘ (q·kᵀ)) · v                                (elementwise-modulated contraction)

A = exp(segsum(g)) is the cumulative decay; mask_level selects dyadic sub-blocks
with per-level scales. H is NOT a softmax mask (it can be 0/negative) — this is a
linear-attention-like full-matrix form, not the canonical single-state K2 law
(sweep verdict). The hierarchical H construction is the pinned external typed
transform — the oracle helpers (``construct_H_matrix`` / ``segsum`` /
``construct_level_mask`` / ``dyadic_level_of``) live with the parity-gate test
(``tests/test_architectures_a4.py``). The level_scales frontend and the
streaming/chunked partial-chunk state (LogLinearAttentionState with
q/k/v/g/level_scales carry) are residual, not claimed. Shared with arch-046
LogLinearMamba2 (which adds the Mamba-2 frontend).
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class BankedLogLinearMixer(torch.nn.Module):
    """The shared A4 hierarchical mixer routed through the public ``dyadic_banked_state`` op.

    Both arch-009 (Log-linear attention) and arch-046 (LogLinearMamba2) use this:
    the banked dyadic state law is identical; only the frontend differs (arch-046
    adds the Mamba-2 discretization). The disjoint dyadic-block and full-matrix
    reference oracles live in the parity-gate test (``tests/test_architectures_a4.py``).
    """

    def __init__(self, num_heads: int, head_dim: int, num_levels: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_levels = num_levels
        H, D, L = num_heads, head_dim, num_levels
        document = {
            "schema_version": 2, "name": "log_linear_banked", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "log_decay", "dtype": "float32", "shape": ["B", "T", H]},
                    {"name": "level_scales", "dtype": "float32", "shape": ["B", "T", H, L]},
                ],
                "nodes": [
                    {
                        "id": "bank", "op": "dyadic_banked_state",
                        "inputs": ["query", "key", "value", "log_decay", "level_scales"],
                        "outputs": ["output"],
                        "params": {
                            "num_levels": L,
                            "roles": {
                                "query": "query", "key": "key", "value": "value",
                                "log_decay": "log_decay", "level_scales": "level_scales",
                            },
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        program = normalize_graph_document(load_graph_recipe_document(document).document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, q, k, v, g, level_scales):
        """q/k/v [B,T,H,D]; g per-head log decay [B,T,H]; level_scales [B,T,H,L_levels]."""
        return self._plan.execute(
            query=q.float(), key=k.float(), value=v.float(),
            log_decay=g.float(), level_scales=level_scales.float(),
        )["output"]


__all__ = ["BankedLogLinearMixer"]
