"""External model module: KATA (arch-073, Flash-KATA / KATA-SPD).

A13 score/reducer client — the SQUARED_SUM reducer law. Verified against the
pinned kata source (kata/parallel_kata_attn.py): the causal score
``A[t,s] = Σ_g (scale·q_g[t]·k_g[s])²`` (concat-SPD; the head splits into
``num_groups`` groups), all scores ≥ 0, future masked to 0; output
``o = (Σ_s A·v_s)/max(Σ_s A, 1)`` — a positive squared-group-dot score with sum
normalization (denominator tracked, no softmax, no epsilon in the safe
division).

The optional data-dependent offset gate ``(<q,k> + a_t·a_s)²``, the GDN-style
decay ``A·exp(min(c_t − c_s, 0))``, qk-norm and the quadratic_sum variant are
residual external variants, not claimed. The Q/K/V projections and group split
are external; the mixer is the typed K1 squared-sum call.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class KATALayer(torch.nn.Module):
    """One KATA-SPD layer: typed K1 squared-sum call (group decomposed)."""

    def __init__(self, num_heads: int, head_dim: int, num_groups: int = 4, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        # Pinned scale = 1/sqrt(E) with E = head_dim/num_groups (per-group width).
        import math
        self.scale = 1.0 / math.sqrt(head_dim // num_groups)
        document = {
            "schema_version": 2, "name": "kata", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "scale", "dtype": "float32", "shape": []},
                ],
                "nodes": [
                    {
                        "id": "attn", "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "scale"], "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "sequence",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": True, "head_map": "equal",
                            "reducer_law": "squared_sum",
                            "squared_sum_groups": num_groups,
                            "scale_rule": "explicit_operand",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "scale": "scale"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scale = torch.tensor(self.scale, dtype=torch.float32, device=query.device)
        return self._plan.execute(query=query, key=key, value=value, scale=scale)["output"]


__all__ = ["KATALayer"]
