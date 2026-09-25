"""External model module: TDA (Threshold Differential Attention, arch-068).

A13 score/reducer client — the THRESHOLD_RELU_POWER reducer law — plus the Diff
combinator. Verified against the pinned tda source (triton_threshold_attention.py):
``out = (ReLU(Q@K^T − τ))^p @ V`` with position-dependent threshold
``τ_i = β·sqrt(2·log(i+1)/d)``, causal mask zeroing j>i before the threshold, and
NO normalization denominator.

The differential variant: optionally L2-normalize q1,k1,q2,k2 (cosine), compute
two threshold-ReLU paths out1/out2, then ``out = out1 − clamp(λ,0,1)·out2`` —
the typed Diff combinator (two K1 threshold calls + the admitted Merge op). The
Q/K/V projections, optional cosine L2-normalization and the λ clamp are external.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def _k1_threshold(node_id, qi, ki, out_name, beta, power):
    return {
        "id": node_id, "op": "weighted_reduce", "inputs": [qi, ki, "value"],
        "outputs": [out_name],
        "params": {
            "query_domain": "sequence", "source_domain": "sequence",
            "selection": "dense", "normalization": "softmax",
            "capacity_policy": "dropless", "deterministic": True,
            "causal": True, "head_map": "equal",
            "reducer_law": "threshold_relu_power",
            "threshold_beta": beta, "relu_power": power,
            "roles": {"query": qi, "key": ki, "value": "value"},
        },
    }


class TDALayer(torch.nn.Module):
    """One TDA layer: typed K1 threshold-ReLU-power call(s) (+ Diff merge)."""

    def __init__(self, num_heads: int, head_dim: int, *,
                 beta: float = 1.0, relu_power: float = 2.0, differential: bool = True,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.differential = differential

        if differential:
            nodes = [
                _k1_threshold("path1", "q1", "k1", "o1", beta, relu_power),
                _k1_threshold("path2", "q2", "k2", "o2", beta, relu_power),
                {"id": "merge", "op": "merge", "inputs": ["o1", "o2"], "outputs": ["output"],
                 "params": {"coefficients": [1.0, -1.0], "scale_operands": ["", "lam"]}},
            ]
            inputs = [
                {"name": "q1", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                {"name": "k1", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                {"name": "q2", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                {"name": "k2", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                {"name": "lam", "dtype": "float32", "shape": []},
            ]
        else:
            nodes = [_k1_threshold("attn", "query", "key", "output", beta, relu_power)]
            inputs = [
                {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
            ]
        document = {"schema_version": 2, "name": "tda", "kind": "kernel_fragment",
                    "graph": {"inputs": inputs, "nodes": nodes, "outputs": ["output"]}}
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, query, key, value, *, query2=None, key2=None, lam: float = 0.5,
                cosine: bool = False):
        if self.differential:
            q1 = F.normalize(query, dim=-1) if cosine else query
            k1 = F.normalize(key, dim=-1) if cosine else key
            q2 = F.normalize(query2, dim=-1) if cosine else query2
            k2 = F.normalize(key2, dim=-1) if cosine else key2
            lam_t = torch.clamp(torch.as_tensor(lam, dtype=torch.float32, device=query.device), 0.0, 1.0)
            return self._plan.execute(q1=q1, k1=k1, q2=q2, k2=k2, value=value, lam=lam_t)["output"]
        return self._plan.execute(query=query, key=key, value=value)["output"]


__all__ = ["TDALayer"]
