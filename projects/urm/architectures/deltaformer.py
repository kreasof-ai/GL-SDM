"""External model module: DeltaFormer (arch-013) / UT strict-causal triangular transform.

UT (typed causal triangular transform) client. Verified against
fla/ops/deltaformer/naive.py @ 864a87f6. Two stages:

1. Strict-past triangular value correction: with ``P = tril_softmax(q·kᵀ/√D,
   strict=True)`` (rows over j<i only), ``u_t = v_t − β_t·Σ_{j<t} P_tj·u_j`` —
   i.e. ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` by forward substitution. This
   is the typed strict-causal triangular transform (a data-dependent operand
   correction, NOT a per-token projection — sweep verdict).
2. Ordinary causal softmax attention ``o = softmax(q·kᵀ/√D, causal incl.
   diagonal) @ u`` — the typed K1 call on the corrected values.

The strict-past correction (stage 1) and its gradients are preserved; stage 2
includes the diagonal. The β frontend and the chunked/kernel forms are external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def strict_causal_value_correction(q, k, v, beta):
    """The typed strict-causal triangular transform: u = (I + diag(β)·strict_tril(P))^{-1}·v.

    q/k/v [B,H,T,D], beta [B,H,T]. Forward substitution, fp32.
    """
    import math
    B, H, T, D = q.shape
    qf, kf, vf = q.float(), k.float(), v.float()
    betaf = beta.float() if beta is not None else torch.ones(B, H, T, device=q.device)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (1.0 / math.sqrt(D))
    # strict tril softmax (j < i only; diagonal and above excluded)
    strict_mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=0)
    scores = scores.masked_fill(strict_mask, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    us = []
    for t in range(T):
        if t == 0:
            us.append(vf[:, :, 0])
        else:
            w = P[:, :, t, :t]                                # [B,H,t]
            u_prev = torch.stack(us, dim=-2)                  # [B,H,t,D]
            us.append(vf[:, :, t] - betaf[:, :, t].unsqueeze(-1) * (w.unsqueeze(-1) * u_prev).sum(-2))
    return torch.stack(us, dim=2)                              # [B,H,T,D]


class DeltaFormerLayer(torch.nn.Module):
    """DeltaFormer: strict-causal value correction (UT transform) + typed K1 over corrected values."""

    def __init__(self, num_heads: int, head_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        document = {
            "schema_version": 2, "name": "deltaformer_attn", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "u", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                ],
                "nodes": [
                    {
                        "id": "attn", "op": "weighted_reduce",
                        "inputs": ["query", "key", "u"], "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "sequence",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": True, "head_map": "equal",
                            "roles": {"query": "query", "key": "key", "value": "u"},
                        },
                    },
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def forward(self, q, k, v, beta=None):
        """q/k/v [B,H,T,D] (head-first); beta [B,H,T] or None."""
        u = strict_causal_value_correction(q, k, v, beta)     # stage 1 (UT transform)
        # stage 2: causal softmax attention over u. The K1 graph takes [B,T,H,D].
        out = self._plan.execute(
            query=q.transpose(1, 2), key=k.transpose(1, 2), u=u.transpose(1, 2)
        )["output"]
        return out.transpose(1, 2)                            # back to [B,H,T,D]


__all__ = ["DeltaFormerLayer", "strict_causal_value_correction"]
