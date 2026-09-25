"""External model module: DeltaFormer (arch-013) / UT strict-causal triangular transform.

UT (typed causal triangular transform) client. Verified against
fla/ops/deltaformer/naive.py @ 864a87f6. Two stages, both through the public graph:

1. Strict-past triangular value correction — the typed ``triangular_solve`` op
   (UT). With ``P = tril_softmax(q·kᵀ/√D, strict=True)`` (rows over j<i only),
   ``u = (I + diag(β)·strict_tril(P))^{-1}·v`` by forward substitution
   (``u_t = v_t − β_t·Σ_{j<t} P_tj·u_j``). This is a data-dependent, read-
   dependent correction the canonical K2 rank-1 law cannot express — the write
   ``u_t`` feeds later reads. The strict-lower ``P`` (softmax over j<i) is
   computed externally; the solve is the typed public op.
2. Ordinary causal softmax attention ``o = softmax(q·kᵀ/√D, causal incl.
   diagonal) @ u`` — the typed K1 call on the corrected values.

The strict-past correction (stage 1) and its gradients are preserved; stage 2
includes the diagonal. The β frontend, the strict-lower softmax P, and the
chunked/kernel forms are external. ``strict_causal_value_correction`` is retained
as the independent serial comparator that the parity gate checks the public op
against.

Layout: the solve runs head-first ``[B,H,T,D]`` (matching the pinned naive); the
K1 attention runs in its ``[B,T,H,D]`` layout; the module transposes between them
at the boundary (external ordinary ops). The two stages are two separate compiled
fragments, each a single typed op — this is the unfused reference composition.
"""

from __future__ import annotations

import math

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def strict_tril_softmax(q, k):
    """P = softmax(q·kᵀ/√D) masked to the STRICT lower triangle (j < i only).

    q/k [B,H,T,D]. External: produces the strict-lower probability operand the
    triangular_solve op consumes. fp32.
    """
    B, H, T, D = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (1.0 / math.sqrt(D))
    strict = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=0)
    scores = scores.masked_fill(strict, float("-inf"))
    P = torch.softmax(scores, dim=-1)
    return torch.nan_to_num(P, nan=0.0)


def strict_causal_value_correction(q, k, v, beta):
    """Independent serial comparator: u = (I + diag(β)·strict_tril(P))^{-1}·v.

    q/k/v [B,H,T,D], beta [B,H,T]. Forward substitution, fp32 — the direct
    transcription of the pinned naive_deltaformer stage-1 recurrence.
    """
    P = strict_tril_softmax(q, k)
    B, H, T, D = q.shape
    vf = v.float()
    betaf = beta.float() if beta is not None else torch.ones(B, H, T, device=q.device)
    us = []
    for t in range(T):
        if t == 0:
            us.append(vf[:, :, 0])
        else:
            w = P[:, :, t, :t]
            u_prev = torch.stack(us, dim=-2)
            us.append(vf[:, :, t] - betaf[:, :, t].unsqueeze(-1) * (w.unsqueeze(-1) * u_prev).sum(-2))
    return torch.stack(us, dim=2)


class DeltaFormerLayer(torch.nn.Module):
    """DeltaFormer: typed triangular_solve value correction (UT) + typed K1 over corrected values."""

    def __init__(self, num_heads: int, head_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        H, D = num_heads, head_dim
        intent_e = CompilationIntent(intent)
        # Stage 1: the UT triangular solve (u = (I + diag(β)·strict_tril(P))^{-1}·v),
        # head-first [B,H,T,D].
        solve_doc = {
            "schema_version": 2, "name": "deltaformer_value_solve", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "probs", "dtype": "float32", "shape": ["B", "H", "T", "T"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", D]},
                ],
                "nodes": [
                    {
                        "id": "value_solve", "op": "triangular_solve",
                        "inputs": ["probs", "beta", "value"], "outputs": ["u"],
                        "params": {"roles": {"probs": "probs", "beta": "beta", "value": "value"}},
                    },
                ],
                "outputs": ["u"],
            },
        }
        self._solve = compile_graph(
            normalize_graph_document(load_graph_recipe_document(solve_doc).document),
            target=target, intent=intent_e,
        )
        # Stage 2: causal softmax attention over the corrected values u, [B,T,H,D].
        attn_doc = {
            "schema_version": 2, "name": "deltaformer_attn", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", H, D]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", H, D]},
                    {"name": "u", "dtype": "float32", "shape": ["B", "S", H, D]},
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
        self._attn = compile_graph(
            normalize_graph_document(load_graph_recipe_document(attn_doc).document),
            target=target, intent=intent_e,
        )

    def correct_values(self, q, k, v, beta=None):
        """Stage 1 only (UT triangular solve), head-first. Exposed for the parity gate."""
        B, H, T, D = q.shape
        P = strict_tril_softmax(q, k)
        betaf = beta.float() if beta is not None else torch.ones(B, H, T, device=q.device)
        return self._solve.execute(probs=P, beta=betaf, value=v.float())["u"]

    def forward(self, q, k, v, beta=None):
        """q/k/v [B,H,T,D] (head-first); beta [B,H,T] or None → output [B,H,T,D]."""
        u_hf = self.correct_values(q, k, v, beta)                    # stage 1 (UT solve) [B,H,T,D]
        out = self._attn.execute(                                    # stage 2 (K1) over [B,T,H,D]
            query=q.transpose(1, 2).float(),
            key=k.transpose(1, 2).float(),
            u=u_hf.transpose(1, 2).float(),
        )["output"]
        return out.transpose(1, 2)                                   # back to [B,H,T,D]


__all__ = ["DeltaFormerLayer", "strict_causal_value_correction", "strict_tril_softmax"]
