"""External model module: PaTH attention (arch-010) / UT Householder operand correction.

UT (typed causal triangular transform) client — reuses the SAME public
``triangular_solve`` op admitted for arch-013 DeltaFormer. Verified against
fla/ops/path_attn/naive.py @ 864a87f6. The single-chunk form factors into public ops:

1. **Householder transform** — ``T_mat = inv(I + strict_tril(w_β wᵀ))``. This is exactly
   the public :class:`~urm.ir.program.TriangularSolve` op with ``P = w wᵀ`` (the Gram),
   ``beta = β`` and ``value`` = the identity basis: ``w_β = w·β`` folds the diagonal into
   the Gram rows, so ``strict_tril(w_β wᵀ) = diag(β)·strict_tril(w wᵀ)``. Verified at
   1.5e-8 against the pinned forward-substitution construction.
2. **Householder score correction** (external typed assembly) — the corrected local score
   ``A_local = tril(qkᵀ) − tril(qwᵀ)@(T_mat @ tril(w_β kᵀ))``. The correction
   ``A_local − tril(qkᵀ) = −tril(qwᵀ)@(T_mat @ tril(w_β kᵀ))`` is ADDITIVE, so it rides
   the K1 ``score_bias`` together with the cumulative gate.
3. **Cumulative-gate bias** (the FoX/RWKV sub-law, external cumsum) — ``scale·(gc_i − gc_j)``
   with ``gc = cumsum(g)``, the same admitted additive-score-bias algebra as FoX.
4. **Causal softmax @ v** — the public K1 call with the combined ``score_bias``.

The short-conv / l2-norm on w frontend and the grouped-KV expansion are external. The
MULTI-CHUNK progressive cross-chunk correction (the ``q_i −= q_i @ H_mat[j]`` loop over
prior chunks) is the residual PaTH-specific schedule — a blockwise right-to-left
score-transform scan with no typed K1 interface — recorded not claimed. The
chunk_size ≥ T (single-chunk) case is the public path here.
"""

from __future__ import annotations

import math

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def _householder_T(w_beta: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """T_mat = I + inv(I + tril(w_β wᵀ, −1)) − I via forward substitution (per chunk).

    w_beta/w [..., C, D]. Returns [..., C, C]. Retained as the independent serial
    comparator that the public triangular_solve path is checked against.
    """
    C = w.shape[-2]
    mask = torch.triu(torch.ones(C, C, dtype=torch.bool, device=w.device), diagonal=0)
    T_mat = -(w_beta @ w.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, C):
        T_mat[..., i, :i] = T_mat[..., i, :i] + (
            T_mat[..., i, :, None] * T_mat[..., :, :i]
        ).sum(-2)
    return T_mat + torch.eye(C, dtype=w.dtype, device=w.device)


class PaTHAttentionLayer(torch.nn.Module):
    """PaTH mixer (single chunk): public triangular_solve Householder + public K1 gated softmax.

    The Householder ``T_mat`` is computed by the public ``triangular_solve`` op; the
    score correction + cumulative-gate bias are assembled externally and ride the K1
    ``score_bias``; the causal softmax @ v is the public K1 call.
    """

    def __init__(self, num_heads: int, head_dim: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        intent_e = CompilationIntent(intent)
        # Public op 1: the Householder triangular solve. probs = w wᵀ (Gram) [B,H,T,T];
        # beta [B,H,T]; value = identity basis columns [B,H,T,T] (D_solve = T).
        solve_doc = {
            "schema_version": 2, "name": "path_householder_solve", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "probs", "dtype": "float32", "shape": ["B", "H", "T", "T"]},
                    {"name": "beta", "dtype": "float32", "shape": ["B", "H", "T"]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "H", "T", "T"]},
                ],
                "nodes": [
                    {
                        "id": "solve", "op": "triangular_solve",
                        "inputs": ["probs", "beta", "value"], "outputs": ["T_mat"],
                        "params": {"roles": {"probs": "probs", "beta": "beta", "value": "value"}},
                    },
                ],
                "outputs": ["T_mat"],
            },
        }
        self._solve = compile_graph(
            normalize_graph_document(load_graph_recipe_document(solve_doc).document),
            target=target, intent=intent_e,
        )
        # Public op 2: the causal softmax K1 with an additive score bias.
        HQ = num_heads
        D = head_dim
        attn_doc = {
            "schema_version": 2, "name": "path_attn", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", HQ, D]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", HQ, D]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", HQ, D]},
                    {"name": "score_bias", "dtype": "float32", "shape": ["B", HQ, "T", "S"]},
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
        self._attn = compile_graph(
            normalize_graph_document(load_graph_recipe_document(attn_doc).document),
            target=target, intent=intent_e,
        )

    def householder_T(self, w, beta):
        """Public triangular_solve Householder transform. w [B,T,H,D], beta [B,T,H].

        Returns T_mat [B,H,T,T] (head-first, matching the pinned construction). Exposed
        for the parity gate against the serial comparator.
        """
        B, T, H, D = w.shape
        wf = w.permute(0, 2, 1, 3).float()                       # [B,H,T,D]
        betaf = beta.permute(0, 2, 1).float()                    # [B,H,T]
        P = wf @ wf.transpose(-1, -2)                            # Gram [B,H,T,T]
        eye = torch.eye(T, device=w.device).expand(B, H, T, T)   # identity basis columns
        return self._solve.execute(probs=P, beta=betaf, value=eye.contiguous())["T_mat"]

    def forward(self, q, k, v, w, beta, g, scale):
        """Single-chunk (chunk_size >= T) path. q/k/v/w [B,T,H,D]; beta [B,T,H]; g [B,T,HQ]."""
        B, T, HQ, D = q.shape
        H = k.shape[2]
        G = HQ // H
        # expand shared KV/w/beta to the query heads (GQA), external
        def exp(x):
            return (x.unsqueeze(3).expand(B, T, H, G, x.shape[-1]).flatten(2, 3) if x.dim() == 4
                    else x.unsqueeze(3).expand(B, T, H, G).flatten(2, 3))
        k, v, w = exp(k), exp(v), exp(w)
        beta = beta.unsqueeze(3).expand(B, T, H, G).flatten(2, 3)
        g_cumsum = g.cumsum(1)                                    # [B,T,HQ]

        # 1. Householder T_mat via the public triangular_solve op (per expanded head).
        wf = w.permute(0, 2, 1, 3).float()                        # [B,HQ,T,D]
        betaf = beta.permute(0, 2, 1).float()                     # [B,HQ,T]
        P = wf @ wf.transpose(-1, -2)
        eye = torch.eye(T, device=q.device).expand(B, HQ, T, T).contiguous()
        T_mat = self._solve.execute(probs=P, beta=betaf, value=eye)["T_mat"]  # [B,HQ,T,T]

        # 2. Householder score correction (external typed assembly), head-first.
        qf = q.permute(0, 2, 1, 3).float()
        kf = k.permute(0, 2, 1, 3).float()
        wbf = (wf * betaf.unsqueeze(-1))                          # w_β [B,HQ,T,D]
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), diagonal=0)
        Twbk = T_mat @ (wbf @ kf.transpose(-1, -2)).masked_fill(upper, 0)
        qw = (qf @ wf.transpose(-1, -2)).tril()
        correction = -(qw @ Twbk)                                 # A_local − tril(qkᵀ) [B,HQ,T,T]

        # 3. Cumulative-gate bias (FoX sub-law, external cumsum).
        gc_hf = g_cumsum.permute(0, 2, 1)                         # [B,HQ,T]
        gc_bias = gc_hf.unsqueeze(-1) - gc_hf.unsqueeze(-2)       # gc_i − gc_j [B,HQ,T,T]

        # 4. Public K1 causal softmax with the combined additive score bias.
        score_bias = (correction + gc_bias) * scale
        out = self._attn.execute(query=q.float(), key=k.float(), value=v.float(),
                                 score_bias=score_bias)["output"]
        return out.to(q.dtype)                                    # [B,T,HQ,D]


__all__ = ["PaTHAttentionLayer"]
