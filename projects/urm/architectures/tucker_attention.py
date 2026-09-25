"""External model module: Tucker attention (arch-070).

Verified composition-now row: Q/K/V and the output are Tucker-factorized —
Q̃ = x·Us_pre[0], K̃ = x·Us_pre[2], Ṽ = x·Us_post[0], the score core
B_pre = Us_pre[1] · Core_pre (per-head score mixing matrix), and the output
folding out = (y·(Us_post[1]·Core_post)) · Us_post[2]ᵀ are all external
stages; the mixer is softmax attention over the core-mixed scores (sweep
row-070, verified against ViT/src/attn/tucker.py @ c3e3d3ce).

The score core B_pre[H, R, T] makes each head's score
``Q̃_h · (K̃_h · B_pre[h])ᵀ / √d`` — an external per-head mixing of the K side,
after which the typed K1 ``weighted_reduce`` graph runs with the *effective*
keys ``K̃·B_pre`` per head. The K1 key-dim scale rule matches the pinned
``sm_scale = head_dim^-0.5`` exactly (the pinned Triton kernel's scale).
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class TuckerAttentionLayer(torch.nn.Module):
    """One Tucker attention layer: factor matrices + typed K1 mixer.

    ``r_pre = (r_q, r_h, r_k)`` and ``r_post = (r_v, r_h, r_o)`` are the Tucker
    ranks; ``None`` defaults to the full (uncompressed) ranks, matching the
    pinned ``_default_ranks``.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        r_pre: tuple[int, int, int] | None = None,
        r_post: tuple[int, int, int] | None = None,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        if r_pre is None:
            r_pre = (n_embd, n_head, n_embd)
        if r_post is None:
            r_post = (n_embd, n_head, n_embd)
        r_q, r_h, r_k = r_pre
        r_v, r_h_post, r_o = r_post
        if r_h != n_head or r_h_post != n_head:
            raise ValueError("the head rank of both cores must equal n_head")

        self.Us_pre = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(n_embd, r_q)),
            torch.nn.Parameter(torch.randn(n_head, r_h)),
            torch.nn.Parameter(torch.randn(n_embd, r_k)),
        ])
        self.Core_pre = torch.nn.Parameter(torch.randn(r_q, r_h, r_k))
        self.Us_post = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(n_embd, r_v)),
            torch.nn.Parameter(torch.randn(n_head, r_h_post)),
            torch.nn.Parameter(torch.randn(n_embd, r_o)),
        ])
        self.Core_post = torch.nn.Parameter(torch.randn(r_v, r_h_post, r_o))
        self.bias_post = torch.nn.Parameter(torch.zeros(n_embd))
        self._init_factors()

        document = {
            "schema_version": 2,
            "name": "tucker_mixer",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", n_head, r_q]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", n_head, r_k]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", n_head, r_v]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value"],
                        "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence",
                            "source_domain": "sequence",
                            "selection": "dense",
                            "normalization": "softmax",
                            "capacity_policy": "dropless",
                            "deterministic": True,
                            "causal": False,  # the pinned FlashAttentionTucker is noncausal
                            "head_map": "equal",
                        },
                    }
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(
            program, target=target, intent=CompilationIntent(intent)
        )

    def _init_factors(self) -> None:
        for param in [self.Core_pre, self.Core_post, *self.Us_pre, *self.Us_post]:
            torch.nn.init.xavier_uniform_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, N, n_embd]`` → ``[B, N, n_embd]``."""
        B, N, _ = x.shape
        x_flat = x.reshape(B * N, self.n_embd)

        # External factor foldings (pinned _tucker_foldings_attn_optimized).
        q_tilde = torch.matmul(x_flat, self.Us_pre[0]).view(B, N, -1)     # [B,N,R]
        k_tilde = torch.matmul(x_flat, self.Us_pre[2]).view(B, N, -1)     # [B,N,T]
        v_tilde = torch.matmul(x_flat, self.Us_post[0]).view(B, N, -1)    # [B,N,Rv]
        R, S, T = self.Core_pre.shape
        core = self.Core_pre.permute(1, 0, 2).reshape(S, R * T)
        b_pre = torch.matmul(self.Us_pre[1], core).view(-1, R, T)         # [H,R,T]

        # The pinned kernel mixes the score core on the Q side:
        # t1[h] = Q̃·B_pre[h], scores[h] = t1[h]·K̃ᵀ (tucker_attn.py:113-161).
        # So the typed K1 call sees per-head effective queries q_eff[h] and the
        # shared K̃ (equal head map = broadcast over heads).
        q_eff = torch.einsum("bnr,hrt->bnht", q_tilde, b_pre)             # [B,N,H,T]
        k_h = k_tilde.unsqueeze(2).expand(B, N, self.n_head, T)           # [B,N,H,T]
        v_h = v_tilde.unsqueeze(2).expand(B, N, self.n_head, v_tilde.shape[-1])

        # The K1 key-dim rule scales by R^-0.5 (the q/k width after factoring),
        # matching the pinned kernel's sm_scale = key_dim^-0.5 with key_dim = R.
        out = self._plan.execute(query=q_eff, key=k_h, value=v_h)["output"]  # [B,N,H,Rv]

        # External output folding (pinned _tucker_foldings_output_optimized).
        R2, S2, T2 = self.Core_post.shape
        core_post = self.Core_post.permute(1, 0, 2).reshape(S2, R2 * T2)
        b_post_flat = torch.matmul(self.Us_post[1], core_post).reshape(self.n_head * R2, T2)
        y_bn_hr = out.permute(0, 1, 2, 3).reshape(B * N, self.n_head * R2)
        w_bn_t = torch.matmul(y_bn_hr, b_post_flat)
        y = torch.matmul(w_bn_t, self.Us_post[2].T).reshape(B, N, self.n_embd)
        return y + self.bias_post


__all__ = ["TuckerAttentionLayer"]
