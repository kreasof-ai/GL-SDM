"""External model module: Conformer relative-position attention (arch-075).

Verified composition-now row: the mixer is exactly noncausal U1.S with an
additive score bias — softmax([(q+u)·kᵀ + rel_shift((q+v)·pᵀ)]/√d)·v — and the
relative-position frontend (linear_pos, pos_bias_u/v, rel_shift) plus the
Conformer block's macaron FFN / depthwise conv / norms / residuals are external
stages (sweep row-075, verified against
espnet2/asr_transducer/encoder/modules/attention.py @ 2950325e).

Composition: the typed K1 ``weighted_reduce`` call receives
``query = linear_q(x) + pos_bias_u`` and the external additive bias
``score_bias = rel_shift((linear_q(x) + pos_bias_v) · linear_pos(pos_enc)ᵀ)/√d``
(built once per forward, shape [B, H, T, T]); the K1 node's key-dim scale rule
matches the pinned ``(matrix_ac + matrix_bd) / sqrt(d_k)`` exactly. The
all-masked-row-zero policy of the K1 contract matches the pinned
``softmax(...).masked_fill(mask, 0.0)``.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Pinned rel_shift: (B, H, T, 2T-1) → (B, H, T, T), a strided reindex."""
    batch_size, n_heads, time1, n = x.shape
    time2 = time1
    batch_stride, n_heads_stride, time1_stride, n_stride = x.stride()
    return x.as_strided(
        (batch_size, n_heads, time1, time2),
        (batch_stride, n_heads_stride, time1_stride - n_stride, n_stride),
        storage_offset=(n_stride * (time1 - 1)),
    )


class ConformerRelPosAttention(torch.nn.Module):
    """One Conformer rel-pos attention layer (full u/v-bias variant).

    External stages: linear_q/k/v/out projections, linear_pos, pos_bias_u/v,
    rel_shift. Mixer: one typed K1 call (noncausal, equal head map, key-dim
    scale, additive score_bias). Streaming/chunk masks and the conv/FFN block
    wrapper are residual external work (sweep blocker), not claimed.
    """

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.d_k = embed_size // num_heads

        self.linear_q = torch.nn.Linear(embed_size, embed_size)
        self.linear_k = torch.nn.Linear(embed_size, embed_size)
        self.linear_v = torch.nn.Linear(embed_size, embed_size)
        self.linear_out = torch.nn.Linear(embed_size, embed_size)
        self.linear_pos = torch.nn.Linear(embed_size, embed_size, bias=False)
        self.pos_bias_u = torch.nn.Parameter(torch.zeros(num_heads, self.d_k))
        self.pos_bias_v = torch.nn.Parameter(torch.zeros(num_heads, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        document = {
            "schema_version": 2,
            "name": "conformer_relpos_attention",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", num_heads, self.d_k]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", num_heads, self.d_k]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", num_heads, self.d_k]},
                    {"name": "score_bias", "dtype": "float32",
                     "shape": ["B", num_heads, "T", "S"]},
                    {"name": "attention_mask", "dtype": "bool", "shape": ["B", "S"]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "score_bias", "attention_mask"],
                        "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence",
                            "source_domain": "sequence",
                            "selection": "dense",
                            "normalization": "softmax",
                            "capacity_policy": "dropless",
                            "deterministic": True,
                            "causal": False,
                            "head_map": "equal",
                            "roles": {
                                "query": "query",
                                "key": "key",
                                "value": "value",
                                "score_bias": "score_bias",
                                "attention_mask": "attention_mask",
                            },
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

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                pos_enc: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pinned signature: q/k/v ``[B, T, E]``, pos_enc ``[B, 2T-1, E]``,
        mask ``[B, S]`` (True = masked out; None = nothing masked)."""
        B = query.size(0)
        T1 = query.size(1)
        # External projections to the K1 boundary layout [B, T, H, d_k].
        q = self.linear_q(query).view(B, -1, self.num_heads, self.d_k)
        k = self.linear_k(key).view(B, -1, self.num_heads, self.d_k)
        v = self.linear_v(value).view(B, -1, self.num_heads, self.d_k)

        # External relative-position bias: rel_shift((q+pos_bias_v)·pᵀ)/√d,
        # computed in [B, H, T, S] (the score layout).
        p = self.linear_pos(pos_enc).view(pos_enc.size(0), -1, self.num_heads, self.d_k)
        q_h = q.transpose(1, 2)  # [B, H, T, d_k]
        q_with_bias_v = q_h + self.pos_bias_v.unsqueeze(1)
        matrix_bd = torch.matmul(q_with_bias_v, p.permute(0, 2, 3, 1))
        score_bias = _rel_shift(matrix_bd) / (self.d_k ** 0.5)

        # Mixer: q carries pos_bias_u; the K1 key-dim scale completes 1/√d.
        # The pinned mask is "True = masked out"; the K1 attention_mask role is
        # "True = visible", so invert it here.
        q_with_bias_u = q + self.pos_bias_u
        visible = ~mask if mask is not None else None
        out = self._plan.execute(
            query=q_with_bias_u, key=k, value=v, score_bias=score_bias,
            attention_mask=visible,
        )["output"]
        return self.linear_out(out.reshape(B, T1, self.num_heads * self.d_k))


__all__ = ["ConformerRelPosAttention"]
