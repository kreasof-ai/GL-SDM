"""External model module: Differential Attention (arch-067, Diff Transformer V1).

Verified combinator row: the Diff combinator — two U1.S calls + a typed
λ-weighted merge — verified against Diff-Transformer/multihead_diffattn.py @
50224e38 (unilm pin). The pinned V1 computes one softmax over the concatenated
2H map then splits; the admitted base plan is two typed K1 calls (softmax over
each half) merged as ``attn1 − λ_full·attn2`` — the same equation, unfused
(the V2 paired-head one-call fusion is a later proven rewrite, separately
gated).

The λ frontend is external: ``λ_full = exp(λq1·λk1) − exp(λq2·λk2) + λ_init``
with ``λ_init = 0.8 − 0.6·exp(−0.3·depth)``, carried as a runtime scale
operand on the merge. The output RMSNorm (subln over 2·head_dim) and the
(1 − λ_init) rescale are external post-stages; rotary and Q/K/V projections
are external pre-stages.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def lambda_init_fn(depth: int) -> float:
    """Pinned lambda_init_fn: 0.8 − 0.6·exp(−0.3·depth)."""
    return 0.8 - 0.6 * (2.718281828459045 ** (-0.3 * depth))


class DifferentialAttentionLayer(torch.nn.Module):
    """One Diff V1 layer: external projections/λ frontend + two typed K1 calls + typed merge."""

    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, depth: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lambda_init = lambda_init_fn(depth)

        # Pinned shapes: q/k project to 2H halves, v to 2·head_dim per head.
        self.q_proj = torch.nn.Linear(embed_dim, 2 * num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(embed_dim, 2 * num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(embed_dim, num_heads * 2 * head_dim, bias=False)
        self.out_proj = torch.nn.Linear(num_heads * 2 * head_dim, embed_dim, bias=False)
        self.lambda_q1 = torch.nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = torch.nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = torch.nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = torch.nn.Parameter(torch.zeros(head_dim).normal_(mean=0, std=0.1))
        self.subln = torch.nn.RMSNorm(2 * head_dim, eps=1e-5)

        # Graph: two typed K1 calls (one per softmax half) + typed merge.
        def k1(node_id, qi, ki, out_name):
            return {
                "id": node_id, "op": "weighted_reduce", "inputs": [qi, ki, "v"],
                "outputs": [out_name],
                "params": {
                    "query_domain": "sequence", "source_domain": "sequence",
                    "selection": "dense", "normalization": "softmax",
                    "capacity_policy": "dropless", "deterministic": True,
                    "causal": True, "head_map": "equal",
                    "roles": {"query": qi, "key": ki, "value": "v"},
                },
            }
        document = {
            "schema_version": 2, "name": "differential_attention", "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "q1", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "k1", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "q2", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "k2", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "v", "dtype": "float32", "shape": ["B", "S", num_heads, 2 * head_dim]},
                    {"name": "lambda_full", "dtype": "float32", "shape": []},
                ],
                "nodes": [
                    k1("attn1", "q1", "k1", "o1"),
                    k1("attn2", "q2", "k2", "o2"),
                    {"id": "merge", "op": "merge", "inputs": ["o1", "o2"], "outputs": ["output"],
                     "params": {"coefficients": [1.0, -1.0], "scale_operands": ["", "lambda_full"]}},
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(program, target=target, intent=CompilationIntent(intent))

    def _lambda_full(self) -> torch.Tensor:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1))
        return lambda_1 - lambda_2 + self.lambda_init

    def forward(self, x: torch.Tensor, rel_pos: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        B, T, _ = x.size()
        H, D, D2 = self.num_heads, self.head_dim, 2 * self.head_dim
        q = self.q_proj(x).view(B, T, 2 * H, D)
        k = self.k_proj(x).view(B, T, 2 * H, D)
        v = self.v_proj(x).view(B, T, H, D2)

        # External rotary (interleaved/GPT-J style, matching the pinned call).
        if rel_pos is not None:
            q = _apply_rotary_interleaved(q, *rel_pos)
            k = _apply_rotary_interleaved(k, *rel_pos)

        # Pinned split: the 2H head dim is viewed as [H, 2] — half 0 is the even
        # head indices, half 1 the odd (interleaved), NOT first-H/second-H.
        q = q.view(B, T, H, 2, D)
        k = k.view(B, T, H, 2, D)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]   # [B,T,H,D] each
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

        out = self._plan.execute(
            q1=q1, k1=k1, q2=q2, k2=k2, v=v, lambda_full=self._lambda_full()
        )["output"]  # [B,T,H,2D]

        # External post-stages: subln over 2·head_dim, (1−λ_init) rescale, out_proj.
        attn = self.subln(out)
        attn = attn * (1 - self.lambda_init)
        return self.out_proj(attn.reshape(B, T, H * D2))


def _apply_rotary_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-J style (interleaved) rotary on ``[B, T, H, D]`` with cos/sin ``[T, D/2]``."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., 0::2], x[..., 1::2]  # even, odd
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


__all__ = ["DifferentialAttentionLayer", "lambda_init_fn"]
