"""External model module: Samba attention branch (arch-053).

Verified composition-now row: the attention branch is plain causal U1.S with a
GQA-capable head map (``n_query_groups``) and external RoPE — verified against
lit_gpt/model.py CausalSelfAttention @ 617c7a0f. The per-layer mixer selection
(config use_mamba/use_retnet/use_gla schedules) and the ``mamba_swa_mlp``
Mamba-then-attention block mode are gated on the Mamba selective-SSM axis (A8)
and are NOT claimed here; this module is the attention branch only.

Composition: the batched Q/K/V projection, RoPE (transcribed pinned
build_rope_cache + apply_rotary_emb_func), and output projection are external;
the typed K1 ``weighted_reduce`` graph (causal, head map from
n_head/n_query_groups) executes through the public compile path. The optional
short-conv (``sc_attn``) and KV-cache/decode paths of the pinned source are
residual external work, not claimed.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcribed from the pinned lit_gpt/model.py build_rope_cache."""
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))
    seq_idx = torch.arange(seq_len, device=device)
    idx_theta = torch.outer(seq_idx, theta)
    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary on ``[B, T, H, D]`` with cos/sin ``[T, D/2]`` (pinned non-interleaved)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[None, :, None, :].to(x.dtype)
    sin = sin[None, :, None, :].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SambaAttentionLayer(torch.nn.Module):
    """Samba attention branch: batched QKV + RoPE + typed causal K1 mixer + out proj."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_size: int,
        n_query_groups: int | None = None,
        *,
        rope_base: int = 10000,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if n_query_groups is None:
            n_query_groups = n_head
        if n_head % n_query_groups != 0:
            raise ValueError("n_head must be divisible by n_query_groups")
        if n_embd != n_head * head_size:
            raise ValueError("n_embd must equal n_head * head_size")
        self.n_head = n_head
        self.n_query_groups = n_query_groups
        self.head_size = head_size
        self.rope_base = rope_base

        shape = (n_head + 2 * n_query_groups) * head_size
        self.attn = torch.nn.Linear(n_embd, shape, bias=False)
        self.proj = torch.nn.Linear(n_embd, n_embd, bias=False)

        if n_query_groups == n_head:
            head_map = "equal"
        elif n_query_groups == 1:
            head_map = "single"
        else:
            head_map = "grouped"
        params: dict[str, object] = {
            "query_domain": "sequence",
            "source_domain": "sequence",
            "selection": "dense",
            "normalization": "softmax",
            "capacity_policy": "dropless",
            "deterministic": True,
            "causal": True,
            "head_map": head_map,
        }
        if head_map == "grouped":
            params["group_size"] = n_head // n_query_groups

        document = {
            "schema_version": 2,
            "name": "samba_attention_branch",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", n_head, head_size]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", n_query_groups, head_size]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", n_query_groups, head_size]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value"],
                        "outputs": ["output"],
                        "params": params,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        q_per_kv = self.n_head // self.n_query_groups
        qkv = self.attn(x).view(B, T, self.n_query_groups, q_per_kv + 2, self.head_size)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, self.n_head, self.head_size)
        k = k.reshape(B, T, self.n_query_groups, self.head_size)
        v = v.reshape(B, T, self.n_query_groups, self.head_size)

        cos, sin = build_rope_cache(T, self.head_size, q.dtype, q.device, base=self.rope_base)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        out = self._plan.execute(query=q, key=k, value=v)["output"]
        return self.proj(out.reshape(B, T, -1))


__all__ = ["SambaAttentionLayer", "apply_rotary_emb", "build_rope_cache"]
