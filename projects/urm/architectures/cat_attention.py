"""External model module: CAT (Compress And Attend) decoder attention (arch-066).

Verified composition-now row: the decoder self-attention mixer is exactly U1.S
(causal softmax) over an explicit structural source set — queries attend
causally within their local block AND to prior compressed tokens at
``kv_idx % block_size == 0`` — carried as the K1 ``attention_mask`` role; the
chunk compressor, adaptive/separator tokens, rotary transform and Q/K/V
projections are external (sweep row-066, verified against
fla/models/cat/modeling_cat.py @ 864a87f6). Efficient sparse traversal of the
structural mask is gated on the indexed-K1 schedule (B[indexed_K1_schedule]);
this module exercises the dense-mask correctness oracle.

Composition: Q/K/V projections (GQA-capable), rotary and optional qk-norm are
external; the structural CAT mask ``(within_block | compressed_token) & causal``
is built externally per forward; the typed K1 ``weighted_reduce`` graph
(causal semantics already encoded in the mask, so the node's causal field is
off and the mask carries the full structural constraint) executes through the
public compile path; the output projection is external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def cat_structural_mask(seq_len: int, block_size: int, device=None) -> torch.Tensor:
    """The pinned CAT mask: (within_block | compressed_token) & causal.

    ``mask[q, kv]`` is True where query ``q`` may attend to key ``kv``:
    keys in the same block as the query, or compressed chunk tokens at
    ``kv % block_size == 0``, restricted to causal visibility.
    """
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    within_block = (q_idx // block_size) == (kv_idx // block_size)
    compressed_token = (kv_idx % block_size) == 0
    causal = q_idx >= kv_idx
    return (within_block | compressed_token) & causal


class CATDecoderAttention(torch.nn.Module):
    """One CAT decoder attention layer: external projections/rotary + typed K1.

    The compressor (CATCompressor) and architectural-token construction are
    separate external stages (sweep blocker) and are not part of this layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
        num_kv_heads: int | None = None,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size

        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        if num_kv_heads == num_heads:
            head_map = "equal"
        elif num_kv_heads == 1:
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
            # The structural mask already encodes causality; the node's causal
            # field stays off so the mask carries the full constraint.
            "causal": False,
            "head_map": head_map,
            "roles": {
                "query": "query",
                "key": "key",
                "value": "value",
                "attention_mask": "attention_mask",
            },
        }
        if head_map == "grouped":
            params["group_size"] = num_heads // num_kv_heads

        document = {
            "schema_version": 2,
            "name": "cat_decoder_attention",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", num_heads, self.head_dim]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", num_kv_heads, self.head_dim]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", num_kv_heads, self.head_dim]},
                    {"name": "attention_mask", "dtype": "bool", "shape": ["B", "T", "S"]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "attention_mask"],
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``[B, T, hidden_size]`` → same, over the CAT structural mask."""
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        mask = cat_structural_mask(T, self.block_size, device=hidden_states.device)
        mask = mask.unsqueeze(0).expand(B, T, T)
        out = self._plan.execute(query=q, key=k, value=v, attention_mask=mask)["output"]
        return self.o_proj(out.reshape(B, T, self.num_heads * self.head_dim))


__all__ = ["CATDecoderAttention", "cat_structural_mask"]
