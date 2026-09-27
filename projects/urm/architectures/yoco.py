"""External model module: YOCO (arch-052).

Verified composition row: two decoders — the self-decoder is Simple-GLA (U2.A:
``S_t = exp(g_t) S_{t-1} + k_t v_t``, head-scalar data-dependent gate
``gk = logsigmoid(gk_proj(x))/gate_logit_normalizer``, after-update read), and
the cross-decoder is softmax attention (U1.S) of per-token queries against a
SHARED KV cache built once by YOCOSharedKVBuilder — verified against
fla/layers/yoco.py @ 864a87f6. The shared-KV cache construction (YOCOSharedKVBuilder
with its kv_norm) and rotary are external; the self-decoder runs the typed K2
graph, the cross-decoder the typed K1 graph. The cache/decode and windowed paths
are residual external work (the sweep also notes K2 gate-VJP concerns for the
cross-decoder).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer
from architectures.head_map_attention import HeadMapAttention


class YOCOSelfDecoder(K2LinearStateLayer):
    """Self-decoder: Simple-GLA (U2.A head-scalar gate)."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, gate_logit_normalizer: int = 8, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="head", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.gate_logit_normalizer = gate_logit_normalizer
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.gk_proj = torch.nn.Linear(hidden_size, num_heads, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        g = F.logsigmoid(self.gk_proj(hidden_states)) / self.gate_logit_normalizer
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


class YOCOCrossDecoder(torch.nn.Module):
    """Cross-decoder: softmax attention over a shared KV cache (U1.S)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        from urm.compiler.normalize.graph import normalize_graph_document
        from urm.compiler.pipeline import CompilationIntent, compile_graph
        from urm.frontend.recipes import load_graph_recipe_document

        document = {
            "schema_version": 2,
            "name": "yoco_cross_attention",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32", "shape": ["B", "T", num_heads, head_dim]},
                    {"name": "key", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "value", "dtype": "float32", "shape": ["B", "S", num_heads, head_dim]},
                    {"name": "attention_mask", "dtype": "bool", "shape": ["B", "T", "S"]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value", "attention_mask"],
                        "outputs": ["output"],
                        "params": {
                            "query_domain": "sequence", "source_domain": "sequence",
                            "selection": "dense", "normalization": "softmax",
                            "capacity_policy": "dropless", "deterministic": True,
                            "causal": False, "head_map": "equal",
                            "roles": {"query": "query", "key": "key", "value": "value",
                                      "attention_mask": "attention_mask"},
                        },
                    }
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        self._plan = compile_graph(
            normalize_graph_document(recipe.document),
            target=target, intent=CompilationIntent(intent),
        )

    def forward(self, hidden_states: torch.Tensor, shared_k: torch.Tensor,
                shared_v: torch.Tensor) -> torch.Tensor:
        """``hidden_states`` ``[B, T, E]``; ``shared_k/shared_v`` ``[B, S, H, D]``."""
        B, T, _ = hidden_states.shape
        S = shared_k.shape[1]
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        # Offset-causal alignment (pinned seqlen_offset = S − T): query i at
        # cache position S−T+i attends to keys 0..S−T+i.
        offset = S - T
        q_idx = torch.arange(T, device=hidden_states.device).unsqueeze(1)
        kv_idx = torch.arange(S, device=hidden_states.device).unsqueeze(0)
        mask = (kv_idx <= (q_idx + offset)).unsqueeze(0).expand(B, T, S)
        out = self._plan.execute(
            query=q, key=shared_k, value=shared_v, attention_mask=mask
        )["output"]
        return self.o_proj(out.reshape(B, T, self.num_heads * self.head_dim))


__all__ = ["YOCOSelfDecoder", "YOCOCrossDecoder"]
