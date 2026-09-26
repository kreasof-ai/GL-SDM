"""External model module: MLA (Multi-head Latent Attention, arch-004).

Verified composition-now row: the mixer is plain causal U1.S over externally
expanded latent heads; the low-rank latent projection/expansion (kv_lora_rank
latent + RMSNorm + per-head k_nope/v expansion, a shared rope key broadcast to
all heads, q split into nope+rope parts) is an ordinary external op (sweep
row-004, against fla/layers/mla.py @ 864a87f6). The pinned source's TODO —
caching only compressed_kv + k_rot and recovering the full k/v — is a recorded
blocker (compressed-cache equality), not claimed here; this module caches the
expanded heads like the pinned prefill path.

Composition: latent q/k/v projections, RMSNorms, rotary on the rope head-dims
and the per-head expansion are external; the typed K1 ``weighted_reduce``
graph (causal, equal head map, key-dim scale = qk_head_dim^-0.5 — the pinned
``self.scaling``) executes through the public compile path; v is padded to
qk_head_dim for the mixer and the output cropped back to v_head_dim, matching
the pinned flash-attn ABI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class MLALayer(torch.nn.Module):
    """One MLA layer: latent projections + rotary + typed causal K1 mixer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        qk_nope_head_dim: int,
        q_lora_rank: int | None = None,
        *,
        rope_theta: float = 10000.0,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.rope_theta = rope_theta

        if q_lora_rank is not None:
            self.q_proj = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, q_lora_rank, bias=False),
                torch.nn.RMSNorm(q_lora_rank),
                torch.nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False),
            )
        else:
            self.q_proj = torch.nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
        self.k_rope = torch.nn.Linear(hidden_size, qk_rope_head_dim, bias=False)
        self.kv_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, kv_lora_rank, bias=False),
            torch.nn.RMSNorm(kv_lora_rank),
            torch.nn.Linear(
                kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
            ),
        )
        self.o_proj = torch.nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        document = {
            "schema_version": 2,
            "name": "mla_mixer",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", num_heads, self.qk_head_dim]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", num_heads, self.qk_head_dim]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", num_heads, self.v_head_dim]},
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
                            "causal": True,
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

    def _rotary(self, x: torch.Tensor, base: float) -> torch.Tensor:
        """Non-interleaved rotary on ``[..., T, H, D]`` over the last dim."""
        *_, T, H, D = x.shape
        d = D // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, device=x.device).float() / D))
        t = torch.arange(T, device=x.device)
        freqs = torch.outer(t, inv_freq)  # [T, D/2]
        cos = freqs.cos()[None, :, None, :].to(x.dtype)
        sin = freqs.sin()[None, :, None, :].to(x.dtype)
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        q_states = self.q_proj(hidden_states).view(B, T, self.num_heads, self.qk_head_dim)
        q_pass, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_pass, k_rot = self.kv_proj(hidden_states), self.k_rope(hidden_states)
        k_rot = k_rot.view(B, T, 1, self.qk_rope_head_dim)
        k_pass = k_pass.view(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Rotary on the rope parts (external); the shared rope key broadcasts.
        q_rot = self._rotary(q_rot, self.rope_theta)
        k_rot = self._rotary(k_rot, self.rope_theta)
        k_rot = k_rot.expand(B, T, self.num_heads, self.qk_rope_head_dim)

        q = torch.cat((q_pass, q_rot), dim=-1)
        k = torch.cat((k_pass, k_rot), dim=-1)
        # v stays at its natural v_head_dim — the K1 kernel handles key_dim ≠
        # value_dim natively (the pinned zero-padding is a semantic no-op).

        out = self._plan.execute(query=q, key=k, value=v)["output"]
        return self.o_proj(out.reshape(B, T, self.num_heads * self.v_head_dim))


__all__ = ["MLALayer"]
