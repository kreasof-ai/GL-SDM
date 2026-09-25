"""External model module: TPA (Tensor Product Attention, arch-069).

Verified composition-now row: the CP-factorized QKV production (CPLinear:
A = W_A(x) per head, B = W_B(x) per rank, RoPE applied to the B_q/B_k factors,
q = bmm(A_q, B_q)/q_rank) is external; the mixer is plain causal softmax
attention (U1.S) — verified against model/T6.py @ c276c80d. The factorized
(low-rank) KV-cache ABI the decomposition enables is residual external work
(sweep blocker), not claimed here.

Composition: the CP projections + rotary are ordinary external operators; the
typed K1 ``weighted_reduce`` graph (causal, equal head map, key-dim scale —
matching the pinned ``F.scaled_dot_product_attention(..., is_causal=True)``)
is executed through the public compile path; the output projection is external.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


class _Rotary(torch.nn.Module):
    """Transcribed from the pinned T6.py Rotary (bf16 cached cos/sin)."""

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        self.register_buffer(
            "inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        )
        self._cached: tuple[int, torch.Tensor, torch.Tensor] | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        if self._cached is None or self._cached[0] != seq_len:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self._cached = (seq_len, freqs.cos().bfloat16(), freqs.sin().bfloat16())
        _, cos, sin = self._cached
        return cos[None, :, None, :], sin[None, :, None, :]


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CPLinear(torch.nn.Module):
    """CP-factorized QKV projection, transcribed from the pinned T6.py CPLinear."""

    def __init__(self, in_features: int, n_head: int, head_dim: int, rank: int, q_rank: int):
        super().__init__()
        self.in_features = in_features
        self.n_head = n_head
        self.head_dim = head_dim
        self.rank = rank
        self.q_rank = q_rank
        self.W_A_q = torch.nn.Linear(in_features, n_head * q_rank, bias=False)
        self.W_A_k = torch.nn.Linear(in_features, n_head * rank, bias=False)
        self.W_A_v = torch.nn.Linear(in_features, n_head * rank, bias=False)
        self.W_B_q = torch.nn.Linear(in_features, q_rank * head_dim, bias=False)
        self.W_B_k = torch.nn.Linear(in_features, rank * head_dim, bias=False)
        self.W_B_v = torch.nn.Linear(in_features, rank * head_dim, bias=False)
        self.rotary = _Rotary(head_dim)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.size()
        A_q = self.W_A_q(x).view(B, T, self.n_head, self.q_rank)
        A_k = self.W_A_k(x).view(B, T, self.n_head, self.rank)
        A_v = self.W_A_v(x).view(B, T, self.n_head, self.rank)
        B_q = self.W_B_q(x).view(B, T, self.q_rank, self.head_dim)
        B_k = self.W_B_k(x).view(B, T, self.rank, self.head_dim)
        B_v = self.W_B_v(x).view(B, T, self.rank, self.head_dim)

        cos, sin = self.rotary(B_q)
        B_q = _apply_rotary_emb(B_q, cos, sin)
        B_k = _apply_rotary_emb(B_k, cos, sin)

        q = torch.bmm(A_q.reshape(B * T, self.n_head, self.q_rank),
                      B_q.reshape(B * T, self.q_rank, self.head_dim)) / self.q_rank
        k = torch.bmm(A_k.reshape(B * T, self.n_head, self.rank),
                      B_k.reshape(B * T, self.rank, self.head_dim)) / self.rank
        v = torch.bmm(A_v.reshape(B * T, self.n_head, self.rank),
                      B_v.reshape(B * T, self.rank, self.head_dim)) / self.rank
        return (
            q.view(B, T, self.n_head, self.head_dim),
            k.view(B, T, self.n_head, self.head_dim),
            v.view(B, T, self.n_head, self.head_dim),
        )


class TPAAttentionLayer(torch.nn.Module):
    """One TPA layer: CP QKV projection + typed causal K1 mixer + output proj."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int,
        rank: int,
        q_rank: int,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.c_qkv = CPLinear(n_embd, n_head, head_dim, rank, q_rank)
        self.c_proj = torch.nn.Linear(n_head * head_dim, n_embd, bias=False)

        document = {
            "schema_version": 2,
            "name": "tpa_mixer",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", n_head, head_dim]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", n_head, head_dim]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", n_head, head_dim]},
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        q, k, v = self.c_qkv(x)
        out = self._plan.execute(query=q, key=k, value=v)["output"]
        return self.c_proj(out.reshape(B, T, self.n_head * self.head_dim))


__all__ = ["CPLinear", "TPAAttentionLayer"]
