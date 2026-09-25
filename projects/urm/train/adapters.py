"""Adapters mapping each architecture's mixer signature to the harness's [B,T,C]→[B,T,C].

The URM architecture modules fall into three interface families:

1. **Sequence mixers** — forward(hidden [B,T,C]) → [B,T,C] (the K2 family, Mamba, etc.):
   registered directly, no adapter.
2. **Operand mixers** — forward(q, k, v, *gates): the pinned law is mixer-level with no
   projections; the adapter adds external q/k/v/o projections and synthesizes the law's
   extra operands (decay gates, routing indices) exactly as the per-architecture parity
   test does. Layers disagree on layout (some take [B,H,T,D], some [B,T,H,D]) — the
   adapter carries a ``layout`` flag.
3. **External compositions** — plain-torch modules that don't route through a URM plan:
   registered with ``public_path=False`` so the report is honest about what is and isn't
   exercising the compiler.

Every adapter owns ONLY the surround (projections + operand synthesis) — never a mixer
equation, which stays in the architecture module.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QKVAdapter(nn.Module):
    """Project [B,T,C] → q/k/v, call the operand mixer, project back to [B,T,C].

    ``layout="bhtd"`` feeds the mixer [B,H,T,D]; ``"bthd"`` feeds [B,T,H,D]. ``extra``
    is a callable ``(adapter, hidden, q, k, v) -> dict`` producing the law's additional
    operands (gates, indices); the dict is passed as **kwargs to the mixer.

    Gates are DERIVED from the hidden state (a projection the adapter owns), never fresh
    randomness — a mixer's gates are functions of the input, and the checkpoint gate's
    determinism depends on it.
    """

    def __init__(self, model_dim, num_heads, head_dim, mixer_factory, *,
                 layout="bhtd", extra=None, gate_out_dim: int | None = None):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.layout = layout
        self.q_proj = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self._mixer = mixer_factory(num_heads, head_dim)
        self._extra = extra
        # Optional gate projection: extra() may use adapter.gate_proj(hidden) to derive
        # gates from the input deterministically.
        self.gate_proj = (
            nn.Linear(model_dim, gate_out_dim, bias=True) if gate_out_dim else None
        )
        self.gate_proj2 = (
            nn.Linear(model_dim, gate_out_dim, bias=True) if gate_out_dim else None
        )

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        if self.layout == "bhtd":
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        extras = self._extra(self, hidden, q, k, v) if self._extra is not None else {}
        out = self._mixer(q, k, v, **extras)
        if not isinstance(out, torch.Tensor):
            out = out[0]
        if self.layout == "bhtd":
            out = out.transpose(1, 2)
        return self.o_proj(out.reshape(B, T, H * D))


def zero_head_gate(adapter, hidden, q, k, v):
    """log_decay ≡ 0 at head scope, [B,H,T] (bhtd layout)."""
    B, _, T = hidden.shape
    return {"g": torch.zeros(B, adapter.num_heads, T, device=hidden.device)}
