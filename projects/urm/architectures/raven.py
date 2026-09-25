"""External model module: Raven (arch-050, router + GSA).

Verified combinator row: an external top-k router injected as dense gates into
the GSA two-stage slot-summary mixer — verified against fla/layers/raven.py +
fla/ops/gsa/naive.py @ 864a87f6.

External router (pinned): ``router = r_proj(x)``; optional training-time Gumbel
noise; ``orig_scores = sigmoid(router)`` (or softmax); topk slot indices over
scores; sigmoid case renormalizes selected weights by their sum; the multihot
slot vector ``s_multihot`` is scattered; the decay ``f`` (GLA-style
``logsigmoid(f_proj)/normalizer`` here) is masked ``f = f·s_multihot`` and
``s = 1 − exp(f)``; then the GSA mixer runs. The router, feature map, q/k norm
and output gate are external; the two-stage mixer reuses the GSA composition
(architectures/abc_gsa.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.abc_gsa import _SlotAttentionBase


class RavenLayer(_SlotAttentionBase):
    """One Raven layer: external top-k router + GSA two-stage slot-summary mixer."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 n_slots: int, topk: int, *, gate_logit_normalizer: int = 8,
                 target: str = "reference", intent: str = "inference"):
        super().__init__(num_heads, head_k_dim, head_v_dim, n_slots, target=target, intent=intent)
        self.hidden_size = hidden_size
        self.topk = topk
        self.gate_logit_normalizer = gate_logit_normalizer
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.r_proj = torch.nn.Linear(hidden_size, num_heads * n_slots, bias=False)
        self.f_proj = torch.nn.Linear(hidden_size, num_heads * n_slots, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)
        self.scale = head_k_dim ** -0.5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv, M = self.num_heads, self.head_k_dim, self.head_v_dim, self.n_slots
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)

        # External router (pinned): sigmoid scores, topk, renormalized weights,
        # multihot scatter. (Gumbel noise is training-only; disabled in eval.)
        router = self.r_proj(hidden_states).view(B, T, H, M)
        orig_scores = torch.sigmoid(router)
        route_idx = orig_scores.topk(self.topk, dim=-1).indices
        topk_weights = torch.gather(orig_scores, dim=-1, index=route_idx)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        s_multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))

        # GLA-style decay masked by the router: f = logsigmoid(f_proj)/norm; f *= s_multihot; s = 1 − exp(f).
        f = F.logsigmoid(self.f_proj(hidden_states).view(B, T, H, M)) / self.gate_logit_normalizer
        f = f * s_multihot
        s = 1 - f.exp()

        # Mixer (to [B,H,T,*] layout for the shared slot base).
        out = self._mixer(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            s.transpose(1, 2), f.transpose(1, 2), self.scale,
        )  # [B,H,T,V]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


__all__ = ["RavenLayer"]
