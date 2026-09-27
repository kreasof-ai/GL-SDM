"""External model modules: Based (arch-020) and ReBased (arch-021).

Both are U2.A normalized linear attention after an external feature map —
verified against fla/layers/based.py + rebased.py and fla/modules/feature_map.py
@ 864a87f6.

- Based (020): a second-order Taylor feature map ``phi`` (approximating
  ``exp(q·k) ≈ 1 + q·k + (q·k)²/2``) applied to q/k, then normalized linear
  attention (``chunk_linear_attn(normalize=True, scale=1)``).
- ReBased (021): a squared-dot polynomial feature map with optional
  gamma/beta/normalize, then normalized linear attention (``scale=1``).

The feature maps are external; the mixer is the typed K2 normalized linear
attention (additive, gate_scope=none, explicit-operand scale 1.0, normalized).
"""

from __future__ import annotations

import math

import torch

from architectures.k2_linear_state import K2LinearStateLayer


def taylor_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Second-order Taylor feature map (fla TaylorFeatureMap.forward).

    The pinned ``flatten_diag_outer_product_off1(x, x)`` returns
    ``(x2_1, x2_2) = (off_diagonal, diagonal)``, and the map concatenates
    ``[1, x/rrd, x2_2/(rd·r2), x2_1/rd]`` — off-diagonal scaled by
    ``1/(rd·√2)``, diagonal by ``1/rd``.
    """
    head_dim = x.shape[-1]
    rd = math.sqrt(head_dim)
    rrd = math.sqrt(rd)
    r2 = math.sqrt(2.0)
    z = torch.einsum("...i,...j->...ij", x, x)
    iu = torch.triu_indices(head_dim, head_dim, offset=1, device=x.device)
    diag = torch.arange(head_dim, device=x.device)
    x2_1 = z[..., iu[0], iu[1]]       # off-diagonal (pinned x2_1)
    x2_2 = z[..., diag, diag]         # diagonal (pinned x2_2)
    return torch.cat(
        [torch.ones_like(x[..., 0:1]), x / rrd, x2_2 / (rd * r2), x2_1 / rd], dim=-1
    )


def rebased_feature_map(x: torch.Tensor) -> torch.Tensor:
    """Squared-dot polynomial feature map (RebasedFeatureMap, no gamma/beta/norm)."""
    return x ** 2


class _FeatureMapLinearAttention(K2LinearStateLayer):
    """Shared base: external feature map + typed normalized U2.A mixer."""

    def __init__(self, hidden_size, num_heads, head_k_dim, head_v_dim, *,
                 target="reference", intent="inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="none", scale_rule="explicit_operand",
                         normalized=True, target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self._map(self.q_proj(hidden_states).view(B, T, H, dk))
        k = self._map(self.k_proj(hidden_states).view(B, T, H, dk))
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        # Feature map widens the key dim; the mixer's state is [B,H,K',V].
        dk_fm = q.shape[-1]
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": torch.zeros(B, H, T, device=q.device),
            "initial_state": torch.zeros(B, H, dk_fm, dv, device=q.device),
            "scale": torch.ones((), dtype=torch.float32, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


class BasedLayer(_FeatureMapLinearAttention):
    """arch-020: Taylor-2 feature map + normalized linear attention."""

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        return taylor_feature_map(x)


class ReBasedLayer(_FeatureMapLinearAttention):
    """arch-021: squared-dot polynomial feature map + normalized linear attention."""

    def _map(self, x: torch.Tensor) -> torch.Tensor:
        return rebased_feature_map(x)


__all__ = ["BasedLayer", "ReBasedLayer", "taylor_feature_map", "rebased_feature_map"]
