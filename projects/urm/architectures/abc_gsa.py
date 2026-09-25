"""External model modules: ABC (arch-048) and GSA (arch-049).

Verified combinator rows (same two-stage slot-summary shape) against
fla/ops/abc/naive.py + fla/ops/gsa/naive.py @ 864a87f6.

Two coupled additive per-slot-decayed accumulators with an external interstage
slot softmax:

    Stage 1: hk_t = hk_{t-1}·exp(g_t)[None, :] + k_t ⊗ s_t   (decay on the SLOT/column axis)
             ok_t = (scale·q_t)ᵀ hk_t
             qv_t = softmax(ok_t)                            (external interstage)
    Stage 2: hv_t = hv_{t-1}·exp(g_t)[:, None] + s_t ⊗ v_t   (decay on the SLOT/row axis)
             o_t  = qv_tᵀ hv_t

**Structural finding (verified this batch):** the slot decay is on the
*column* axis in stage 1 and the *row* axis in stage 2. The closed K2
channel-diagonal gate decays the *key* axis of the state, so it covers stage 2
(slot = key axis) but NOT stage 1 (slot = value axis). Stage 1 is therefore an
external recurrence here; stage 2 is the typed K2 call. The sweep's "two U2.A
calls" is imprecise on the stage-1 gate axis — recorded honestly in the recipe.

- ABC (048): g derived from slot logits (z = logcumsumexp(s), g_t = z_{t-1} − z_t,
  s = exp(s − z)), external.
- GSA (049): g explicitly supplied per slot, external.
"""

from __future__ import annotations

import torch

from architectures.k2_linear_state import K2LinearStateLayer


class _SlotAttentionBase(torch.nn.Module):
    """Two-stage slot-summary mixer: external stage-1 recurrence + typed K2 stage 2."""

    def __init__(self, num_heads: int, head_k_dim: int, head_v_dim: int, n_slots: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.n_slots = n_slots
        # Stage 2: additive channel-gate over the slot axis (key axis) — typed K2.
        self._stage2 = K2LinearStateLayer(
            hidden_size=1, num_heads=num_heads, head_k_dim=n_slots, head_v_dim=head_v_dim,
            delta=False, gate_scope="channel", scale_rule="one",
            target=target, intent=intent,
        )

    def _stage1(self, q, k, s, g, scale):
        """Stage-1 slot summary hk_t (decay on the column/slot axis) — external.

        q/k [B,H,T,K], s/g [B,H,T,M] → ok [B,H,T,M].
        """
        B, H, T, K = k.shape
        M = s.shape[-1]
        hk = torch.zeros(B, H, K, M, dtype=torch.float32, device=q.device)
        ok = torch.zeros(B, H, T, M, dtype=torch.float32, device=q.device)
        for t in range(T):
            g_i = g[:, :, t].exp()                          # [B,H,M]
            hk = hk * g_i.unsqueeze(-2) + k[:, :, t].unsqueeze(-1) * s[:, :, t].unsqueeze(-2)
            ok[:, :, t] = ((q[:, :, t] * scale).unsqueeze(-1) * hk).sum(-2)
        return ok

    def _mixer(self, q, k, v, s, g, scale):
        """q/k [B,H,T,K], v [B,H,T,V], s/g [B,H,T,M] → o [B,H,T,V]."""
        B, H, T = q.shape[:3]
        ok = self._stage1(q, k, s, g, scale)
        qv = torch.softmax(ok, dim=-1)                     # external interstage
        stage2 = self._stage2._run_mixer({
            "query": qv, "key": s, "value": v,
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g,
            "initial_state": torch.zeros(B, H, self.n_slots, self.head_v_dim, device=q.device),
        })
        return stage2["output"]


class ABCLayer(_SlotAttentionBase):
    """arch-048: g derived from slot logits (cumulative-softmax slot gates)."""

    def __init__(self, *args, target: str = "reference", intent: str = "inference", **kwargs):
        super().__init__(*args, target=target, intent=intent, **kwargs)
        self.scale = self.head_k_dim ** -0.5

    def forward(self, q, k, v, s):
        """q/k [B,H,T,K], v [B,H,T,V], s (slot logits) [B,H,T,M]; g derived externally."""
        z = s.float().logcumsumexp(2)
        g = torch.cat((z[:, :, :1], z[:, :, :-1]), 2) - z
        s_norm = torch.exp(s - z)
        return self._mixer(q, k, v, s_norm, g, self.scale)


class GSALayer(_SlotAttentionBase):
    """arch-049: g explicitly supplied per slot."""

    def __init__(self, *args, target: str = "reference", intent: str = "inference", **kwargs):
        super().__init__(*args, target=target, intent=intent, **kwargs)
        self.scale = self.head_k_dim ** -0.5

    def forward(self, q, k, v, s, g):
        """q/k [B,H,T,K], v [B,H,T,V], s [B,H,T,M], g (per-slot log-decay) [B,H,T,M]."""
        return self._mixer(q, k, v, s, g, self.scale)


__all__ = ["ABCLayer", "GSALayer"]
