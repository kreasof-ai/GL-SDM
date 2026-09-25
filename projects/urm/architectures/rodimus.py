"""External model module: Rodimus (arch-036).

Verified composition row: a GLA channel-decay mixer (``fused_recurrent_gla``,
``state_v_first=True``) with an external input-gate frontend — ``k =
l2norm(k)·it_gate`` where ``it_gate = softplus(g_gate) ** sigmoid(τ_gate)``,
and the channel decay ``rt_gate_log = −softplus(g_gate)·sigmoid(τ_gate)`` —
verified against fla/layers/rodimus.py @ 864a87f6. There is no
fla/ops/rodimus; the mixer is the GLA call. External stages: the d_inner
expansion, the input/tau gate projections, L2-norm × input gate, and the
v = i_gate ⊙ hidden construction. Cache/chunked paths and the MLP/short-conv
wrapper are residual external work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class RodimusLayer(K2LinearStateLayer):
    """One Rodimus mixer layer: external gated frontend + typed U2.A channel-gate mixer."""

    def __init__(self, d_inner: int, mem_size: int, *,
                 target: str = "reference", intent: str = "inference"):
        # Single head over the mem width (head_k_dim = mem_size); the mixer law
        # is per-channel over the mem dim.
        super().__init__(d_inner, num_heads=1, head_k_dim=mem_size, head_v_dim=mem_size,
                         delta=False, gate_scope="channel", scale_rule="one",
                         target=target, intent=intent)
        self.d_inner = d_inner
        self.mem_size = mem_size
        self.k_proj = torch.nn.Linear(d_inner, mem_size, bias=False)
        self.q_proj = torch.nn.Linear(d_inner, mem_size, bias=False)
        self.i_gate_proj = torch.nn.Linear(d_inner, d_inner, bias=False)
        self.g_gate_proj = torch.nn.Linear(d_inner, mem_size, bias=True)
        self.tau_gate_proj = torch.nn.Linear(d_inner, mem_size, bias=True)
        self.k_norm_eps = 1e-5

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states here is the d_inner-wide activation (post gate_proj/norm).
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states)                          # [B,T,M]
        k = self.k_proj(hidden_states)
        v = self.i_gate_proj(hidden_states) * hidden_states     # [B,T,d_inner]
        g_gate = F.softplus(self.g_gate_proj(hidden_states).float())
        tau_gate = torch.sigmoid(self.tau_gate_proj(hidden_states).float())
        it_gate = g_gate ** tau_gate                            # [B,T,M]
        rt_gate_log = (-g_gate) * tau_gate                      # [B,T,M] channel log-decay
        k = F.normalize(k.float(), dim=-1, eps=self.k_norm_eps) * it_gate

        # v is d_inner-wide; the mixer's value dim is the mem width in the
        # pinned layer via an up-projection — here we keep v at d_inner and run
        # the mixer at [B,1,T,M]×[B,1,T,d_inner].
        out = self._run_mixer({
            "query": q.unsqueeze(1), "key": k.unsqueeze(1), "value": v.unsqueeze(1),
            "beta": torch.ones(B, 1, T, device=q.device),
            "log_decay": rt_gate_log.unsqueeze(1),
            "initial_state": torch.zeros(B, 1, self.mem_size, self.d_inner, device=q.device),
        })["output"]  # [B,1,T,d_inner]
        return out.squeeze(1)


__all__ = ["RodimusLayer"]
