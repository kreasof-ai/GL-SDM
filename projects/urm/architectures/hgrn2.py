"""External model module: HGRN2 (arch-024).

Verified composition row: the GLA channel-decay mixer in a value-first state
layout — ``q = swish(q_proj)``, ``g = logsigmoid(f_proj)``, ``k = 1 − exp(g)``
(the complement gate as the "key"), ``v = i_proj``, then chunk/fused GLA with
``gk = g``, ``state_v_first=True`` — verified against fla/layers/hgrn2.py @
864a87f6. The lower-bound logaddexp variant and short conv are external; the
typed K2 channel-gate mixer runs through the public path. Note
``state_v_first`` is purely a physical layout (sweep), so the mixer is the
same U2.A channel-gate law.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.k2_linear_state import K2LinearStateLayer


class HGRN2Layer(K2LinearStateLayer):
    """One HGRN2 layer: external projections + typed U2.A channel-gate mixer."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="channel", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.f_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.i_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = F.silu(self.q_proj(hidden_states)).view(B, T, H, dk)
        f = self.f_proj(hidden_states)
        v = self.i_proj(hidden_states).view(B, T, H, dv)
        g = F.logsigmoid(f).view(B, T, H, dk)          # [B,T,H,K]
        k = 1 - g.exp()                                # complement gate as the "key"
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


__all__ = ["HGRN2Layer"]
