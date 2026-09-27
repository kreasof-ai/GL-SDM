"""External model module: LightNet (arch-022).

Verified composition row: a GLA channel-decay mixer with a normalized-key
frontend — keys normalized via log-cumsum-exp over time: ``z_t =
logcumsumexp(k)``, ``k'_t = exp(k_t − z_t)``, ``g_t = z_{t-1} − z_t`` — feeding
the channel-diagonal GLA law (``fused_recurrent_gla``, ``state_v_first=True``)
— verified against fla/layers/lightnet.py @ 864a87f6. The log-cumsum-exp
frontend, projections and the short conv are external; the typed K2 channel-
gate mixer runs through the public path. The decode path (logcumsumexp from a
cached state) and padding-mask handling are residual external work, not claimed.
"""

from __future__ import annotations

import torch

from architectures.k2_linear_state import K2LinearStateLayer


class LightNetLayer(K2LinearStateLayer):
    """One LightNet layer: external normalized-key frontend + typed U2.A channel-gate mixer."""

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="channel", scale_rule="key_dim_rsqrt",
                         target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self.q_proj(hidden_states).view(B, T, H, dk)
        k = self.k_proj(hidden_states).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)

        # External normalized-key frontend (pinned prefill path, no padding):
        # z = logcumsumexp(k) over time, k' = exp(k − z), g = z_prev − z.
        k_float = k.float()
        z = k_float.logcumsumexp(1)                                   # [B, T, H, K]
        k_new = torch.exp(k_float - z)
        k_new = torch.nan_to_num(k_new, nan=0.0, posinf=0.0).to(k.dtype)
        z_prev = torch.cat((torch.zeros_like(z[:, :1]), z[:, :-1]), dim=1)
        g = (z_prev - z).to(k.dtype)                                  # [B, T, H, K]

        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k_new.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": g.transpose(1, 2),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


__all__ = ["LightNetLayer"]
