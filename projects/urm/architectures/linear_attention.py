"""External model module: Linear Attention (arch-015).

Verified composition row (U2.A, no decay): additive linear attention
``S_t = S_{t-1} + k_t v_t``, ``o_t = (q_t·scale) S_t`` (after-update read, G=I)
— verified against fla/ops/linear_attn + fla/layers/linear_attn.py @ 864a87f6.
The optional normalized variant keeps a cumulative-key denominator state and
divides the read by ``(q·scale)·k_cum + ε`` (the K2 descriptor's
``normalized=True`` path). The feature map (elu/relu/hadamard/t2r/dpfp) applied
to q/k is external; this module takes the mapped q/k (or applies elu+1 by
default). Short conv, cache/chunked paths are residual external work.
"""

from __future__ import annotations

import torch

from architectures.k2_linear_state import K2LinearStateLayer


class LinearAttentionLayer(K2LinearStateLayer):
    """One linear-attention layer: external projections + typed U2.A mixer.

    ``normalize=True`` enables the normalized variant (denominator state) of the
    K2 descriptor; the mixer then returns the normalized output.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_k_dim: int, head_v_dim: int,
                 *, normalize: bool = False, target: str = "reference", intent: str = "inference"):
        super().__init__(hidden_size, num_heads, head_k_dim, head_v_dim,
                         delta=False, gate_scope="none", scale_rule="key_dim_rsqrt",
                         normalized=normalize, target=target, intent=intent)
        self.q_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, num_heads * head_k_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden_size, num_heads * head_v_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_v_dim, hidden_size, bias=False)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        # Default fla linear-attention feature map: elu + 1.
        return torch.nn.functional.elu(x) + 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, _ = hidden_states.shape
        H, dk, dv = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self._feature_map(self.q_proj(hidden_states)).view(B, T, H, dk)
        k = self._feature_map(self.k_proj(hidden_states)).view(B, T, H, dk)
        v = self.v_proj(hidden_states).view(B, T, H, dv)
        out = self._run_mixer({
            "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
            "beta": torch.ones(B, H, T, device=q.device),
            "log_decay": torch.zeros(B, H, T, device=q.device),
            "initial_state": torch.zeros(B, H, dk, dv, device=q.device),
        })["output"]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * dv))


__all__ = ["LinearAttentionLayer"]
