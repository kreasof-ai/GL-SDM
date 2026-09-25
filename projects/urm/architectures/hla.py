"""External model module: HLA (arch-074, higher-order linear attention).

Verified combinator row (flagship Nest), masked second-order UNNORMALIZED
causal case (paper Eq. 3.3 + Algorithm 1, the default operator) — against the
pinned paper (HLA.pdf, arXiv 2510.27258, @ 484fef2b).

The Nest combinator shape is two dependent additive prefix accumulations —
``S_t = S_{t-1} + k_t k_tᵀ`` (the inner (q,k,k) pass producing the per-token
summary z_j = S_jᵀ q_j) feeding the outer ``o_t = q_tᵀ(S_t C_t − G_t)`` — with
the strict-causality correction ``G_t = G_{t-1} + ΔS_t·C_{t-1}`` arising from
the nesting. The per-token prefix MATRICES (S_t, C_t, G_t) are the carried
state bundle; the K2 contract's read (``q_tᵀ M_t``, a vector) does not expose
the matrix, so the prefix scan and the correction recurrence are the external
Nest structure (the sweep: "two unfused U2.A calls plus the correction
summaries"). This module implements Algorithm 1's serial recurrence exactly.

The normalized variant (Eq. 3.4, +ε), decay γ (Eq. 4.1) and ridge λ (Alg. 1
line 7) are explicitly NOT claimed (sweep verdict); VJP through the nested pair
is not derived in the source and is covered here by torch autograd through the
external recurrence.
"""

from __future__ import annotations

import torch


class HLALayer(torch.nn.Module):
    """Masked second-order unnormalized HLA (Algorithm 1, serial recurrence).

    No projections in the pinned Algorithm 1 — q/k/v are the inputs. The layer
    is the external Nest structure: the two prefix accumulations plus the
    strict-causality correction.
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """``query``/``key`` ``[B, H, T, K]``, ``value`` ``[B, H, T, V]`` → ``[B, H, T, V]``.

        Transcribed from the pinned Algorithm 1 (default masked unnormalized
        path): per-token ΔS = k kᵀ, ΔC = q vᵀ, Δm = q; exclusive-prefix scan
        (γ=1, no ridge/normalize); G_t += ΔS_t·C_{t-1}; o_t = q_tᵀ(S_t C_t − G_t).
        """
        B, H, T, K = key.shape
        V = value.shape[-1]
        device, dtype = key.device, torch.float32

        q = query.to(dtype)
        k = key.to(dtype)
        v = value.to(dtype)

        S = torch.zeros(B, H, K, K, device=device, dtype=dtype)
        C = torch.zeros(B, H, K, V, device=device, dtype=dtype)
        G = torch.zeros(B, H, K, V, device=device, dtype=dtype)
        outs = []
        for t in range(T):
            dS = torch.einsum("bhk,bhl->bhkl", k[:, :, t], k[:, :, t])
            dC = torch.einsum("bhk,bhv->bhkv", q[:, :, t], v[:, :, t])
            # Inclusive prefixes; the G correction uses the EXCLUSIVE C (C_{t-1}).
            G = G + torch.einsum("bhkl,bhlv->bhkv", dS, C)
            S = S + dS
            C = C + dC
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S @ C - G))
        return torch.stack(outs, dim=2)


__all__ = ["HLALayer"]
