"""External model modules: Mamba-1 (arch-043) and Mamba-2 / SSD (arch-044).

A8 read/write-coupling sub-law — selective SSMs with input-precomputable
coefficients. Verified against the pinned mamba_ssm source
(mamba_ssm/ops/selective_scan_interface.py, selective_scan_ref).

Mamba-1 (043): a per-channel diagonal SSM. State ``x [B,D,N]`` (one N-dim state
per channel); the recurrence is
    ``x_i = exp(δ_i·A)·x_{i-1} + δ_i·B_i·u_i``,  ``y_i = C_i·x_i``
with a DIAGONAL transition ``exp(δ_i·A)`` (per-channel, from the input-dependent
step size δ_i and the [D,N] parameter A) and an input-modulated write
``δ_i·B_i·u_i``. The δ/B/C are DATA-DEPENDENT (selective), input-precomputable.
This is a diagonal vector-state recurrence — structurally distinct from the K2
matrix state (the K2 contract's [H,K,V] matrix).

Mamba-2 (044): the semiseparable (chunked) form — within a chunk the diagonal
blocks give ``Y = (CBᵀ ∘ L)X`` with ``L = exp(segsum(A·dt))`` the intra-chunk
decay mask; off-diagonal blocks factor through a boundary-state recurrence with
decay_chunk = exp(segsum(A·dt) at chunk ends). The cross-chunk boundary carry is
the defining structure; residual here (the within-chunk diagonal-block form is
verified).

The δ/B/C/A discretization frontend, the D·u skip, the optional z gate, and the
conv short-circuit are external.
"""

from __future__ import annotations

import torch


def selective_scan_diag(u, delta, A, B, C, delta_bias=None, delta_softplus=True):
    """Mamba-1 diagonal SSM recurrence (the pinned selective_scan_ref equation).

    u [B,D,L]; delta [B,D,L]; A [D,N]; B/C [B,N,L] (variable) or [D,N].
    State x [B,D,N]; x_i = exp(δ_i·A)·x_{i-1} + δ_i·B_i·u_i; y_i = C_i·x_i.
    Returns y [B,D,L]. fp32.
    """
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    B_, D, N = u.shape[0], A.shape[0], A.shape[1]
    x = A.new_zeros((B_, D, N))
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    if B.dim() >= 3:  # variable B [B,N,L]
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:             # static B [D,N]
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    ys = []
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() >= 3:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        else:
            y = torch.einsum("bdn,dn->bd", x, C)
        ys.append(y)
    return torch.stack(ys, dim=2)  # [B,D,L]


class Mamba1Layer(torch.nn.Module):
    """One Mamba-1 selective-scan layer (diagonal SSM)."""

    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # External selective frontend: δ/B/C projections and the A parameter.
        self.A_log = torch.nn.Parameter(torch.randn(d_model, d_state))
        self.in_proj = torch.nn.Linear(d_model, d_model * 2 + d_state * 2 + d_model, bias=False)
        self.dt_bias = torch.nn.Parameter(torch.rand(d_model))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states [B,L,D] → output [B,L,D]."""
        B, L, D = hidden_states.shape
        N = self.d_state
        proj = self.in_proj(hidden_states)  # [B,L, 2D + 2N + D]
        u, delta, Bm, Cm, _skip = proj.split([D, D, N, N, D], dim=-1)
        A = -self.A_log.exp()               # [D,N] log-decay rates (negative)
        out = selective_scan_diag(
            u.transpose(1, 2), delta.transpose(1, 2), A,
            Bm.transpose(1, 2), Cm.transpose(1, 2),
            delta_bias=self.dt_bias, delta_softplus=True,
        )
        return out.transpose(1, 2)


class Mamba2Layer(torch.nn.Module):
    """One Mamba-2 / SSD layer (semiseparable selective SSM, recurrent reference form).

    Mamba-2's within-chunk diagonal blocks and cross-chunk boundary carry reduce to
    the SAME diagonal SSM recurrence as Mamba-1 (x_i = exp(δ_i·A)·x_{i-1} +
    δ_i·B_i·u_i, y_i = C_i·x_i) with A broadcast per (head, state) and dt per head.
    Verified against the pinned ssd recurrent form. The chunked semiseparable kernel
    (chunk_state/state_passing/chunk_scan) is a physical schedule — residual.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.A_log = torch.nn.Parameter(torch.randn(n_heads))          # per-head A
        self.dt_bias = torch.nn.Parameter(torch.rand(n_heads))
        self.in_proj = torch.nn.Linear(d_model, d_model + n_heads + d_state + d_state, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states [B,L,D] → output [B,L,D]."""
        B, L, D = hidden_states.shape
        H, P, N = self.n_heads, self.head_dim, self.d_state
        proj = self.in_proj(hidden_states)
        x, dt, Bm, Cm = proj.split([D, H, N, N], dim=-1)
        A = -self.A_log.exp()                                          # [H]
        # Broadcast A to [H*P, N] (per channel) and dt to per-channel.
        A_full = A.unsqueeze(-1).expand(H, N).unsqueeze(1).expand(H, P, N).reshape(H * P, N)
        dt_full = dt.unsqueeze(-1).expand(B, L, H, P).reshape(B, L, H * P)
        out = selective_scan_diag(
            x.transpose(1, 2), dt_full.transpose(1, 2), A_full,
            Bm.transpose(1, 2), Cm.transpose(1, 2),
            delta_bias=self.dt_bias.unsqueeze(-1).expand(H, P).reshape(H * P),
            delta_softplus=True,
        )
        return out.transpose(1, 2)


__all__ = ["Mamba1Layer", "Mamba2Layer", "selective_scan_diag"]
