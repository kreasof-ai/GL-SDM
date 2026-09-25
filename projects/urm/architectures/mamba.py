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
decay_chunk = exp(segsum(A·dt) at chunk ends). Its recurrent core maps EXACTLY
onto the canonical additive K2 law (scalar-per-head transition →
``gate_scope=head``, ``delta=False``, state ``M[dstate, head_dim]``) and executes
through the public K2 path in ``Mamba2K2Layer`` — verified against the pinned
``ssd_chunk_scan_combined_ref`` and the diagonal scan. The chunked semiseparable
kernel (chunk_state/state_passing/chunk_scan) is a physical schedule — residual.

Mamba-1's diagonal gate ``exp(dt_d·A_{d,n})`` varies over BOTH state dims and does
NOT fit ``LinearDeltaSpec`` (its channel gate decays only the key axis); admitting
it needs a new elementwise diagonal-affine law, which is a single-client physical
branch — residual per the two-client rule.

The δ/B/C/A discretization frontend, the D·u skip, the optional z gate, and the
conv short-circuit are external.
"""

from __future__ import annotations

import torch

from architectures.k2_linear_state import K2LinearStateLayer


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


class Mamba1K2Layer(K2LinearStateLayer):
    """Mamba-1's diagonal SSM routed through the public elementwise-gated K2.

    The diagonal gate ``exp(dt_d·A_{d,n})`` varies over BOTH state dims — a full
    per-element gate — expressed as the K2 ``gate_scope=elementwise`` law. The
    mapping treats each channel ``d`` as an independent head (H=D), with a
    degenerate key axis (K=1, key=query=1) and the SSM state dim as the value axis
    (V=N): the state is ``M[D, 1, N] = x`` (the per-channel N-dim SSM state), the
    additive write is ``dt·B·u`` (value, dim N), the per-element gate is ``dt·A``
    (``[D, 1, N]``), and the after-update read with query=1 returns the state, so
    ``y = C·x`` is applied externally.

    Reference-tier admission: Mamba-1 is the single source client for the full
    elementwise gate (HGRN's vector-state channel gate is adjacent but not a
    structural match), so no native branch per the two-client rule. Verified
    against the pinned selective_scan_ref (via selective_scan_diag) at 0.0. The
    selective frontend (dt/B/C projections, A_log/dt_bias discretization), the D·u
    skip, the z gate and conv are external.
    """

    def __init__(self, d_model: int, d_state: int, *,
                 target: str = "reference", intent: str = "inference"):
        # H = d_model (one "head" per channel), K = 1, V = d_state.
        super().__init__(
            hidden_size=d_model, num_heads=d_model, head_k_dim=1, head_v_dim=d_state,
            delta=False, gate_scope="elementwise", scale_rule="one",
            target=target, intent=intent,
        )
        self.d_model = d_model
        self.d_state = d_state
        self.A_log = torch.nn.Parameter(torch.randn(d_model, d_state))
        self.dt_bias = torch.nn.Parameter(torch.rand(d_model))
        self.in_proj = torch.nn.Linear(d_model, d_model * 2 + d_state * 2 + d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states [B,L,D] → output [B,L,D]."""
        B, L, D = hidden_states.shape
        N = self.d_state
        proj = self.in_proj(hidden_states)
        u, delta, Bm, Cm, _skip = proj.split([D, D, N, N, D], dim=-1)
        A = -self.A_log.exp()                                   # [D,N]
        dt = torch.nn.functional.softplus(delta + self.dt_bias)  # [B,L,D]
        # K2 operands ([B,H=D,T,*]): key=query=1 (K=1), value=dt·B·u (V=N), gate=dt·A ([D,1,N]).
        ones_k = torch.ones(B, D, L, 1, device=hidden_states.device)
        val = torch.einsum("bld,bln,bld->bdln", dt, Bm, u)      # dt·B·u -> [B,D,L,N]
        ld = torch.einsum("bld,dn->bdln", dt, A).unsqueeze(3)   # dt·A -> [B,D,L,1,N]
        out = self._run_mixer({
            "query": ones_k, "key": ones_k, "value": val,
            "beta": torch.ones(B, D, L, device=hidden_states.device),
            "log_decay": ld,
            "initial_state": torch.zeros(B, D, 1, N, device=hidden_states.device),
        })["output"]                                            # [B,H=D,T=L,V=N] = x trajectory
        # y_t = C_t · x_t (contract the state dim N). out is [B,D,L,N]; Cm is [B,L,N].
        y = torch.einsum("bdln,bln->bld", out, Cm)              # [B,L,D]
        return y


class Mamba2K2Layer(K2LinearStateLayer):
    """Mamba-2's recurrent core routed through the public head-gated matrix K2.

    Mamba-2's scalar-per-head transition maps EXACTLY onto the canonical additive
    K2 law (``delta=False``, ``gate_scope=head``, ``scale_rule=one``): the state is
    the matrix ``M[N, P]`` (dstate × head_dim), the head gate is ``exp(A_h·dt_t)``,
    the additive write is ``B_t ⊗ (dt_t·u_t)`` (key=B, value=dt·u), and the
    after-update read is ``C_tᵀ M_t`` (query=C). Verified against the pinned
    ``ssd_chunk_scan_combined_ref`` and the diagonal scan (0.0). The selective
    frontend (dt/B/C projections, A_log/dt_bias discretization), the D·u skip, the
    z gate, conv and the chunked semiseparable kernel are external.

    Unlike Mamba-1 (whose diagonal gate ``exp(dt_d·A_{d,n})`` varies over both state
    dims and needs a new elementwise law — residual, single-client), Mamba-2's
    scalar-per-head gate fits the existing ``gate_scope=head`` descriptor, so the
    recurrent core executes through the public K2 path here.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, d_state: int, *,
                 target: str = "reference", intent: str = "inference"):
        super().__init__(
            hidden_size=d_model, num_heads=n_heads, head_k_dim=d_state, head_v_dim=head_dim,
            delta=False, gate_scope="head", scale_rule="one",
            target=target, intent=intent,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        # External selective frontend: dt/B/C projections and the A parameter.
        self.A_log = torch.nn.Parameter(torch.randn(n_heads))           # per-head A
        self.dt_bias = torch.nn.Parameter(torch.rand(n_heads))
        self.in_proj = torch.nn.Linear(d_model, d_model + n_heads + d_state + d_state, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states [B,L,D] → output [B,L,D]."""
        B, L, D = hidden_states.shape
        H, P, N = self.n_heads, self.head_dim, self.d_state
        proj = self.in_proj(hidden_states)
        x, dt, Bm, Cm = proj.split([D, H, N, N], dim=-1)
        # External discretization: A = -exp(A_log) (per head); dt = softplus(dt + dt_bias).
        A = -self.A_log.exp()                                           # [H]
        dt = torch.nn.functional.softplus(dt + self.dt_bias)            # [B,L,H]
        # K2 operands ([B,H,L,*]): query=C[N], key=B[N], value=dt·u[P], log_decay=A_h·dt.
        Cm_e = Cm.reshape(B, L, 1, N).expand(B, L, H, N).permute(0, 2, 1, 3)   # [B,H,L,N]
        Bm_e = Bm.reshape(B, L, 1, N).expand(B, L, H, N).permute(0, 2, 1, 3)   # [B,H,L,N]
        u_r = x.reshape(B, L, H, P).permute(0, 2, 1, 3)                          # [B,H,L,P]
        dt_hl = dt.permute(0, 2, 1)                                            # [B,H,L]
        val = u_r * dt_hl.unsqueeze(-1)                                        # [B,H,L,P] = dt·u
        log_decay = A.reshape(1, H, 1) * dt_hl                                 # [B,H,L] = A_h·dt
        out = self._run_mixer({
            "query": Cm_e, "key": Bm_e, "value": val,
            "beta": torch.ones(B, H, L, device=hidden_states.device),
            "log_decay": log_decay,
            "initial_state": torch.zeros(B, H, N, P, device=hidden_states.device),
        })["output"]                                                            # [B,H,L,P]
        return out.permute(0, 2, 1, 3).reshape(B, L, D)


__all__ = ["Mamba1Layer", "Mamba1K2Layer", "Mamba2Layer", "Mamba2K2Layer", "selective_scan_diag"]
