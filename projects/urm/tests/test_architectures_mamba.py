"""Parity gates for arch-043 Mamba-1 + arch-044 Mamba-2 / SSD (A8 read/write coupling).

Verified against the pinned mamba_ssm source (mamba_ssm/ops/selective_scan_interface.py
selective_scan_ref, the sweep's verification origin): the diagonal SSM recurrence
x_i = exp(δ_i·A)·x_{i-1} + δ_i·B_i·u_i, y_i = C_i·x_i with input-precomputable
(selective) δ/B/C. Mamba-2 shares the same recurrence (A per-head broadcast, dt per
head). The selective frontend, D·u skip, z gate, conv, and the Mamba-2 chunked
semiseparable kernel are residual.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.mamba import Mamba1Layer, Mamba2K2Layer, Mamba2Layer, selective_scan_diag

D, N, L, H, P = 3, 4, 6, 2, 4


def test_mamba2_core_routes_through_public_k2():
    """Mamba-2's recurrent core executes through the public head-gated matrix K2.

    The scalar-per-head transition maps onto the canonical additive K2 law
    (delta=False, gate_scope=head): state M[dstate, head_dim], gate exp(A_h·dt),
    write B⊗(dt·u), read CᵀM. Verified against the pinned ssd_chunk_scan_combined_ref
    and the diagonal-scan layer sharing the same external frontend.
    """
    import sys
    sys.path.insert(0, "/tmp/urm-comparator-pins/mamba")
    from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref
    torch.manual_seed(5)
    d_model = H * P
    layer = Mamba2K2Layer(d_model, H, P, N)
    x = torch.randn(1, 8, d_model)
    with torch.no_grad():
        out_k2 = layer(x)
        proj = layer.in_proj(x)
        xx, dt, Bm, Cm = proj.split([d_model, H, N, N], dim=-1)
        A = -layer.A_log.exp()
        dt = torch.nn.functional.softplus(dt + layer.dt_bias)
        expected = ssd_chunk_scan_combined_ref(
            xx.reshape(1, 8, H, P), dt, A, Bm.unsqueeze(2), Cm.unsqueeze(2), 4, dt_softplus=False
        ).reshape(1, 8, d_model)
    err = (out_k2 - expected).abs().max().item()
    assert err < 2e-3, f"Mamba2 public-K2 vs pinned chunked SSD: max abs err {err}"


def test_mamba2_k2_matches_diagonal_scan_layer():
    """The public-K2 Mamba-2 core == the diagonal-scan layer on shared weights."""
    torch.manual_seed(7)
    d_model = H * P
    k2 = Mamba2K2Layer(d_model, H, P, N)
    diag = Mamba2Layer(d_model, H, P, N)
    with torch.no_grad():
        diag.in_proj.weight.copy_(k2.in_proj.weight)
        diag.A_log.copy_(k2.A_log)
        diag.dt_bias.copy_(k2.dt_bias)
        x = torch.randn(2, 5, d_model)
        err = (k2(x) - diag(x)).abs().max().item()
    assert err < 1e-4, f"Mamba2 public-K2 vs diagonal-scan: max abs err {err}"


def test_mamba1_selective_scan_matches_pinned_ref():
    torch.manual_seed(5)
    B = 2
    u = torch.randn(B, D, L); delta = torch.rand(B, D, L) * 0.5
    A = -torch.rand(D, N); Bm = torch.randn(B, N, L); Cm = torch.randn(B, N, L)
    with torch.no_grad():
        actual = selective_scan_diag(u, delta, A, Bm, Cm, delta_softplus=False)
    import sys
    sys.path.insert(0, "/tmp/urm-comparator-pins/mamba")
    from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
    expected = selective_scan_ref(u, delta, A, Bm, Cm, delta_softplus=False)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"mamba1 selective_scan parity: max abs err {err}"


def test_mamba1_selectivity_is_active():
    """δ is input-dependent: changing δ changes the decay, so the output changes."""
    torch.manual_seed(7)
    B = 1
    u = torch.randn(B, D, L); A = -torch.rand(D, N); Bm = torch.randn(B, N, L); Cm = torch.randn(B, N, L)
    with torch.no_grad():
        out1 = selective_scan_diag(u, torch.rand(B, D, L) * 0.5, A, Bm, Cm, delta_softplus=False)
        out2 = selective_scan_diag(u, torch.rand(B, D, L) * 0.5, A, Bm, Cm, delta_softplus=False)
    assert (out1 - out2).abs().max().item() > 1e-3


def test_mamba1_layer_composition_and_gradients():
    torch.manual_seed(9)
    d_model = D
    layer = Mamba1Layer(d_model, N)
    x = torch.randn(2, L, d_model)
    out = layer(x)
    assert out.shape == (2, L, d_model)
    out.square().sum().backward()
    assert layer.in_proj.weight.grad is not None


def test_mamba2_recurrence_matches_pinned_chunked_ref():
    """Mamba-2's chunked semiseparable form reduces to the diagonal SSM recurrence."""
    import sys
    sys.path.insert(0, "/tmp/urm-comparator-pins/mamba")
    from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref
    from einops import repeat, rearrange
    torch.manual_seed(5)
    B, Lq, Hh, Pp, Nn, Gg = 1, 8, 2, 4, 4, 1
    x = torch.randn(B, Lq, Hh, Pp)
    dt = torch.rand(B, Lq, Hh) * 0.5
    A = -torch.rand(Hh)
    Bm = torch.randn(B, Lq, Gg, Nn); Cm = torch.randn(B, Lq, Gg, Nn)
    with torch.no_grad():
        expected = ssd_chunk_scan_combined_ref(x, dt, A, Bm, Cm, 4, dt_softplus=False)
        A_full = repeat(A, "h -> (h p) n", p=Pp, n=Nn)
        dt_full = repeat(dt, "b l h -> b l (h p)", p=Pp)
        xf = rearrange(x, "b l h p -> b (h p) l")
        Bf = rearrange(Bm, "b l g n -> b g n l")[:, 0]
        Cf = rearrange(Cm, "b l g n -> b g n l")[:, 0]
        actual = selective_scan_diag(xf, dt_full.transpose(1, 2), A_full, Bf, Cf, delta_softplus=False)
        actual = rearrange(actual, "b (h p) l -> b l h p", h=Hh, p=Pp)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"mamba2 chunked-ref vs recurrent: max abs err {err}"


def test_mamba2_layer_composition_and_gradients():
    torch.manual_seed(11)
    d_model = H * P
    layer = Mamba2Layer(d_model, H, P, N)
    x = torch.randn(2, L, d_model)
    out = layer(x)
    assert out.shape == (2, L, d_model)
    out.square().sum().backward()
    assert layer.in_proj.weight.grad is not None
    assert layer.A_log.grad is not None
