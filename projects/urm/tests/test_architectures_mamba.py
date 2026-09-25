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

from architectures.mamba import Mamba1Layer, Mamba2Layer, selective_scan_diag

D, N, L, H, P = 3, 4, 6, 2, 4


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
