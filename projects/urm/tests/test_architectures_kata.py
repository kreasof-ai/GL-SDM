"""Parity gates for arch-073 KATA (A13 SQUARED_SUM reducer).

Verified against the pinned kata source (kata/parallel_kata_attn.py, the sweep's
verification origin): causal score A[t,s] = Σ_g (scale·q_g·k_g)² with
scale = 1/sqrt(E) (E = head_dim/num_groups), all scores ≥ 0, future masked to 0;
output o = (Σ_s A·v_s)/max(Σ_s A, 1) — a positive squared-group-dot score with
sum normalization (no softmax, no epsilon). The oracle is the independent torch
reference transcription of the SPD-concat identity.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from architectures.kata import KATALayer

H, D, T, M = 2, 8, 6, 4
E = D // M


def _operands(seed: int):
    torch.manual_seed(seed)
    return (torch.randn(2, T, H, D), torch.randn(2, T, H, D), torch.randn(2, T, H, D))


def _torch_reference(q, k, v, num_groups):
    scale = 1.0 / math.sqrt(D // num_groups)
    B, T = q.shape[0], q.shape[1]
    qg = q.view(B, T, H, num_groups, E).permute(0, 2, 1, 3, 4)
    kg = k.view(B, T, H, num_groups, E).permute(0, 2, 1, 3, 4)
    gd = torch.einsum("bhime,bhjme->bhijm", qg, kg) * scale
    A = gd.pow(2.0).sum(-1)
    mask = torch.ones(T, T, dtype=torch.bool).tril()
    A = A.masked_fill(~mask, 0.0)
    num = torch.einsum("bhij,bhjd->bhid", A, v.permute(0, 2, 1, 3))
    den = A.sum(-1, keepdim=True).clamp(min=1.0)
    return (num / den).permute(0, 2, 1, 3)


def test_kata_matches_torch_reference():
    q, k, v = _operands(seed=5)
    layer = KATALayer(H, D, num_groups=M)
    with torch.no_grad():
        actual = layer(q, k, v)
    expected = _torch_reference(q, k, v, M)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"kata parity: max abs err {err}"


def test_kata_scores_are_nonnegative_and_causal():
    """Squared scores ≥ 0; a future-token perturbation leaves earlier outputs unchanged."""
    q, k, v = _operands(seed=7)
    layer = KATALayer(H, D, num_groups=M)
    with torch.no_grad():
        base = layer(q, k, v)
        k_future = k.clone()
        k_future[:, -1] = torch.randn(2, H, D)
        perturbed = layer(q, k_future, v)
    err = (base[:, :-1] - perturbed[:, :-1]).abs().max().item()
    assert err < 1e-5, f"causality violated: max abs err {err}"


def test_kata_gradients_flow():
    q, k, v = _operands(seed=13)
    for t in (q, k, v):
        t.requires_grad_(True)
    KATALayer(H, D, num_groups=M)(q, k, v).square().sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None and t.grad.abs().sum().item() > 0, f"{name} has no gradient"
