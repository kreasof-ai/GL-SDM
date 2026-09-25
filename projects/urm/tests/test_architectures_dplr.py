"""Parity gates for arch-032 DPLR (A8 decay∘low-rank composed transition).

Verified against the pinned fla source
(fla/ops/generalized_delta_rule/dplr/naive.py @ 864a87f6, the sweep's
verification origin):

    lr_read = α_tᵀ·S_{t-1}                                (off the PRE-decay state)
    S_t     = Diag(exp(gk_t))·S_{t-1} + k_t·v_tᵀ + β_t·lr_read
    o_t     = (q_t·K^-0.5)ᵀ·S_t                           (read after update)

The typed K2 low-rank call (the descriptor's ``alpha``/``low_rank_beta`` roles
bound to the naive's α/β) matches the pinned naive on identical operands
(fp32, forward AND final state) on the reference tier and — CUDA permitting —
the native tier, and backward produces finite gradients to the rank-1 factors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.dplr import DPLRLayer

H, K, V, T = 2, 4, 4, 5


def _operands(seed: int, device: str = "cpu"):
    torch.manual_seed(seed)
    q = torch.randn(2, H, T, K, device=device)
    k = torch.randn(2, H, T, K, device=device)
    v = torch.randn(2, H, T, V, device=device)
    alpha = torch.randn(2, H, T, K, device=device) * 0.2   # rank-1 read factor
    beta = torch.randn(2, H, T, K, device=device) * 0.2    # rank-1 write factor
    gk = torch.nn.functional.logsigmoid(torch.randn(2, H, T, K, device=device))  # log decay
    return q, k, v, alpha, beta, gk


def _pinned_dplr():
    from benchmarks.comparators.fla_k2 import fla_op
    return fla_op("fla.ops.generalized_delta_rule.dplr.naive.dplr_recurrence")


@pytest.mark.parametrize("target", ["reference", "native"])
def test_dplr_matches_pinned_dplr_naive(target):
    """Forward + final state, fp32, vs the pinned fla DPLR naive on both tiers."""
    if target == "native" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native tier")
    device = "cuda" if target == "native" else "cpu"
    q, k, v, alpha, beta, gk = _operands(seed=5, device=device)
    layer = DPLRLayer(H, K, V, target=target)
    with torch.no_grad():
        actual, actual_state = layer(q, k, v, alpha, beta, gk)
    dplr = _pinned_dplr()
    expected, expected_state = dplr(q, k, v, alpha, beta, gk)
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, f"dplr ({target}) parity: max abs err {err}"
    state_err = (actual_state - expected_state).abs().max().item()
    assert state_err < 1e-4, f"dplr ({target}) final-state parity: max abs err {state_err}"


@pytest.mark.parametrize("target", ["reference", "native"])
def test_dplr_low_rank_term_is_active(target):
    """The rank-1 transition (α/β) must affect the output (not just decay + write)."""
    if target == "native" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native tier")
    device = "cuda" if target == "native" else "cpu"
    q, k, v, alpha, beta, gk = _operands(seed=9, device=device)
    layer = DPLRLayer(H, K, V, target=target)
    with torch.no_grad():
        out1, _ = layer(q, k, v, alpha, beta, gk)
        out2, _ = layer(q, k, v, torch.zeros_like(alpha), beta, gk)  # alpha=0 kills it
    assert (out1 - out2).abs().max().item() > 1e-3, "low-rank transition has no effect"


@pytest.mark.parametrize("target", ["reference", "native"])
def test_dplr_gradients_flow_to_all_operands(target):
    """Backward produces finite, nonzero gradients to every operand (incl. α/β/gk)."""
    if target == "native" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the native tier")
    device = "cuda" if target == "native" else "cpu"
    operands = [t.requires_grad_(True) for t in _operands(seed=13, device=device)]
    layer = DPLRLayer(H, K, V, target=target, intent="training")
    out, final_state = layer(*operands)
    (out.square().sum() + final_state.square().sum()).backward()
    for name, t in zip(("q", "k", "v", "alpha", "beta", "gk"), operands):
        assert t.grad is not None, f"{target}: d{name} missing"
        assert torch.isfinite(t.grad).all().item(), f"{target}: d{name} not finite"
        assert t.grad.abs().sum().item() > 0, f"{target}: d{name} is zero"
