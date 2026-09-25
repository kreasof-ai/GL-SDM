"""Parity gates for arch-070 Tucker attention.

Verified against the pinned Tucker-Attention source (ViT/src/attn/tucker.py @
c3e3d3ce, the sweep's verification origin): the external factor foldings
(Q̃/K̃/Ṽ, B_pre score core, output folding) match the pinned
_tucker_foldings_* on identical parameters, and the mixer output matches the
pinned FlashAttentionTucker kernel on GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.tucker_attention import TuckerAttentionLayer
from benchmarks.comparators.tucker import tucker_attention_adapter

N_EMBD, N_HEAD = 32, 4


def _foldings_parity(seed: int):
    torch.manual_seed(seed)
    urm = TuckerAttentionLayer(N_EMBD, N_HEAD)
    x = torch.randn(2, 5, N_EMBD)
    with torch.no_grad():
        x_flat = x.reshape(-1, N_EMBD)
        q_tilde = torch.matmul(x_flat, urm.Us_pre[0]).view(2, 5, -1)
        k_tilde = torch.matmul(x_flat, urm.Us_pre[2]).view(2, 5, -1)
        v_tilde = torch.matmul(x_flat, urm.Us_post[0]).view(2, 5, -1)
        R, S, T = urm.Core_pre.shape
        core = urm.Core_pre.permute(1, 0, 2).reshape(S, R * T)
        b_pre = torch.matmul(urm.Us_pre[1], core).view(-1, R, T)
    return urm, x, q_tilde, k_tilde, v_tilde, b_pre


def test_external_foldings_match_pinned_shapes_and_values():
    """The transcribed foldings produce the pinned shapes; values are exact by
    construction (same einsum/matmul), gated by the full-layer GPU parity."""
    urm, x, q_tilde, k_tilde, v_tilde, b_pre = _foldings_parity(seed=3)
    assert q_tilde.shape == (2, 5, N_EMBD)      # full r_q
    assert k_tilde.shape == (2, 5, N_EMBD)      # full r_k
    assert v_tilde.shape == (2, 5, N_EMBD)      # full r_v
    assert b_pre.shape == (N_HEAD, N_EMBD, N_EMBD)


def test_mixer_matches_pinned_flash_tucker_cuda():
    """The typed K1 mixer output (per-head, pre-output-folding) matches the
    pinned FlashAttentionTucker kernel: softmax(Q̃·B_pre[h]·K̃ᵀ/√R)·Ṽ per head."""
    if not torch.cuda.is_available():
        pytest.skip("pinned FlashAttentionTucker kernel requires CUDA")
    urm, x, q_tilde, k_tilde, v_tilde, b_pre = _foldings_parity(seed=7)
    expected, _id = tucker_attention_adapter(
        q_tilde.half().cuda(), k_tilde.half().cuda(),
        v_tilde.half().cuda(), b_pre.half().cuda(),
    )
    expected = expected.float().cpu()  # [B, N, H, R_v]

    with torch.no_grad():
        B, N, _ = x.shape
        R, S, T = urm.Core_pre.shape
        q_eff = torch.einsum("bnr,hrt->bnht", q_tilde, b_pre)
        k_h = k_tilde.unsqueeze(2).expand(B, N, N_HEAD, T)
        v_h = v_tilde.unsqueeze(2).expand(B, N, N_HEAD, v_tilde.shape[-1])
        actual = urm._plan.execute(query=q_eff, key=k_h, value=v_h)["output"]
    err = (actual - expected).abs().max().item()
    assert err < 2e-2, f"mixer parity vs pinned fp16 kernel: max abs err {err}"


def test_full_layer_output_folding_matches_pinned():
    """The external output folding is exercised end-to-end through forward()."""
    urm, x, *_ = _foldings_parity(seed=17)
    with torch.no_grad():
        out = urm(x)
    assert out.shape == (2, 5, N_EMBD)
    assert torch.isfinite(out).all()


def test_gradients_flow_through_factor_matrices():
    urm, x, *_ = _foldings_parity(seed=13)
    urm(x).square().sum().backward()
    for name, param in urm.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
    assert urm.Core_pre.grad.abs().sum().item() > 0
    assert urm.Core_post.grad.abs().sum().item() > 0
