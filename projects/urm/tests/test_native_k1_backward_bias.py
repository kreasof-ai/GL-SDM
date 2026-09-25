"""Parity gate: the native K1 two-pass backward is correct with score_bias/mask.

The native online-softmax backward runs two SMEM-safe passes (query-parallel dq
+ key-parallel dk/dv) at ``head_dim <= 64``. Those passes originally ignored the
``score_bias``/``attention_mask`` operands entirely, so any layer whose K1 score
carries an additive bias (FoX cumulative gate, PaTH Householder correction, ...)
produced wrong cotangents on the native tier. These tests pin forward AND
cotangent parity against a manual PyTorch reference (and torch SDPA for the
plain-causal path) at the SMEM-bound configs that motivated the two-pass split.

CUDA + Triton are required; the tests skip cleanly when either is unavailable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

import torch.nn.functional as F

from urm.backends.triton.k1.online_softmax import execute_online_softmax

DEV = "cuda"
LOG2E = 1.4426950408889634


def _manual_reference(q, k, v, *, bias=None, bool_mask=None, causal=True, scale):
    """Manual causal softmax attention: softmax((q@k.T)*scale + bias) @ v.

    Operands are BTHD; the reduction runs head-major in fp32 so it stays a
    faithful oracle for the fp32 native accumulation.
    """
    qh = q.permute(0, 2, 1, 3).float()  # [B,H,T,D]
    kh = k.permute(0, 2, 1, 3).float()
    vh = v.permute(0, 2, 1, 3).float()
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale  # [B,H,Tq,Tk]
    if bias is not None:
        scores = scores + bias.float()
    tq, tk = scores.shape[-2], scores.shape[-1]
    if causal:
        causal_mask = torch.ones(tq, tk, dtype=torch.bool, device=scores.device).tril_(
            diagonal=tk - tq
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if bool_mask is not None:
        m = bool_mask
        if m.dim() == 2:
            m = m.view(1, 1, *m.shape)
        elif m.dim() == 3:
            m = m.unsqueeze(1)
        scores = scores.masked_fill(~m, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, vh)  # [B,H,Tq,DV]
    return out.permute(0, 2, 1, 3)  # back to BTHD


def _operands(seed, b, h, t, d, *, bias=False, bool_mask=False):
    torch.manual_seed(seed)
    q = torch.randn(b, t, h, d, device=DEV)
    k = torch.randn(b, t, h, d, device=DEV)
    v = torch.randn(b, t, h, d, device=DEV)
    score_bias = (
        torch.randn(b, h, t, t, device=DEV) * 0.5 if bias else None
    )
    if bool_mask:
        # Boolean keep-mask: causal-and then drop a deterministic band so the
        # mask is not redundant with causality.
        keep = torch.ones(b, h, t, t, dtype=torch.bool, device=DEV).tril()
        band = torch.ones(t, t, dtype=torch.bool, device=DEV).tril(diagonal=-3)
        keep = keep & band
    else:
        keep = None
    return q, k, v, score_bias, keep


@pytest.mark.parametrize("t", [128, 512])
def test_native_k1_backward_with_score_bias_causal(t):
    """score_bias (causal): forward + dq/dk/dv/dbias parity vs manual reference."""
    b, h, d = 2, 4, 64
    scale = d ** -0.5
    q, k, v, score_bias, _ = _operands(11, b, h, t, d, bias=True)

    torch.manual_seed(123)  # shared cotangent for both runs
    cot = torch.randn(b, t, h, d, device=DEV)

    def run_native():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, score_bias)]
        out = execute_online_softmax(
            *leaves[:3], attention_mask=None, score_bias=leaves[3],
            causal=True, scale=scale,
        )
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    def run_reference():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, score_bias)]
        out = _manual_reference(*leaves[:3], bias=leaves[3], causal=True, scale=scale)
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    out_n, grads_n = run_native()
    out_r, grads_r = run_reference()

    fwd_err = (out_n - out_r).abs().max().item()
    assert fwd_err < 1e-4, f"forward parity: max abs err {fwd_err}"
    for name, g_n, g_r in zip(("dq", "dk", "dv", "dbias"), grads_n, grads_r):
        assert torch.isfinite(g_n).all(), f"d{name} not finite"
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"cotangent {name}: max abs err {err}"


@pytest.mark.parametrize("t", [128, 512])
def test_native_k1_backward_plain_causal_matches_sdpa(t):
    """No bias: dq/dk/dv parity vs torch SDPA (is_causal=True)."""
    b, h, d = 2, 4, 64
    scale = d ** -0.5
    q, k, v, _, _ = _operands(21, b, h, t, d)

    torch.manual_seed(321)
    cot = torch.randn(b, t, h, d, device=DEV)

    def run_native():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        out = execute_online_softmax(
            *leaves, attention_mask=None, score_bias=None, causal=True, scale=scale,
        )
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    def run_sdpa():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        qh, kh, vh = (x.permute(0, 2, 1, 3) for x in leaves)  # [B,H,T,D]
        out = F.scaled_dot_product_attention(
            qh, kh, vh, is_causal=True, scale=scale
        )
        out = out.permute(0, 2, 1, 3)
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    out_n, grads_n = run_native()
    out_s, grads_s = run_sdpa()
    fwd_err = (out_n - out_s).abs().max().item()
    assert fwd_err < 1e-4, f"forward parity vs SDPA: max abs err {fwd_err}"
    for name, g_n, g_s in zip(("dq", "dk", "dv"), grads_n, grads_s):
        assert torch.isfinite(g_n).all(), f"d{name} not finite"
        err = (g_n - g_s).abs().max().item()
        assert err < 1e-3, f"cotangent {name} vs SDPA: max abs err {err}"


@pytest.mark.parametrize("t", [128, 512])
def test_native_k1_backward_with_bool_mask_causal(t):
    """Boolean attention_mask (causal): forward + dq/dk/dv parity vs reference."""
    b, h, d = 2, 4, 64
    scale = d ** -0.5
    q, k, v, _, keep = _operands(31, b, h, t, d, bool_mask=True)

    torch.manual_seed(555)
    cot = torch.randn(b, t, h, d, device=DEV)

    def run_native():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        out = execute_online_softmax(
            *leaves, attention_mask=keep, score_bias=None, causal=True, scale=scale,
        )
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    def run_reference():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        out = _manual_reference(
            *leaves, bool_mask=keep, causal=True, scale=scale
        )
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    out_n, grads_n = run_native()
    out_r, grads_r = run_reference()
    fwd_err = (out_n - out_r).abs().max().item()
    assert fwd_err < 1e-4, f"forward parity (bool mask): max abs err {fwd_err}"
    for name, g_n, g_r in zip(("dq", "dk", "dv"), grads_n, grads_r):
        assert torch.isfinite(g_n).all(), f"d{name} not finite"
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"cotangent {name} (bool mask): max abs err {err}"


def test_native_k1_backward_additive_float_mask_grad():
    """A floating-point (additive) attention_mask receives a cotangent too."""
    b, h, t, d = 2, 4, 128, 64
    scale = d ** -0.5
    q, k, v, _, _ = _operands(41, b, h, t, d)
    float_mask = torch.randn(b, h, t, t, device=DEV) * 0.25

    torch.manual_seed(777)
    cot = torch.randn(b, t, h, d, device=DEV)

    def run_native():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, float_mask)]
        out = execute_online_softmax(
            *leaves[:3], attention_mask=leaves[3], score_bias=None,
            causal=True, scale=scale,
        )
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    def run_reference():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v, float_mask)]
        out = _manual_reference(*leaves[:3], bias=leaves[3], causal=True, scale=scale)
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    out_n, grads_n = run_native()
    out_r, grads_r = run_reference()
    fwd_err = (out_n - out_r).abs().max().item()
    assert fwd_err < 1e-4, f"forward parity (float mask): max abs err {fwd_err}"
    for name, g_n, g_r in zip(("dq", "dk", "dv", "dmask"), grads_n, grads_r):
        assert torch.isfinite(g_n).all(), f"d{name} not finite"
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"cotangent {name} (float mask): max abs err {err}"
