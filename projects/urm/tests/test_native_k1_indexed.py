"""Parity gate: the native Triton indexed K1 gather-attend (the A2 law).

The native provider ``urm_native_k1_indexed_gather_v1`` executes the indexed
gather-attend — gather the per-query source set given by the external
``gather_indices`` route ([B, HK, T, W], -1 = padding → masked), softmax over
the W gathered scores, weighted value sum — with fp32 accumulation. The forward
loops the W slots gathering per-row K/V; the backward computes dq directly and
scatters dk/dv to the gathered source positions through relaxed atomics (the K3
native policy), so cross-program accumulation order is not guaranteed.

These tests pin forward AND cotangent parity (dq/dk/dv) against the Torch
reference indexed path (``urm.backends.torch.k1``), including GQA (HQ != HK)
and -1 padding / fully-masked rows. They also run two A2 clients (MoBA,
Longformer) forward+backward natively on CUDA with finite cotangents, matching
the reference tier. CUDA + Triton are required; the tests skip cleanly without
either.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.backends.contract import ProviderRequest
from urm.backends.torch.k1 import k1_softmax_attention as torch_k1
from urm.backends.triton.k1 import K1NativeIndexedTritonProvider, execute_indexed_k1
from urm.ir.program import K1Descriptor, K1ReducerLaw, K1ScoreLaw

DEV = "cuda"
INDEXED = K1Descriptor(indexed=True)


def _request(descriptor):
    return ProviderRequest(
        family="k1", descriptor=descriptor, mode="inference", accumulation_dtype="float32"
    )


def test_indexed_provider_decline_contract():
    """The native indexed provider serves ONLY indexed DOT+SOFTMAX descriptors."""
    provider = K1NativeIndexedTritonProvider()
    # Accepts the indexed softmax law (CUDA present in this environment).
    assert provider.decline(_request(INDEXED)) is None
    # Declines the dense softmax law (owned by the dense native anchor)...
    assert provider.decline(_request(K1Descriptor())) is not None
    # ...the channel-decay score law (here declined as non-indexed)...
    assert (
        provider.decline(_request(K1Descriptor(score_law=K1ScoreLaw.CHANNEL_DECAY)))
        is not None
    )
    # ...and an indexed non-softmax reducer.
    assert (
        provider.decline(
            _request(
                K1Descriptor(
                    indexed=True,
                    reducer_law=K1ReducerLaw.THRESHOLD_RELU_POWER,
                    threshold_beta=0.5,
                    relu_power=2.0,
                )
            )
        )
        is not None
    )


def _gather_indices(seed, b, hk, t, w, s, *, pad=True, masked_row=False):
    """A random external route [B, HK, T, W] of source positions (int64)."""
    torch.manual_seed(seed)
    idx = torch.randint(0, s, (b, hk, t, w), device=DEV)
    if pad:
        idx[:, :, :2, -1] = -1  # trailing slot padded on the first queries
    if masked_row:
        idx[:, :, t // 2, :] = -1  # a fully masked query row (returns zero)
    return idx


def _parity(seed, b, t, hq, hk, d, s, w, *, pad=True, masked_row=False):
    """Forward + dq/dk/dv parity vs the Torch reference indexed path."""
    torch.manual_seed(seed + 1000)
    q = torch.randn(b, t, hq, d, device=DEV)
    k = torch.randn(b, s, hk, d, device=DEV)
    v = torch.randn(b, s, hk, d, device=DEV)
    idx = _gather_indices(seed, b, hk, t, w, s, pad=pad, masked_row=masked_row)
    cot = torch.randn(b, t, hq, d, device=DEV)
    scale = d ** -0.5

    def run_native():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        out = execute_indexed_k1(*leaves, idx.float(), scale=scale)
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    def run_reference():
        leaves = [x.clone().requires_grad_(True) for x in (q, k, v)]
        out = torch_k1(
            *leaves, descriptor=INDEXED, gather_indices=idx.float()
        )
        grads = torch.autograd.grad(out, leaves, grad_outputs=cot)
        return out, grads

    out_n, grads_n = run_native()
    out_r, grads_r = run_reference()
    fwd_err = (out_n - out_r).abs().max().item()
    assert fwd_err < 1e-4, f"forward parity: max abs err {fwd_err}"
    for name, g_n, g_r in zip(("dq", "dk", "dv"), grads_n, grads_r):
        assert torch.isfinite(g_n).all(), f"d{name} not finite"
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"cotangent {name}: max abs err {err}"
    return fwd_err


def test_native_indexed_forward_backward_equal_heads():
    """Equal head map (HK == HQ), with -1 padding."""
    _parity(0, 2, 16, 2, 2, 16, 16, 5)


def test_native_indexed_forward_backward_gqa():
    """Grouped head map: HQ != HK (each query head shares its KV head's route)."""
    _parity(1, 2, 16, 4, 2, 16, 16, 6)


def test_native_indexed_fully_masked_row_returns_zero():
    """A query whose gathered set is entirely -1 padding returns a zero row."""
    _parity(2, 2, 16, 2, 2, 16, 16, 4, masked_row=True)


@pytest.mark.parametrize("w", [2, 8, 17, 64])
def test_native_indexed_gather_widths(w):
    """The W loop is correct across small and power-of-2 gather widths."""
    _parity(3 + w, 2, 24, 2, 2, 32, 24, w)


def test_native_indexed_head_dim_64():
    """The SMEM-bound config: head_dim=64 stays under the A10G's 101KB."""
    _parity(11, 2, 64, 4, 2, 64, 64, 64)


def test_native_indexed_no_padding():
    """A dense (fully valid) route with no -1 padding."""
    _parity(21, 2, 16, 2, 2, 16, 16, 5, pad=False)


# --- A2 clients run forward+backward natively on CUDA ---

def _client_backward_finite(layer, args, n_diff=3):
    """The native-tier client forward+backward must produce finite cotangents.

    Only the first ``n_diff`` operands (q, k, v) carry cotangents; the rest
    (the integer gather route) are passed but not differentiated.
    """
    leaves = tuple(
        x.clone().requires_grad_(i < n_diff) for i, x in enumerate(args)
    )
    out = layer(*leaves)
    cot = torch.randn_like(out)
    grads = torch.autograd.grad(out, leaves[:n_diff], grad_outputs=cot)
    assert torch.isfinite(out).all(), "client forward not finite"
    for name, g in zip(("dq", "dk", "dv"), grads):
        assert torch.isfinite(g).all(), f"client d{name} not finite"
    return out, grads, cot


def test_moba_layer_native_matches_reference():
    from architectures.moba import MoBALayer

    torch.manual_seed(5)
    h, d, t = 2, 16, 16
    q = torch.randn(1, t, h, d, device=DEV)
    k = torch.randn(1, t, h, d, device=DEV)
    v = torch.randn(1, t, h, d, device=DEV)
    # MoBA routes internally (forward(q, k, v)); the route is deterministic, so
    # both tiers gather the same source set.
    native = MoBALayer(h, d, chunk_size=4, topk=2, target="native").to(DEV)
    out_n, grads_n, cot = _client_backward_finite(native, (q, k, v))

    reference = MoBALayer(h, d, chunk_size=4, topk=2, target="reference").to(DEV)
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    out_r = reference(qr, kr, vr)
    grads_r = torch.autograd.grad(out_r, (qr, kr, vr), grad_outputs=cot)

    fwd_err = (out_n - out_r).abs().max().item()
    assert fwd_err < 1e-4, f"MoBA native-vs-reference forward: {fwd_err}"
    for name, g_n, g_r in zip(("dq", "dk", "dv"), grads_n, grads_r):
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"MoBA native-vs-reference d{name}: {err}"


def test_longformer_layer_native_matches_reference():
    from architectures.longformer import LongformerLayer

    torch.manual_seed(7)
    h, d, t = 2, 16, 16
    q = torch.randn(1, t, h, d, device=DEV)
    k = torch.randn(1, t, h, d, device=DEV)
    v = torch.randn(1, t, h, d, device=DEV)
    native = LongformerLayer(h, d, window=3, target="native").to(DEV)
    gather = native.build_gather_indices(t, global_positions=[0], device=DEV)
    out_n, grads_n, cot = _client_backward_finite(native, (q, k, v, gather))

    reference = LongformerLayer(h, d, window=3, target="reference").to(DEV)
    qr, kr, vr = (x.clone().requires_grad_(True) for x in (q, k, v))
    out_r = reference(qr, kr, vr, gather)
    grads_r = torch.autograd.grad(out_r, (qr, kr, vr), grad_outputs=cot)

    fwd_err = (out_n - out_r).abs().max().item()
    assert fwd_err < 1e-4, f"Longformer native-vs-reference forward: {fwd_err}"
    for name, g_n, g_r in zip(("dq", "dk", "dv"), grads_n, grads_r):
        err = (g_n - g_r).abs().max().item()
        assert err < 1e-3, f"Longformer native-vs-reference d{name}: {err}"
