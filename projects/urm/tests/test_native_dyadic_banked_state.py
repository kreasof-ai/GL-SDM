"""Gate: the native Triton DyadicBankedState tier (A4) matches the torch reference.

The native provider (``urm_native_dyadic_banked_state_v1``,
``urm.backends.triton.k2.dyadic_banks``) runs the banked dyadic recurrence
in the decay-forward form with fp32 accumulation: one program owns a
(batch·head, value-block) fragment and walks the token axis in order, keeping
the ``num_levels−1`` bank slots in a program-exclusive fp32 global state buffer
(the ordered carry-cascade lifecycle is structural, never atomic). The backward
is the exact adjoint reverse scan against the per-step post-decay slots saved by
the forward; the per-token operand cotangents reduce across value blocks with
relaxed atomics (the K3 native backward policy).

This module gates forward AND cotangent parity against the pinned torch
reference (``urm.backends.torch.dyadic_banked_state``) in fp32, the structured
decline surface, and the two native-tier clients — BankedLogLinearMixer
(arch-009) and LogLinearMamba2Layer (arch-046) — bound to the native anchor
with finite gradients and reference-tier parity.

CUDA + Triton are required; the tests skip cleanly when either is unavailable.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from architectures.log_linear_attention import BankedLogLinearMixer
from architectures.log_linear_mamba2 import LogLinearMamba2Layer
from urm.backends.contract import ProviderFamily, ProviderRequest
from urm.backends.torch.k2.dyadic_banks import dyadic_banked_state_forward
from urm.backends.triton.k2.dyadic_banks import (
    NATIVE_DYADIC_BANKED_STATE_NAME,
    DyadicBankedStateNativeTritonProvider,
    dyadic_banked_state,
)
from urm.ir.program import DyadicBankedState

DEV = "cuda"


def _operands(seed: int, B: int, T: int, H: int, D: int, L: int):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D, device=DEV)
    k = torch.randn(B, T, H, D, device=DEV)
    v = torch.randn(B, T, H, D, device=DEV)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=DEV))
    ls = torch.randn(B, T, H, L, device=DEV)
    return q, k, v, g, ls


def _request(num_levels: int, **overrides) -> ProviderRequest:
    fields = dict(
        family=ProviderFamily.DYADIC_BANKED_STATE,
        descriptor=DyadicBankedState(
            name="bank",
            inputs=("query", "key", "value", "log_decay", "level_scales"),
            outputs=("output",),
            num_levels=num_levels,
        ),
        mode="training",
        accumulation_dtype="float32",
    )
    fields.update(overrides)
    return ProviderRequest(**fields)


@pytest.mark.parametrize(
    "B, T, H, D, L",
    [
        (2, 8, 2, 8, 4),    # exact dyadic blocks
        (2, 7, 3, 16, 4),   # partial dyadic block tail
        (2, 13, 2, 32, 5),  # partial dyadic block tail
        (2, 24, 4, 64, 6),  # the client head_dim
        (1, 64, 4, 64, 7),  # full-length hierarchy
    ],
)
def test_native_dyadic_banked_state_forward_matches_reference(B, T, H, D, L):
    """Native forward parity vs the pinned torch reference, fp32, incl. partial blocks."""
    q, k, v, g, ls = _operands(5 + T, B, T, H, D, L)
    expected = dyadic_banked_state_forward(q, k, v, g, ls, L)
    actual = dyadic_banked_state(q, k, v, g, ls, L)
    assert actual.dtype == torch.float32
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, f"forward parity: max abs err {err}"


@pytest.mark.parametrize("B, T, H, D, L", [(2, 8, 2, 8, 4), (2, 24, 4, 64, 6)])
def test_native_dyadic_banked_state_cotangents_match_reference(B, T, H, D, L):
    """Operand cotangents (q, k, v, log_decay, level_scales) match the reference."""
    base = _operands(17 + T, B, T, H, D, L)
    cotangent = torch.randn(B, T, H, D, device=DEV)

    def run(fn):
        operands = [x.clone().requires_grad_(True) for x in base]
        out = fn(*operands, L)
        return torch.autograd.grad(out, operands, cotangent)

    native = run(dyadic_banked_state)
    reference = run(dyadic_banked_state_forward)
    for name, actual, expected in zip(("q", "k", "v", "g", "ls"), native, reference):
        err = (actual - expected).abs().max().item()
        assert err < 1e-4, f"cotangent d{name}: max abs err {err}"


def test_native_dyadic_banked_state_provider_decline_is_structured():
    """The provider declines honestly: wrong descriptor type, non-fp32 accumulation."""
    provider = DyadicBankedStateNativeTritonProvider()
    assert provider.decline(_request(4)) is None
    wrong = _request(4, descriptor=object())
    assert provider.decline(wrong) is not None
    half = _request(4, accumulation_dtype="float16")
    assert "float32" in provider.decline(half)


def test_native_target_binds_native_anchor():
    """target="native" selects the native anchor; target="reference" the reference one."""
    native = BankedLogLinearMixer(4, 64, 4, target="native", intent="training")
    reference = BankedLogLinearMixer(4, 64, 4, target="reference", intent="training")
    native_anchors = [s.anchor for s in native._plan.compilation.plan.steps]
    reference_anchors = [s.anchor for s in reference._plan.compilation.plan.steps]
    assert native_anchors == [NATIVE_DYADIC_BANKED_STATE_NAME]
    assert reference_anchors == ["urm.unified.dyadic_banked_state.reference.v1"]


def test_banked_log_linear_mixer_native_training_matches_reference_tier():
    """BankedLogLinearMixer(4, 64, 4, native, training): finite grads + reference parity."""
    torch.manual_seed(11)
    B, T, H, D, L = 2, 64, 4, 64, 4
    base = _operands(23, B, T, H, D, L)
    cotangent = torch.randn(B, T, H, D, device=DEV)
    native = BankedLogLinearMixer(H, D, L, target="native", intent="training")
    reference = BankedLogLinearMixer(H, D, L, target="reference", intent="training")

    def run(mixer):
        operands = [x.clone().requires_grad_(True) for x in base]
        out = mixer(*operands)
        assert torch.isfinite(out).all()
        return out, torch.autograd.grad(out, operands, cotangent)

    out_n, grads_n = run(native)
    out_r, grads_r = run(reference)
    assert (out_n - out_r).abs().max().item() < 1e-4
    for name, actual, expected in zip(("q", "k", "v", "g", "ls"), grads_n, grads_r):
        assert actual is not None and torch.isfinite(actual).all()
        err = (actual - expected).abs().max().item()
        assert err < 1e-4, f"mixer cotangent d{name}: max abs err {err}"


def test_log_linear_mamba2_native_training_matches_reference_tier():
    """LogLinearMamba2Layer(256, 4, 64, 4, native, training): the Mamba-2 frontend
    driving the native banked op; weights shared via load_state_dict."""
    reference = LogLinearMamba2Layer(256, 4, 64, 4, target="reference", intent="training")
    native = LogLinearMamba2Layer(256, 4, 64, 4, target="native", intent="training")
    native.load_state_dict(reference.state_dict())
    reference = reference.to(DEV)
    native = native.to(DEV)
    anchors = [s.anchor for s in native._mixer._plan.compilation.plan.steps]
    assert anchors == [NATIVE_DYADIC_BANKED_STATE_NAME]

    torch.manual_seed(3)
    B, T, H, D = 2, 64, 4, 64
    x = torch.randn(B, T, 256, device=DEV)
    cotangent = torch.randn(B, T, H, D, device=DEV)

    x_r = x.clone().requires_grad_(True)
    out_r = reference(x_r)
    grads_r = torch.autograd.grad(out_r, [x_r, *reference.parameters()], cotangent)
    x_n = x.clone().requires_grad_(True)
    out_n = native(x_n)
    assert torch.isfinite(out_n).all()
    grads_n = torch.autograd.grad(out_n, [x_n, *native.parameters()], cotangent)

    assert (out_n - out_r).abs().max().item() < 1e-4
    assert torch.isfinite(grads_n[0]).all()
    assert (grads_n[0] - grads_r[0]).abs().max().item() < 1e-4
    for (name, _), actual, expected in zip(
        reference.named_parameters(), grads_n[1:], grads_r[1:]
    ):
        assert actual is not None and torch.isfinite(actual).all()
        err = (actual - expected).abs().max().item()
        assert err < 2e-4, f"mamba2 parameter cotangent {name}: max abs err {err}"
