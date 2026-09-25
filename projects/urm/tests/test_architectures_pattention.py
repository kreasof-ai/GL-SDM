"""Parity gates for arch-057 Pattention (parameter-token attention).

Verified against the pinned TokenFormer source (megatron/model/tokenformer.py @
4d56c73f, the sweep's verification origin): the parameter-domain mixer
(plain-inner-product scores, external nonlinear normalizer, value contraction)
matches the pinned Pattention module on identical parameter tokens across all
three normalizer variants, and parameter-token gradients flow.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.pattention import PattentionLayer
from benchmarks.comparators.pattention import patention_adapter

IN_DIM, OUT_DIM, N_TOKENS = 16, 24, 8
NORMALIZERS = ("softmax", "gelu_l2_norm", "l2_norm_gelu")


def _build_pair(seed: int, norm: str, scale: float | None = None):
    torch.manual_seed(seed)
    x = torch.randn(2, 5, IN_DIM)
    expected, pinned, _id = patention_adapter(
        x,
        input_channels=IN_DIM, output_channels=OUT_DIM, param_token_num=N_TOKENS,
        norm_activation_type=norm, seed=seed, scale=scale,
    )
    urm = PattentionLayer(IN_DIM, OUT_DIM, N_TOKENS, norm_activation_type=norm)
    with torch.no_grad():
        urm.key_param_tokens.copy_(pinned.key_param_tokens)
        urm.value_param_tokens.copy_(pinned.value_param_tokens)
    return expected, urm, x, scale


@pytest.mark.parametrize("norm", NORMALIZERS)
def test_layer_matches_pinned_pattention(norm):
    expected, urm, x, scale = _build_pair(seed=5, norm=norm)
    with torch.no_grad():
        actual = urm(x, scale=scale)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"{norm} parity: max abs err {err}"


def test_non_unit_scale_matches_pinned():
    expected, urm, x, _ = _build_pair(seed=11, norm="softmax", scale=0.7)
    with torch.no_grad():
        actual = urm(x, scale=0.7)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"scaled parity: max abs err {err}"


def test_parameter_token_gradients_flow():
    """Gradients reach both parameter-token banks through the contraction."""
    _expected, urm, x, scale = _build_pair(seed=17, norm="gelu_l2_norm")
    urm(x).square().sum().backward()
    assert urm.key_param_tokens.grad is not None
    assert urm.key_param_tokens.grad.abs().sum().item() > 0
    assert urm.value_param_tokens.grad is not None
    assert urm.value_param_tokens.grad.abs().sum().item() > 0
