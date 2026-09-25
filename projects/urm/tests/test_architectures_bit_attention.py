"""Parity gates for arch-014 BitAttention.

Two verification levels, both against the pinned fla source (the sweep's
verification origin, fla @ 864a87f6):

1. Projection parity — ``architectures.bit_attention.BitLinear`` vs the pinned
   fused Triton kernel ``LayerNormLinearQuantFn`` on identical weights:
   forward values AND the straight-through gradient policy (GPU; the pinned
   kernels are Triton).

2. Layer parity — ``BitAttentionLayer`` vs the composition of the transcribed
   BitLinear projections with the pinned fla naive attention mixer (CPU, fp32),
   isolating the K1 graph execution against the verified U1.S equation.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.bit_attention import (
    BitAttentionLayer,
    BitLinear,
    activation_quant,
    weight_quant,
)
from benchmarks.comparators.fla_attention_naive import fla_naive_attention_adapter
from benchmarks.comparators.fla_bitlinear import fla_fused_bitlinear_adapter


def _bitlinear_pair(seed: int, device: str, in_f: int = 32, out_f: int = 32):
    torch.manual_seed(seed)
    norm_weight = torch.randn(in_f, device=device)
    norm_bias = torch.zeros(in_f, device=device)
    weight = torch.randn(out_f, in_f, device=device)
    x = torch.randn(2, 7, in_f, device=device)
    return x, norm_weight, norm_bias, weight


def test_activation_and_weight_quant_match_pinned_equation():
    # Mirror the pinned definitions exactly on a known vector.
    x = torch.tensor([[0.0, 1.0, -2.0, 3.5]])
    scale_x = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    expected_x = (x * scale_x).round().clamp(-128, 127) / scale_x
    assert torch.equal(activation_quant(x), expected_x)

    w = torch.tensor([[0.1, -0.4, 2.0, -1.0]])
    scale_w = 1.0 / w.abs().mean().clamp(min=1e-5)
    expected_w = (w * scale_w).round().clamp(-1, 1) / scale_w
    assert torch.equal(weight_quant(w), expected_w)
    # 1.58-bit: quantized weights take values in {-1, 0, +1} before rescaling.
    assert set(torch.unique(expected_w * scale_w).tolist()) <= {-1.0, 0.0, 1.0}


def test_bitlinear_forward_matches_pinned_fused_kernel_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused BitLinear kernels require CUDA")
    x, norm_weight, norm_bias, weight = _bitlinear_pair(seed=11, device="cuda")
    module = BitLinear(32, 32, bias=False).to("cuda")
    with torch.no_grad():
        module.norm.weight.copy_(norm_weight)
        module.weight.copy_(weight)

    expected, _identity = fla_fused_bitlinear_adapter(
        x, norm_weight, norm_bias, weight, None
    )
    actual = module(x)
    err = (actual - expected).abs().max().item()
    assert err < 2e-4, f"BitLinear forward vs pinned fused kernel: max abs err {err}"


def test_bitlinear_gradient_policy_matches_pinned_cuda():
    """The pinned STE gradient: identity through rounding, real grads through RMSNorm."""
    if not torch.cuda.is_available():
        pytest.skip("pinned fused BitLinear kernels require CUDA")
    x, norm_weight, norm_bias, weight = _bitlinear_pair(seed=17, device="cuda")

    # Pinned path.
    x_p = x.clone().requires_grad_(True)
    nw_p = norm_weight.clone().requires_grad_(True)
    w_p = weight.clone().requires_grad_(True)
    expected, _ = fla_fused_bitlinear_adapter(x_p, nw_p, norm_bias, w_p, None)
    expected.square().sum().backward()

    # URM external-module path.
    module = BitLinear(32, 32, bias=False).to("cuda")
    with torch.no_grad():
        module.norm.weight.copy_(norm_weight)
        module.weight.copy_(weight)
    x_u = x.clone().requires_grad_(True)
    module(x_u).square().sum().backward()

    for name, a, b in (
        ("dx", x_u.grad, x_p.grad),
        ("d(norm_weight)", module.norm.weight.grad, nw_p.grad),
        ("d(weight)", module.weight.grad, w_p.grad),
    ):
        err = (a - b).abs().max().item()
        scale = b.abs().max().item() + 1e-12
        assert err / scale < 1e-3, f"{name}: max abs err {err} (scale {scale})"


def test_bit_attention_layer_matches_pinned_mixer_cpu():
    """Full layer: transcribed BitLinear projections + K1 graph vs the pinned
    naive attention equation over the same projections."""
    torch.manual_seed(23)
    module = BitAttentionLayer(hidden_size=32, num_heads=4, num_kv_heads=2)
    hidden = torch.randn(2, 7, 32)
    B, T, _ = hidden.shape
    with torch.no_grad():
        q = module.q_proj(hidden).view(B, T, 4, 8)
        k = module.k_proj(hidden).view(B, T, 2, 8)
        v = module.v_proj(hidden).view(B, T, 2, 8)
        expected, _ = fla_naive_attention_adapter(q, k, v, causal=True)
        expected = module.o_proj(expected.reshape(B, T, -1))
        actual = module(hidden)
    err = (actual - expected).abs().max().item()
    assert err < 2e-5, f"layer parity: max abs err {err}"


def test_layer_gradient_flows_through_quantized_projections():
    module = BitAttentionLayer(hidden_size=32, num_heads=4)
    hidden = torch.randn(2, 7, 32)
    module(hidden).square().sum().backward()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert param.grad.abs().sum().item() > 0, f"{name} gradient is exactly zero"
