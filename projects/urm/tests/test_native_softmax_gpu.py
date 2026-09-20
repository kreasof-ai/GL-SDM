"""Differential tests for the URM native tiled online K1 implementation."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from urm.compiler.unified_mixer import MixerBackend, compile_mixer
from urm.frontend.mixer_recipes import softmax_attention_spec


def _dtype_cases():
    return (
        (torch.float32, 2e-4, 8e-4),
        (torch.float16, 2e-2, 3e-2),
        (torch.bfloat16, 4e-2, 6e-2),
    )


@pytest.mark.parametrize("dtype,atol,rtol", _dtype_cases())
@pytest.mark.parametrize("mask_kind", ("boolean", "additive"))
def test_native_k1_matches_reference_outputs_and_gradients(dtype, atol, rtol, mask_kind):
    if not torch.cuda.is_available():
        pytest.skip("native K1 requires CUDA")
    torch.manual_seed(8021)
    batch, q_len, k_len, q_heads, kv_heads, key_dim, value_dim = 2, 3, 5, 4, 2, 7, 13
    base_q = torch.randn(batch, q_heads, q_len, key_dim, device="cuda", dtype=dtype)
    base_k = torch.randn(batch, kv_heads, k_len, key_dim, device="cuda", dtype=dtype)
    base_v = torch.randn(batch, kv_heads, k_len, value_dim, device="cuda", dtype=dtype)
    # Non-contiguous BTHD views exercise the explicit layout conversion path.
    tensors = [
        base_q.transpose(1, 2),
        base_k.transpose(1, 2),
        base_v.transpose(1, 2),
    ]
    bias_source = torch.randn(1, q_heads, 1, k_len, device="cuda", dtype=torch.float32)
    bias_source.requires_grad_()
    if mask_kind == "boolean":
        mask = torch.ones(batch, 1, q_len, k_len, device="cuda", dtype=torch.bool)
        mask[0, 0, 1, :] = False
    else:
        mask = torch.zeros(batch, q_len, k_len, device="cuda", dtype=torch.float32)
        mask[0, 1, :] = float("-inf")
        mask.requires_grad_()

    spec = softmax_attention_spec("native_k1", score_bias=True)
    dtype_name = str(dtype).removeprefix("torch.")
    reference = compile_mixer(spec, backend=MixerBackend.REFERENCE, dtype=dtype_name)
    native = compile_mixer(spec, backend=MixerBackend.NATIVE, dtype=dtype_name)
    assert native.anchor == "urm_native_k1_online_softmax_v1"

    def execute(plan):
        q, k, v = [tensor.detach().requires_grad_() for tensor in tensors]
        bias_leaf = bias_source.detach().requires_grad_()
        score_bias = bias_leaf.expand(batch, q_heads, q_len, k_len)
        route_mask = mask.detach()
        if mask_kind == "additive":
            route_mask.requires_grad_()
        result = plan.execute(
            query=q,
            key=k,
            value=v,
            attention_mask=route_mask,
            score_bias=score_bias,
        ).output
        grad_output = loss_gradient
        gradients = torch.autograd.grad(
            (result * grad_output).sum(),
            (q, k, v, bias_leaf, *((route_mask,) if route_mask.requires_grad else ())),
        )
        return result, gradients

    loss_gradient = torch.randn(
        batch, q_len, q_heads, value_dim, device="cuda", dtype=dtype
    )
    expected, expected_grads = execute(reference)
    actual, actual_grads = execute(native)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert len(actual_grads) == len(expected_grads)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)
    # The selected fully masked row is defined as exact zero with a zero Q VJP.
    assert torch.count_nonzero(actual[0, 1]) == 0
    assert torch.count_nonzero(actual_grads[0][0, 1]) == 0
    if mask_kind == "additive":
        assert torch.count_nonzero(actual_grads[-1][0, 1]) == 0


@pytest.mark.parametrize("dtype,atol,rtol", _dtype_cases())
def test_native_k1_supports_noncausal_query_longer_than_key(dtype, atol, rtol):
    if not torch.cuda.is_available():
        pytest.skip("native K1 requires CUDA")
    torch.manual_seed(8022)
    query = torch.randn(1, 5, 2, 7, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn(1, 3, 1, 7, device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(1, 3, 1, 9, device="cuda", dtype=dtype, requires_grad=True)
    spec = softmax_attention_spec("cross_attention", causal=False)
    dtype_name = str(dtype).removeprefix("torch.")
    expected = compile_mixer(spec, dtype=dtype_name).execute(
        query=query, key=key, value=value
    ).output
    actual = compile_mixer(spec, backend=MixerBackend.NATIVE, dtype=dtype_name).execute(
        query=query, key=key, value=value
    ).output
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
