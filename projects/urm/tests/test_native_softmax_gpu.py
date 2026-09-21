"""Differential tests for the URM native tiled online K1 implementation."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from urm.backends.triton.softmax.online import execute_online_softmax
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


def _reference_attention(query, key, value, attention_mask, score_bias, causal, scale):
    """Independent BTHD softmax reference used to check native K1 gradients."""
    q = query.permute(0, 2, 1, 3).float()
    k = key.permute(0, 2, 1, 3).float()
    v = value.permute(0, 2, 1, 3).float()
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if attention_mask is not None:
        if attention_mask.dtype is torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask.float()
    if score_bias is not None:
        scores = scores + score_bias.float()
    if causal:
        tq, tk = scores.shape[-2], scores.shape[-1]
        visible = torch.arange(tk, device=scores.device)[None, :] <= (
            torch.arange(tq, device=scores.device)[:, None] + tk - tq
        )
        scores = scores.masked_fill(~visible, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    # Fully masked rows must produce exact zeros, matching the native kernel.
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    return torch.matmul(probabilities, v).permute(0, 2, 1, 3)


# Singleton dimensions exercised one at a time across batch, head, query, and
# key axes, plus a fully singleton bias and an expanded (stride-zero) view.
_SINGLETON_BIAS_SHAPES = (
    (1, 4, 3, 5),  # broadcast over batch
    (2, 1, 3, 5),  # broadcast over head
    (2, 4, 1, 5),  # broadcast over query
    (2, 4, 3, 1),  # broadcast over key
    (1, 4, 1, 5),  # broadcast over batch and query (the reported case)
    (1, 1, 1, 1),  # broadcast over every dimension
)


@pytest.mark.parametrize("bias_shape", _SINGLETON_BIAS_SHAPES)
@pytest.mark.parametrize("expand_input", (False, True))
def test_native_k1_singleton_bias_gradient_stays_in_bounds(bias_shape, expand_input):
    """Singleton-shaped score_bias must accumulate broadcast gradients in bounds.

    Regression: the backward pass preserved nonzero strides for singleton
    dimensions, so a bias shaped e.g. [1, H, 1, K] broadcast to [B, H, Q, K]
    wrote gradients through offsets beyond its allocation instead of reducing
    the broadcast contributions into the single element.
    """
    if not torch.cuda.is_available():
        pytest.skip("native K1 requires CUDA")
    torch.manual_seed(20260921)
    batch, q_len, k_len, q_heads, key_dim, value_dim = 2, 3, 5, 4, 8, 8
    scale = key_dim**-0.5

    query = torch.randn(batch, q_len, q_heads, key_dim, device="cuda")
    key = torch.randn(batch, k_len, q_heads, key_dim, device="cuda")
    value = torch.randn(batch, k_len, q_heads, value_dim, device="cuda")
    bias_source = torch.randn(*bias_shape, device="cuda", dtype=torch.float32)
    grad_output = torch.randn(batch, q_len, q_heads, value_dim, device="cuda")

    def run(native):
        q = query.detach().requires_grad_()
        k = key.detach().requires_grad_()
        v = value.detach().requires_grad_()
        leaf = bias_source.detach().requires_grad_()
        # Expanded views carry stride zero on the broadcast axes; direct
        # singleton tensors carry their raw strides. Both must be correct.
        bias = leaf.expand(batch, q_heads, q_len, k_len) if expand_input else leaf
        if native:
            output = execute_online_softmax(
                q, k, v,
                attention_mask=None,
                score_bias=bias,
                causal=False,
                scale=scale,
            )
        else:
            output = _reference_attention(q, k, v, None, bias, False, scale)
        grads = torch.autograd.grad((output * grad_output).sum(), (q, k, v, leaf))
        return output, grads

    expected_out, expected_grads = run(native=False)
    actual_out, actual_grads = run(native=True)
    torch.testing.assert_close(actual_out, expected_out, atol=2e-4, rtol=8e-4)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=8e-4)


@pytest.mark.parametrize("mask_shape", _SINGLETON_BIAS_SHAPES)
def test_native_k1_singleton_additive_mask_gradient_stays_in_bounds(mask_shape):
    """Singleton-shaped additive masks must also reduce broadcast gradients."""
    if not torch.cuda.is_available():
        pytest.skip("native K1 requires CUDA")
    torch.manual_seed(20260922)
    batch, q_len, k_len, q_heads, key_dim, value_dim = 2, 3, 5, 4, 8, 8
    scale = key_dim**-0.5

    query = torch.randn(batch, q_len, q_heads, key_dim, device="cuda")
    key = torch.randn(batch, k_len, q_heads, key_dim, device="cuda")
    value = torch.randn(batch, k_len, q_heads, value_dim, device="cuda")
    mask_source = torch.randn(*mask_shape, device="cuda", dtype=torch.float32)
    grad_output = torch.randn(batch, q_len, q_heads, value_dim, device="cuda")

    def run(native):
        q = query.detach().requires_grad_()
        k = key.detach().requires_grad_()
        v = value.detach().requires_grad_()
        leaf = mask_source.detach().requires_grad_()
        if native:
            output = execute_online_softmax(
                q, k, v,
                attention_mask=leaf,
                score_bias=None,
                causal=False,
                scale=scale,
            )
        else:
            output = _reference_attention(q, k, v, leaf, None, False, scale)
        grads = torch.autograd.grad((output * grad_output).sum(), (q, k, v, leaf))
        return output, grads

    expected_out, expected_grads = run(native=False)
    actual_out, actual_grads = run(native=True)
    torch.testing.assert_close(actual_out, expected_out, atol=2e-4, rtol=8e-4)
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=8e-4)


def test_native_k1_gradient_strides_never_exceed_allocation():
    """The computed gradient strides must keep every write inside the buffer.

    The backward kernel indexes the gradient with the full broadcast target
    [B, H, Q, K], so the maximum reachable offset must stay within the logical
    gradient allocation even though the kernel loops over the broadcast shape.
    """
    from urm.backends.triton.softmax.online import _gradient_strides_4d

    broadcast_target = (2, 4, 3, 5)  # [B, H, Q, K]
    for shape in (
        (1, 4, 1, 5),
        (2, 1, 3, 5),
        (1, 1, 1, 1),
        (2, 4, 3, 5),
        (3, 5),  # rank-2 -> [1, 1, Q, K]
        (2, 3, 5),  # rank-3 -> [B, 1, Q, K]
    ):
        tensor = torch.zeros(*shape)
        strides = _gradient_strides_4d(tensor)
        logical = (
            shape
            if len(shape) == 4
            else (1, 1, *shape)
            if len(shape) == 2
            else (shape[0], 1, *shape[1:])
        )
        # The kernel iterates over the broadcast target, so use its extents.
        max_offset = sum(
            (extent - 1) * stride
            for extent, stride in zip(broadcast_target, strides)
        )
        assert max_offset < math.prod(logical), (shape, strides, max_offset)


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
