"""Actual pinned CUDA baseline, autocast boundary, padding and state lifetime."""
import pytest
import torch

from architectures.sdm_chunked import chunked_sparse_delta_memory
from extra.comparators.sdm.cuda import pinned_cuda_write_read

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='SDM CUDA requires GPU')


def _inputs(dtype, tokens=13):
    torch.manual_seed(137)
    p, s, d, r, w = 2, 64, 64, 4, 4
    wi = torch.rand(p, tokens, s, device='cuda').argsort(-1)[..., :w].sort(-1).values
    ri = torch.rand(p, tokens, s, device='cuda').argsort(-1)[..., :r].sort(-1).values
    tensors = [torch.randn(p, s, d, device='cuda', dtype=dtype) * .1,
               torch.randn(p, tokens, r, device='cuda', dtype=dtype).softmax(-1),
               torch.randn(p, tokens, w, device='cuda', dtype=dtype).softmax(-1),
               torch.randn(p, tokens, d, device='cuda', dtype=dtype) * .1,
               torch.rand(p, tokens, 1, device='cuda') * .5,
               -torch.rand(p, tokens, 1, device='cuda') * .15]
    return wi, ri, tensors


def _serial(wi, ri, inputs):
    memory, rw, ww, values, beta, g = inputs
    state = memory.float()
    p, t, d = values.shape
    outputs = []
    for token in range(t):
        index = wi[:, token, :, None].expand(p, -1, d)
        rows = state.gather(1, index) * g[:, token].float().exp().unsqueeze(1)
        weights = ww[:, token, :, None].float()
        error = beta[:, token].float() * (values[:, token].float() - (rows * weights).sum(1))
        state = state.scatter(1, index, rows + weights * error[:, None])
        reads = state.gather(1, ri[:, token, :, None].expand(p, -1, d))
        outputs.append((reads * rw[:, token, :, None].float()).sum(1))
    return torch.stack(outputs, 1).to(values.dtype), state.to(memory.dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cuda_matches_recurrence_with_autocast_padding_and_terminal_gradients(dtype):
    wi, ri, base = _inputs(dtype)
    actual_inputs = [x.detach().clone().requires_grad_() for x in base]
    ref_inputs = [x.detach().clone().requires_grad_() for x in base]
    m, rw, ww, v, beta, g = actual_inputs
    cy, cm = torch.randn_like(v), torch.randn_like(m)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        actual = pinned_cuda_write_read(m, ri, rw, write_indices=wi, write_weights=ww,
            values=v, beta=beta, log_decay=g, chunk_size=8, grad_final_memory=cm)
    state_before_backward = actual[1].clone()
    expected = _serial(wi, ri, ref_inputs)
    # The pin's fused FP32 Triton matmuls use their default TF32 input mode.
    # The independent FP64 tests above certify our algebra more strictly.
    tolerance = 4e-3 if dtype == torch.bfloat16 else 1e-4
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(m.detach(), base[0], atol=0, rtol=0)
    actual_grad = torch.autograd.grad((actual[0].float() * cy.float()).sum(), actual_inputs)
    expected_grad = torch.autograd.grad((expected[0].float() * cy.float()).sum()
                                       + (expected[1].float() * cm.float()).sum(), ref_inputs)
    torch.testing.assert_close(actual[1], state_before_backward, atol=0, rtol=0)
    for name, a, b in zip(('memory', 'read_weights', 'write_weights', 'values', 'beta', 'decay'),
                          actual_grad, expected_grad):
        error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert error < (.025 if dtype == torch.bfloat16 else .001), (name, float(error))


@pytest.mark.parametrize('compiled', [False, True])
def test_chunked_bf16_all_gradients_agree_with_actual_cuda(compiled):
    wi, ri, base = _inputs(torch.bfloat16)
    ours = [x.detach().clone().requires_grad_() for x in base]
    upstream = [x.detach().clone().requires_grad_() for x in base]
    cy, cm = torch.randn_like(base[3]), torch.randn_like(base[0])
    def kwargs(leaves):
        return dict(write_indices=wi, write_weights=leaves[2], values=leaves[3],
                    beta=leaves[4], log_decay=leaves[5], chunk_size=8)
    output = chunked_sparse_delta_memory(ours[0], ri, ours[1], **kwargs(ours), compiled=compiled)
    expected = pinned_cuda_write_read(upstream[0], ri, upstream[1], **kwargs(upstream),
                                     grad_final_memory=cm)
    for a, b in zip(output, expected):
        torch.testing.assert_close(a, b, atol=2e-3, rtol=.02)
    agrad = torch.autograd.grad((output[0].float() * cy.float()).sum()
                               + (output[1].float() * cm.float()).sum(), ours)
    bgrad = torch.autograd.grad((expected[0].float() * cy.float()).sum(), upstream)
    for name, a, b in zip(('memory', 'read_weights', 'write_weights', 'values', 'beta', 'decay'),
                          agrad, bgrad):
        error = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8)
        assert error < .025, (name, float(error))


@pytest.mark.parametrize('execution', ['torch-chunked', 'upstream-cuda'])
def test_external_layer_commits_terminal_state_after_backward(execution):
    from architectures.sdm_memory import SparseDeltaMemoryLayer
    torch.manual_seed(51)
    layer = SparseDeltaMemoryLayer(64, 1, 64, 64, 4, 4, 2,
        execution=execution, chunk_size=8, compile_state=False).cuda()
    x = torch.randn(2, 13, 64, device='cuda', requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = layer(x)
    terminal = layer._pending_state.detach().clone()
    assert terminal.abs().sum() > 0
    assert torch.count_nonzero(layer.persistent_memory) == 0
    output.float().square().sum().backward()
    torch.testing.assert_close(layer._pending_state.detach(), terminal, atol=0, rtol=0)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    assert torch.isfinite(x.grad).all()
    layer.detach_state()
    torch.testing.assert_close(layer.persistent_memory, terminal.float(), atol=0, rtol=0)
    assert layer._pending_state is None
    assert layer._state_is_zero is False
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        assert torch.isfinite(layer(x.detach())).all()
    layer.reset_state()
    assert torch.count_nonzero(layer.persistent_memory) == 0
    assert layer._pending_state is None
    assert layer._state_is_zero is True


def test_registry_pair_executes_cuda_with_matching_frontend_parameter_gradients():
    from train.registry import get_mixer
    from train.upstream import UPSTREAM_STATEFUL_BUILDERS, UPSTREAM_TIER
    torch.manual_seed(73)
    ours = get_mixer('sdm').builder(128, 2, 64, 'training', 'native', batch_size=2).cuda()
    baseline = UPSTREAM_STATEFUL_BUILDERS['sdm']()(128, 2, 64, 'training', batch_size=2).cuda()
    baseline.load_state_dict(ours.state_dict())
    assert UPSTREAM_TIER['sdm'] == 'production-kernel'
    assert ours.chunk_size == 128 and baseline.chunk_size == 64
    assert ours.execution == 'torch-chunked' and baseline.execution == 'upstream-cuda'
    assert get_mixer('sdm').public_path is False
    # Verify the production call executes rather than merely constructing an adapter.
    calls = []
    original = baseline._upstream_kernel
    class TracedKernel:
        @staticmethod
        def apply(*args):
            calls.append(True)
            return original.apply(*args)
    baseline._upstream_kernel = TracedKernel
    a = torch.randn(2, 65, 128, device='cuda', requires_grad=True)
    b = a.detach().clone().requires_grad_()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out, ref = ours(a), baseline(b)
    assert calls
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=.03)
    cotangent = torch.randn_like(out)
    out.backward(cotangent)
    ref.backward(cotangent)
    for name, x, y in [('input', a.grad, b.grad),
                       *[(name, x.grad, y.grad) for (name, x), (_, y) in
                         zip(ours.named_parameters(), baseline.named_parameters())]]:
        assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
        error = (x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-8)
        assert error < .04, (name, float(error))
