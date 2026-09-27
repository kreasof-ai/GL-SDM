"""Independent FP64 algebra/VJP checks for the native K3 chunk schedule."""
import pytest
import torch

pytest.importorskip('triton')
from urm.backends.triton.k3.sparse_state import chunked_sparse_state_update


def _case(tokens=19, decay=-.4, dtype=torch.float64, device='cpu'):
    torch.manual_seed(37)
    p, s, d, w, r = 2, 9, 4, 4, 3
    wi = torch.rand(p, tokens, s, device=device).argsort(-1)[..., :w].sort(-1).values
    ri = torch.rand(p, tokens, s, device=device).argsort(-1)[..., :r].sort(-1).values
    leaves = [torch.randn(p, s, d, device=device, dtype=dtype) * .1,
              torch.randn(p, tokens, r, device=device, dtype=dtype).softmax(-1),
              torch.randn(p, tokens, w, device=device, dtype=dtype).softmax(-1),
              torch.randn(p, tokens, d, device=device, dtype=dtype) * .1,
              torch.rand(p, tokens, 1, device=device, dtype=dtype) * .5,
              torch.full((p, tokens, 1), decay, device=device, dtype=dtype)]
    return wi, ri, [x.requires_grad_() for x in leaves]


def _serial(wi, ri, memory, rw, ww, values, beta, g, read_before_update=False):
    """Independent recurrence, retaining FP64 including state boundaries."""
    state = memory
    p, t, d = values.shape
    outputs = []
    for token in range(t):
        reads = state.gather(1, ri[:, token, :, None].expand(p, -1, d))
        if read_before_update:
            outputs.append((reads * rw[:, token, :, None]).sum(1))
        index = wi[:, token, :, None].expand(p, -1, d)
        rows = state.gather(1, index) * g[:, token].exp().unsqueeze(1)
        weights = ww[:, token, :, None]
        retrieved = (rows * weights).sum(1)
        error = beta[:, token] * (values[:, token] - retrieved)
        state = state.scatter(1, index, rows + weights * error[:, None])
        reads = state.gather(1, ri[:, token, :, None].expand(p, -1, d))
        if not read_before_update:
            outputs.append((reads * rw[:, token, :, None]).sum(1))
    return torch.stack(outputs, 1), state


@pytest.mark.parametrize('chunk_size', [1, 4, 8, 32])
@pytest.mark.parametrize('decay', [0., -.4, -20., -1000.])
@pytest.mark.parametrize('read_before_update', [False, True])
def test_forward_final_state_and_every_gradient_match_recurrence(chunk_size, decay, read_before_update):
    wi, ri, leaves = _case(decay=decay)
    memory, rw, ww, values, beta, g = leaves
    original = memory.detach().clone()
    actual = chunked_sparse_state_update(memory, ri, rw, write_indices=wi,
        write_weights=ww, values=values, beta=beta, log_decay=g, chunk_size=chunk_size,
        read_before_update=read_before_update)
    expected = _serial(wi, ri, *leaves, read_before_update=read_before_update)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(memory.detach(), original, atol=0, rtol=0)
    cotangents = [torch.randn_like(x) for x in actual]
    actual_grad = torch.autograd.grad(sum((x * v).sum() for x, v in zip(actual, cotangents)), leaves)
    expected_grad = torch.autograd.grad(sum((x * v).sum() for x, v in zip(expected, cotangents)), leaves)
    for name, a, b in zip(('memory', 'read_weights', 'write_weights', 'values', 'beta', 'decay'),
                          actual_grad, expected_grad):
        assert torch.isfinite(a).all(), name
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10, msg=name)


def test_zero_snapshot_specialization_preserves_all_learned_gradients():
    wi, ri, leaves = _case()
    leaves[0] = torch.zeros_like(leaves[0], requires_grad=False)
    memory, rw, ww, v, beta, g = leaves
    kwargs = dict(write_indices=wi, write_weights=ww, values=v, beta=beta,
                  log_decay=g, chunk_size=8)
    normal = chunked_sparse_state_update(memory, ri, rw, **kwargs)
    specialized = chunked_sparse_state_update(memory, ri, rw, **kwargs, zero_initial_state=True)
    for a, b in zip(normal, specialized):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    ngrad = torch.autograd.grad(sum(x.square().sum() for x in normal), leaves[1:])
    sgrad = torch.autograd.grad(sum(x.square().sum() for x in specialized), leaves[1:])
    for a, b in zip(ngrad, sgrad):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_zero_snapshot_specialization_rejects_trainable_initial_memory():
    wi, ri, leaves = _case()
    memory, rw, ww, v, beta, g = leaves
    with pytest.raises(ValueError, match='initial-memory gradients'):
        chunked_sparse_state_update(memory, ri, rw, write_indices=wi, write_weights=ww,
            values=v, beta=beta, log_decay=g, zero_initial_state=True)


def test_two_transactions_preserve_snapshot_and_cross_transaction_gradients():
    wi, ri, leaves = _case(tokens=17)
    memory, rw, ww, v, beta, g = leaves
    def call(state, start, stop, serial=False):
        parts = [x[:, start:stop] for x in (rw, ww, v, beta, g)]
        if serial:
            return _serial(wi[:, start:stop], ri[:, start:stop], state, *parts)
        return chunked_sparse_state_update(state, ri[:, start:stop], parts[0],
            write_indices=wi[:, start:stop], write_weights=parts[1], values=parts[2],
            beta=parts[3], log_decay=parts[4], chunk_size=4)
    first_y, snapshot = call(memory, 0, 7)
    saved = snapshot.detach().clone()
    second_y, final = call(snapshot, 7, 17)
    ref_first_y, ref_snapshot = call(memory, 0, 7, serial=True)
    ref_second_y, ref_final = call(ref_snapshot, 7, 17, serial=True)
    for a, b in zip((first_y, second_y, final), (ref_first_y, ref_second_y, ref_final)):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-11)
    agrad = torch.autograd.grad(second_y.square().sum() + final.square().sum(), leaves)
    bgrad = torch.autograd.grad(ref_second_y.square().sum() + ref_final.square().sum(), leaves)
    for a, b in zip(agrad, bgrad):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(snapshot.detach(), saved, atol=0, rtol=0)
