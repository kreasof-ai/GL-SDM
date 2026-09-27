import copy
import pytest
import torch
from gl_sdm.memory import MemoryView, WriteProposal, read, propose_write, merge, commit
from gl_sdm.model import create_model
from gl_sdm.experiments.metrics import parameter_counts
from gl_sdm.experiments.verify import check


def config(**changes):
    return {"arch_type": "gl_sdm", "hidden_size": 32, "head_dim": 16,
            "num_hidden_layers": 1, "vocab_size": 64, "dtype": "float32",
            "gl_slots": 16, "gl_reads": 2, "gl_writes": 2,
            "gl_max_steps": 3, "gl_reasoning": "fixed", **changes}


def proposal(view, targets, mass=0.5, indices=None):
    requests = torch.arange(2, device=view.values.device)
    idx = torch.tensor([[[1, 2]], [[1, 2]]], device=view.values.device) if indices is None else indices
    weights = torch.tensor([[[0.25, 0.75]], [[0.75, 0.25]]], device=view.values.device, requires_grad=True)
    beta = torch.full((2, 1, 1), 0.3, device=view.values.device, requires_grad=True)
    decay = torch.full_like(beta, -0.1, requires_grad=True)
    p = propose_write(view, requests, idx, weights, targets, beta, decay, torch.full((2,), mass, device=view.values.device))
    return p, (weights, beta, decay)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_snapshot_duplicate_commit_and_gradients(device):
    torch.manual_seed(42)
    values = torch.randn(2, 1, 4, 3, device=device, requires_grad=True)
    target = torch.randn(2, 1, 3, device=device, requires_grad=True)
    view = MemoryView(values)
    before = values.detach().clone()
    p1, inputs = proposal(view, target)
    p2, inputs2 = proposal(view, target * 2)
    buffer = merge(view, [p1, p2])
    actual = commit(view, buffer)
    expected = commit(view, buffer, reference=True)
    torch.testing.assert_close(actual.values, expected.values, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(values, before, atol=0, rtol=0)
    assert actual.version == 1
    for _ in range(3):
        torch.testing.assert_close(commit(view, buffer).values, actual.values, atol=0, rtol=0)
    watched = (values, target, *inputs, *inputs2)
    grad = torch.autograd.grad(actual.values.square().sum(), watched, retain_graph=True)
    ref_grad = torch.autograd.grad(expected.values.square().sum(), watched)
    for a, b in zip(grad, ref_grad):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="stale"):
        commit(actual, buffer)
    with pytest.raises(ValueError, match="stale"):
        commit(MemoryView(values), buffer)
    with pytest.raises(ValueError, match="snapshot"):
        merge(actual, [p1])


def test_snapshot_reads_and_write_timing_differ():
    values = torch.ones(2, 1, 4, 3)
    view = MemoryView(values)
    p, _ = proposal(view, torch.zeros(2, 1, 3), mass=1)
    idx = torch.tensor([[[1, 2]], [[1, 2]]])
    weight = torch.full((2, 1, 2), 0.5)
    initial = read(view, torch.arange(2), idx, weight)
    after = commit(view, merge(view, [p]))
    torch.testing.assert_close(read(view, torch.arange(2), idx, weight), initial, atol=0, rtol=0)
    assert not torch.equal(read(after, torch.arange(2), idx, weight), initial)


def test_commit_does_not_cancel_updates_at_unrelated_addresses():
    view = MemoryView(torch.zeros(1, 1, 4, 1))
    p = WriteProposal(0, torch.tensor([0, 3]), torch.tensor([[1e8], [0.125]]), view.lineage)
    result = commit(view, merge(view, [p]))
    assert result.values[0, 0, 3, 0].item() == 0.125


@pytest.mark.parametrize("policy", ["merged", "final", "every_step"])
@pytest.mark.parametrize("reasoning", ["fixed", "adaptive"])
def test_full_model_reference_gradients_and_continuation(policy, reasoning):
    check(config(gl_write_policy=policy, gl_reasoning=reasoning), "cpu")


def test_one_bank_and_one_reasoner_across_depth():
    torch.manual_seed(1)
    a = create_model(config(gl_max_steps=2))
    torch.manual_seed(1)
    b = create_model(config(gl_max_steps=8))
    assert len(a.blocks) == len(b.blocks) == 1
    assert parameter_counts(a) == parameter_counts(b)
    assert parameter_counts(a)["memory_params"] == 2 * 16 * 16
    assert len([n for n, p in a.named_parameters() if getattr(p, "_sdm_memory_bank", False)]) == 1
    bf16 = a.bfloat16()
    assert bf16.blocks[0].bank.memory.dtype == torch.float32


def test_adaptive_halting_changes_actual_execution_and_has_gradient():
    model = create_model(config(gl_reasoning="adaptive", gl_max_steps=4))
    block = model.blocks[0]
    x = torch.randint(64, (2, 5))
    y = torch.randint(64, x.shape)
    with torch.no_grad():
        block.halt.weight.zero_()
        block.halt.bias.fill_(8)
    model(x, y)
    assert block.last_depth.eq(1).all()
    with torch.no_grad():
        block.halt.bias.fill_(-8)
    _, _, ponder = model(x, y)
    assert block.last_depth.eq(4).all()
    ponder.backward()
    assert block.halt.bias.grad.abs().sum() > 0
    assert torch.isfinite(block.halt.bias.grad).all()


def test_shared_bank_policies_produce_different_outputs_and_request_isolation():
    torch.manual_seed(23)
    model = create_model(config()).eval()
    other = copy.deepcopy(model)
    other.blocks[0].write_policy = "every_step"
    x = torch.randint(64, (2, 7))
    with torch.inference_mode():
        full, cache = model.prefill(x)
        early, _ = other.prefill(x)
        assert not torch.allclose(full, early)
        assert cache[0].view.version == cache[0].tokens == 7
        a, ac = model.prefill(x[:1])
        b, bc = model.prefill(x[1:])
        torch.testing.assert_close(torch.cat([a, b]), full, atol=1e-6, rtol=1e-5)
        with pytest.raises(ValueError, match="cache belongs"):
            other.decode(x[:, :1], cache)
        with pytest.raises(ValueError, match="cache belongs"):
            model.decode(x[:1, :1], cache)


def test_product_key_ties_are_stable_and_unique():
    model = create_model(config())
    router = model.blocks[0].attn
    with torch.no_grad():
        router.q.weight.zero_()
        router.q.bias.zero_()
    indices, weights = router.route(router.q, torch.ones(2, 32), 4)
    torch.testing.assert_close(indices, torch.arange(4).expand(2, 2, 4))
    torch.testing.assert_close(weights, torch.full((2, 2, 4), 0.25))


def test_adaptive_execution_removes_halted_requests():
    model = create_model(config(gl_reasoning="adaptive", gl_max_steps=4)).eval()
    block = model.blocks[0]
    with torch.no_grad():
        for layer in (block.input_proj, block.attn.proj, block.mlp.proj):
            layer.weight.zero_()
            layer.bias.zero_()
        block.halt.weight.zero_()
        block.halt.weight[0, 0] = 8
        block.halt.bias.zero_()
    calls = []
    hook = block.mlp.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    with torch.inference_mode():
        block(torch.stack([torch.ones(2, 32), -torch.ones(2, 32)]))
    hook.remove()
    assert block.last_depth.tolist() == [[1, 1], [4, 4]]
    assert calls == [2, 1, 1, 1, 2, 1, 1, 1]


def test_reads_share_one_version_until_commit():
    from unittest.mock import patch
    from gl_sdm.layers import router, token
    model = create_model(config()).eval()
    reads, commits = [], []
    original_read, original_commit = router.read, token.commit
    def observe_read(view, *args, **kwargs):
        reads.append(view.version)
        return original_read(view, *args, **kwargs)
    def observe_commit(view, *args, **kwargs):
        commits.append(view.version)
        return original_commit(view, *args, **kwargs)
    with patch.object(router, "read", observe_read), patch.object(token, "commit", observe_commit), torch.inference_mode():
        _, cache = model.prefill(torch.randint(64, (2, 3)))
    assert reads == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert commits == [0, 1, 2] and cache[0].view.version == 3
