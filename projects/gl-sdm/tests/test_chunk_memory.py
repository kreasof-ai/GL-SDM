from unittest.mock import patch
import pytest
import torch
from gl_sdm.model import create_model
from gl_sdm.experiments.verify import check
from gl_sdm.memory import MemoryView, WriteProposal, merge, commit
from test_global_memory import config


@pytest.mark.parametrize("reasoning", ["fixed", "adaptive"])
@pytest.mark.parametrize("policy", ["merged", "final"])
@pytest.mark.parametrize("backend,device", [("torch", "cpu"),
    pytest.param("urm", "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_chunk_full_reference(reasoning, policy, backend, device):
    check(config(gl_chunk_size=16, gl_reasoning=reasoning,
                 gl_write_policy=policy, gl_memory_backend=backend), device)


def test_absolute_commit_clock_and_frozen_snapshot():
    model = create_model(config(gl_chunk_size=4)).eval()
    import gl_sdm.memory.routing as module
    versions = []
    original = module.routed_read
    def observe(block, memory, *args):
        versions.append(memory.data_ptr())
        return original(block, memory, *args)
    with torch.inference_mode(), patch.object(module, "routed_read", observe):
        x = torch.randint(64, (2, 9))
        full, fc = model.prefill(x)
        assert fc[0].view.version == 2 and fc[0].tokens == 9
        assert len(set(versions[:4])) == 1
        assert versions[0] != versions[4]
        pieces, cache = [], model.new_cache(2)
        for t in range(9):
            out, cache = model.decode(x[:, t:t+1], cache)
            assert cache[0].view.version == (t + 1) // 4
            pieces.append(out)
        torch.testing.assert_close(torch.cat(pieces, 1), full, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(cache[0].view.values, fc[0].view.values, atol=1e-6, rtol=1e-5)
        assert cache[0].local["k"].shape[2] == 1  # bounded within-chunk KV


def test_keyed_commit_is_independent_of_proposal_arrival_order():
    view = MemoryView(torch.zeros(1, 1, 4, 1))
    deltas = [1e8, -1e8, .125]
    proposals = [WriteProposal(0, torch.tensor([3]), torch.tensor([[d]]),
        view.lineage, torch.tensor([30 + t])) for t, d in enumerate(deltas)]
    a = commit(view, merge(view, proposals))
    b = commit(view, merge(view, [proposals[2], proposals[0], proposals[1]]))
    torch.testing.assert_close(a.values, b.values, atol=0, rtol=0)
    assert a.values.flatten()[3] == .125


def test_chunk_highest_ties_match_urm_contract():
    from gl_sdm.memory.routing import torch_routed_read
    _, index, weights = torch_routed_read(torch.zeros(1, 2, 16, 3), torch.zeros(1, 5, 2, 8), 4)
    torch.testing.assert_close(index, torch.arange(12, 16).expand(1, 5, 2, 4))
    torch.testing.assert_close(weights, torch.full_like(weights, .25))


def test_chunk_rejects_noncausal_every_step():
    with pytest.raises(ValueError, match="causality"):
        create_model(config(gl_chunk_size=4, gl_write_policy="every_step"))


def test_chunk_act_compacts_finished_tokens():
    import gl_sdm.runtime.dense as module
    model = create_model(config(gl_chunk_size=4, gl_reasoning="adaptive", gl_max_steps=4)).eval()
    block = model.blocks[0]
    with torch.no_grad():
        for layer in (block.input_proj, block.attn.proj, block.mlp.proj, block.local_context.proj):
            layer.weight.zero_(); layer.bias.zero_()
        block.halt.weight.zero_(); block.halt.weight[0, 0] = 8
        block.halt.bias.zero_()
    calls = []
    original = module.dense_step
    def observe(current, *args):
        calls.append(current.shape[0])
        return original(current, *args)
    with patch.object(module, "dense_step", observe), torch.inference_mode():
        block(torch.stack([torch.ones(4, 32), -torch.ones(4, 32)]))
    assert calls == [8, 4, 4, 4]
    assert block.last_depth.tolist() == [[1] * 4, [4] * 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_urm_colliding_read_gradients():
    from gl_sdm.memory.backends.urm import routed_snapshot_read
    from gl_sdm.memory.routing import torch_routed_read
    torch.manual_seed(42)
    memory = torch.randn(2, 2, 16, 8, device="cuda", requires_grad=True)
    scores = torch.randn(2, 17, 2, 8, device="cuda", requires_grad=True)
    # Many queries reuse the same small bank; catches unique-query backwards.
    out, idx, weight = routed_snapshot_read(memory, scores, 2)
    ref, ridx, rw = torch_routed_read(memory, scores, 2, True)
    torch.testing.assert_close(idx.long(), ridx)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    grad = torch.autograd.grad(out.square().sum(), (memory, scores), retain_graph=True)
    expected = torch.autograd.grad(ref.square().sum(), (memory, scores))
    for a, b in zip(grad, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_production_bank_padded_read_and_operand_gradients():
    from gl_sdm.memory.backends.urm import routed_snapshot_read
    from gl_sdm.memory.routing import torch_routed_read
    torch.manual_seed(42)
    memory = torch.randn(2, 8, 4096, 64, device="cuda", requires_grad=True)
    scores = torch.randn(2, 65, 8, 128, device="cuda", dtype=torch.bfloat16).float().requires_grad_()
    out, index, weights = routed_snapshot_read(torch.nn.functional.pad(memory, (0, 64)), scores, 8)
    out = out[..., :64]
    ref, ri, rw = torch_routed_read(memory, scores, 8)
    torch.testing.assert_close(index.long(), ri)
    torch.testing.assert_close(weights, rw, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    cotangent = torch.randn_like(out)
    actual = torch.autograd.grad(out, (memory, scores), cotangent, retain_graph=True)
    expected = torch.autograd.grad(ref, (memory, scores), cotangent)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_compiled_bf16_training_loss_and_gradients():
    torch.manual_seed(1234)
    c = config(gl_chunk_size=8, dtype="bfloat16", gl_memory_backend="urm")
    model = create_model(c).cuda()
    compiled = create_model({**c, "gl_compile": True}).cuda()
    compiled.load_state_dict(model.state_dict(), strict=True)
    x = torch.randint(64, (2, 17), device="cuda")
    y = torch.randint_like(x, 64)
    a, _, _ = model(x, y)
    b, _, _ = compiled(x, y)
    torch.testing.assert_close(a, b, atol=.02, rtol=1e-4)
    a.backward(); b.backward()
    for (name, p), (_, q) in zip(model.named_parameters(), compiled.named_parameters(), strict=True):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad.float(), q.grad.float(), atol=.03, rtol=.12, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_padded_width64_full_reference():
    check(config(hidden_size=128, head_dim=64, gl_slots=64, gl_reads=4, gl_writes=4,
        gl_chunk_size=8, gl_memory_backend="urm", gl_compile=True), "cuda", length=23)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_training_graph_updates_gradients_inputs_and_learning_rate():
    from gl_sdm.experiments.train import update, optimizers
    torch.manual_seed(17)
    c = config(gl_chunk_size=8, gl_memory_backend="urm", gl_compile=True,
        seq_len=17, batch_size=34, mbs=1)
    model = create_model(c).cuda()
    captured = create_model({**c, "gl_cuda_graph": True}).cuda()
    captured.load_state_dict(model.state_dict(), strict=True)
    a, b = optimizers(model, c, "cuda"), optimizers(captured, c, "cuda")
    for step in range(3):
        x = torch.randint(64, (2, 17), device="cuda")
        y = torch.randint_like(x, 64)
        a[0].param_groups[0]["lr"] = b[0].param_groups[0]["lr"] = .001 / (step + 1)
        captured.zero_grad(set_to_none=True)  # callbacks must not detach captured gradients
        loss = update(model, a, x, y, c)
        graph_loss = update(captured, b, x, y, {**c, "gl_cuda_graph": True})
        assert abs(loss - graph_loss) < 1e-5
        for (name, p), (_, q) in zip(model.named_parameters(), captured.named_parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5, msg=name)
    torch.testing.assert_close(captured.last_training_depth, model.last_training_depth)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_chunk_executes_urm_backward_without_token_kernels():
    from contextlib import ExitStack
    import gl_sdm.memory.backends.token as local
    import urm.backends.triton.k3.sparse_state as upstream
    model = create_model(config(gl_chunk_size=8, gl_memory_backend="urm")).cuda()
    x = torch.randint(64, (2, 17), device="cuda")
    y = torch.randint_like(x, 64)
    with ExitStack() as stack:
        for name in ("route", "read", "propose"):
            stack.enter_context(patch.object(local, name, side_effect=AssertionError("token kernel reached")))
        backward = stack.enter_context(patch.object(upstream, "_sparse_state_read_backward",
            wraps=upstream._sparse_state_read_backward))
        model(x, y)[0].backward()
        assert backward.call_count > 0
