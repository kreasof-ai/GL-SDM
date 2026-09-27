"""Native transactions against independent equations, including all gradients."""
import copy
import json
from pathlib import Path
import pytest
import torch
from gl_sdm.memory import MemoryView, WriteProposal, read, propose_write, merge, commit
from gl_sdm.model import create_model
from gl_sdm.experiments.verify import check
from test_global_memory import config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def test_frozen_dependency_and_native_read_plan():
    from gl_sdm.memory.backends.urm import URM_REVISION, verify_dependency, read_plan
    root = Path(__file__).resolve().parents[3]
    assert URM_REVISION in (root / "shared/requirements-urm.txt").read_text()
    assert URM_REVISION in (root / "projects/gl-sdm/pyproject.toml").read_text()
    assert verify_dependency()["revision"] == URM_REVISION
    plan = read_plan(67, 4, 37, 3).serialized_plan()
    assert plan["escape_hatch_count"] == 0
    assert plan["steps"][0]["anchor"] == "urm_native_sparse_state_mixer_v0"
    json.dumps(plan)


@pytest.mark.parametrize("half,count", [(4, 2), (3, 5), (8, 64), (64, 8)])
def test_routing_preserves_selection_ties_and_all_score_gradients(half, count):
    from gl_sdm.memory.backends.token import route
    torch.manual_seed(13)
    router = create_model(config(hidden_size=32, head_dim=16, gl_slots=half ** 2)).cuda().blocks[0].attn
    scores = torch.randn(3, 2, 2 * half, device="cuda", requires_grad=True)
    # Adversarial ties, including signed zero, must prefer smaller addresses.
    with torch.no_grad():
        scores[0].zero_()
        scores[0, :, 0] = -0.0
    class Projection:
        def __call__(self, _):
            return scores
    idx, w = route(scores, count)
    ridx, rw = router.route(Projection(), scores, count)
    torch.testing.assert_close(idx, ridx, atol=0, rtol=0)
    torch.testing.assert_close(w, rw, atol=2e-7, rtol=2e-6)
    assert (idx[..., 1:] > idx[..., :-1]).all()
    assert idx[0, 0].tolist() == list(range(count))
    incoming = torch.randn_like(w)
    g = torch.autograd.grad((w * incoming).sum(), scores, retain_graph=True)[0]
    rg = torch.autograd.grad((rw * incoming).sum(), scores)[0]
    torch.testing.assert_close(g, rg, atol=3e-7, rtol=3e-6)


def test_read_proposal_commit_all_gradients_and_compact_saved_rows():
    torch.manual_seed(91)
    memory = torch.randn(3, 2, 67, 37, device="cuda", requires_grad=True)
    requests = torch.tensor([2, 0], device="cuda")  # ACT subset; noncanonical request order
    idx = torch.tensor([[[0, 4, 66], [1, 5, 8]], [[2, 3, 9], [0, 11, 66]]], device="cuda")
    weights = torch.randn(2, 2, 3, device="cuda").softmax(-1).detach().requires_grad_()
    targets = torch.randn(2, 2, 37, device="cuda", requires_grad=True)
    beta = torch.full((2, 2, 1), 0.3, device="cuda", requires_grad=True)
    decay = torch.full((2, 2, 1), -0.12, device="cuda", requires_grad=True)
    mass = torch.tensor([0.0, 0.7], device="cuda", requires_grad=True)
    watched = (memory, weights, targets, beta, decay, mass)
    view = MemoryView(memory)
    before = memory.detach().clone()
    saved = []
    def pack(t):
        saved.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        retrieved = read(view, requests, idx, weights, backend="urm")
        p = propose_write(view, requests, idx, weights, targets, beta, decay, mass, backend="urm")
        # Different proposals collide at every address; include both state and
        # reading cotangents to test gradient ownership across transactions.
        p2 = propose_write(view, requests, idx, weights, targets * 2, beta, decay, 1 - mass, backend="urm")
        updated = commit(view, merge(view, [p, p2]), backend="urm")
    assert tuple(memory.shape) not in saved
    assert (2, 2, 3, 37) in saved
    rread = read(view, requests, idx, weights, reference=True)
    rp = propose_write(view, requests, idx, weights, targets, beta, decay, mass, reference=True)
    rp2 = propose_write(view, requests, idx, weights, targets * 2, beta, decay, 1 - mass, reference=True)
    rupdated = commit(view, merge(view, [rp, rp2]), reference=True)
    torch.testing.assert_close(retrieved, rread, atol=5e-7, rtol=2e-6)
    torch.testing.assert_close(p.deltas, rp.deltas, atol=5e-7, rtol=2e-6)
    torch.testing.assert_close(updated.values, rupdated.values, atol=1e-6, rtol=2e-6)
    torch.testing.assert_close(memory, before, atol=0, rtol=0)
    rc, mc = torch.randn_like(retrieved), torch.randn_like(memory)
    g = torch.autograd.grad((retrieved * rc).sum() + (updated.values * mc).sum(), watched, retain_graph=True)
    rg = torch.autograd.grad((rread * rc).sum() + (rupdated.values * mc).sum(), watched)
    for actual, expected in zip(g, rg, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for _ in range(4):
        torch.testing.assert_close(commit(view, merge(view, [p, p2]), backend="urm").values, updated.values, atol=0, rtol=0)


def test_commit_stable_collisions_preserves_small_unrelated_updates():
    view = MemoryView(torch.zeros(1, 1, 4, 1, device="cuda", requires_grad=True))
    idx = torch.tensor([0, 3, 0, 0, 3], device="cuda")
    delta = torch.tensor([[1e8], [0.125], [-1e8], [1.0], [0.25]], device="cuda", requires_grad=True)
    p = WriteProposal(0, idx, delta, view.lineage)
    result = commit(view, merge(view, [p]), backend="urm")
    assert result.values.flatten().tolist() == [1.0, 0.0, 0.0, 0.375]
    result.values.sum().backward()
    torch.testing.assert_close(delta.grad, torch.ones_like(delta))
    torch.testing.assert_close(view.values.grad, torch.ones_like(view.values))
    with pytest.raises(ValueError, match="stale"):
        commit(result, merge(view, [p]), backend="urm")


@pytest.mark.parametrize("policy", ["merged", "final", "every_step"])
@pytest.mark.parametrize("reasoning", ["fixed", "adaptive"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_native_full_model_reference_and_continuation(policy, reasoning, dtype):
    check(config(gl_memory_backend="urm", gl_write_policy=policy, gl_reasoning=reasoning, dtype=dtype), "cuda", length=23)


def test_native_adaptive_subset_really_stops_and_matches_torch():
    torch.manual_seed(31)
    model = create_model(config(gl_memory_backend="urm", gl_reasoning="adaptive", gl_max_steps=4)).cuda().eval()
    block = model.blocks[0]
    with torch.no_grad():
        for layer in (block.input_proj, block.attn.proj, block.mlp.proj):
            layer.weight.zero_()
            layer.bias.zero_()
        block.halt.weight.zero_()
        block.halt.weight[0, 0] = 8
        block.halt.bias.zero_()
    reference = copy.deepcopy(block)
    reference.attn.backend = "torch"
    calls = []
    hook = block.mlp.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[0]))
    x = torch.stack([torch.ones(2, 32, device="cuda"), -torch.ones(2, 32, device="cuda")])
    with torch.inference_mode():
        actual = block(x)[0]
        expected = reference(x)[0]
    hook.remove()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert block.last_depth.tolist() == [[1, 1], [4, 4]]
    assert calls == [2, 1, 1, 1, 2, 1, 1, 1]


def test_native_backend_never_falls_back_on_cpu():
    model = create_model(config(gl_memory_backend="urm"))
    with pytest.raises(ValueError, match="CUDA"):
        model.hidden(torch.ones(2, 3, dtype=torch.long))
