"""Untied depth, shared snapshots, sliding continuity and sparse-state gradients."""
import copy
from unittest.mock import patch
import pytest
import torch
from gl_sdm.model import create_model
from gl_sdm.layers.global_layer import GlobalLayer
from gl_sdm.memory import MemoryBank
from gl_sdm.experiments.verify import check
from gl_sdm.experiments.evaluate import _blocks_forward
from gl_sdm.experiments.metrics import parameter_counts, utilization


def config(**changes):
    return dict(arch_type="gl_sdm", vocab_size=64, hidden_size=32, head_dim=16,
                num_hidden_layers=16, dtype="float32", gl_layer_pattern=["local", "local", "global", "local"],
                gl_slots=16, gl_reads=2, gl_writes=2, gl_chunk_size=4,
                gl_local_window=6, gl_memory_backend="torch", seq_len=9, **changes)


def test_distinct_layers_share_exactly_one_bank_and_count_executed_weights():
    model = create_model(config())
    assert len(model.blocks) == 16
    assert [i for i, b in enumerate(model.blocks) if isinstance(b, GlobalLayer)] == [2, 6, 10, 14]
    assert len([b for b in model.modules() if isinstance(b, MemoryBank)]) == 1
    assert len({id(b.mlp.fc.weight) for b in model.blocks}) == 16
    counts = parameter_counts(copy.deepcopy(model))
    assert counts["memory_params"] == 2 * 16 * 16
    result = utilization(model, 18, 1, None)
    assert result["write_tokens"] == 16
    assert result["physical_layers"] == 16 and result["global_layers"] == 4
    assert result["weight_loops"] == 0
    assert result["estimated_training_flops_6nd"] == 6 * (result["once_params"] * 18 + result["write_params"] * 16)


def test_chunk_snapshot_is_shared_across_depth_and_local_history_survives_commit():
    model = create_model(config()).eval()
    records = []
    original = GlobalLayer.forward
    def observe(layer, x, snapshot, physical=None):
        records.append((snapshot.version, id(snapshot.values)))
        return original(layer, x, snapshot, physical)
    x = torch.randint(64, (1, 9))
    with torch.inference_mode(), patch.object(GlobalLayer, "forward", observe):
        full, full_cache = model.prefill(x)
        cache = model.new_cache(1)
        parts = []
        for start, end in [(0, 3), (3, 4), (4, 7), (7, 9)]:
            logits, cache = model.prefill(x[:, start:end], cache)
            parts.append(logits)
            if end == 4:
                assert cache[0].view.version == 1
                assert cache[0].local[0]["tokens"] == 4
                assert cache[0].local[0]["k"].shape[2] == 4
    for chunk in range(3):
        group = records[chunk * 4:(chunk + 1) * 4]
        assert len(set(group)) == 1 and group[0][0] == chunk
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(cache[0].view.values, full_cache[0].view.values, atol=2e-6, rtol=2e-5)
    assert cache[0].local[0]["tokens"] == 9
    assert cache[0].local[0]["k"].shape[2] == 5


def test_causality_and_delayed_write_gradients_for_every_global_layer():
    torch.manual_seed(7)
    model = create_model(config()).eval()
    x = torch.randint(64, (1, 9))
    with torch.no_grad():
        full = model.hidden(x)[0]
        for start in (2, 4, 7):
            changed = x.clone()
            changed[:, start:] = (changed[:, start:] + 1) % 64
            torch.testing.assert_close(model.hidden(changed)[0][:, :start], full[:, :start], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(_blocks_forward(model, x), full)
    model.train()
    model.hidden(x)[0][:, -1].square().sum().backward()
    assert model.bank.memory.grad is not None
    for block in model.blocks:
        if isinstance(block, GlobalLayer):
            for name in ("k", "v", "beta", "decay"):
                gradient = getattr(block.attn, name).weight.grad
                assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_identical_token_proposals_are_summed_without_chunk_averaging():
    model = create_model(config())
    layer = model.blocks[2]
    view = model.bank.snapshot(1)
    x = torch.randn(1, 1, 32)
    single = layer.propose(x, view, 0, 2, 4, 16)
    repeated = layer.propose(x.expand(-1, 2, -1), view, 0, 2, 4, 16)
    torch.testing.assert_close(repeated.deltas.reshape(2, -1, 16)[0], single.deltas)
    torch.testing.assert_close(repeated.deltas.reshape(2, -1, 16)[1], single.deltas)
    # Changing the commit interval must not scale a token's proposed delta.
    longer = layer.propose(x, view, 0, 2, 128, 16)
    torch.testing.assert_close(longer.deltas, single.deltas)


def test_stack_reference_checkpoint_and_continuation():
    check(config(), "cpu", length=23)


def test_loops_are_rejected_in_the_new_stack():
    with pytest.raises(ValueError, match="no reasoning loops"):
        create_model(config(gl_max_steps=4))


def test_sliding_window_excludes_old_tokens_and_uses_absolute_rotary_positions():
    from gl_sdm.layers.attention import Transformer
    layer = Transformer({**config(), "attention_window": 4}, 0).eval()
    x = torch.randn(1, 13, 32)
    with torch.no_grad():
        full = layer(x)
        changed = x.clone()
        changed[:, 0] += 5
        torch.testing.assert_close(layer(changed)[:, 4:], full[:, 4:], atol=0, rtol=0)
        cache = {}
        parts = [layer(x[:, :6], cache), layer(x[:, 6:11], cache),
                 layer(x[:, 11:12], cache), layer(x[:, 12:], cache)]
    torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-7, rtol=2e-5)
    assert cache["tokens"] == 13 and cache["k"].shape[2] == 3


def test_benchmark_releases_request_caches_between_inference_samples():
    import weakref
    from gl_sdm.experiments.benchmark import run
    cfg = config()
    cfg.update(seq_len=20, batch_size=40, mbs=1)
    model = create_model(cfg)
    original = model.new_cache
    references = []
    def fresh(batch):
        assert all(ref() is None for ref in references), "benchmark retained a previous request"
        cache = original(batch)
        references.append(weakref.ref(cache[0]))
        return cache
    with patch.object(model, "new_cache", fresh):
        result = run(model, cfg, "cpu", iterations=2, warmup=1)
    assert result["training"]["weight_loops"] == 0
    assert len(references) == 6 and all(ref() is None for ref in references)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native URM needs CUDA")
def test_large_route_compiled_read_backward_and_override_restoration():
    from gl_sdm.memory.backends.urm import routed_read_plan, routed_snapshot_read
    from gl_sdm.memory.routing import torch_routed_read
    from urm.backends.triton.k3.route_generation import TritonSparseRouteBackend
    from urm.ir.program import DType, SparseRouteSelectionSpec
    original = TritonSparseRouteBackend.support_status
    spec = SparseRouteSelectionSpec(1, 3, 262144, 8, DType.FLOAT32)
    assert not original(spec).supported
    plan = routed_read_plan(1, 3, 262144, 128, 8, True)
    assert TritonSparseRouteBackend.support_status is original
    assert not original(spec).supported
    assert all(step["anchor"].startswith("urm_native") for step in plan.serialized_plan()["steps"])
    torch.manual_seed(7)
    memory = torch.randn(1, 1, 262144, 64, device="cuda", requires_grad=True)
    scores = torch.randn(1, 3, 1, 1024, device="cuda", requires_grad=True)
    with torch.no_grad():
        scores[:, 0].zero_()
        scores[:, 2].floor_()
    oracle_memory = memory.detach().clone().requires_grad_()
    oracle_scores = scores.detach().clone().requires_grad_()
    values, index, weights = routed_snapshot_read(torch.nn.functional.pad(memory, (0, 64)), scores, 8, True)
    expected, expected_index, expected_weights = torch_routed_read(oracle_memory, oracle_scores, 8)
    torch.testing.assert_close(index.long(), expected_index)
    torch.testing.assert_close(weights, expected_weights, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(values[..., :64], expected, atol=2e-6, rtol=2e-5)
    incoming = torch.randn_like(expected)
    (values[..., :64] * incoming).sum().backward()
    (expected * incoming).sum().backward()
    torch.testing.assert_close(memory.grad, oracle_memory.grad, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(scores.grad, oracle_scores.grad, atol=3e-6, rtol=3e-5)
    assert not original(spec).supported
    with pytest.raises(Exception, match="factor extent"):
        routed_read_plan(1, 3, 262144, 128, 8, False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native URM needs CUDA")
def test_large_route_complete_stack_reference_and_cache():
    cfg = config()
    cfg.update(gl_slots=262144, gl_reads=8, gl_writes=8,
               gl_memory_backend="urm", gl_urm_large_route_override=True,
               verify_batch_size=1, residual_dtype="float32")
    check(cfg, "cuda", length=23)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native URM needs CUDA")
def test_large_route_override_preserves_hardware_dtype_and_shape_declines():
    from gl_sdm.memory.backends.urm import large_route_override
    from urm.backends.triton.k3.route_generation import TritonSparseRouteBackend
    from urm.ir.program import DType, SparseRouteSelectionSpec
    original = TritonSparseRouteBackend.support_status
    with large_route_override(True):
        for slots, width, dtype in [(262144, 4, DType.FLOAT32),
                                    (1048576, 8, DType.FLOAT32),
                                    (262144, 8, DType.BFLOAT16)]:
            spec = SparseRouteSelectionSpec(1, 3, slots, width, dtype)
            assert not TritonSparseRouteBackend.support_status(spec).supported
        with patch("torch.cuda.is_available", return_value=False):
            spec = SparseRouteSelectionSpec(1, 3, 262144, 8, DType.FLOAT32)
            status = TritonSparseRouteBackend.support_status(spec)
            assert not status.supported and status.code == "unsupported_hardware"
    assert TritonSparseRouteBackend.support_status is original
