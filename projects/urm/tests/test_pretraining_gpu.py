"""Short GPU correctness gates for the independently owned pretraining model."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from urm.adapters.sparse_delta_memory import probe_sdm_support
from urm.pretraining import FP32AdamW, PretrainingConfig, URMDecoderLM


def _config() -> PretrainingConfig:
    return PretrainingConfig(
        vocab_size=128,
        sequence_length=16,
        layers=1,
        width=64,
        heads=1,
        value_dim=64,
        slots_per_partition=64,
        reads=8,
        writes=8,
        microbatch=1,
        gradient_accumulation=1,
    )


def _step(model, tokens, targets):
    optimizer = FP32AdamW(model.named_parameters(), lr=1e-4)
    model.reset_state()
    logits, loss = model(tokens, targets)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer.step()
    model.detach_state()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in gradients.values())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    return (
        logits.detach().float(),
        float(loss.item()),
        gradients,
        model.state_checksums(),
    )


@pytest.mark.skipif(
    not probe_sdm_support().supported, reason="pinned upstream SDM is unavailable"
)
def test_seed_matched_sparse_pretraining_step_matches() -> None:
    config = _config()
    generator = torch.Generator(device="cuda").manual_seed(991)
    tokens = torch.randint(
        0, config.vocab_size, (1, 16), device="cuda", generator=generator
    )
    targets = torch.randint(
        0, config.vocab_size, (1, 16), device="cuda", generator=generator
    )
    rows = []
    for backend in ("upstream_sdm", "urm_native"):
        torch.manual_seed(191)
        torch.cuda.manual_seed_all(191)
        rows.append(
            _step(
                URMDecoderLM(config, backend).cuda().to(torch.bfloat16),
                tokens,
                targets,
            )
        )
    upstream, native = rows
    torch.testing.assert_close(native[0], upstream[0], atol=0.03, rtol=0.01)
    assert abs(native[1] - upstream[1]) <= 0.01
    for name in upstream[2]:
        left, right = upstream[2][name].flatten(), native[2][name].flatten()
        assert torch.nn.functional.cosine_similarity(left, right, dim=0) >= 0.95
        torch.testing.assert_close(right, left, atol=0.02, rtol=0.05)
    assert (
        max(
            abs(a["mean"] - b["mean"])
            for a, b in zip(upstream[3], native[3], strict=True)
        )
        <= 2e-6
    )


def test_native_pretraining_uses_compiler_serialized_schedule() -> None:
    model = URMDecoderLM(_config(), "urm_native").cuda().to(torch.bfloat16)
    mixers = model.sparse_mixers()
    assert len(mixers) == 1
    plan = mixers[0]._executor.serialized_plan()
    dispatch = [step for step in plan["steps"] if step["kind"] == "anchor_dispatch"]
    assert dispatch[0]["anchor"] == "urm_native_sparse_memory_e2e_v0"
    assert dispatch[0]["launch_config"]["schedule_family"] == (
        "native_route_then_partition_scan"
    )


@pytest.mark.parametrize("backend", ["upstream_sdm", "urm_native"])
def test_sparse_pretraining_fullgraph_training_path(backend: str) -> None:
    if backend == "upstream_sdm" and not probe_sdm_support().supported:
        pytest.skip("pinned upstream SDM is unavailable")
    config = _config()
    torch.manual_seed(813)
    model = URMDecoderLM(config, backend).cuda().to(torch.bfloat16)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch._dynamo.config.error_on_recompile = True
    compiled = torch.compile(model, fullgraph=True, dynamic=False)
    tokens = torch.arange(16, device="cuda").view(1, 16) % config.vocab_size
    logits, loss = compiled(tokens, tokens)
    loss.backward()
    model.detach_state()
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    counters = dict(torch._dynamo.utils.counters)
    assert sum(counters.get("graph_break", {}).values()) == 0
    assert counters["stats"]["unique_graphs"] == 1
