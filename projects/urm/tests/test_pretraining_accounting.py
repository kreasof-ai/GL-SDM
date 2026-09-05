"""Dependency-light accounting gates for the model-level pretraining lane."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from urm.pretraining import (
    FP32AdamW,
    PretrainingConfig,
    model_memory_ledger,
    model_parameter_count,
    semantic_training_flops,
)


def test_frozen_model_parameter_and_flop_ledgers() -> None:
    config = PretrainingConfig()
    assert model_parameter_count(config) == 124_651_008
    ledger = semantic_training_flops(config)
    assert ledger.useful_total == 3_098_778_811_392
    assert ledger.backward_included is True
    assert ledger.uncredited_route_selection_comparisons == 452_984_832
    assert "uncredited_route_selection_comparisons" not in {
        "embedding_projection",
        "normalization",
        "mlp",
        "logits_and_loss",
        "sparse_score_generation",
        "sparse_route_normalization",
        "sparse_state_update_read",
        "optimizer",
    }


def test_fp32_optimizer_and_memory_accounting() -> None:
    model = torch.nn.Linear(5, 3, bias=False).to(torch.bfloat16)
    optimizer = FP32AdamW(model.named_parameters())
    model(torch.ones(2, 5, dtype=torch.bfloat16)).float().sum().backward()
    before = optimizer.master[0].clone()
    updates = optimizer.step(record_updates=True)
    ledger = model_memory_ledger(model, optimizer)
    assert all(tensor.dtype is torch.float32 for tensor in optimizer.state_tensors())
    assert ledger == {
        "parameter_bytes": 30,
        "gradient_bytes": 30,
        "optimizer_state_bytes": 180,
        "persistent_state_bytes": 0,
    }
    assert updates["final_norm"]["l2"] == pytest.approx(
        float((optimizer.master[0] - before).norm())
    )


def test_frozen_toml_matches_primary_contract() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "pretraining_step.toml"
    frozen = tomllib.loads(path.read_text(encoding="utf-8"))
    model = frozen["model"]
    assert frozen["freeze_status"] == "pre_measurement"
    assert (
        model["layers"],
        model["width"],
        model["heads"],
        model["value_dim"],
        model["sequence_length"],
        model["vocab_size"],
    ) == (12, 768, 12, 64, 1024, 50_304)
    assert (model["slots_per_partition"], model["reads"], model["writes"]) == (
        4096,
        64,
        64,
    )
    assert frozen["flops"]["sorting_padding_recomputation_credited"] is False


def test_model_benchmark_has_no_direct_native_backend_shortcut() -> None:
    root = Path(__file__).parents[1]
    benchmark = (root / "benchmarks" / "pretraining_step.py").read_text(
        encoding="utf-8"
    )
    model = (root / "src" / "urm" / "pretraining.py").read_text(encoding="utf-8")
    assert "TritonSparseMemoryBackend" not in benchmark
    assert "TritonSparseMemoryBackend" not in model
    assert "compile_sparse_memory_plan(spec)" in model


@pytest.mark.parametrize("width", [65, 128])
def test_model_rejects_route_width_above_factor_extent(width: int) -> None:
    with pytest.raises(ValueError, match="factor extent"):
        PretrainingConfig(reads=width)
