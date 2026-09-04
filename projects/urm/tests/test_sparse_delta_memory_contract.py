"""CPU-safe contract and compiler tests for the optional original-SDM adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from urm.adapters.sparse_delta_memory import (
    EXPECTED_SDM_COMMIT,
    MODE_INFERENCE,
    MODE_READ_ONLY,
    SDMAdapterConfig,
    SDMAddressTrace,
    SDMOperationSpec,
    SDMState,
    SDMTraceOrigin,
    _checkout_root,
    sdm_upstream_identity,
)


def test_upstream_pin_and_install_identity_are_explicit() -> None:
    assert EXPECTED_SDM_COMMIT == "183e7df809131b80ad4393741029d0f20fc3640b"
    identity = sdm_upstream_identity()
    assert identity["expected_commit"] == EXPECTED_SDM_COMMIT
    assert identity["repository"].endswith("facebookresearch/sparse-delta-memory")
    if identity.get("status") != "not_applicable":
        assert identity["license"] == "CC-BY-NC-4.0"
        assert "PYTHONPATH" in identity["installation"]
        assert identity["source_usage"].startswith("external checkout")


def test_checkout_root_requires_real_git_checkout(tmp_path) -> None:
    module = tmp_path / "lingua" / "sparse_delta_memory" / "layer.py"
    module.parent.mkdir(parents=True)
    module.touch()
    assert _checkout_root(str(module)) is None


def test_upstream_kernel_source_is_not_vendored() -> None:
    project_root = Path(__file__).resolve().parents[1]
    assert not (project_root / "src" / "lingua" / "sparse_delta_memory").exists()


def test_trace_rejects_duplicates_cross_partition_and_empty() -> None:
    weights = torch.ones((1, 1, 2), dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="duplicates"):
        SDMAddressTrace._from_product_key(
            torch.tensor([[[1, 1]]]),
            weights,
            torch.tensor([[[2, 3]]]),
            weights,
            slots_per_partition=64,
        )
    with pytest.raises(ValueError, match="partition"):
        SDMAddressTrace._from_product_key(
            torch.tensor([[[1, 2]], [[2, 66]]]),
            torch.ones((2, 1, 2)),
            torch.tensor([[[3, 4]], [[67, 68]]]),
            torch.ones((2, 1, 2)),
            slots_per_partition=64,
        )
    with pytest.raises(ValueError, match="empty"):
        SDMAddressTrace._from_product_key(
            torch.empty((1, 0, 1), dtype=torch.int64),
            torch.empty((1, 0, 1)),
            torch.empty((1, 0, 1), dtype=torch.int64),
            torch.empty((1, 0, 1)),
            slots_per_partition=64,
        )


def test_cpu_execution_is_explicitly_rejected_at_the_call_boundary() -> None:
    trace = SDMAddressTrace._from_product_key(
        torch.tensor([[[1]]]),
        torch.ones((1, 1, 1)),
        torch.tensor([[[2]]]),
        torch.ones((1, 1, 1)),
        slots_per_partition=64,
    )
    with pytest.raises(ValueError, match="require CUDA"):
        SDMOperationSpec.from_call(
            trace,
            SDMState(torch.zeros((64, 7))),
            mode=MODE_READ_ONLY,
            config=SDMAdapterConfig(
                64,
                7,
                1,
                1,
                16,
                MODE_READ_ONLY,
                torch.device("cuda", 0),
                torch.float32,
            ),
        )


@pytest.mark.parametrize(
    "weights,match",
    [
        ([float("nan"), 1.0], "finite"),
        ([-0.1, 1.1], "nonnegative"),
        ([0.4, 0.4], "Softmax normalization"),
    ],
)
def test_trace_weight_certification_rejects_invalid_values(weights, match) -> None:
    tensor = torch.tensor([[weights]], dtype=torch.float32)
    with pytest.raises(ValueError, match=match):
        SDMAddressTrace._from_product_key(
            torch.tensor([[[1, 2]]]),
            tensor,
            torch.tensor([[[3, 4]]]),
            torch.tensor([[[0.5, 0.5]]]),
            slots_per_partition=64,
        )


def test_certified_trace_detects_post_construction_mutation() -> None:
    trace = SDMAddressTrace._from_product_key(
        torch.tensor([[[1]]]),
        torch.ones((1, 1, 1)),
        torch.tensor([[[2]]]),
        torch.ones((1, 1, 1)),
        slots_per_partition=64,
    )
    assert trace.origin is SDMTraceOrigin.PRODUCT_KEY_GENERATED
    trace.write_weights.add_(1)
    with pytest.raises(ValueError, match="mutated after construction"):
        trace._require_intact()


@pytest.mark.parametrize(
    "config,match",
    [
        (
            SDMAdapterConfig(
                128, 7, 1, 1, 16, MODE_READ_ONLY, torch.device("cpu"), torch.float32
            ),
            "slots",
        ),
        (
            SDMAdapterConfig(
                64, 9, 1, 1, 16, MODE_READ_ONLY, torch.device("cpu"), torch.float32
            ),
            "memory",
        ),
        (
            SDMAdapterConfig(
                64, 7, 2, 1, 16, MODE_READ_ONLY, torch.device("cpu"), torch.float32
            ),
            "widths",
        ),
        (
            SDMAdapterConfig(
                64, 7, 1, 1, 16, MODE_INFERENCE, torch.device("cpu"), torch.float32
            ),
            "mode",
        ),
    ],
)
def test_dispatch_rejects_trace_adapter_configuration_drift(config, match) -> None:
    trace = SDMAddressTrace._from_product_key(
        torch.tensor([[[1]]]),
        torch.ones((1, 1, 1)),
        torch.tensor([[[2]]]),
        torch.ones((1, 1, 1)),
        slots_per_partition=64,
    )
    with pytest.raises(ValueError, match=match):
        SDMOperationSpec.from_call(
            trace,
            SDMState(torch.zeros((64, 7))),
            mode=MODE_READ_ONLY,
            config=config,
        )
