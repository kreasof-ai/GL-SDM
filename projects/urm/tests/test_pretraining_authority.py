"""Regression gates for actual execution, matching, and timing authority."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
from audit_sparse_memory_launches import verify_launches
from pretraining_projection import project_state_replacement
from pretraining_step import _compare_correctness, _one_step, load_frozen_config

from urm.pretraining import FP32AdamW


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False)])
        self.state_checksums = Mock(return_value=[])
        self.reset_state = Mock()
        self.detach_state = Mock()
        self.capture_mixer_input_gradients = Mock()

    def forward(self, tokens, targets):
        logits = self.blocks[0](tokens) * self.weight
        return logits, (logits - targets).square().mean()

    def sparse_mixers(self):
        return []


def test_timed_boundary_calls_execution_wrapper_without_diagnostics():
    model = ToyModel()
    wrapper = Mock(wraps=model)
    wrapper._orig_mod = model
    optimizer = FP32AdamW(model.named_parameters())
    _, config = load_frozen_config()
    batch = (torch.ones(1, 2, 8), torch.zeros(1, 2, 8))
    report, gradients = _one_step(
        wrapper,
        optimizer,
        [batch] * 4,
        config,
        gradient_clip=1.0,
        record_correctness=False,
        torch=torch,
    )
    assert wrapper.call_count == 4
    assert model.detach_state.call_count == 4
    model.state_checksums.assert_not_called()
    assert report == {} and gradients == []
    assert optimizer.step_count == 1


def test_comparison_rejects_unmatched_initial_optimizer():
    with pytest.raises(RuntimeError, match="matched parameters"):
        _compare_correctness(
            {"matched_initial": {"optimizer": "a"}},
            {"matched_initial": {"optimizer": "b"}},
            torch,
            {},
        )


def test_compiled_wrapper_preserves_parameter_gradient_groups():
    model = ToyModel()
    optimizer = FP32AdamW(model.named_parameters())
    graph = torch.compile(model, backend="eager", fullgraph=True)
    config = SimpleNamespace(sequence_length=2, gradient_accumulation=2)
    batch = (torch.ones(1, 2, 8), torch.zeros(1, 2, 8))
    report, _ = _one_step(
        graph,
        optimizer,
        [batch] * 2,
        config,
        gradient_clip=1.0,
        record_correctness=True,
        torch=torch,
    )
    assert set(report["gradient_norms"]) == {"block_0_weight", "final_norm"}
    assert len(report["sampled_logits"]) == 2


def test_intermediate_microbatch_failure_is_not_hidden_by_last_state():
    frozen, _ = load_frozen_config()

    def process(mean):
        states = [[{"sum": x, "mean": x, "elements": 1}] for x in (mean, 0.0, 0.0, 0.0)]
        report = {
            "sampled_logits": [[[0.0]]],
            "loss_before": 0.0,
            "loss_after": 0.0,
            "gradient_norms": {"a": 0.0},
            "parameter_updates": {"a": {"l2": 0.0, "max_abs": 0.0}},
            "persistent_state_checksums": states[-1],
            "microbatch_persistent_state_checksums": states,
            "nonfinite": {"parameters": 0},
        }
        return {"matched_initial": {}, "correctness": [report] * 5}

    result = _compare_correctness(
        process(0.0), process(3e-6), torch, frozen["correctness"]["bfloat16"]
    )
    assert not result["passed"]
    assert result["steps"][0]["persistent_state_checksum_normalized_max_abs"] == 3e-6


def test_threefold_screen_is_not_model_acceptance():
    projection = project_state_replacement(2.365, 0.8, 3.0)
    assert projection["threefold_screen_passed"]
    assert projection["predicted_native_upstream_ratio"] == pytest.approx(1.1036666667)
    assert not projection["projection_gate_passed"]
    assert (
        project_state_replacement(2.365, 0.2, 100.0)["required_speedup_for_1_05"]
        is None
    )


@pytest.mark.parametrize("backward", [False, True])
def test_launch_audit_rejects_actual_state_launch_drift(backward):
    root = Path(__file__).parents[1]
    artifact = json.loads(
        (root / "results/pretraining-step/launch-audit-eager-v2.json").read_text()
    )
    _, config = load_frozen_config()
    verify_launches(artifact["observed_launches"], artifact["plan"], config)
    records = copy.deepcopy(artifact["observed_launches"])
    target = next(
        row
        for row in records
        if "sparse_state_update" in row["kernel"]
        and ("backward" in row["kernel"]) == backward
    )
    target["constants"]["BLOCK_D"] = 64
    with pytest.raises(AssertionError):
        verify_launches(records, artifact["plan"], config)
