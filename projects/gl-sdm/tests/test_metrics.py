import copy
import pytest
import torch
from gl_sdm.metrics import gl_parameter_groups, parameter_counts, utilization
from gl_sdm.model import create_model
from test_global_memory import config


def test_tied_depth_changes_work_without_multiplying_once_only_parameters():
    one = create_model(config(gl_max_steps=1, seq_len=8, gl_chunk_size=4))
    four = create_model(config(gl_max_steps=4, seq_len=8, gl_chunk_size=4))
    a = utilization(one, 16, 1, 70e12)
    b = utilization(four, 16, 1, 70e12)
    assert parameter_counts(one) == parameter_counts(four)
    assert a["unique_parameter_6nd_pct"] == b["unique_parameter_6nd_pct"]
    assert a["reasoner_token_passes"] == 16 and b["reasoner_token_passes"] == 64
    assert a["write_token_passes"] == 8 and b["write_token_passes"] == 32
    # The new work comes from the reasoner and the preceding chunk's writes.
    added = 6 * (48 * a["reasoner_params"] + 24 * a["write_params"])
    assert b["estimated_training_flops_6nd"] - a["estimated_training_flops_6nd"] == added
    assert a["mfu_6nd_pct"] < b["mfu_6nd_pct"] < 4 * a["mfu_6nd_pct"]
    assert sum(gl_parameter_groups(one).values()) == a["active_params"]


def test_act_counts_observed_depth_at_write_positions_not_maximum_or_mean():
    model = create_model(config(gl_reasoning="adaptive", gl_max_steps=4, seq_len=4, gl_chunk_size=2))
    early = utilization(model, 4, 1, 70e12, torch.tensor([1, 1, 4, 4]))
    late = utilization(model, 4, 1, 70e12, torch.tensor([4, 4, 1, 1]))
    assert early["mean_reasoning_steps"] == late["mean_reasoning_steps"] == 2.5
    assert early["reasoner_token_passes"] == late["reasoner_token_passes"] == 10
    assert early["write_token_passes"] == 2 and late["write_token_passes"] == 8
    assert late["estimated_training_flops_6nd"] - early["estimated_training_flops_6nd"] == 6 * 6 * early["write_params"]


@pytest.mark.parametrize("length,expected", [(3, 0), (4, 0), (5, 8), (8, 8), (9, 16)])
def test_partial_and_complete_terminal_chunks_do_not_count_omitted_writes(length, expected):
    model = create_model(config(gl_max_steps=2, seq_len=length, gl_chunk_size=4))
    result = utilization(model, length, 1, None)
    assert result["write_token_passes"] == expected
    assert result["mfu_6nd_pct"] is result["unique_parameter_6nd_pct"] is None


def test_final_policy_counts_projections_that_are_executed_with_zero_mass():
    merged = create_model(config(gl_max_steps=3, seq_len=8, gl_chunk_size=4))
    final = copy.deepcopy(merged)
    final.blocks[0].write_policy = "final"
    assert utilization(merged, 8, 1, 70e12)["mfu_6nd_pct"] == utilization(final, 8, 1, 70e12)["mfu_6nd_pct"]


def test_token_clock_does_not_apply_chunk_write_pruning():
    model = create_model(config(gl_max_steps=3, seq_len=8))
    result = utilization(model, 8, 1, 70e12)
    assert result["reasoner_token_passes"] == result["write_token_passes"] == 24


def test_invalid_depth_telemetry_is_rejected():
    model = create_model(config(gl_reasoning="adaptive", seq_len=4, gl_chunk_size=2))
    for depth in (torch.tensor([1]), torch.tensor([1., 2., 2., float("nan")]),
                  torch.tensor([1., 2., 2., 1.5]), torch.tensor([0, 1, 1, 1])):
        with pytest.raises(ValueError, match="telemetry|depths"):
            utilization(model, 4, 1, 70e12, depth)
    with pytest.raises(ValueError, match="observed"):
        utilization(model, 4, 1, 70e12)


def test_actual_act_forward_supplies_depth_for_training_metrics():
    model = create_model(config(gl_reasoning="adaptive", gl_max_steps=4, seq_len=5, gl_chunk_size=4))
    with torch.no_grad():
        model.blocks[0].halt.weight.zero_()
        model.blocks[0].halt.bias.fill_(8)
    model(torch.randint(64, (2, 5)), torch.randint(64, (2, 5)))
    result = utilization(model, 10, 1, 70e12)
    assert result["reasoner_token_passes"] == 10
    assert result["write_token_passes"] == 8
