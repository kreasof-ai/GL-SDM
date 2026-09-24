"""Native inner-state / convolution / solve executors vs the NumPy canonical executors.

Each test builds operands with ``_rng_operands`` from
``benchmarks/representation_coverage.py`` (the RECURRENCE section), runs the
native CUDA path in ``urm/backends/triton/recurrence/inner_state.py`` in fp32,
and compares against the float64 NumPy canonical executor in
``urm/oracles/nonlinear_recurrence.py`` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from representation_coverage import _rng_operands  # noqa: E402
from benchmarks.recipe_catalog import load_kernel_recipe  # noqa: E402
from urm.backends.reference.numpy import nonlinear_recurrence as canonical  # noqa: E402
from urm.backends.triton.k2 import inner_state as native  # noqa: E402

ATOL = 1e-4
RTOL = 1e-4


def _operands(name):
    return _rng_operands(load_kernel_recipe(name).spec, seed=0)


def _cuda(value):
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value, dtype=torch.float32, device="cuda")
    return value


def _torch_operands(operands, *names):
    return {name: _cuda(operands[name]) for name in names}


def _assert_close(actual, expected, label):
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(),
        np.asarray(expected, dtype=np.float64),
        atol=ATOL,
        rtol=RTOL,
        err_msg=label,
    )


def test_layernorm_inner_state():
    ops = _operands("ttt_linear_core")
    args = _torch_operands(ops, "query", "key", "value", "w", "b", "eta")
    chunk_size = ops["chunk_size"]
    out, (memory, memory_bias) = native.execute_layernorm_inner_state(
        **args, chunk_size=chunk_size
    )
    ref_out, (ref_memory, ref_memory_bias) = canonical.layernorm_inner_state(
        ops["query"], ops["key"], ops["value"], ops["w"], ops["b"], ops["eta"],
        chunk_size=chunk_size,
    )
    _assert_close(out, ref_out, "layernorm_inner_state output")
    _assert_close(memory, ref_memory, "layernorm_inner_state memory")
    _assert_close(
        memory_bias, ref_memory_bias.squeeze(-2), "layernorm_inner_state memory_bias"
    )


def test_momentum_inner_state():
    ops = _operands("titans_linear_memory_core")
    args = _torch_operands(ops, "query", "key", "value", "w", "b", "theta", "alpha", "eta")
    chunk_size = ops["chunk_size"]
    out, memory = native.execute_momentum_inner_state(**args, chunk_size=chunk_size)
    ref_out, ref_memory = canonical.momentum_inner_state(
        ops["query"], ops["key"], ops["value"], ops["w"], ops["b"],
        ops["theta"], ops["alpha"], ops["eta"], chunk_size=chunk_size,
    )
    _assert_close(out, ref_out, "momentum_inner_state output")
    _assert_close(memory, ref_memory, "momentum_inner_state memory")


def test_regularized_solve():
    ops = _operands("mesa_net_core")
    args = _torch_operands(ops, "query", "key", "value", "log_decay", "beta", "lamb")
    out, (h_kk, h_kv) = native.execute_regularized_solve(**args)
    ref_out, (ref_kk, ref_kv) = canonical.regularized_solve(
        ops["query"], ops["key"], ops["value"],
        ops["log_decay"], ops["beta"], ops["lamb"],
    )
    _assert_close(out, ref_out, "regularized_solve output")
    _assert_close(h_kk, ref_kk, "regularized_solve h_kk")
    _assert_close(h_kv, ref_kv, "regularized_solve h_kv")


def test_second_order_cumsum():
    ops = _operands("hla_second_order_core")
    args = _torch_operands(ops, "query", "key", "value")
    out, state = native.execute_second_order_cumsum(**args)
    ref_out, ref_state = canonical.second_order_cumsum(
        ops["query"], ops["key"], ops["value"]
    )
    assert state is None and ref_state is None
    _assert_close(out, ref_out, "second_order_cumsum output")


def test_fft_convolution():
    ops = _operands("hyena_fftconv_core")
    args = _torch_operands(ops, "query", "kernel", "direct")
    out, state = native.execute_fft_convolution(**args)
    ref_out, ref_state = canonical.hyena_fft_convolution(
        ops["query"], ops["kernel"], ops["direct"]
    )
    assert state is None and ref_state is None
    _assert_close(out, ref_out, "fft_convolution output")


def test_two_stage_fft_convolution():
    ops = _operands("h3_ssm_fft_core")
    args = _torch_operands(
        ops, "query", "key", "value", "ssm_kernel", "ssm_k_kernel", "ssm_k_direct", "skip"
    )
    out, state = native.execute_two_stage_fft_convolution(**args)
    ref_out, ref_state = canonical.two_stage_fft_convolution(
        ops["query"], ops["key"], ops["value"],
        ops["ssm_kernel"], ops["ssm_k_kernel"], ops["ssm_k_direct"], ops["skip"],
    )
    assert state is None and ref_state is None
    _assert_close(out, ref_out, "two_stage_fft_convolution output")


@pytest.mark.parametrize("recipe", ["abc_core", "gsa_core"])
def test_slot_attention_two_stage(recipe):
    ops = _operands(recipe)
    if recipe == "abc_core":
        # ABC derives slot_weights and log_decay from slot_logits via a
        # cumulative log-sum-exp over time (the composition derivation).
        slot_logits = np.asarray(ops["slot_logits"], dtype=np.float64)
        cumulative = np.apply_along_axis(
            lambda x: np.logaddexp.accumulate(x), 1, slot_logits
        )
        log_decay = (
            np.concatenate((cumulative[:, :1], cumulative[:, :-1]), axis=1) - cumulative
        )
        slot_weights = np.exp(slot_logits - cumulative)
    else:
        slot_weights = ops["slot_weights"]
        log_decay = ops["log_decay"]
    args = _torch_operands(ops, "query", "key", "value")
    args["slot_weights"] = _cuda(slot_weights)
    args["log_decay"] = _cuda(log_decay)
    out, (key_state, value_state) = native.execute_slot_attention_two_stage(**args)
    ref_out, (ref_key_state, ref_value_state) = canonical.slot_attention_two_stage(
        ops["query"], ops["key"], ops["value"], slot_weights, log_decay,
        group_size=ops["query"].shape[2] // ops["key"].shape[2],
    )
    _assert_close(out, ref_out, f"{recipe} slot_attention_two_stage output")
    _assert_close(key_state, ref_key_state, f"{recipe} slot_attention_two_stage key_state")
    _assert_close(
        value_state, ref_value_state, f"{recipe} slot_attention_two_stage value_state"
    )


def test_momentum_delta():
    ops = _operands("momentum_delta_core")
    args = _torch_operands(
        ops, "query", "key", "value", "p", "log_alpha", "log_mu", "beta", "eta"
    )
    out, (state, momentum) = native.execute_momentum_delta(**args)
    ref_out, (ref_state, ref_momentum) = canonical.momentum_delta(
        ops["query"], ops["key"], ops["value"], ops["p"],
        ops["log_alpha"], ops["log_mu"], ops["beta"], ops["eta"],
    )
    _assert_close(out, ref_out, "momentum_delta output")
    _assert_close(state, ref_state, "momentum_delta state")
    _assert_close(momentum, ref_momentum, "momentum_delta momentum")


def test_gated_oja():
    ops = _operands("gated_oja_core")
    args = _torch_operands(ops, "query", "key", "value", "beta")
    args["gate"] = _cuda(ops["gv"])
    out, state = native.execute_gated_oja(**args)
    ref_out, ref_state = canonical.gated_oja(
        ops["query"], ops["key"], ops["value"], ops["gv"], ops["beta"]
    )
    _assert_close(out, ref_out, "gated_oja output")
    _assert_close(state, ref_state, "gated_oja state")
