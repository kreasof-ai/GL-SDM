"""Native Triton kernels for the distinguished K2 nonlinear recurrences must
match their NumPy canonical executors (``urm.oracles.nonlinear_recurrence``)
directly, in fp32.

Each test builds the recipe operands with ``_rng_operands`` (the RECURRENCE
section of ``benchmarks/representation_coverage.py``), converts them to CUDA
torch float32, runs the native kernel, and compares against the float64
canonical executor. They must agree to ~1e-4 (fp32 kernel vs fp64 oracle).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for native nonlinear recurrence validation",
        allow_module_level=True,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import representation_coverage as rc  # noqa: E402

from urm.frontend.mixer_recipes import named_mixer_recipe  # noqa: E402
from urm.oracles import nonlinear_recurrence as nl  # noqa: E402
from urm.backends.triton.recurrence import nonlinear as native  # noqa: E402


def _cuda_operands(operand_dict):
    """Convert NumPy operands to CUDA torch float32 (scalars pass through)."""
    out = {}
    for name, value in operand_dict.items():
        if isinstance(value, (int, float, bool)):
            out[name] = value
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in "iu":
            out[name] = torch.as_tensor(arr, dtype=torch.int64, device="cuda")
        else:
            out[name] = torch.as_tensor(arr, dtype=torch.float32, device="cuda")
    return out


def _operands_for(recipe_name, seed=0):
    spec = named_mixer_recipe(recipe_name).spec
    return rc._rng_operands(spec, seed=seed)


def _max_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def test_tanh_rnn_native_matches_canonical():
    ops = _operands_for("rnn_core")
    ref_out, ref_state = nl.tanh_rnn(
        ops["query"], ops["weight"], ops["initial_state"]
    )
    t = _cuda_operands(ops)
    out, state = native.execute_tanh_rnn(
        query=t["query"], weight=t["weight"], initial_state=t["initial_state"]
    )
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(state.cpu().numpy(), ref_state) < 1e-4


def test_gated_rnn_native_matches_canonical():
    ops = _operands_for("gru_core")
    ref_out, ref_state = nl.gated_rnn(
        ops["query"], ops["weight"], ops["forget_input"], ops["forget_weight"],
        ops["reset_input"], ops["reset_weight"], ops["initial_state"],
    )
    t = _cuda_operands(ops)
    out, state = native.execute_gated_rnn(
        query=t["query"], weight=t["weight"], forget_input=t["forget_input"],
        forget_weight=t["forget_weight"], reset_input=t["reset_input"],
        reset_weight=t["reset_weight"], initial_state=t["initial_state"],
    )
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(state.cpu().numpy(), ref_state) < 1e-4


def test_multiplicative_rnn_native_matches_canonical():
    ops = _operands_for("m2rnn_core")
    ref_out, ref_state = nl.multiplicative_rnn(
        ops["query"], ops["key"], ops["value"], ops["weight"],
        ops["forget_input"], ops["initial_state"],
    )
    t = _cuda_operands(ops)
    out, state = native.execute_multiplicative_rnn(
        query=t["query"], key=t["key"], value=t["value"], weight=t["weight"],
        forget_input=t["forget_input"], initial_state=t["initial_state"],
    )
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(state.cpu().numpy(), ref_state) < 1e-4


def test_rwkv4_scalar_state_native_matches_canonical():
    ops = _operands_for("rwkv4_memory_core")
    ref_out, ref_state = nl.rwkv4_scalar_state(
        ops["w"], ops["u"], ops["k"], ops["v"], ops["state"]
    )
    t = _cuda_operands(ops)
    out, state = native.execute_rwkv4_scalar_state(
        w=t["w"], u=t["u"], key=t["k"], value=t["v"], state_input=t["state"]
    )
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(state.cpu().numpy(), ref_state) < 1e-4


def test_rwkv6_bonus_corrected_native_matches_canonical():
    ops = _operands_for("rwkv6_memory_core")
    ref_out, ref_state = nl.rwkv6_bonus_corrected(
        ops["query"], ops["key"], ops["value"], ops["log_decay"], ops["bonus"],
        initial_state=None,
    )
    t = _cuda_operands(ops)
    out, state = native.execute_rwkv6_bonus_corrected(
        query=t["query"], key=t["key"], value=t["value"],
        log_decay=t["log_decay"], bonus=t["bonus"], initial_state=None,
    )
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(state.cpu().numpy(), ref_state) < 1e-4


def test_mamba2_structured_ssm_native_matches_canonical():
    ops = _operands_for("mamba2_ssm_core")
    ref_out, ref_state = nl.mamba2_structured_ssm(
        ops["x"], ops["dt"], ops["A"], ops["B"], ops["C"], initial_states=None,
    )
    t = _cuda_operands(ops)
    out, state = native.execute_mamba2_structured_ssm(
        x=t["x"], dt=t["dt"], A=t["A"], B=t["B"], C=t["C"], initial_states=None,
    )
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(state.cpu().numpy(), ref_state) < 1e-4


def test_trapezoidal_ssm_native_matches_canonical():
    ops = _operands_for("mamba3_siso_core")
    ref_out, ref_state = nl.trapezoidal_ssm(
        ops["query"], ops["key"], ops["value"], ops["adt"], ops["dt"],
        ops["trap"], ops["query_bias"], ops["key_bias"], ops["angles"],
    )
    t = _cuda_operands(ops)
    out, state = native.execute_trapezoidal_ssm(
        query=t["query"], key=t["key"], value=t["value"], adt=t["adt"],
        dt=t["dt"], trap=t["trap"], query_bias=t["query_bias"],
        key_bias=t["key_bias"], angles=t["angles"],
    )
    ref_angle, ref_ssm, ref_key, ref_value = ref_state
    angle, ssm, key_state, value_state = state
    assert _max_err(out.cpu().numpy(), ref_out) < 1e-4
    assert _max_err(angle.cpu().numpy(), ref_angle) < 1e-4
    assert _max_err(ssm.cpu().numpy(), ref_ssm) < 1e-4
    assert _max_err(key_state.cpu().numpy(), ref_key) < 1e-4
    assert _max_err(value_state.cpu().numpy(), ref_value) < 1e-4
