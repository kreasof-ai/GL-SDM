"""Native K1 attention-variant kernels mirrored against the NumPy canonical core.

Each test runs a covered K1 attention-variant recipe through the URM-native
(Triton) execution path and compares the output to the float64 canonical core
(:func:`urm.oracles.composition.execute_canonical`) on the recipe's operands.
The canonical core runs in float64 and the native kernels in float32, so the
tolerance is float32 precision (~1e-4, asserted on the relative error).

The native plans are constructed directly (``CompiledMixerPlan`` with
``backend=NATIVE``) because the compile-time native anchor gate that admits these
operations is widened separately; these tests pin the kernel/dispatch contract
that gate will expose.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from urm.compiler.execution import NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME
from urm.compiler.unified_mixer import CompiledMixerPlan
from urm.frontend.mixer_recipes import named_mixer_recipe
from urm.ir.mixer import MixerBackend, MixerIntent
from urm.oracles.composition import execute_canonical

# fp32 native kernels vs the fp64 canonical core: allow fp32 precision.
ATOL = 1e-4
RTOL = 1e-4

_B, _T, _H, _K, _V = 1, 5, 2, 4, 4


def _native_plan(recipe_name: str) -> CompiledMixerPlan:
    recipe = named_mixer_recipe(recipe_name)
    return CompiledMixerPlan(
        spec=recipe.spec,
        intent=MixerIntent.INFERENCE,
        anchor=NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME,
        backend=MixerBackend.NATIVE,
        recipe=recipe,
        compile_dtype="float32",
    )


def _to_torch_operands(operands: dict) -> dict:
    out = {}
    for name, value in operands.items():
        if isinstance(value, (int, float, bool)):
            out[name] = value
        else:
            out[name] = torch.as_tensor(
                np.asarray(value), dtype=torch.float32, device="cuda"
            )
    return out


def _to_canonical_operands(operands: dict) -> dict:
    return {
        name: (value if isinstance(value, (int, float)) else np.asarray(value, np.float64))
        for name, value in operands.items()
    }


def _run_against_canonical(recipe_name: str, operands: dict) -> tuple[float, float]:
    plan = _native_plan(recipe_name)
    canonical = execute_canonical(plan.spec, **_to_canonical_operands(operands))["output"]
    native = plan.execute(**_to_torch_operands(operands)).output.detach().cpu().numpy()
    abs_err = float(np.abs(native - canonical).max())
    rel_err = abs_err / max(float(np.abs(canonical).max()), 1e-12)
    return abs_err, rel_err


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("native K1 requires CUDA")


def test_projected_tucker_matches_canonical():
    rng = np.random.default_rng(0)
    r = 3  # low-rank query width
    operands = {
        "query": rng.normal(size=(_B, _T, r)),
        "B_pre": rng.normal(size=(_H, r, _K)),
        "key": rng.normal(size=(_B, _T, _K)),
        "value": rng.normal(size=(_B, _T, _V)),
    }
    abs_err, rel_err = _run_against_canonical("tucker_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


def test_local_window_longformer_matches_canonical():
    rng = np.random.default_rng(1)
    operands = {
        "query": rng.normal(size=(_B, _T, _H, _K)),
        "key": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _V)),
        "attention_window": 2,
    }
    abs_err, rel_err = _run_against_canonical("longformer_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


def test_gated_wall_matches_canonical():
    rng = np.random.default_rng(2)
    operands = {
        "query": rng.normal(size=(_B, _T, _H, _K)),
        "key": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _V)),
        "g": -rng.uniform(0, 0.3, size=(_B, _T, _H, _K)),
    }
    abs_err, rel_err = _run_against_canonical("wall_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


def test_positional_parallax_matches_canonical():
    rng = np.random.default_rng(3)
    operands = {
        "query": rng.normal(size=(_B, _T, _H, _K)),
        "r": rng.normal(size=(_B, _T, _H, _K)),
        "key": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _K)),
    }
    abs_err, rel_err = _run_against_canonical("parallax_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


@pytest.mark.parametrize("num_groups", (1, 2, 4))
def test_positive_feature_kata_matches_canonical(num_groups):
    rng = np.random.default_rng(4)
    operands = {
        "query": rng.normal(size=(_B, _T, _H, _K)),
        "key": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _V)),
        "num_groups": num_groups,
    }
    abs_err, rel_err = _run_against_canonical("kata_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


def test_thresholded_tda_matches_canonical():
    rng = np.random.default_rng(5)
    operands = {
        "query_a": rng.normal(size=(_B, _T, _H, _K)),
        "query_b": rng.normal(size=(_B, _T, _H, _K)),
        "key_a": rng.normal(size=(_B, _T, _H, _K)),
        "key_b": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _V)),
        "beta": np.asarray(0.5),
        "lambda_weight": np.asarray(0.5),
    }
    abs_err, rel_err = _run_against_canonical("tda_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


def test_delta_transform_deltaformer_matches_canonical():
    rng = np.random.default_rng(6)
    operands = {
        "query": rng.normal(size=(_B, _T, _H, _K)),
        "key": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _V)),
        "beta": rng.uniform(0.1, 0.9, size=(_B, _T, _H)),
    }
    abs_err, rel_err = _run_against_canonical("deltaformer_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"


def test_differential_matches_canonical():
    """The already-native differential composition still matches the canonical core."""
    rng = np.random.default_rng(7)
    operands = {
        "query_a": rng.normal(size=(_B, _T, _H, _K)),
        "query_b": rng.normal(size=(_B, _T, _H, _K)),
        "key_a": rng.normal(size=(_B, _T, _H, _K)),
        "key_b": rng.normal(size=(_B, _T, _H, _K)),
        "value": rng.normal(size=(_B, _T, _H, _V)),
        "lambda_weight": rng.uniform(0.1, 0.9, size=(_H,)),
    }
    abs_err, rel_err = _run_against_canonical("differential_attention_core", operands)
    assert rel_err < RTOL, f"relative error {rel_err:.3e} (abs {abs_err:.3e})"
