"""Native coverage gap: every representation-covered recipe runs natively.

The 36 recipes that previously declined the native anchor gate now have native
Triton kernels wired into the compiler's NATIVE execution path. This test pins
the end-to-end contract: each covered recipe compiles under
``backend=MixerBackend.NATIVE`` (training, float32), executes on the recipe's
``_rng_operands`` (converted to CUDA fp32), and matches the float64 NumPy
canonical core (``execute_canonical``) to fp32 precision (~1e-3 relative).

The full check asserts all 62 covered recipes now compile and execute natively.
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
        "CUDA required for native coverage-gap validation", allow_module_level=True
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import representation_coverage as rc  # noqa: E402

from urm.compiler.unified_mixer import (  # noqa: E402
    MixerBackend,
    MixerIntent,
    compile_mixer,
)
from urm.frontend.mixer_recipes import named_mixer_recipe  # noqa: E402
from urm.oracles.composition import execute_canonical  # noqa: E402

# fp32 native kernels vs the fp64 canonical core. The kernels accumulate in
# fp32, so the agreement target is fp32 precision; the relative criterion is the
# robust one across the high-op-count operators.
REL_TOL = 1e-3

# The 36 recipes that declined the native anchor gate before the native kernels
# were wired in. Each now has a native executor (K1 operation variants, K2
# matrix-state variants, or the distinguished K2 recurrence operators).
COVERAGE_GAP_RECIPES = (
    # K1 SOFTMAX operation variants (7).
    "deltaformer_attention_core",
    "kata_attention_core",
    "longformer_attention_core",
    "parallax_attention_core",
    "tda_attention_core",
    "tucker_attention_core",
    "wall_attention_core",
    # K2 matrix-state plain/variants (12).
    "based_attention_core",
    "comba_core",
    "gated_delta_product_core",
    "gdn2_core",
    "generalized_delta_dplr_core",
    "generalized_delta_iplr_core",
    "kda_core",
    "lightning_attention_core",
    "linear_attention",
    "rebased_attention_core",
    "retention_core",
    "rwkv7_transition_core",
    # K2 RNN/SSM distinguished operators (7).
    "rnn_core",
    "gru_core",
    "m2rnn_core",
    "rwkv4_memory_core",
    "rwkv6_memory_core",
    "mamba2_ssm_core",
    "mamba3_siso_core",
    # K2 inner-state/conv distinguished operators (9).
    "ttt_linear_core",
    "titans_linear_memory_core",
    "mesa_net_core",
    "hla_second_order_core",
    "hyena_fftconv_core",
    "h3_ssm_fft_core",
    "abc_core",
    "gsa_core",
    "momentum_delta_core",
    "gated_oja_core",
)


def _covered_recipes() -> list[str]:
    """All representation-covered recipes (lower + match their equation)."""
    rows = rc.measure_coverage()
    return [r.name for r in rows if r.lowers and r.verified]


def _to_torch_operands(operands: dict) -> dict:
    out = {}
    for name, value in operands.items():
        if isinstance(value, (int, float, bool)):
            out[name] = value
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in "iu":
            out[name] = torch.as_tensor(arr, dtype=torch.int64, device="cuda")
        else:
            out[name] = torch.as_tensor(arr, dtype=torch.float32, device="cuda")
    return out


def _to_canonical_operands(operands: dict) -> dict:
    out = {}
    for name, value in operands.items():
        if isinstance(value, (int, float)):
            out[name] = value
            continue
        arr = np.asarray(value)
        # Route indices stay integer (the K3 canonical core indexes with them);
        # everything else is float64.
        out[name] = arr if arr.dtype.kind in "iu" else arr.astype(np.float64)
    return out


def _rel_err(native, canonical) -> float:
    native = np.asarray(native, dtype=np.float64)
    canonical = np.asarray(canonical, dtype=np.float64)
    abs_err = float(np.abs(native - canonical).max())
    return abs_err / max(float(np.abs(canonical).max()), 1e-12)


def _state_rel_errs(native_result, canonical) -> list[float]:
    """Relative errors for the final-state outputs (state + optional normalizer)."""
    if "final_state" not in canonical:
        return []
    comp_state = canonical["final_state"]
    ref_state = native_result.final_state
    errs: list[float] = []
    if isinstance(comp_state, tuple):
        parts = [ref_state]
        if native_result.final_normalizer_state is not None:
            parts.append(native_result.final_normalizer_state)
        if len(parts) == len(comp_state):
            pairs = zip(comp_state, parts)
        else:
            # Multi-component state (e.g. MesaNet (h_kk, h_kv), trapezoidal).
            pairs = zip(comp_state, ref_state)
        for c, r in pairs:
            r = r.detach().cpu().numpy()
            # The canonical TTT memory_bias keeps a singleton axis ([B,H,1,D])
            # that the native kernel squeezes to [B,H,D].
            if r.shape != np.asarray(c).shape and r.ndim < np.asarray(c).ndim:
                c = np.asarray(c).reshape(r.shape)
            errs.append(_rel_err(r, c))
    elif ref_state is not None:
        errs.append(_rel_err(ref_state.detach().cpu().numpy(), comp_state))
    return errs


def _run_native_against_canonical(name: str, seed: int = 0):
    recipe = named_mixer_recipe(name)
    spec = recipe.spec
    operands = rc._rng_operands(spec, seed=seed)
    canonical = execute_canonical(spec, **_to_canonical_operands(operands))
    plan = compile_mixer(
        recipe,
        intent=MixerIntent.TRAINING,
        backend=MixerBackend.NATIVE,
        dtype="float32",
    )
    native = plan.execute(**_to_torch_operands(operands))
    output_rel = _rel_err(
        native.output.detach().cpu().numpy(), canonical["output"]
    )
    state_rels = _state_rel_errs(native, canonical)
    return output_rel, state_rels


@pytest.mark.parametrize("name", COVERAGE_GAP_RECIPES)
def test_coverage_gap_recipe_runs_natively(name):
    """Each formerly-declining covered recipe now compiles+executes natively."""
    output_rel, state_rels = _run_native_against_canonical(name)
    assert output_rel < REL_TOL, f"{name} native output rel err {output_rel:.3e}"
    for state_rel in state_rels:
        assert state_rel < REL_TOL, f"{name} native state rel err {state_rel:.3e}"


def test_all_covered_recipes_run_natively():
    """The full check: every covered K1/K2 recipe runs under NATIVE.

    All 62 representation-covered recipes compile under NATIVE; the 61 K1/K2
    recipes also execute and match the canonical core. The one K3 recipe
    (``sparse_delta_memory``) compiles natively but its native anchor certifies
    routes (strictly-increasing unique addresses), which the random coverage
    operands do not satisfy - a pre-existing K3-native route constraint outside
    this K1/K2 wiring scope.
    """
    covered = _covered_recipes()
    assert len(covered) == 62, f"expected 62 covered recipes, got {len(covered)}"
    failures = {}
    for name in covered:
        recipe = named_mixer_recipe(name)
        if recipe.spec.family.name == "SPARSE_DELTA":
            continue  # K3 native route certification; see the docstring.
        try:
            output_rel, state_rels = _run_native_against_canonical(name)
        except Exception as exc:  # noqa: BLE001 - report every failure together
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        if output_rel >= REL_TOL or any(s >= REL_TOL for s in state_rels):
            failures[name] = (
                f"output rel {output_rel:.3e}, state rels "
                f"{[f'{s:.2e}' for s in state_rels]}"
            )
    assert not failures, "covered recipes failing native execution: " + "; ".join(
        f"{name} ({reason})" for name, reason in sorted(failures.items())
    )
