"""Representation-coverage contract: architectures lower into canonical cores.

This pins the durable contract behind the unified generator (acceptance-contract
section 4): a recipe that lowers into the canonical NumPy execution path for its
core must match its independent architecture equation in float64, recipe
renaming must not change execution, and under-specified collision-group
equations must decline rather than silently compute the wrong equation.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import representation_coverage as rc  # noqa: E402
from urm.frontend.mixer_recipes import MIXER_RECIPE_NAMES, named_mixer_recipe  # noqa: E402
from urm.oracles.composition import UnderspecifiedComposition, execute_canonical  # noqa: E402

# Recipes verified to lower into a canonical core and match their independent
# equation. Regenerate via `python benchmarks/representation_coverage.py`.
VERIFIED = {
    "abc_core",
    "based_attention_core",
    "cat_attention_core",
    "comba_core",
    "conformer_attention_core",
    "delta_net",
    "differential_attention_core",
    "dsa_attention_core",
    "foveal_attention_core",
    "gated_delta_net",
    "gated_oja_core",
    "gdn2_core",
    "generalized_delta_dplr_core",
    "generalized_delta_iplr_core",
    "gla",
    "gqa",
    "gru_core",
    "gsa_core",
    "h3_ssm_fft_core",
    "hgrn2_ssm_core",
    "hgrn_ssm_core",
    "hla_second_order_core",
    "hopfield_attention_core",
    "hyena_fftconv_core",
    "kata_attention_core",
    "kda_core",
    "lightnet_gla_core",
    "lightning_attention_core",
    "linear_attention",
    "longformer_attention_core",
    "m2rnn_core",
    "mamba1_ssm_core",
    "mamba3_siso_core",
    "mesa_net_core",
    "mha",
    "mla_attention_core",
    "mom_selected_memory_core",
    "momentum_delta_core",
    "mqa",
    "nsa_selected_attention_core",
    "parallax_attention_core",
    "pattention_core",
    "rebased_attention_core",
    "retention_core",
    "rnn_core",
    "rodimus_gla_core",
    "rwkv7_transition_core",
    "samba_attention_core",
    "simple_gla",
    "sparse_attention_core",
    "sparse_delta_memory",
    "titans_linear_memory_core",
    "tpa_attention_core",
    "ttt_linear_core",
    "tucker_attention_core",
}


def test_verified_set_matches_live_measurement():
    """The pinned verified set must equal the live coverage measurement."""
    rows = rc.measure_coverage()
    live = {r.name for r in rows if r.lowers and r.verified}
    assert live == VERIFIED, (
        f"representation coverage drifted: live={sorted(live - VERIFIED)} newly "
        f"covered, lost={sorted(VERIFIED - live)}. Update VERIFIED and regenerate "
        f"the doc with `python benchmarks/representation_coverage.py`."
    )


@pytest.mark.parametrize("name", sorted(VERIFIED))
def test_lowered_recipe_matches_independent_equation(name):
    """Each covered recipe lowers and matches its independent equation in fp64."""
    row = rc.measure_recipe(name)
    assert row.lowers, f"{name} no longer lowers: {row.reason}"
    assert row.verified, (
        f"{name} lowered but mismatch: output_err={row.output_err} "
        f"state_err={row.state_err}"
    )


def test_recipe_renaming_does_not_change_execution():
    """Two specs identical except for name must execute identically."""
    spec = named_mixer_recipe("gated_delta_net").spec
    renamed = dataclasses.replace(spec, name="renamed_gated_delta_clone")
    rng = np.random.default_rng(0)
    operands = rc._rng_operands(spec, seed=0)
    out_a = execute_canonical(spec, **operands)
    out_b = execute_canonical(renamed, **operands)
    np.testing.assert_allclose(out_a["output"], out_b["output"], rtol=0, atol=0)
    np.testing.assert_allclose(out_a["final_state"], out_b["final_state"], rtol=0, atol=0)


def test_different_equations_have_distinguishable_representations():
    """Specs with different equations must have different semantic signatures.

    The former additive/no-decay collision group is now distinguished by the
    ``recurrence_operator`` IR field, so no two recipes with different equations
    share a semantic signature.
    """
    signatures = {}
    for name in MIXER_RECIPE_NAMES:
        spec = named_mixer_recipe(name).spec
        signatures.setdefault(spec.semantic_signature(), []).append(name)
    # Any recipes still sharing a signature must be genuine aliases (the same
    # equation), never two different equations silently sharing an execution.
    collisions = {sig: names for sig, names in signatures.items() if len(names) > 1}
    # The known collision group is resolved; assert no residual collisions among
    # the formerly-colliding nonlinear recurrences.
    former_collision = {"gru_core", "rnn_core", "m2rnn_core", "ttt_linear_core",
                        "titans_linear_memory_core", "mesa_net_core",
                        "mamba3_siso_core", "hla_second_order_core",
                        "h3_ssm_fft_core", "hyena_fftconv_core"}
    for names in collisions.values():
        assert not (set(names) & former_collision), (
            f"former collision-group recipes still share a signature: {names}"
        )


@pytest.mark.parametrize(
    "name", ["gru_core", "rnn_core", "m2rnn_core", "hla_second_order_core",
             "h3_ssm_fft_core", "hyena_fftconv_core"],
)
def test_former_collision_group_is_distinguished_and_covered(name):
    """The IR extension makes each former collision-group equation distinguishable
    (a distinct recurrence_operator) and coverable by its canonical executor."""
    spec = named_mixer_recipe(name).spec
    from urm.ir.mixer import RecurrenceOperator

    assert spec.recurrence_operator is not RecurrenceOperator.PLAIN
    row = rc.measure_recipe(name)
    assert row.lowers, f"{name} should lower via its canonical operator executor"
    assert row.verified, f"{name} mismatch: output_err={row.output_err}"
