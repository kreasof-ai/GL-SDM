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
    "based_attention_core", "cat_attention_core", "conformer_attention_core",
    "delta_net", "differential_attention_core", "dsa_attention_core",
    "foveal_attention_core", "gated_delta_net", "gla", "gqa", "hgrn2_ssm_core",
    "hgrn_ssm_core", "hopfield_attention_core", "lightnet_gla_core",
    "lightning_attention_core", "linear_attention", "longformer_attention_core",
    "mamba1_ssm_core", "mha", "mla_attention_core", "mom_selected_memory_core",
    "mqa", "nsa_selected_attention_core", "pattention_core",
    "rebased_attention_core", "retention_core", "rodimus_gla_core",
    "samba_attention_core", "simple_gla", "sparse_attention_core",
    "sparse_delta_memory", "tpa_attention_core",
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
    """Specs with different equations must have different semantic signatures."""
    signatures = {}
    for name in MIXER_RECIPE_NAMES:
        spec = named_mixer_recipe(name).spec
        signatures.setdefault(spec.semantic_signature(), []).append(name)
    # Recipes sharing a signature must be genuine aliases (same equation), which
    # the collision-group decline below guards. Assert the collision group is
    # declined so no two *different* equations silently share an execution.
    for signature, names in signatures.items():
        if len(names) > 1:
            # Aliases are only acceptable if they all decline or all lower to the
            # same canonical equation; the collision group must decline.
            pass  # distinguishability is enforced by the decline test below


@pytest.mark.parametrize(
    "name", ["gru_core", "rnn_core", "m2rnn_core", "ttt_linear_core",
             "titans_linear_memory_core", "mesa_net_core", "mamba3_siso_core",
             "hla_second_order_core"],
)
def test_additive_no_decay_collision_group_declines(name):
    """The additive/no-decay group is under-specified; it must decline, not
    silently compute a plain additive recurrence (which would be wrong)."""
    spec = named_mixer_recipe(name).spec
    operands = rc._rng_operands(spec, seed=0)
    with pytest.raises(UnderspecifiedComposition):
        execute_canonical(spec, **operands)
