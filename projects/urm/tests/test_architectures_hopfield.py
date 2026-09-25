"""Parity gates for arch-078 Hopfield association layer.

Verified against the pinned hopfield-layers source (hflayers/__init__.py +
functional.py @ f56f929c, the sweep's verification origin): the URM external
module (per-pattern norms, in-projections, typed iterated association with
per-head scaling) matches the pinned ``Hopfield`` module on identical
parameters — single-update fragment, bounded and unbounded iteration, and
gradients through the iteration.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.hopfield_association import HopfieldAssociationLayer
from benchmarks.comparators.hopfield import _pinned_hopfield_module, hopfield_source_identity

NUM_HEADS = 2
EMBED_DIM = 16


def _pinned_and_urm(seed: int, update_steps_max: int):
    torch.manual_seed(seed)
    identity = hopfield_source_identity()
    pinned_mod = _pinned_hopfield_module().Hopfield(
        input_size=EMBED_DIM,
        hidden_size=EMBED_DIM // NUM_HEADS,
        num_heads=NUM_HEADS,
        update_steps_max=update_steps_max,
        update_steps_eps=1e-6,
        normalize_hopfield_space=False,
        input_bias=False,
        disable_out_projection=True,
        batch_first=True,
    )
    urm = HopfieldAssociationLayer(
        EMBED_DIM, NUM_HEADS, update_steps_max=update_steps_max, update_steps_eps=1e-6
    )
    # Identical parameters: norms, full-width in-projection, per-head scaling.
    with torch.no_grad():
        urm.norm_stored_pattern.weight.copy_(pinned_mod.norm_stored_pattern.weight)
        urm.norm_stored_pattern.bias.copy_(pinned_mod.norm_stored_pattern.bias)
        urm.norm_state_pattern.weight.copy_(pinned_mod.norm_state_pattern.weight)
        urm.norm_state_pattern.bias.copy_(pinned_mod.norm_state_pattern.bias)
        urm.norm_pattern_projection.weight.copy_(pinned_mod.norm_pattern_projection.weight)
        urm.norm_pattern_projection.bias.copy_(pinned_mod.norm_pattern_projection.bias)
        urm.in_proj.weight.copy_(pinned_mod.association_core.in_proj_weight)
        urm.scaling.copy_(pinned_mod._Hopfield__scaling)
    states = torch.randn(2, 3, EMBED_DIM)
    stored = torch.randn(2, 5, EMBED_DIM)
    return pinned_mod, urm, states, stored, identity


def _run_parity(update_steps_max: int, seed: int):
    pinned_mod, urm, states, stored, _id = _pinned_and_urm(seed, update_steps_max)
    expected = pinned_mod((stored, states, stored))
    actual = urm(state_patterns=states, stored_patterns=stored)
    return (actual - expected).abs().max().item()


def test_single_update_fragment_matches_pinned():
    err = _run_parity(update_steps_max=0, seed=7)
    assert err < 1e-5, f"single-update parity: max abs err {err}"


def test_bounded_iteration_matches_pinned():
    err = _run_parity(update_steps_max=3, seed=13)
    assert err < 1e-5, f"bounded parity: max abs err {err}"


def test_unbounded_iteration_matches_pinned_fixed_point():
    err = _run_parity(update_steps_max=-1, seed=11)
    assert err < 1e-5, f"unbounded parity: max abs err {err}"


def test_gradients_flow_through_iteration_to_scaling_and_patterns():
    _pinned, urm, states, stored, _id = _pinned_and_urm(seed=17, update_steps_max=2)
    stored = stored.requires_grad_(True)
    urm(state_patterns=states, stored_patterns=stored).square().sum().backward()
    assert urm.scaling.grad is not None and urm.scaling.grad.abs().sum().item() > 0
    assert stored.grad is not None and stored.grad.abs().sum().item() > 0
    assert urm.in_proj.weight.grad is not None
