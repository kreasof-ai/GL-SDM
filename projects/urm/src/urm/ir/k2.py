"""K2 semantic family: structured recurrence (linear/delta state updates).

This module is the canonical IR home for the K2 contract. For each independent
partition, state ``M`` has shape ``[K, Dv]``; with a nonnegative diagonal
transition ``G_t``, scalar write strength ``beta_t``, and correction choice
``c`` in ``{0,1}``::

    Z_t = G_t M_(t-1)
    h_t = k_t^T Z_t
    delta_t = beta_t * (v_t - c * h_t)
    M_t = Z_t + k_t delta_t^T
    y_t = scale * q_t^T M_t

``c=0`` is an additive linear update; ``c=1`` is a delta-corrected update. The
family also covers diagonal SSM layouts, factored/low-rank transitions, static
and input-conditioned decay, normalized (denominator-state) variants, and the
pinned in-place slot-table decode step. The shared, backend independent spec
lives in :mod:`urm.ir.graph`; this module owns the K2-specific semantic
contract and its validation boundary.

Ownership
---------
- Contract: `docs/kernels/linear-delta.md`
- Native implementation: `urm.backends.triton.k2.diagonal`
  (diagonal SSM) and `urm.backends.triton.k2.second_order`; the in-place
  slot-table decode step is bound through the ATMA gated-delta adapter.
- Reference/oracle: the reference matrix/diagonal recurrence executors in the
  compiler.
- Limitations: the native K2 anchor currently supports the diagonal SSM subset;
  matrix-state recurrences run through the reference or pinned library anchors.
  The HLA recurrence composition is not yet derived/validated as a reusable
  operation (see the compiler charter). Per-channel transitions and normalized
  variants are separately tested capabilities, not silently represented.
- Conformance tests: `tests/test_gated_delta_rule_adapter.py` and the K2 cases
  in `tests/test_unified_mixer.py`.
"""

from __future__ import annotations

from urm.ir.graph import MixerKernelFamily, RecurrentLayout, UnifiedMixerSpec


def is_recurrence_family(spec: UnifiedMixerSpec) -> bool:
    """Whether a unified spec belongs to the K2 structured-recurrence family."""
    return spec.family is MixerKernelFamily.RECURRENCE


def is_diagonal_layout(spec: UnifiedMixerSpec) -> bool:
    """Whether a K2 spec uses the diagonal (SSM) state layout."""
    return is_recurrence_family(spec) and (
        spec.recurrent_layout is RecurrentLayout.DIAGONAL
    )


def validate_recurrence_contract(spec: UnifiedMixerSpec) -> None:
    """Raise unless ``spec`` is a well-formed K2 contract.

    The authoritative field-level invariants are enforced by
    :meth:`UnifiedMixerSpec.__post_init__`; this boundary asserts family
    membership so callers get a K2-specific error rather than a generic one.
    """
    if not is_recurrence_family(spec):
        raise ValueError(
            f"K2 recurrence contract requires family=RECURRENCE, got {spec.family.value}"
        )


__all__ = [
    "is_diagonal_layout",
    "is_recurrence_family",
    "validate_recurrence_contract",
]
