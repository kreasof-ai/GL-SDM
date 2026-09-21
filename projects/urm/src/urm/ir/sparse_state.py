"""K3 semantic family: ordered sparse-state operations (sparse delta memory).

This module is the canonical IR home for the K3 contract. Work is independent
per partition/head. Memory ``M`` has shape ``[S, D]``; for token ``t``, dense
vectors ``w_t, q_t`` of shape ``[S]`` encode sparse write/read weights, ``v_t``
has shape ``[D]``, and ``selected[t, s]`` is the explicit write-selection mask.
For scalar ``g_t <= 0`` the diagonal ``G_t[s,s] = exp(g_t * selected[t,s])``
decays each selected slot once, before retrieval and update::

    Z_t = G_t M_(t-1)
    h_t = w_t^T Z_t
    delta_t = beta_t (v_t - h_t)
    M_t = Z_t + w_t delta_t^T
    y_t = q_t^T M_t                       # after-update read

Write indices are unique within a token; a selected slot decays even when its
weight is zero. The shared, backend independent spec lives in
:mod:`urm.ir.mixer`; this module owns the K3-specific semantic contract and its
validation boundary.

Ownership
---------
- Contract: `docs/kernels/sparse-delta.md`
- Native implementations: `urm.backends.triton.sparse_state.memory`
  (score-to-state pipeline), `urm.backends.triton.sparse_state.route_backend`
  (route production), and `urm.backends.triton.sparse_state.backend`
  (certified route-state mixer). The executable plan binding lives in
  `urm.runtime.sparse_memory`.
- Reference/oracle: `urm.oracles.sparse_slot` (independent recurrence, chunked
  solve, and analytical reverse recurrence in float64).
- Limitations: the contract does not cover state-dependent routing or a
  nonlinear state update; before-update reads require a separately derived read
  operator. The chunked rewrite is a real-arithmetic reassociation, not bitwise
  equivalence under BF16 state rounding.
- Conformance tests: `tests/test_sparse_slot_formulation.py`,
  `tests/test_sparse_delta_memory_contract.py`,
  `tests/test_sparse_state_mixer_contract.py`,
  `tests/test_sparse_memory_native_gpu.py`, and the other
  `tests/test_sparse_*` modules.
"""

from __future__ import annotations

from urm.ir.mixer import MixerKernelFamily, UnifiedMixerSpec


def is_sparse_state_family(spec: UnifiedMixerSpec) -> bool:
    """Whether a unified spec belongs to the K3 ordered sparse-state family."""
    return spec.family is MixerKernelFamily.SPARSE_DELTA


def validate_sparse_state_contract(spec: UnifiedMixerSpec) -> None:
    """Raise unless ``spec`` is a well-formed K3 contract.

    The authoritative field-level invariants are enforced by
    :meth:`UnifiedMixerSpec.__post_init__`; this boundary asserts family
    membership so callers get a K3-specific error rather than a generic one.
    """
    if not is_sparse_state_family(spec):
        raise ValueError(
            f"K3 sparse-state contract requires family=SPARSE_DELTA, got {spec.family.value}"
        )


__all__ = [
    "is_sparse_state_family",
    "validate_sparse_state_contract",
]
