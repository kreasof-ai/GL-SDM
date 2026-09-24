"""K1 semantic family: normalized routed reduction (softmax attention and kin).

This module is the canonical IR home for the K1 contract. K1 computes, for each
query row over its visible keys::

    scores = scale * Q K^T + bias
    P = softmax(scores)
    Y = P V

The family also covers the typed K1 variants expressed through
:class:`~urm.ir.mixer.K1Operation` (differential, thresholded, projected,
local-window, positive-feature, selected-read, block-routed, positional, gated,
depth, path/delta transforms, and the polar family). The shared, backend
independent spec lives in :mod:`urm.ir.mixer`; this module owns the K1-specific
semantic contract and its validation boundary.

Ownership
---------
- Contract: `docs/kernels/softmax-attention.md`
- Native implementation: `urm.backends.triton.softmax.online` (tiled online
  softmax with recomputed backward) behind
  `urm.backends.triton.softmax.online_backend.TritonOnlineSoftmaxBackend`;
  fused row-scale routed reduction in
  `urm.backends.triton.softmax.routed_scale_epilogue`; plain routed reduction in
  `urm.backends.triton.softmax.routed_reduce`.
- Reference/oracle: the reference executor in the compiler and the NumPy oracle
  `urm.oracles.routed`.
- Limitations: native K1 requires CUDA and BTHD rank-4 layout, key/value widths
  up to 128, and float32 additive-mask gradients. Cache ownership, decode
  positions, dropout, sparse traversal efficiency, larger dimensions, and
  end-to-end layer training/inference are unqualified (see
  `docs/planning/coverage.md`).
- Conformance tests: `tests/test_native_softmax_gpu.py`,
  `tests/test_dense_attention_adapter.py`, and the K1 cases in
  `tests/test_unified_mixer.py`.
"""

from __future__ import annotations

from urm.ir.mixer import K1Operation, MixerKernelFamily, UnifiedMixerSpec

K1_OPERATIONS: frozenset[K1Operation] = frozenset(K1Operation)


def is_softmax_family(spec: UnifiedMixerSpec) -> bool:
    """Whether a unified spec belongs to the K1 normalized-routed-reduction family."""
    return spec.family is MixerKernelFamily.SOFTMAX


def validate_softmax_contract(spec: UnifiedMixerSpec) -> None:
    """Raise unless ``spec`` is a well-formed K1 contract.

    The authoritative field-level invariants are enforced by
    :meth:`UnifiedMixerSpec.__post_init__`; this boundary asserts family
    membership so callers get a K1-specific error rather than a generic one.
    """
    if not is_softmax_family(spec):
        raise ValueError(
            f"K1 softmax contract requires family=SOFTMAX, got {spec.family.value}"
        )


__all__ = [
    "K1_OPERATIONS",
    "is_softmax_family",
    "validate_softmax_contract",
]
