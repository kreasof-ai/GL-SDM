"""Canonical per-family kernel signatures shared by every backend tier.

Each family has ONE canonical function (same name, same role order, same
batched shapes, same closed descriptor, same return) implemented identically
by the NumPy oracle, the Torch reference and the native Triton schedule. There
is no layout conversion and no per-tier signature drift in the canonical
functions: a backend is interchangeable because it implements the same
function, not because an adapter reshapes its operands.

The canonical functions live in each family package:

- K1 ``k1_softmax_attention(query, key, value, *, descriptor, score_bias,
  attention_mask, scale)`` — ``providers/k1/{numpy,torch,triton}.py``.
  Operands ``[B, T, H, D]``; output ``[B, T, H, Dv]``.
- K2 ``linear_delta_state(initial_state, keys, queries, values, beta,
  log_decay, *, spec, scale)`` — ``providers/k2/{numpy,torch}.py`` and the
  native wrapper in ``providers/k2/triton_matrix.py``. ``initial_state``
  ``[B,H,K,V]``; keys/queries ``[B,H,T,K]``; values ``[B,H,T,V]``; returns
  ``(output [B,H,T,V], final_state [B,H,K,V])``.
- K3 ``sparse_delta_state(memory, read_addresses, read_weights, *,
  write_addresses, write_weights, values, beta, log_decay, spec)`` —
  ``providers/k3/{numpy,torch}.py``. Address-index operand form shared across
  tiers. Memory ``[B,S,D]``; addresses/weights ``[B,T,W]``; values ``[B,T,D]``;
  beta/log_decay ``[B,T,1]``; returns ``(readings [B,T,D], updated_memory)``.

The role orders below make the canonical operand lists importable. Legacy
keyword aliases (e.g. ``torch_sparse_state_mixer``) are kept as thin
back-compat shims over the canonical functions; new code uses the canonical
names only.
"""

from __future__ import annotations

# Canonical role order for each family's kernel function.
K1_ROLES = ("query", "key", "value")
K1_OPTIONAL_ROLES = ("score_bias", "attention_mask", "scale")

K2_ROLES = ("initial_state", "keys", "queries", "values", "beta", "log_decay")
K2_OPTIONAL_ROLES = ("scale",)

K3_ROLES = (
    "memory",
    "read_addresses",
    "read_weights",
    "write_addresses",
    "write_weights",
    "values",
    "beta",
    "log_decay",
)

__all__ = [
    "K1_OPTIONAL_ROLES",
    "K1_ROLES",
    "K2_OPTIONAL_ROLES",
    "K2_ROLES",
    "K3_ROLES",
]
