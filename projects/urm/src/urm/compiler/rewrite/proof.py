"""Proof and obligation vocabulary for verified rewrites.

A rewrite rule is a *verified contract*: it carries an equivalence
classification with a numerical envelope, a certified backward (or an explicit
forward-only restriction), saved-state/recomputation requirements, and the
semantic obligations the executing plan must still honor. These types are the
proof side of the rewrite system; the rule contracts live in
:mod:`urm.compiler.rewrite.rules` and the deterministic engine in
:mod:`urm.compiler.rewrite.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from urm.ir.program import DType


class EquivalenceClass(StrEnum):
    """What kind of equivalence a verified rewrite promises.

    ``EXACT`` is reserved for rewrites whose supported execution model
    promises exact or bitwise-equivalent results on every supported dtype.
    Reassociations that change floating-point operation order are
    ``FLOATING_POINT`` and must carry a dtype-specific numerical envelope;
    they are algebraically justified over real arithmetic only.
    """

    EXACT = "exact"
    FLOATING_POINT = "floating_point"


class SavedStatePolicy(StrEnum):
    NONE = "none"
    SAVE_TENSORS = "save_tensors"
    RECOMPUTE = "recompute"


class ForwardOnlyRestriction(StrEnum):
    """Why a rule may be forward-only."""

    NOT_FORWARD_ONLY = "not_forward_only"
    BACKWARD_UNVERIFIED = "backward_unverified_this_prototype"


class BackwardStrategy(StrEnum):
    """How the verified backward is obtained."""

    LINEARITY = "linearity"
    TILE_RECOMPUTE = "tile_recompute"
    MATERIALIZED_AUTOGRAD = "materialized_autograd"


@dataclass(frozen=True, slots=True)
class BackwardContract:
    """A certified backward for a rewrite, per supported dtype.

    A rule with ``backward_contract=None`` is forward-only and is rejected by
    training compilations. Certification is evidence-linked: the dtypes listed
    here are exactly those exercised by committed differential tests.
    """

    strategy: BackwardStrategy
    verified_dtypes: tuple[DType, ...]
    tolerance_envelope: dict[str, float]
    evidence: str

    def covers(self, dtype: DType) -> bool:
        return dtype in self.verified_dtypes


@dataclass(frozen=True, slots=True)
class Obligation:
    """A semantic duty the executing plan must still honor."""

    kind: str
    subject_op: str
    detail: str


__all__ = [
    "BackwardContract",
    "BackwardStrategy",
    "EquivalenceClass",
    "ForwardOnlyRestriction",
    "Obligation",
    "SavedStatePolicy",
]
