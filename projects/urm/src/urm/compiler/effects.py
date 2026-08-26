"""Explicit effect system for semantic operations.

State mutation, collision policy, ordering, and version/commit behavior are
effects in URM, not hidden implementation details. A rewrite may move an
operation only across effect boundaries the rule explicitly preserves; the
rewrite engine enforces this mechanically.

Effect algebra used by the engine:

- PURE composes with everything.
- REDUCTION is associative reassociation only (rule-declared tolerance).
- ORDERED_STATE_TRANSITION and TRANSACTIONAL_COMMIT are barriers: nothing may
  move across them unless a registered rule proves the movement safe.
- ATOMIC_ACCUMULATION allows intra-op reassociation under a declared envelope.
- COLLECTIVE_COMMUNICATION is a first-class effect: remote exchange is never
  disguised as an ordinary local tensor operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, StrEnum


class EffectKind(Flag):
    """Bit set of effects an operation reads or writes."""

    PURE = 0
    READS_STATE = 1 << 0
    WRITES_STATE = 1 << 1
    ORDERED = 1 << 2
    ATOMIC = 1 << 3
    COLLECTIVE = 1 << 4
    COMMITS = 1 << 5

    def describes(self, *kinds: EffectKind) -> bool:
        return all((self & kind) == kind for kind in kinds)


class EffectClass(StrEnum):
    """Named effect classes used in op declarations and rewrite contracts."""

    PURE = "pure"
    REDUCTION = "reduction"
    STATE_ACCESS = "state_access"
    ORDERED_STATE_TRANSITION = "ordered_state_transition"
    ATOMIC_ACCUMULATION = "atomic_accumulation"
    BUFFERED_MUTATION = "buffered_mutation"
    COLLECTIVE_COMMUNICATION = "collective_communication"
    TRANSACTIONAL_COMMIT = "transactional_commit"


_CLASS_FLAGS: dict[EffectClass, EffectKind] = {
    EffectClass.PURE: EffectKind.PURE,
    EffectClass.REDUCTION: EffectKind.ATOMIC,
    EffectClass.STATE_ACCESS: EffectKind.READS_STATE,
    EffectClass.ORDERED_STATE_TRANSITION: (
        EffectKind.READS_STATE | EffectKind.WRITES_STATE | EffectKind.ORDERED
    ),
    EffectClass.ATOMIC_ACCUMULATION: (EffectKind.WRITES_STATE | EffectKind.ATOMIC),
    EffectClass.BUFFERED_MUTATION: EffectKind.WRITES_STATE,
    EffectClass.COLLECTIVE_COMMUNICATION: EffectKind.COLLECTIVE,
    EffectClass.TRANSACTIONAL_COMMIT: (EffectKind.WRITES_STATE | EffectKind.COMMITS),
}

# Effects that act as movement barriers for rules that do not explicitly
# preserve them.
BARRIERS: frozenset[EffectClass] = frozenset(
    {
        EffectClass.ORDERED_STATE_TRANSITION,
        EffectClass.TRANSACTIONAL_COMMIT,
        EffectClass.BUFFERED_MUTATION,
        EffectClass.COLLECTIVE_COMMUNICATION,
    }
)


@dataclass(frozen=True, slots=True)
class EffectSignature:
    """What an operation does to the world while it runs."""

    reads: frozenset[EffectClass] = frozenset()
    writes: frozenset[EffectClass] = frozenset()

    @property
    def all_classes(self) -> frozenset[EffectClass]:
        return self.reads | self.writes

    @property
    def is_pure(self) -> bool:
        return not self.all_classes - {EffectClass.PURE}

    def flags(self) -> EffectKind:
        result = EffectKind.PURE
        for cls in self.all_classes:
            result |= _CLASS_FLAGS[cls]
        return result


PURE = EffectSignature()

REDUCING = EffectSignature(writes=frozenset({EffectClass.REDUCTION}))

STATE_READ_EFFECT = EffectSignature(reads=frozenset({EffectClass.STATE_ACCESS}))

ATOMIC_ACCUMULATE = EffectSignature(writes=frozenset({EffectClass.ATOMIC_ACCUMULATION}))

ORDERED_STATE = EffectSignature(
    reads=frozenset({EffectClass.ORDERED_STATE_TRANSITION}),
    writes=frozenset({EffectClass.ORDERED_STATE_TRANSITION}),
)

BUFFERED_WRITE = EffectSignature(writes=frozenset({EffectClass.BUFFERED_MUTATION}))

COMMIT = EffectSignature(writes=frozenset({EffectClass.TRANSACTIONAL_COMMIT}))

COLLECTIVE = EffectSignature(reads=frozenset({EffectClass.COLLECTIVE_COMMUNICATION}))


def crossing_effects(moved_over: EffectSignature) -> set[EffectClass]:
    """Barrier classes an operation would cross when moved over ``moved_over``."""
    return set(moved_over.all_classes) & set(BARRIERS)
