"""Locality levels and movement constraints.

URM's locality ladder is ordered from the smallest resident scope to the
largest. A rewrite may not move an operation to a locality that cannot hold its
operands, and it may not move an operation across a locality boundary that the
rule does not explicitly justify.

    REGISTER (scalar) < LANE (pair-local) < TILE < BLOCK < DEVICE < MESH
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Locality(StrEnum):
    """Where data lives while an anchor works on it."""

    REGISTER = "register"  # scalar / register-local within one program instance
    LANE = "lane"  # pair-local across lanes of one warp/wavefront
    TILE = "tile"  # tile-resident for the lifetime of one mainloop
    BLOCK = "block"  # block/workgroup shared memory
    DEVICE = "device"  # device-local (HBM)
    MESH = "mesh"  # distributed across devices; access implies communication

    @property
    def rank(self) -> int:
        return _ORDER.index(self)

    def covers(self, other: Locality) -> bool:
        """True when this scope can hold data at ``other``'s scope."""
        return self.rank >= other.rank

    def __lt__(self, other: object) -> bool:
        return self._cmp(other) < 0

    def __le__(self, other: object) -> bool:
        return self._cmp(other) <= 0

    def __gt__(self, other: object) -> bool:
        return self._cmp(other) > 0

    def __ge__(self, other: object) -> bool:
        return self._cmp(other) >= 0

    def _cmp(self, other: object) -> int:
        if not isinstance(other, Locality):
            return NotImplemented
        return (self.rank > other.rank) - (self.rank < other.rank)


_ORDER = (
    Locality.REGISTER,
    Locality.LANE,
    Locality.TILE,
    Locality.BLOCK,
    Locality.DEVICE,
    Locality.MESH,
)


@dataclass(frozen=True, slots=True)
class LocalityConstraint:
    """Accepted window of localities for one operand or result.

    ``min`` is the smallest scope that can represent the value (e.g. a partial
    sum needs at least TILE residency); ``max`` is the largest scope at which
    the semantic still means the same thing without extra communication.
    """

    min: Locality = Locality.REGISTER
    max: Locality = Locality.MESH

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"locality window inverted: {self.min} > {self.max}")

    def accepts(self, level: Locality) -> bool:
        return self.min <= level <= self.max

    def intersect(self, other: LocalityConstraint) -> LocalityConstraint | None:
        lo = max(self.min, other.min)
        hi = min(self.max, other.max)
        if lo > hi:
            return None
        return LocalityConstraint(min=lo, max=hi)


REGISTER_LOCAL = LocalityConstraint(Locality.REGISTER, Locality.REGISTER)
TILE_LOCAL = LocalityConstraint(Locality.TILE, Locality.TILE)
DEVICE_LOCAL = LocalityConstraint(Locality.DEVICE, Locality.DEVICE)


def movement_allowed(source: Locality, target: Locality) -> bool:
    """A rewrite may keep or shrink scope freely; growing scope past DEVICE
    turns memory access into communication and is never implicit."""
    if target <= source:
        return True
    return target is Locality.DEVICE and source < Locality.DEVICE
