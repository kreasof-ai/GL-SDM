"""Bounded schedule spaces: exhaustive enumeration and analytical ranking.

Two complementary roles:

- **Routed-epilogue problem facts** (:class:`ScheduleProblem`,
  :class:`SchedulePoint`, :func:`is_legal`): the first real schedule-selection
  problem. Every knob maps to a capability the Triton prototype implements.
- **Generic model sweeps** (:func:`legal_assignments`,
  :func:`exhaustive_optimum`): imperative brute force over a
  :class:`ConstraintModel`. These spaces are small by construction, so
  exhaustive enumeration is the legality oracle: a solver verdict is only
  trusted when it agrees with this sweep on bounded problems.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from urm.compiler.constraints import (
    Assignment,
    BoolVar,
    ConstraintModel,
    EnumVar,
    IntVar,
)

# -- Generic exhaustive reference over constraint models ------------------------


def iter_assignments(model: ConstraintModel) -> Iterator[Assignment]:
    """Deterministic product enumeration of every variable assignment.

    BoolVar -> {False, True}; IntVar -> full inclusive range; EnumVar ->
    value strings. Order is fixed by variable declaration then domain order.
    """
    domains: list[tuple[str, tuple[bool | int | str, ...]]] = []
    for variable in model.variables:
        if isinstance(variable, BoolVar):
            domains.append((variable.name, (False, True)))
        elif isinstance(variable, IntVar):
            domains.append(
                (variable.name, tuple(range(variable.lower, variable.upper + 1)))
            )
        else:
            assert isinstance(variable, EnumVar)
            domains.append((variable.name, variable.values))
    names = [name for name, _ in domains]
    for combination in itertools.product(*(values for _, values in domains)):
        yield dict(zip(names, combination, strict=True))


def legal_assignments(model: ConstraintModel) -> list[Assignment]:
    """Every assignment satisfying all named constraints, in stable order."""
    return [
        assignment
        for assignment in iter_assignments(model)
        if all(constraint.holds(assignment) for constraint in model.constraints)
    ]


def rank_lexicographic(
    model: ConstraintModel, assignments: list[Assignment]
) -> list[tuple[Assignment, tuple[int, ...]]]:
    """Rank assignments by the model's lexicographic objective terms."""
    scored: list[tuple[Assignment, tuple[int, ...]]] = []
    for assignment in assignments:
        values = []
        for term in model.objectives:
            raw = term.expression.evaluate(assignment)
            values.append(-raw if term.sense.value == "maximize" else raw)
        scored.append((assignment, tuple(values)))
    scored.sort(key=lambda pair: pair[1])
    return scored


@dataclass(frozen=True, slots=True)
class ExhaustiveResult:
    legal_count: int
    total_count: int
    best_assignment: Assignment | None
    best_objectives: tuple[int, ...] | None
    elapsed_ms: float


def exhaustive_optimum(model: ConstraintModel) -> ExhaustiveResult:
    """Reference brute-force optimum; also the legality ground truth."""
    started = time.perf_counter()
    legal = legal_assignments(model)
    ranked = rank_lexicographic(model, legal) if legal else []
    elapsed = (time.perf_counter() - started) * 1000.0
    if not ranked:
        total = sum(1 for _ in iter_assignments(model)) or 1
        return ExhaustiveResult(
            legal_count=0,
            total_count=total,
            best_assignment=None,
            best_objectives=None,
            elapsed_ms=elapsed,
        )
    total = sum(1 for _ in iter_assignments(model))
    return ExhaustiveResult(
        legal_count=len(legal),
        total_count=total,
        best_assignment=ranked[0][0],
        best_objectives=ranked[0][1],
        elapsed_ms=elapsed,
    )


# -- Routed-scale row-scale epilogue schedule space -------------------------------


class PlanKind(StrEnum):
    BASE = "base"  # trusted v1 reduction + materialized row-scale transform
    FUSED = "fused"  # experimental typed epilogue anchor


class GradValuesDecomposition(StrEnum):
    PER_QUERY = "per_query"  # one program per query; sequential routes (det)
    PER_ROUTE = "per_route"  # one program per route edge; atomic accumulation


class GradValuesSchedule(StrEnum):
    FULL_ROW = "full_row"  # single program loops the whole value dimension
    SEGMENTED = "segmented"  # grid over value-dimension segments


SUPPORTED_BLOCKS = (32, 64, 128, 256)
SUPPORTED_WARPS = (1, 2, 4, 8)
SUPPORTED_STAGES = (1, 2, 4)


@dataclass(frozen=True, slots=True)
class SchedulePoint:
    """One concrete point of the routed-epilogue schedule space."""

    plan: str
    block_d: int
    num_warps: int
    num_stages: int
    grad_values_decomposition: str
    grad_values_schedule: str
    dtype: str

    def as_dict(self) -> dict[str, str | int]:
        return {
            "plan": self.plan,
            "block_d": self.block_d,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "grad_values_decomposition": self.grad_values_decomposition,
            "grad_values_schedule": self.grad_values_schedule,
            "dtype": self.dtype,
        }

    @property
    def stable_key(self) -> str:
        return "|".join(
            f"{key}={value}" for key, value in sorted(self.as_dict().items())
        )


@dataclass(frozen=True, slots=True)
class ScheduleProblem:
    """Shape/device facts bounding the routed-epilogue schedule space."""

    queries: int = 1024
    sources: int = 512
    route_width: int = 8
    value_dim: int = 1024
    dtypes: tuple[str, ...] = ("float32", "float16", "bfloat16")
    training: bool = True
    deterministic: bool = False
    fused_anchor_available: bool = True
    fused_backward_dtypes: frozenset[str] = frozenset(
        {"float32", "float16", "bfloat16"}
    )
    base_backward_dtypes: frozenset[str] = frozenset({"float32", "float16", "bfloat16"})
    max_route_width: int = 128
    shared_mem_bytes_per_block: int = 64 * 1024


DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


def dtype_bytes(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype, 2)


def shared_memory_estimate(point: SchedulePoint, value_dim: int, dtype: str) -> int:
    """Analytical shared-memory staging estimate for one schedule point.

    The prototype stages BLOCK_D-wide tiles per pipeline stage:
    ``stages * BLOCK_D * dtype_bytes``. Register-resident accumulators are not
    shared memory and are estimated separately by resource checks.
    """
    del value_dim
    return point.num_stages * point.block_d * dtype_bytes(dtype)


def is_legal(point: SchedulePoint, problem: ScheduleProblem) -> bool:
    """Imperative legality filter - the reference implementation."""
    if point.plan not in {p.value for p in PlanKind}:
        return False
    if point.block_d not in SUPPORTED_BLOCKS or point.num_warps not in SUPPORTED_WARPS:
        return False
    if point.num_stages not in SUPPORTED_STAGES:
        return False
    # Tile/vector compatibility: BLOCK_D covers whole 32-lane warp tiles.
    if point.block_d % min(32, point.num_warps * 32) != 0:
        return False
    # Route width limits
    if problem.route_width > problem.max_route_width:
        return False
    if problem.route_width > problem.sources:
        return False
    # Shared memory bound
    if (
        shared_memory_estimate(point, problem.value_dim, point.dtype)
        > problem.shared_mem_bytes_per_block
    ):
        return False
    # Dtype support
    if point.dtype not in problem.dtypes:
        return False
    # Backward completeness per plan under training
    if problem.training:
        if point.plan == PlanKind.FUSED.value:
            if not problem.fused_anchor_available:
                return False
            if point.dtype not in problem.fused_backward_dtypes:
                return False
        elif point.dtype not in problem.base_backward_dtypes:
            return False
    # Deterministic mode: forward kernels are sequential per program and
    # deterministic, but EVERY implemented grad-value lowering (per-query,
    # per-route, v1 autograd) accumulates through relaxed cross-program
    # atomics whose float-add order is not bitwise reproducible. A
    # deterministic *training* compilation therefore has no legal schedule;
    # deterministic inference remains fully available.
    if problem.deterministic and problem.training:
        return False
    return not (problem.deterministic and not problem.fused_anchor_available)


def enumerate_space(problem: ScheduleProblem) -> list[SchedulePoint]:
    """Full cartesian product of the bounded space (deterministic order)."""
    points: list[SchedulePoint] = []
    for plan, block, warps, stages, decomp, sched, dtype in itertools.product(
        [p.value for p in PlanKind],
        SUPPORTED_BLOCKS,
        SUPPORTED_WARPS,
        SUPPORTED_STAGES,
        [d.value for d in GradValuesDecomposition],
        [s.value for s in GradValuesSchedule],
        problem.dtypes,
    ):
        points.append(
            SchedulePoint(
                plan=plan,
                block_d=block,
                num_warps=warps,
                num_stages=stages,
                grad_values_decomposition=decomp,
                grad_values_schedule=sched,
                dtype=dtype,
            )
        )
    return points


def legal_schedules(problem: ScheduleProblem) -> list[SchedulePoint]:
    """Reference enumeration of the legal subspace."""
    return [point for point in enumerate_space(problem) if is_legal(point, problem)]


def heuristic_schedule(problem: ScheduleProblem) -> SchedulePoint:
    """The current production heuristic, as a comparable schedule point.

    Mirrors ``anchors.routed_reduction_epilogue._forward_launch``: BLOCK_D is
    the next power of two of VALUE_DIM clamped to [32, 256], warps follow the
    query-count rule, and grads use per-query decomposition with segmented D.
    """
    block = min(256, max(32, 2 ** math.ceil(math.log2(max(1, problem.value_dim)))))
    plan = (
        PlanKind.FUSED.value if problem.fused_anchor_available else PlanKind.BASE.value
    )
    dtype = next(
        (d for d in ("bfloat16", "float16", "float32") if d in problem.dtypes),
        problem.dtypes[0],
    )
    return SchedulePoint(
        plan=plan,
        block_d=block,
        num_warps=2,
        num_stages=SUPPORTED_STAGES[0],
        grad_values_decomposition=GradValuesDecomposition.PER_QUERY.value,
        grad_values_schedule=GradValuesSchedule.SEGMENTED.value,
        dtype=dtype,
    )


__all__ = [
    "SUPPORTED_BLOCKS",
    "SUPPORTED_STAGES",
    "SUPPORTED_WARPS",
    "ExhaustiveResult",
    "GradValuesDecomposition",
    "GradValuesSchedule",
    "PlanKind",
    "SchedulePoint",
    "ScheduleProblem",
    "dtype_bytes",
    "enumerate_space",
    "exhaustive_optimum",
    "heuristic_schedule",
    "is_legal",
    "iter_assignments",
    "legal_assignments",
    "legal_schedules",
    "rank_lexicographic",
    "shared_memory_estimate",
]
