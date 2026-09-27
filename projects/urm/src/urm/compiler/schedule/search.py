"""Compiler-owned bounded schedule search: solve -> verify -> retry.

This module closes the loop between candidate selection and kernel lowering:

    selected rewrite candidate
      -> CANDIDATE-BOUND schedule model (kernel_plan.build_schedule_model)
      -> Z3 optimization (optional ``solver`` extra) OR deterministic
         heuristic fallback lifted to a complete assignment
      -> independent imperative verification (the SAME verifier both ways)
      -> bounded nogood/retry within ``SolverLimits.max_nogoods``
      -> a serializable :class:`ScheduleDecision` consumed by lowering

Nothing here imports Torch/Triton or Z3 at module import time; solving
imports lazily, so CPU-only installs work. Serialized decisions carry plain
data - never solver objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from urm.compiler.solve.constraints import Assignment, ConstraintModel
from urm.compiler.common.diagnostics import CompilerError, Diagnostic, DiagnosticCode
from urm.compiler.schedule.space import SchedulePoint


@dataclass(frozen=True, slots=True)
class ScheduleAttempt:
    """One bounded-search attempt: what was tried and what happened."""

    index: int
    selection_policy: str
    schedule: dict[str, str | int]
    verified: bool
    rejection_reasons: tuple[str, ...] = ()
    nogood_added: bool = False
    nogood_budget_exhausted: bool = False
    nogood_forbidden: dict[str, bool | int | str] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "selection_policy": self.selection_policy,
            "schedule": dict(self.schedule),
            "verified": self.verified,
            "rejection_reasons": list(self.rejection_reasons),
            "nogood_added": self.nogood_added,
            "nogood_budget_exhausted": self.nogood_budget_exhausted,
            "nogood_forbidden": (
                dict(self.nogood_forbidden) if self.nogood_forbidden else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    """The verified schedule the executable plan will actually run."""

    schedule_point: SchedulePoint
    launch_config: dict[str, str | int]
    selection_policy: str
    model_hash: str
    objective_values: tuple[int, ...] = ()
    solver_statistics: dict[str, float | int | str] = field(default_factory=dict)
    verification_checks_run: tuple[str, ...] = ()
    attempts: tuple[ScheduleAttempt, ...] = ()
    fallback_used: bool = False
    rejected_assignments: int = 0
    nogoods_added: int = 0
    recoveries: int = 0
    retry_budget_exhausted: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schedule": self.schedule_point.as_dict(),
            "launch_config": dict(self.launch_config),
            "selection_policy": self.selection_policy,
            "model_hash": self.model_hash,
            "objective_values": list(self.objective_values),
            "solver_statistics": dict(self.solver_statistics),
            "verification_checks_run": list(self.verification_checks_run),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "fallback_used": self.fallback_used,
            "rejected_assignments": self.rejected_assignments,
            "nogoods_added": self.nogoods_added,
            "recoveries": self.recoveries,
            "retry_budget_exhausted": self.retry_budget_exhausted,
        }


Verifier = Callable[[ConstraintModel, Assignment], object]


def launch_config_of(point: SchedulePoint) -> dict[str, str | int]:
    """Concrete decoded launch configuration handed to the real launcher."""
    return {
        "plan": point.plan,
        "block_d": point.block_d,
        "num_warps": point.num_warps,
        "num_stages": point.num_stages,
        "grad_values_decomposition": point.grad_values_decomposition,
        "grad_values_schedule": point.grad_values_schedule,
        "dtype": point.dtype,
    }


def nogood_count(model: ConstraintModel) -> int:
    from urm.compiler.solve.constraints import Nogood

    return sum(1 for constraint in model.constraints if isinstance(constraint, Nogood))


class CompilationSearch:
    """Bounded solve/verify/retry loop over one candidate-bound model.

    Every unverified assignment is rejected BEFORE any lowering can see it;
    verification failures add an exact nogood derived from the failed
    attempt's own assignment (never a stale optimized one) and the search
    re-solves. Retries stop at ``max_nogoods`` and surface a structured
    :class:`CompilerError` instead of looping forever.
    """

    def __init__(
        self,
        *,
        model: ConstraintModel,
        problem_hint: Mapping[str, object] | None = None,
        max_nogoods: int,
        verifier: Verifier | None = None,
    ) -> None:
        del problem_hint
        self.model = model
        self.max_nogoods = max_nogoods
        self._verifier = verifier


    def _next_assignment(
        self,
    ) -> tuple[Assignment, tuple[int, ...], dict[str, float | int | str], str]:
        """Best next assignment: Z3 optimum when available, else heuristic."""
        from urm.compiler.solve.z3 import (
            FeasibilityStatus,
            OptimizationPass,
            SolverLimits,
            z3_available,
        )

        if z3_available():
            result = OptimizationPass(SolverLimits(max_nogoods=self.max_nogoods)).run(
                self.model
            )
            if result.status is FeasibilityStatus.UNSAT:
                raise _no_schedule_error(result.diagnostics)
            if result.status is FeasibilityStatus.SAT and result.assignment is not None:
                return (
                    result.assignment,
                    result.objective_values,
                    dict(result.statistics),
                    "solver_guided",
                )

        from urm.compiler.select.model import exhaustive_schedule_sweep

        _legal, ranked, _total = exhaustive_schedule_sweep(self.model)
        stats: dict[str, float | int | str] = {
            "fallback_policy": "cost_heuristic",
            "legal_points": len(ranked),
        }
        if not ranked:
            raise _no_schedule_error(())
        best_assignment, objectives = ranked[0]
        return best_assignment, tuple(objectives), stats, "cost_heuristic"

    def _add_nogood(
        self,
        assignment: Assignment,
        *,
        explanation: str,
        origin_kind: str,
    ) -> tuple[str | None, dict[str, bool | int | str] | None]:
        """Exact nogood excluding THIS assignment; (None, _) = budget gone."""
        existing = nogood_count(self.model)
        if existing >= self.max_nogoods:
            return None, None
        from urm.compiler.solve.constraints import make_nogood

        forbidden = {name: assignment[name] for name in sorted(assignment)}
        name = f"nogood_{origin_kind}_{existing + 1}"
        self.model.add_constraint(
            make_nogood(
                name=name,
                explanation=explanation[:300],
                origin_kind=origin_kind,
                origin_id=self.model.metadata.get("candidate_id", "unknown"),
                forbidden=forbidden,
            )
        )
        return name, forbidden

    def _verify(self, assignment: Assignment):
        if self._verifier is not None:
            return self._verifier(self.model, assignment)
        from urm.compiler.select.model import verify_schedule_assignment

        return verify_schedule_assignment(self.model, assignment)


    def run(self) -> ScheduleDecision:
        from urm.compiler.select.model import decode_schedule_point

        attempts: list[ScheduleAttempt] = []
        rejected = 0
        budget_exhausted = False

        while True:
            assignment, objective_values, solver_stats, policy = self._next_assignment()
            report = self._verify(assignment)
            checks_run = tuple(getattr(report, "checks_run", ()) or ())
            failures = tuple(getattr(report, "failures", ()) or ())
            point = decode_schedule_point(self.model, assignment)
            index = len(attempts)

            if not report.ok:
                rejected += 1
                reasons = tuple(f.message for f in failures[:4])
                added, forbidden = self._add_nogood(
                    assignment,
                    explanation=(
                        "rejected by independent verification: "
                        + "; ".join(reasons[:2])
                    ),
                    origin_kind="verification_failure",
                )
                attempts.append(
                    ScheduleAttempt(
                        index=index,
                        selection_policy=policy,
                        schedule=point.as_dict(),
                        verified=False,
                        rejection_reasons=reasons,
                        nogood_added=added is not None,
                        nogood_budget_exhausted=added is None,
                        nogood_forbidden=forbidden,
                    )
                )
                if added is None:
                    budget_exhausted = True
                    break
                continue

            attempts.append(
                ScheduleAttempt(
                    index=index,
                    selection_policy=policy,
                    schedule=point.as_dict(),
                    verified=True,
                )
            )
            return ScheduleDecision(
                schedule_point=point,
                launch_config=launch_config_of(point),
                selection_policy=policy,
                model_hash=self.model.summary_hash(),
                objective_values=tuple(int(v) for v in objective_values),
                solver_statistics=solver_stats,
                verification_checks_run=checks_run,
                attempts=tuple(attempts),
                fallback_used=policy == "cost_heuristic",
                rejected_assignments=rejected,
                nogoods_added=nogood_count(self.model),
                retry_budget_exhausted=budget_exhausted,
            )

        last = attempts[-1]
        raise CompilerError(
            (
                Diagnostic(
                    code=DiagnosticCode.NOGOOD_RETRY_LIMIT,
                    message=(
                        f"bounded schedule search exhausted its retry budget "
                        f"after {len(attempts)} attempts; last attempt "
                        f"{last.index} ({last.selection_policy}) chose "
                        f"{last.schedule}"
                    ),
                ),
            )
        )


def _no_schedule_error(diagnostics) -> CompilerError:
    items = [
        Diagnostic(
            code=DiagnosticCode.UNSAT_CONSTRAINTS,
            message=("the candidate-bound schedule model has no legal schedule"),
        ),
        *tuple(diagnostics)[:4],
    ]
    return CompilerError(tuple(items))


__all__ = [
    "CompilationSearch",
    "ScheduleAttempt",
    "ScheduleDecision",
    "launch_config_of",
    "nogood_count",
]
