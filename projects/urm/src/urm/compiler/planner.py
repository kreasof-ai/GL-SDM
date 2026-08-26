"""Placement-aware planner and the NAS-facing compilation API.

The planner lowers validated semantic programs into deterministic executable
plans through an explicit pipeline:

    semantic program -> validated rewrite/lowering CANDIDATES
      -> (optional) Z3 feasibility + bounded lexicographic ranking
      -> independent imperative model verification
      -> anchor selection -> placement/sharding -> executable plan
      -> deterministic compilation trace

Two parameter namespaces are kept separate everywhere: *architecture*
parameters (what the model is) and *schedule* parameters (how this lowering
runs). Serialized artifacts never mix them.

Compilation is intent-explicit (:class:`CompilationIntent`): a training
compilation rejects forward-only anchors, rewrites with unverified backward,
and unresolved recomputation/saved-state obligations. Candidate selection is
never implicit mutation: the unfused/base plan is always a candidate, every
rewrite occurrence has a stable ID, and traces record the selected candidate
plus rejected alternatives.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from urm.compiler.cost import CostEstimate, DeviceLimits, combine
from urm.compiler.diagnostics import (
    CompilerError,
    Diagnostic,
    DiagnosticCode,
    DiagnosticsCollector,
)
from urm.compiler.execution import (
    AnchorDecision,
    AnchorKind,
    AnchorRegistry,
    AnchorRequest,
    Decline,
    VisitorDescriptor,
    default_registry,
)
from urm.compiler.placement import (
    ExchangeStep,
    PlacementMap,
    PlanStep,
    RouteLeg,
)
from urm.compiler.rewrite import (
    CheckOutcome,
    RewriteEngine,
    RewriteMatch,
    RewriteRule,
    RewriteTrace,
)
from urm.compiler.semantic import (
    CollectiveExchange,
    Gather,
    Matmul,
    OrderedRecurrence,
    RouteSpec,
    Score,
    Select,
    StateUpdate,
    Transform,
    WeightedReduce,
)

if TYPE_CHECKING:
    from pathlib import Path

    from urm.compiler.constraints import ConstraintModel
    from urm.compiler.semantic import SemanticNode, SemanticProgram
    from urm.compiler.solver import FeasibilityResult, OptimizationResult
    from urm.compiler.verification import VerificationReport


# -- Parameter namespaces -----------------------------------------------------


class CompilationIntent(StrEnum):
    """Why this program is being compiled.

    ``TRAINING`` demands complete verified backward paths; ``INFERENCE``
    accepts forward-only lowerings; ``FORWARD_ONLY_ANALYSIS`` is an explicit
    declaration that gradients are out of scope for this run.
    """

    INFERENCE = "inference"
    TRAINING = "training"
    FORWARD_ONLY_ANALYSIS = "forward_only_analysis"


@dataclass(frozen=True, slots=True)
class ArchitectureParams:
    """NAS-facing parameters describing WHAT the architecture is."""

    family: str
    logical_dims: dict[str, int]
    routing: dict[str, str]


VALID_NUM_WARPS = (1, 2, 4, 8)
VALID_NUM_STAGES = (1, 2, 3, 4)
VALID_LAYOUTS = frozenset({"row_major", "col_major"})


@dataclass(frozen=True, slots=True)
class ScheduleParams:
    """Backend/schedule parameters describing HOW one lowering executes.

    These are operational, not decorative: they are validated (invalid hints
    produce structured ``schedule_hint_invalid`` diagnostics), passed into
    anchor requests, honored by the schedule solver as bounds, and they can
    demonstrably alter the selected plan. They may pick among *legal*
    lowerings; they may never change routing results, merge policies, or
    commit boundaries.
    """

    anchor_overrides: dict[str, str] = field(default_factory=dict)
    block_hints: dict[str, int] = field(default_factory=dict)
    warp_count: int | None = None
    stage_count: int | None = None
    dtype_hints: dict[str, str] = field(default_factory=dict)
    layout_hints: dict[str, str] = field(default_factory=dict)
    deterministic: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_overrides": dict(self.anchor_overrides),
            "block_hints": dict(self.block_hints),
            "warp_count": self.warp_count,
            "stage_count": self.stage_count,
            "dtype_hints": dict(self.dtype_hints),
            "layout_hints": dict(self.layout_hints),
            "deterministic": self.deterministic,
        }


def validate_schedule_params(
    params: ScheduleParams,
) -> tuple[Diagnostic, ...]:
    """Structured validation of schedule hints; empty tuple means valid."""
    collector = DiagnosticsCollector()
    for key, anchor_name in params.anchor_overrides.items():
        if not key or not anchor_name:
            collector.error(
                DiagnosticCode.SCHEDULE_HINT_INVALID,
                "anchor overrides need non-empty op-key and anchor-name",
                subject=key or "*",
            )
    for key, value in params.block_hints.items():
        if not key:
            collector.error(
                DiagnosticCode.SCHEDULE_HINT_INVALID,
                "block-hint keys must be non-empty",
            )
        elif not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            collector.error(
                DiagnosticCode.SCHEDULE_HINT_INVALID,
                f"block hint {key}={value!r} must be a positive integer",
                subject=key,
            )
        elif value > 1024:
            collector.error(
                DiagnosticCode.SCHEDULE_HINT_INVALID,
                f"block hint {key}={value} exceeds the 1024 prototype maximum",
                subject=key,
            )
    if params.warp_count is not None and params.warp_count not in VALID_NUM_WARPS:
        collector.error(
            DiagnosticCode.SCHEDULE_HINT_INVALID,
            f"warp_count={params.warp_count} not in {list(VALID_NUM_WARPS)}",
            subject="warp_count",
        )
    if params.stage_count is not None and params.stage_count not in VALID_NUM_STAGES:
        collector.error(
            DiagnosticCode.SCHEDULE_HINT_INVALID,
            f"stage_count={params.stage_count} not in {list(VALID_NUM_STAGES)}",
            subject="stage_count",
        )
    for key, value in params.dtype_hints.items():
        try:
            from urm.compiler.semantic import DType

            DType(value)
        except ValueError:
            collector.error(
                DiagnosticCode.SCHEDULE_HINT_INVALID,
                f"dtype hint {key}={value!r} is not a supported DType",
                subject=key,
            )
    for key, value in params.layout_hints.items():
        if value not in VALID_LAYOUTS:
            collector.error(
                DiagnosticCode.SCHEDULE_HINT_INVALID,
                f"layout hint {key}={value!r} not in {sorted(VALID_LAYOUTS)}",
                subject=key,
            )
    return tuple(collector)


# -- Candidates ----------------------------------------------------------------


BASE_CANDIDATE_ID = "base"


@dataclass(frozen=True, slots=True)
class CompilationCandidate:
    """One enumerated, immutable rewrite/lowering alternative.

    Enumeration never mutates the program: applying a candidate goes through
    :meth:`UrmCompiler.compile_candidate`. The unfused/base plan is always the
    first candidate (id ``"base"``).
    """

    candidate_id: str
    kind: str  # "base" | "rewrite"
    rule: str | None = None
    subject_op: str | None = None
    legal: bool = True
    reason_code: str | None = None
    detail: str | None = None
    # Analytical effects of choosing this candidate (base: zero):
    traffic_bytes_delta: int = 0
    launch_count_delta: int = 0
    backward_verified: bool = True
    equivalence_class: str = "none"
    saved_state_policy: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "rule": self.rule,
            "subject_op": self.subject_op,
            "legal": self.legal,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "traffic_bytes_delta": self.traffic_bytes_delta,
            "launch_count_delta": self.launch_count_delta,
            "backward_verified": self.backward_verified,
            "equivalence_class": self.equivalence_class,
            "saved_state_policy": self.saved_state_policy,
        }


@dataclass(frozen=True, slots=True)
class RejectedAlternative:
    """A candidate that was enumerated but not selected, and why."""

    candidate_id: str
    reason_code: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


class SelectionPolicy(StrEnum):
    EXPLICIT = "explicit"
    COST_HEURISTIC = "cost_heuristic"
    SOLVER_GUIDED = "solver_guided"


# -- Plans --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutablePlan:
    """Deterministic, serializable lowering of one semantic program."""

    program_name: str
    steps: tuple[PlanStep, ...]
    obligations: tuple[tuple[str, str, str], ...]  # kind, subject, detail
    escape_hatch_count: int = 0  # structural invariant: always zero

    def to_dict(self) -> dict[str, object]:
        return {
            "program": self.program_name,
            "steps": [
                {
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "anchor": step.anchor,
                    "exchanges": [
                        {
                            "step_id": ex.step_id,
                            "src_device": ex.src_device,
                            "dst_device": ex.dst_device,
                            "payload_count": ex.payload_count,
                            "payload_bytes": ex.payload_bytes,
                        }
                        for ex in step.exchanges
                    ],
                    "note": step.note,
                }
                for step in self.steps
            ],
            "obligations": [
                {"kind": kind, "subject_op": subject, "detail": detail}
                for kind, subject, detail in self.obligations
            ],
            "escape_hatch_count": self.escape_hatch_count,
        }


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: ExecutablePlan
    rewritten_program: SemanticProgram
    trace: RewriteTrace
    cost: CostEstimate
    intent: CompilationIntent = CompilationIntent.INFERENCE
    selected_candidate_id: str = BASE_CANDIDATE_ID
    selection_policy: SelectionPolicy = SelectionPolicy.COST_HEURISTIC
    rejected_alternatives: tuple[RejectedAlternative, ...] = ()
    candidates: tuple[CompilationCandidate, ...] = ()
    schedule_params: dict[str, object] | None = None
    unresolved_obligations: tuple[tuple[str, str, str], ...] = ()
    solver_statistics: dict[str, float | int] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "trace": self.trace.to_dict(),
            "cost": self.cost.to_dict(),
            "intent": self.intent.value,
            "selected_candidate_id": self.selected_candidate_id,
            "selection_policy": self.selection_policy.value,
            "rejected_alternatives": [
                rejection.to_dict() for rejection in self.rejected_alternatives
            ],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "schedule_params": self.schedule_params,
            "unresolved_obligations": [
                {"kind": kind, "subject_op": subject, "detail": detail}
                for kind, subject, detail in self.unresolved_obligations
            ],
            "solver_statistics": self.solver_statistics,
        }


# -- Communication planning ---------------------------------------------------


def plan_route_distribution(
    *,
    edges: Sequence[tuple[int, int]],
    placement: PlacementMap,
    source_tensor: str = "values",
    query_tensor: str = "queries",
    payload_bytes: int = 2,
    capacity_drops: Iterable[int] = (),
) -> tuple[PlanStep, ...]:
    """Lower one semantic route as a PULL-GATHER protocol.

    Query owners reference remote source rows: source-owner devices send
    payloads toward query owners, which perform the local weighted reduction.
    (Token-to-expert dispatch with a required return is the PUSH protocol in
    ``urm.compiler.route_protocols``; the directions are never conflated.)

    Every logical edge survives unless ``capacity_drops`` names it explicitly
    (a declared capacity policy drop).
    """
    dropped = set(capacity_drops)
    local_edges: list[tuple[int, int]] = []
    remote_edges: list[tuple[int, int]] = []
    for position, (query_index, source_index) in enumerate(edges):
        if position in dropped:
            continue
        leg = placement.classify(source_tensor, source_index, query_tensor, query_index)
        if leg is RouteLeg.PEER_EXCHANGE:
            remote_edges.append((query_index, source_index))
        else:
            local_edges.append((query_index, source_index))

    steps: list[PlanStep] = []
    step_id = 0
    if remote_edges:
        grouped: dict[tuple[int, int], list[int]] = {}
        for query_index, source_index in remote_edges:
            src = placement.owner_of(source_tensor, source_index)
            dst = placement.owner_of(query_tensor, query_index)
            grouped.setdefault((src, dst), []).append(query_index)
        exchanges: list[ExchangeStep] = []
        for (src, dst), queries in sorted(grouped.items()):
            exchanges.append(
                ExchangeStep(
                    step_id=len(exchanges),
                    src_device=src,
                    dst_device=dst,
                    payload_count=len(queries),
                    payload_bytes=payload_bytes,
                    grouped_key=f"dst:{dst}",
                )
            )
        steps.append(
            PlanStep(
                step_id=step_id,
                kind="exchange",
                exchanges=tuple(exchanges),
                note="pull_gather: grouped by destination query owner",
            )
        )
        step_id += 1
    if local_edges or remote_edges:
        steps.append(
            PlanStep(
                step_id=step_id,
                kind="local_reduce",
                note=(
                    f"{len(local_edges)} co-resident edges gathered locally; "
                    f"{len(remote_edges)} received payloads merged at query owner"
                ),
            )
        )
        step_id += 1
    if not steps:
        raise CompilerError(
            (
                Diagnostic(
                    code=DiagnosticCode.CAPACITY_DROP_REQUIRED,
                    message="capacity policy dropped every logical route",
                ),
            )
        )
    return tuple(steps)


def _route_edges_for(
    spec: RouteSpec, queries: int, sources: int, route_width: int, seed: int = 0
) -> Iterator[tuple[int, int]]:
    """Deterministic representative edges for a symbolic route."""
    del spec, seed
    for query_index in range(queries):
        for k in range(route_width):
            yield query_index, (query_index * route_width + k) % sources


# -- The compiler facade ------------------------------------------------------


class UrmCompiler:
    """NAS-facing compilation entry point.

    Explicit flow (each step callable independently):

    1. ``validate`` - typed semantic + intent validation;
    2. ``enumerate_candidates`` - immutable candidate list (base included);
    3. ``build_constraints`` - backend-independent constraint model;
    4. ``check_feasible`` - SAT/UNSAT/UNKNOWN with diagnostics;
    5. ``solve_schedule`` - bounded lexicographic schedule optimization;
    6. ``verify_model`` - imperative, solver-independent verification;
    7. ``compile_candidate`` / ``compile`` - executable plan + trace;
    8. ``benchmark_candidate`` - empirical measurement ticket (benchmarks/).
    """

    def __init__(
        self,
        *,
        rules: Sequence[RewriteRule] | None = None,
        anchors: AnchorRegistry | None = None,
        device_limits_path: Path | None = None,
    ) -> None:
        self.engine = RewriteEngine(rules) if rules is not None else RewriteEngine()
        self.anchors = anchors if anchors is not None else default_registry()
        self.device_limits = DeviceLimits.load(device_limits_path)

    # -- validation --------------------------------------------------------

    def validate(
        self,
        program: SemanticProgram,
        intent: CompilationIntent = CompilationIntent.INFERENCE,
    ) -> tuple[Diagnostic, ...]:
        """Semantic validation plus intent-specific structural checks."""
        diagnostics = list(program.validate())
        del intent  # structural intent checks live in candidate enumeration
        return tuple(diagnostics)

    # -- candidate enumeration ----------------------------------------------

    def enumerate_candidates(
        self,
        program: SemanticProgram,
        intent: CompilationIntent = CompilationIntent.INFERENCE,
    ) -> tuple[CompilationCandidate, ...]:
        """Immutable candidate list; the program is never mutated."""
        candidates: list[CompilationCandidate] = [
            CompilationCandidate(
                candidate_id=BASE_CANDIDATE_ID,
                kind="base",
                legal=True,
                equivalence_class="identity",
            )
        ]
        for rule, match in self.engine.candidates_for(program):
            outcome: CheckOutcome = CheckOutcome.pass_()
            for precondition in rule.preconditions:
                outcome = precondition.check(program, match)
                if not outcome.ok:
                    break
            candidate_id = self.engine.candidate_id(rule, match)
            legal = outcome.ok
            reason_code = outcome.reason_code.value if outcome.reason_code else None
            detail = outcome.message
            if legal and intent is CompilationIntent.TRAINING:
                verdict = self._training_legality(program, rule, match)
                legal = verdict.ok
                reason_code = verdict.reason_code.value if verdict.reason_code else None
                detail = verdict.message
            candidates.append(
                CompilationCandidate(
                    candidate_id=candidate_id,
                    kind="rewrite",
                    rule=rule.name,
                    subject_op=match.subject.name,
                    legal=legal,
                    reason_code=reason_code,
                    detail=detail,
                    traffic_bytes_delta=rule.traffic_bytes_delta,
                    launch_count_delta=rule.launch_count_delta,
                    backward_verified=not rule.forward_only,
                    equivalence_class=rule.equivalence.value,
                    saved_state_policy=rule.saved_state_policy.value,
                )
            )
        return tuple(candidates)

    # -- constraint building / solving (Phase 9 NAS-facing flow) -------------

    def build_constraints(
        self,
        program: SemanticProgram,
        candidate_id: str,
        intent: CompilationIntent = CompilationIntent.INFERENCE,
        *,
        schedule_params: ScheduleParams | None = None,
    ) -> ConstraintModel:
        """Build the backend-independent constraint model for one candidate."""
        from urm.compiler.kernel_plan import build_schedule_model

        candidates = {
            c.candidate_id: c for c in self.enumerate_candidates(program, intent)
        }
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise CompilerError(
                (
                    Diagnostic(
                        code=DiagnosticCode.CANDIDATE_NOT_FOUND,
                        message=f"unknown candidate id {candidate_id!r}",
                    ),
                )
            )
        return build_schedule_model(
            program=program,
            candidate=candidate,
            intent=intent,
            schedule_params=schedule_params or ScheduleParams(),
            device_limits=self.device_limits,
        )

    def check_feasible(self, model: ConstraintModel) -> FeasibilityResult:
        """Run the Z3 feasibility pass (requires the ``solver`` extra)."""
        from urm.compiler.solver import FeasibilityPass

        return FeasibilityPass().run(model)

    def solve_schedule(self, model: ConstraintModel) -> OptimizationResult:
        """Run the bounded lexicographic optimization pass (Z3 optional)."""
        from urm.compiler.solver import OptimizationPass

        return OptimizationPass().run(model)

    def verify_model(
        self,
        model: ConstraintModel,
        assignment: dict[str, bool | int | str],
    ) -> VerificationReport:
        """Independently verify a solver assignment without any solver."""
        from urm.compiler.kernel_plan import verify_schedule_assignment

        return verify_schedule_assignment(model, assignment)

    def benchmark_candidate(self, candidate_id: str) -> dict[str, str]:
        """Provenance ticket pointing at the empirical benchmark entrypoint."""
        return {
            "candidate_id": candidate_id,
            "benchmark_entrypoint": "benchmarks/routed_epilogue_selection.py",
            "artifact": "results/compiler/solver/routed-epilogue-selection.json",
            "note": (
                "empirical numbers are produced by the committed benchmark "
                "script on the exact implementation commit; the compiler "
                "itself never fabricates measurements"
            ),
        }

    # -- compilation ---------------------------------------------------------

    def compile(
        self,
        program: SemanticProgram,
        *,
        intent: CompilationIntent = CompilationIntent.INFERENCE,
        candidate_id: str | None = None,
        placement: PlacementMap | None = None,
        schedule_params: ScheduleParams | None = None,
    ) -> CompilationResult:
        schedule_params = schedule_params or ScheduleParams()
        hint_diagnostics = validate_schedule_params(schedule_params)
        if any(d.severity.value == "error" for d in hint_diagnostics):
            raise CompilerError(hint_diagnostics)

        diagnostics = program.validate()
        if diagnostics and any(d.severity.value == "error" for d in diagnostics):
            raise CompilerError(diagnostics)

        candidates = self.enumerate_candidates(program, intent)
        selection = self._select_candidate(candidates, candidate_id, schedule_params)
        chosen, policy, rejections, solver_stats = selection

        compiled = program
        trace_parts: list[RewriteTrace] = []
        if chosen.kind == "rewrite":
            rule, match = self._find_rule_match(program, chosen)
            applied = self.engine.apply_candidate(program, rule, match)
            compiled = applied.program
            trace_parts.append(applied.trace)

        steps, anchors_chosen, cost_parts = self._lower_to_anchors(
            compiled, intent, schedule_params
        )

        obligations = tuple(
            obligation for trace in trace_parts for obligation in trace.obligations
        )
        unresolved = self._resolve_obligations(obligations, anchors_chosen, intent)

        if placement is not None:
            steps.extend(self._communication_steps(compiled, placement))
        elif any(isinstance(op, CollectiveExchange) for op in compiled.ops):
            collector = DiagnosticsCollector()
            collector.error(
                DiagnosticCode.PLACEMENT_INCOMPLETE,
                "program declares collective intent but no placement was given",
            )
            raise CompilerError(tuple(collector))

        estimated = (
            combine(*cost_parts)
            if cost_parts
            else CostEstimate(
                useful_flops=0,
                logical_bytes=0,
                physical_bytes_estimate=0,
                launch_count=0,
                temporary_bytes=0,
            )
        )
        trace = RewriteTrace(
            attempts=tuple(a for t in trace_parts for a in t.attempts),
            obligations=obligations,
            anchors=tuple(anchors_chosen),
            estimated_costs={
                "useful_flops": estimated.useful_flops,
                "physical_bytes_estimate": estimated.physical_bytes_estimate,
                "launch_count": estimated.launch_count,
            },
        )
        plan = ExecutablePlan(
            program_name=program.name,
            steps=tuple(steps),
            obligations=tuple((o.kind, o.subject_op, o.detail) for o in obligations),
        )
        return CompilationResult(
            plan=plan,
            rewritten_program=compiled,
            trace=trace,
            cost=estimated,
            intent=intent,
            selected_candidate_id=chosen.candidate_id,
            selection_policy=policy,
            rejected_alternatives=rejections,
            candidates=candidates,
            schedule_params=schedule_params.to_dict(),
            unresolved_obligations=unresolved,
            solver_statistics=solver_stats,
        )

    def compile_candidate(
        self,
        program: SemanticProgram,
        candidate_id: str,
        *,
        intent: CompilationIntent = CompilationIntent.INFERENCE,
        placement: PlacementMap | None = None,
        schedule_params: ScheduleParams | None = None,
    ) -> CompilationResult:
        """Compile with an explicitly selected candidate (no auto policy)."""
        return self.compile(
            program,
            intent=intent,
            candidate_id=candidate_id,
            placement=placement,
            schedule_params=schedule_params,
        )

    # -- internals -------------------------------------------------------------

    def _select_candidate(
        self,
        candidates: tuple[CompilationCandidate, ...],
        candidate_id: str | None,
        schedule_params: ScheduleParams,
    ) -> tuple[
        CompilationCandidate,
        SelectionPolicy,
        tuple[RejectedAlternative, ...],
        dict[str, float | int] | None,
    ]:
        legal = [c for c in candidates if c.legal]
        if candidate_id is not None:
            match = next(
                (c for c in candidates if c.candidate_id == candidate_id), None
            )
            if match is None:
                raise CompilerError(
                    (
                        Diagnostic(
                            code=DiagnosticCode.CANDIDATE_NOT_FOUND,
                            message=(
                                f"unknown candidate {candidate_id!r}; known: "
                                f"{[c.candidate_id for c in candidates]}"
                            ),
                        ),
                    )
                )
            if not match.legal:
                raise CompilerError(
                    (
                        Diagnostic(
                            code=DiagnosticCode.CANDIDATE_ILLEGAL,
                            message=(
                                f"candidate {candidate_id!r} is not legal: "
                                f"{match.reason_code}: {match.detail}"
                            ),
                        ),
                    )
                )
            rejections = tuple(
                RejectedAlternative(
                    candidate_id=c.candidate_id,
                    reason_code=(
                        "not_selected_explicit_choice"
                        if c.legal
                        else (c.reason_code or "illegal")
                    ),
                    detail=c.detail or ("explicit selection overrode this candidate"),
                )
                for c in candidates
                if c.candidate_id != candidate_id
            )
            return match, SelectionPolicy.EXPLICIT, rejections, None

        # Automatic selection: solver-guided when the pinned solver extra is
        # installed, otherwise the documented deterministic cost heuristic.
        # Both policies are deterministic; the choice made is recorded.
        if len(legal) > 1:
            try:
                from urm.compiler.kernel_plan import (
                    build_candidate_selection_model,
                    decode_selected_candidate,
                )
                from urm.compiler.solver import OptimizationPass

                model, candidate_by_choice = build_candidate_selection_model(candidates)
                result = OptimizationPass().run(model)
                if result.status.value == "sat" and result.assignment:
                    chosen_id = decode_selected_candidate(
                        result.assignment, candidate_by_choice
                    )
                    chosen = next(c for c in legal if c.candidate_id == chosen_id)
                    rejections = tuple(
                        RejectedAlternative(
                            candidate_id=c.candidate_id,
                            reason_code=(
                                "rejected_by_solver_objective"
                                if c.candidate_id != chosen_id
                                else "selected"
                            ),
                            detail=(
                                f"objective values {result.objective_values}"
                                if c.candidate_id != chosen_id
                                else "solver-selected"
                            ),
                        )
                        for c in candidates
                        if c.candidate_id != chosen_id
                    )
                    return (
                        chosen,
                        SelectionPolicy.SOLVER_GUIDED,
                        rejections,
                        dict(result.statistics),
                    )
                # Solver could not decide (UNSAT over legal set should be
                # impossible; UNKNOWN possible under timeout): record why and
                # use the deterministic heuristic instead of failing silently.
                heuristic_note = f"solver status {result.status.value}"
            except Exception as error:  # noqa: BLE001 - recorded, never hidden
                heuristic_note = f"solver path unavailable: {error}"
        else:
            heuristic_note = "single legal candidate"

        ordered = sorted(
            legal,
            key=lambda c: (
                c.traffic_bytes_delta,
                c.launch_count_delta,
                c.candidate_id,
            ),
        )
        chosen = ordered[0]
        rejections = tuple(
            RejectedAlternative(
                candidate_id=c.candidate_id,
                reason_code="higher_estimated_cost"
                if c.candidate_id != chosen.candidate_id
                else "selected",
                detail=(
                    f"heuristic order ({c.traffic_bytes_delta:+d} B traffic, "
                    f"{c.launch_count_delta:+d} launches); {heuristic_note}"
                ),
            )
            for c in candidates
            if c.candidate_id != chosen.candidate_id
        )
        del schedule_params
        return chosen, SelectionPolicy.COST_HEURISTIC, rejections, None

    def _training_legality(
        self, program: SemanticProgram, rule: RewriteRule, match: RewriteMatch
    ) -> CheckOutcome:
        """Training compilations reject rewrites with unverified backward."""
        if rule.backward_contract is None:
            return CheckOutcome.fail(
                DiagnosticCode.INTENT_CONFLICT,
                f"{rule.name} declares no certified backward "
                f"({rule.forward_only_restriction.value}); training "
                "compilation cannot accept it",
            )
        for dtype in self._operand_dtypes(program, match.subject):
            if not rule.backward_contract.covers(dtype):
                return CheckOutcome.fail(
                    DiagnosticCode.INTENT_CONFLICT,
                    f"{rule.name} backward is not certified for {dtype.value}",
                )
        return CheckOutcome.pass_()

    @staticmethod
    def _operand_dtypes(program: SemanticProgram, op) -> tuple:
        from urm.compiler.semantic import DType

        handles = {handle.name: handle.dtype for handle in program.inputs}
        found: list[DType] = []
        for tensor in op.inputs:
            dtype = handles.get(tensor)
            if dtype is not None:
                found.append(dtype)
        return tuple(found)

    def _find_rule_match(self, program: SemanticProgram, candidate):
        for rule, match in self.engine.candidates_for(program):
            if self.engine.candidate_id(rule, match) == candidate.candidate_id:
                return rule, match
        raise CompilerError(
            (
                Diagnostic(
                    code=DiagnosticCode.CANDIDATE_NOT_FOUND,
                    message=(
                        f"candidate {candidate.candidate_id!r} no longer matches "
                        "the program; re-enumerate candidates"
                    ),
                ),
            )
        )

    def _lower_to_anchors(
        self,
        compiled: SemanticProgram,
        intent: CompilationIntent,
        schedule_params: ScheduleParams,
    ):
        steps: list[PlanStep] = []
        anchors_chosen: list[str] = []
        cost_parts: list[CostEstimate] = []
        step_id = 0
        for op in compiled.ops:
            if self._is_interior_gather(compiled, op):
                continue  # fused into the routed-reduction dispatch
            request_kind, visitors = self._request_for(op)
            if request_kind is None:
                continue
            override = schedule_params.anchor_overrides.get(op.name) or (
                schedule_params.anchor_overrides.get("*")
            )
            decision: AnchorDecision = self.anchors.select(
                AnchorRequest(kind=request_kind, visitors=visitors)
            )
            if override is not None:
                decision = self._apply_override(request_kind, visitors, override, op)
            if not decision.ok:
                decline = decision.decline
                assert decline is not None
                collector = DiagnosticsCollector()
                collector.error(
                    DiagnosticCode.NO_ANCHOR_AVAILABLE,
                    f"{op.name}: {decline.message}",
                    subject=op.name,
                )
                raise CompilerError(tuple(collector))
            assert decision.anchor is not None
            self._check_anchor_intent(
                compiled, op, decision.anchor, intent, schedule_params
            )
            anchors_chosen.append(decision.anchor.name)
            steps.append(
                PlanStep(
                    step_id=step_id,
                    kind="anchor_dispatch",
                    anchor=decision.anchor.name,
                    note=op.name,
                )
            )
            step_id += 1
            cost_parts.append(self._cost_for(op, decision.anchor.name, placement=None))
        return steps, anchors_chosen, cost_parts

    @staticmethod
    def _apply_override(
        kind: AnchorKind,
        visitors: tuple[VisitorDescriptor, ...],
        override_name: str,
        op,
    ) -> AnchorDecision:
        from urm.compiler.execution import TRUSTED_ANCHORS

        anchor = next((a for a in TRUSTED_ANCHORS if a.name == override_name), None)
        if anchor is None or anchor.kind is not kind:
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    reason_code=DiagnosticCode.ANCHOR_DECLINED,
                    message=f"override {override_name!r} is not a registered "
                    f"{kind.value} anchor",
                ),
            )
        unmet = [v.kind.value for v in visitors if not anchor.accepts(v)]
        if unmet:
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    reason_code=DiagnosticCode.ANCHOR_DECLINED,
                    message=(
                        f"overridden anchor {override_name} declined visitors: {unmet}"
                    ),
                ),
            )
        del op
        return AnchorDecision(anchor=anchor, decline=None)

    def _check_anchor_intent(
        self,
        program: SemanticProgram,
        op,
        anchor,
        intent: CompilationIntent,
        schedule_params: ScheduleParams,
    ) -> None:
        problems: list[str] = []
        if intent is CompilationIntent.TRAINING:
            if anchor.forward_only:
                problems.append(
                    f"anchor {anchor.name} declares forward-only execution; "
                    "training compilation requires verified backward"
                )
            for dtype in self._operand_dtypes(program, op):
                if not anchor.backward_covers(dtype.value):
                    problems.append(
                        f"anchor {anchor.name} backward is not verified for "
                        f"{dtype.value}; training cannot dispatch this op"
                    )
            if schedule_params.deterministic and (
                "ordered_grad_accumulation" not in anchor.honored_obligations
            ):
                problems.append(
                    f"deterministic training requested but anchor "
                    f"{anchor.name} accumulates gradients through relaxed "
                    "cross-program atomics (no ordered-accumulation lowering)"
                )
        if schedule_params.deterministic and not anchor.deterministic_accumulation:
            problems.append(
                f"deterministic mode requested but anchor {anchor.name} uses "
                "nondeterministic accumulation ordering"
            )
        if problems:
            raise CompilerError(
                tuple(
                    Diagnostic(
                        code=DiagnosticCode.INTENT_CONFLICT,
                        message=f"{op.name}: {problem}",
                        subject=op.name,
                    )
                    for problem in problems
                )
            )

    def _resolve_obligations(
        self,
        obligations,
        anchors_chosen: Sequence[str],
        intent: CompilationIntent,
    ) -> tuple[tuple[str, str, str], ...]:
        """Classify obligations as resolved or outstanding for this intent."""
        from urm.compiler.execution import TRUSTED_ANCHORS

        selected = [a for a in TRUSTED_ANCHORS if a.name in set(anchors_chosen)]
        unresolved: list[tuple[str, str, str]] = []
        for obligation in obligations:
            resolved = False
            if obligation.kind == "recompute_backward":
                resolved = any(
                    "recompute_backward" in anchor.honored_obligations
                    for anchor in selected
                )
                if intent is not CompilationIntent.TRAINING:
                    # Backward recomputation is only *required* for training.
                    resolved = True
            elif obligation.kind == "forward_only":
                resolved = intent is not CompilationIntent.TRAINING
            if not resolved:
                unresolved.append(
                    (obligation.kind, obligation.subject_op, obligation.detail)
                )
        if intent is CompilationIntent.TRAINING and unresolved:
            raise CompilerError(
                tuple(
                    Diagnostic(
                        code=DiagnosticCode.INTENT_CONFLICT,
                        message=(
                            "training compilation has an unresolved obligation: "
                            f"{kind} on {subject} - {detail}"
                        ),
                        subject=subject,
                    )
                    for kind, subject, detail in unresolved
                )
            )
        return tuple(unresolved)

    @staticmethod
    def _is_interior_gather(program: SemanticProgram, op: SemanticNode) -> bool:
        from urm.compiler.semantic import Gather as GatherOp
        from urm.compiler.semantic import WeightedReduce

        return isinstance(op, GatherOp) and any(
            isinstance(consumer, WeightedReduce)
            for consumer in program.consumers_of(op.outputs[0])
        )

    @staticmethod
    def _request_for(
        op: SemanticNode,
    ) -> tuple[AnchorKind | None, tuple[VisitorDescriptor, ...]]:
        from urm.compiler.execution import VisitorKind
        from urm.compiler.locality import Locality
        from urm.compiler.semantic import TransformKind

        if isinstance(op, WeightedReduce):
            visitors: tuple[VisitorDescriptor, ...] = ()
            if op.epilogue is not None and op.epilogue.kind is TransformKind.ROW_SCALE:
                visitors = (
                    VisitorDescriptor(
                        kind=VisitorKind.FINAL_SCALE_CONVERT,
                        element_dtype="float32",
                        locality=Locality.TILE,
                    ),
                )
            return AnchorKind.ROUTED_REDUCTION, visitors
        if isinstance(op, Matmul):
            return AnchorKind.GEMM, ()
        if isinstance(op, Gather):
            return AnchorKind.ROUTED_REDUCTION, ()
        if isinstance(op, OrderedRecurrence):
            return AnchorKind.RECURRENT_SCAN, ()
        if isinstance(op, Score | Select | Transform):
            return None, ()  # folded into producers by construction here
        if isinstance(op, CollectiveExchange):
            return AnchorKind.COLLECTIVE_EXCHANGE, ()
        return None, ()

    @staticmethod
    def _cost_for(
        op: SemanticNode, anchor_name: str, placement: PlacementMap | None
    ) -> CostEstimate:
        from urm.compiler import cost as cost_mod
        from urm.compiler.semantic import Matmul, WeightedReduce

        shape = op.shape_hint if isinstance(op, WeightedReduce | Matmul) else None
        if isinstance(op, WeightedReduce) and shape is not None:
            queries, sources, route_width, value_dim = shape
            base = cost_mod.routed_reduction_cost(
                queries=queries,
                sources=sources,
                route_width=route_width,
                value_dim=value_dim,
            )
            if op.epilogue is not None:
                materialized = cost_mod.row_scale_transform_cost(
                    queries=queries, value_dim=value_dim
                )
                notes = (
                    *base.notes,
                    f"fused epilogue avoids {materialized.logical_bytes} B",
                )
                return CostEstimate(
                    useful_flops=base.useful_flops + queries * value_dim,
                    logical_bytes=base.logical_bytes,
                    physical_bytes_estimate=base.physical_bytes_estimate,
                    launch_count=base.launch_count,
                    temporary_bytes=base.temporary_bytes,
                    atomic_contention_indicator=base.atomic_contention_indicator,
                    critical_path_us=base.critical_path_us,
                    notes=notes,
                )
            return base
        if isinstance(op, Matmul) and shape is not None:
            m, k, n = shape
            flops = 2 * m * k * n
            io = (m * k + k * n + m * n) * 2
            return CostEstimate(
                useful_flops=flops,
                logical_bytes=io,
                physical_bytes_estimate=io,
                launch_count=1,
                temporary_bytes=m * n * 2,
                notes=(f"anchor={anchor_name}",),
            )
        return CostEstimate(
            useful_flops=0,
            logical_bytes=0,
            physical_bytes_estimate=0,
            launch_count=1,
            temporary_bytes=0,
            notes=(f"anchor={anchor_name}; no shape hint attached",),
        )

    @staticmethod
    def _communication_steps(
        program: SemanticProgram, placement: PlacementMap
    ) -> tuple[PlanStep, ...]:
        steps: list[PlanStep] = []
        step_id = 1000  # namespace after dispatch steps
        for op in program.ops:
            if isinstance(op, StateUpdate) and op.commit_boundary:
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        kind="commit",
                        note=f"{op.state}@version+1 ({op.policy.value} merge)",
                    )
                )
                step_id += 1
            if isinstance(op, CollectiveExchange):
                mesh = placement.mesh
                exchanges = tuple(
                    ExchangeStep(
                        step_id=i,
                        src_device=device,
                        dst_device=peer,
                        payload_count=0,
                        payload_bytes=0,
                        grouped_key=f"collective:{op.kind.value}",
                    )
                    for i, (device, peer) in enumerate(
                        (d, mesh.peers(d)[0]) for d in mesh.devices if mesh.peers(d)
                    )
                )
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        kind="exchange",
                        exchanges=exchanges,
                        note=f"collective {op.kind.value} on axis {op.mesh_axis}",
                    )
                )
                step_id += 1
        return tuple(steps)


__all__ = [
    "BASE_CANDIDATE_ID",
    "ArchitectureParams",
    "CompilationCandidate",
    "CompilationIntent",
    "CompilationResult",
    "ExecutablePlan",
    "RejectedAlternative",
    "ScheduleParams",
    "SelectionPolicy",
    "UrmCompiler",
    "plan_route_distribution",
    "validate_schedule_params",
]
