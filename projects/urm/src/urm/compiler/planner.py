"""Placement-aware planner and the NAS-facing compilation API.

The planner lowers validated (and optionally rewritten) semantic programs into
deterministic executable plans:

    semantic program -> verified rewrites -> anchor selection
      -> placement/sharding decision -> executable plan + analytical cost
      -> deterministic compilation trace

Two parameter namespaces are kept separate everywhere: *architecture*
parameters (what the model is) and *schedule* parameters (how this lowering
runs). Serialized artifacts never mix them.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
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
    RewriteRule,
    RewriteTrace,
)
from urm.compiler.semantic import (
    CollectiveExchange,
    Gather,
    OrderedRecurrence,
    RouteSpec,
    Score,
    Select,
    SemanticNode,
    SemanticProgram,
    StateUpdate,
    Transform,
)

if TYPE_CHECKING:
    from pathlib import Path


# -- Parameter namespaces -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArchitectureParams:
    """NAS-facing parameters describing WHAT the architecture is."""

    family: str
    logical_dims: dict[str, int]
    routing: dict[str, str]


@dataclass(frozen=True, slots=True)
class ScheduleParams:
    """Backend/schedule parameters describing HOW one lowering executes."""

    anchor_overrides: dict[str, str] = field(default_factory=dict)
    block_hints: dict[str, int] = field(default_factory=dict)
    dtype_hints: dict[str, str] = field(default_factory=dict)


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
class CandidateSummary:
    """One enumerated rewrite/lowering candidate and its legality."""

    rule: str | None
    subject_op: str
    legal: bool
    reason_code: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class CompilationResult:
    plan: ExecutablePlan
    rewritten_program: SemanticProgram
    trace: RewriteTrace
    cost: CostEstimate
    candidates: tuple[CandidateSummary, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "trace": self.trace.to_dict(),
            "cost": self.cost.to_dict(),
            "candidates": [
                {
                    "rule": candidate.rule,
                    "subject_op": candidate.subject_op,
                    "legal": candidate.legal,
                    "reason_code": candidate.reason_code,
                    "detail": candidate.detail,
                }
                for candidate in self.candidates
            ],
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
    """Lower one semantic route onto a placed mesh.

    Every logical edge survives unless ``capacity_drops`` names it explicitly
    (a declared capacity policy drop). Co-resident destinations become local
    access; remote destinations become grouped peer exchanges; a final local
    reduction merges payloads per query owner.
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
                note="grouped by destination device",
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
                    f"{len(remote_edges)} received payloads merged"
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

    1. construct/validate typed semantic programs;
    2. enumerate legal rewrite/lowering candidates;
    3. reject invalid architectures with structured diagnostics;
    4. compile a chosen candidate to an executable plan;
    5. obtain analytical cost features without running training;
    6. benchmark selected candidates when hardware allows (see benchmarks/).
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

    def validate(self, program: SemanticProgram) -> tuple[Diagnostic, ...]:
        return program.validate()

    # -- candidate enumeration ----------------------------------------------

    def enumerate_candidates(
        self, program: SemanticProgram
    ) -> tuple[CandidateSummary, ...]:
        summaries: list[CandidateSummary] = []
        for rule, match in self.engine.candidates_for(program):
            outcome: CheckOutcome = CheckOutcome.pass_()
            for precondition in rule.preconditions:
                outcome = precondition.check(program, match)
                if not outcome.ok:
                    break
            summaries.append(
                CandidateSummary(
                    rule=rule.name,
                    subject_op=match.subject.name,
                    legal=outcome.ok,
                    reason_code=outcome.reason_code.value
                    if outcome.reason_code
                    else None,
                    detail=outcome.message,
                )
            )
        return tuple(summaries)

    # -- compilation ---------------------------------------------------------

    def compile(
        self,
        program: SemanticProgram,
        *,
        placement: PlacementMap | None = None,
        schedule_params: ScheduleParams | None = None,
    ) -> CompilationResult:
        del schedule_params  # reserved: anchor overrides land with backends
        diagnostics = program.validate()
        if diagnostics and any(d.severity.value == "error" for d in diagnostics):
            raise CompilerError(diagnostics)

        rewrite_result = self.engine.apply(program)
        compiled = rewrite_result.program

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
            decision: AnchorDecision = self.anchors.select(
                AnchorRequest(kind=request_kind, visitors=visitors)
            )
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
            cost_parts.append(self._cost_for(op, decision.anchor.name, placement))

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
            attempts=rewrite_result.trace.attempts,
            obligations=rewrite_result.trace.obligations,
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
            obligations=tuple(
                (o.kind, o.subject_op, o.detail)
                for o in rewrite_result.trace.obligations
            ),
        )
        return CompilationResult(
            plan=plan,
            rewritten_program=compiled,
            trace=trace,
            cost=estimated,
            candidates=self.enumerate_candidates(program),
        )

    # -- internals -------------------------------------------------------------

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
        from urm.compiler.semantic import Matmul, TransformKind, WeightedReduce

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
    "ArchitectureParams",
    "CandidateSummary",
    "CompilationResult",
    "ExecutablePlan",
    "ScheduleParams",
    "UrmCompiler",
    "plan_route_distribution",
]
