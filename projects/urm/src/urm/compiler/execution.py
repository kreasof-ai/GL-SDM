"""Execution layer: trusted anchors and their typed, locality-constrained visitors.

An execution anchor is a *trusted* lowering target: an existing production
kernel family (GEMM, attention, recurrent scan, grouped GEMM, routed
reduction, page gather/update, collective exchange) or a generated kernel that
has passed its differential gates.

Anchors expose constrained visitors - typed descriptors of the extra work a
program may do inside the anchor's lifetime (prologue, epilogue, side output).
Visitors are NOT Python callables: they are data interpreted by registered
anchor implementations, so no arbitrary tensor callback can enter the core IR.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from urm.compiler.diagnostics import DiagnosticCode
from urm.compiler.effects import ORDERED_STATE, PURE, EffectSignature
from urm.compiler.locality import Locality, LocalityConstraint


class AnchorKind(StrEnum):
    """Trusted execution-anchor classes."""

    GEMM = "gemm"
    ATTENTION = "attention"
    RECURRENT_SCAN = "recurrent_scan"
    GROUPED_GEMM = "grouped_gemm"
    ROUTED_REDUCTION = "routed_reduction"
    PAGE_GATHER_UPDATE = "page_gather_update"
    SPARSE_DELTA_MEMORY = "sparse_delta_memory"
    COLLECTIVE_EXCHANGE = "collective_exchange"


class VisitorKind(StrEnum):
    """Constrained visitor vocabulary exposed by anchors."""

    ELEMENTWISE_MAP = "elementwise_map"
    PAIRWISE_MAP = "pairwise_map"
    VECTOR_LOAD_STORE = "vector_load_store"
    TILE_LOAD_STORE = "tile_load_store"
    PARTIAL_REDUCTION = "partial_reduction"  # row/column partial reductions
    STATEFUL_TILE_TRANSFORM = "stateful_tile_transform"
    SIDE_OUTPUT = "auxiliary_side_output"
    FINAL_SCALE_CONVERT = "final_scaling_conversion"


@dataclass(frozen=True, slots=True)
class VisitorDescriptor:
    """A typed visitor instance an anchor may execute in its lifetime.

    ``locality`` bounds where the visited values live; ``accumulation_dtype``
    pins numeric semantics. Anchors reject visitors they cannot honor.
    """

    kind: VisitorKind
    element_dtype: str  # DType name; a string keeps this module dependency-light
    accumulation_dtype: str = "float32"
    locality: Locality = Locality.TILE
    arity: int = 1

    def __post_init__(self) -> None:
        if self.arity < 1:
            raise ValueError("visitor arity must be >= 1")


@dataclass(frozen=True, slots=True)
class ExecutionAnchor:
    """One trusted lowering target with its capability contract."""

    kind: AnchorKind
    name: str
    trusted: bool = True  # only verified kernels carry True
    experimental: bool = False  # compiler-generated capability under evaluation
    effect: EffectSignature = PURE
    operand_locality: LocalityConstraint = field(
        default_factory=lambda: LocalityConstraint()
    )
    result_locality: LocalityConstraint = field(
        default_factory=lambda: LocalityConstraint(
            min=Locality.DEVICE, max=Locality.DEVICE
        )
    )
    supported_visitors: frozenset[VisitorKind] = frozenset()
    # Capability contract extensions (solver-facing facts):
    forward_only: bool = False
    backward_verified_dtypes: frozenset[str] = frozenset()
    """Dtypes whose backward passes committed differential gates."""
    honored_obligations: frozenset[str] = frozenset()
    """Rewrite obligations this anchor resolves (e.g. ``recompute_backward``)."""
    deterministic_accumulation: bool = True
    commit_capable: bool = False
    consumes_launch_config: bool = False
    schedulable: bool = False
    supported_plan_kinds: frozenset[str] = frozenset()
    required_visitors: frozenset[VisitorKind] = frozenset()
    required_semantic_inputs: tuple[str, ...] = ()
    supported_blocks: tuple[int, ...] = ()
    supported_warps: tuple[int, ...] = ()
    supported_stages: tuple[int, ...] = ()
    supported_decompositions: tuple[str, ...] = ()
    supported_schedules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schedulable:
            if not self.consumes_launch_config:
                raise ValueError(
                    f"schedulable anchor {self.name!r} must set consumes_launch_config=True"
                )
            if not self.supported_plan_kinds:
                raise ValueError(
                    f"schedulable anchor {self.name!r} must declare nonempty supported_plan_kinds"
                )
            from urm.compiler.schedule_space import (
                GradValuesDecomposition,
                GradValuesSchedule,
                PlanKind,
            )

            valid_plans = {p.value for p in PlanKind}
            for p in self.supported_plan_kinds:
                if p not in valid_plans:
                    raise ValueError(
                        f"anchor {self.name!r} contains unrecognized plan kind {p!r}; valid: {sorted(valid_plans)}"
                    )

            for attr, val_name in (
                ("supported_blocks", "blocks"),
                ("supported_warps", "warps"),
                ("supported_stages", "stages"),
            ):
                vals = getattr(self, attr)
                if not vals:
                    raise ValueError(
                        f"schedulable anchor {self.name!r} must declare nonempty {attr}"
                    )
                if len(vals) != len(set(vals)):
                    raise ValueError(
                        f"anchor {self.name!r} contains duplicate {val_name}: {vals}"
                    )
                for v in vals:
                    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                        raise ValueError(
                            f"anchor {self.name!r} contains invalid {val_name} value {v!r}; must be positive integer"
                        )

            valid_decomps = {d.value for d in GradValuesDecomposition}
            if not self.supported_decompositions:
                raise ValueError(
                    f"schedulable anchor {self.name!r} must declare nonempty supported_decompositions"
                )
            if len(self.supported_decompositions) != len(
                set(self.supported_decompositions)
            ):
                raise ValueError(
                    f"anchor {self.name!r} contains duplicate decompositions: {self.supported_decompositions}"
                )
            for d in self.supported_decompositions:
                if d not in valid_decomps:
                    raise ValueError(
                        f"anchor {self.name!r} contains unrecognized decomposition {d!r}; valid: {sorted(valid_decomps)}"
                    )

            valid_scheds = {s.value for s in GradValuesSchedule}
            if not self.supported_schedules:
                raise ValueError(
                    f"schedulable anchor {self.name!r} must declare nonempty supported_schedules"
                )
            if len(self.supported_schedules) != len(set(self.supported_schedules)):
                raise ValueError(
                    f"anchor {self.name!r} contains duplicate schedules: {self.supported_schedules}"
                )
            for s in self.supported_schedules:
                if s not in valid_scheds:
                    raise ValueError(
                        f"anchor {self.name!r} contains unrecognized schedule {s!r}; valid: {sorted(valid_scheds)}"
                    )

    def accepts(self, visitor: VisitorDescriptor) -> bool:
        return visitor.kind in self.supported_visitors and self.result_locality.accepts(
            visitor.locality
        )

    def backward_covers(self, dtype_name: str) -> bool:
        return dtype_name in self.backward_verified_dtypes


@dataclass(frozen=True, slots=True)
class Decline:
    """An explicit refusal; never a silent semantic change."""

    reason_code: DiagnosticCode
    message: str


@dataclass(frozen=True, slots=True)
class AnchorRequest:
    """What the planner asks for before choosing a lowering."""

    kind: AnchorKind
    visitors: tuple[VisitorDescriptor, ...] = ()
    schedule_params: dict[str, str | int | float | bool] | None = None
    semantic_op: object | None = None


AnchorSelector = Callable[[AnchorRequest], "AnchorDecision | None"]
SDMSupportProbe = Callable[[], object]


@dataclass(frozen=True, slots=True)
class AnchorDecision:
    anchor: ExecutionAnchor | None
    decline: Decline | None

    @property
    def ok(self) -> bool:
        return self.anchor is not None


def make_selector(
    anchors: Sequence[ExecutionAnchor],
) -> AnchorSelector:
    """Build a selector over explicit anchor instances.

    The selector returns ``None`` to abstain so later selectors can answer.
    Within its catalog it scans every anchor of the requested kind: the first
    one that accepts all visitors wins; a refusal is returned only after no
    compatible anchor was found, so experimental anchors can still answer.
    """

    def _select(request: AnchorRequest) -> AnchorDecision | None:
        first_refusal: Decline | None = None
        for anchor in anchors:
            if anchor.kind is not request.kind or not anchor.trusted:
                continue
            unmet = [
                visitor.kind.value
                for visitor in request.visitors
                if not anchor.accepts(visitor)
            ]
            if unmet:
                if first_refusal is None:
                    first_refusal = Decline(
                        reason_code=DiagnosticCode.ANCHOR_DECLINED,
                        message=f"anchor {anchor.name} declined visitors: {unmet}",
                    )
                continue
            request_visitor_kinds = {v.kind for v in request.visitors}
            missing_required = [
                v.value
                for v in anchor.required_visitors
                if v not in request_visitor_kinds
            ]
            if missing_required:
                if first_refusal is None:
                    first_refusal = Decline(
                        reason_code=DiagnosticCode.ANCHOR_DECLINED,
                        message=(
                            f"anchor {anchor.name} requires missing visitors: "
                            f"{missing_required}"
                        ),
                    )
                continue
            return AnchorDecision(anchor=anchor, decline=None)
        return (
            AnchorDecision(anchor=None, decline=first_refusal)
            if first_refusal is not None
            else None
        )

    return _select


class AnchorRegistry:
    """Deterministic selection over registered anchors.

    Selectors run in registration order. A backend that cannot support a
    program must decline with a reason - silently changing semantics violates
    the URM charter.
    """

    def __init__(self) -> None:
        self._selectors: list[AnchorSelector] = []

    def register(self, selector: AnchorSelector) -> None:
        self._selectors.append(selector)

    def select(self, request: AnchorRequest) -> AnchorDecision:
        for selector in tuple(self._selectors):
            decision = selector(request)
            if decision is not None:
                return decision
        return AnchorDecision(
            anchor=None,
            decline=Decline(
                reason_code=DiagnosticCode.NO_ANCHOR_AVAILABLE,
                message=f"no registered selector answered request for {request.kind}",
            ),
        )


# -- Standard anchor catalog -------------------------------------------------
# Descriptors of the production families URM lowers onto. These carry no code;
# concrete kernels remain in urm.backends / urm.adapters / upstream packages.

SDM_EXTERNAL_ANCHOR_NAME = "facebook_sparse_delta_memory_183e7df_external_adapter"


TRUSTED_ANCHORS: tuple[ExecutionAnchor, ...] = (
    ExecutionAnchor(
        kind=AnchorKind.GEMM,
        name="torch_linear",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset({VisitorKind.FINAL_SCALE_CONVERT}),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name="flash_attention_adapter",
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_gated_delta_rule_adapter",
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.GROUPED_GEMM,
        name="grouped_gemm_reserved",
        trusted=False,  # reserved until the MoE comparator lands
    ),
    ExecutionAnchor(
        kind=AnchorKind.ROUTED_REDUCTION,
        name="routed_reduction_v1",
        # The frozen v1 kernel has no epilogue capability: a requested row-scale
        # epilogue must route to the experimental anchor instead.
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset({VisitorKind.SIDE_OUTPUT}),
        consumes_launch_config=False,
        schedulable=False,
    ),
    ExecutionAnchor(
        kind=AnchorKind.ROUTED_REDUCTION,
        name="routed_reduction_row_scale_epilogue_v0",
        # Compiler-generated fused-epilogue capability from the validated tranche.
        # Backward covers weights, values AND row scale via tile recomputation;
        # certification evidence lives in the rewrite contract and in
        # tests/test_compiler_epilogue_gpu.py. It becomes fully trusted only
        # while its differential and performance gates hold; v1 remains the
        # default without visitors.
        experimental=True,
        result_locality=LocalityConstraint(min=Locality.TILE, max=Locality.DEVICE),
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        honored_obligations=frozenset({"recompute_backward"}),
        supported_visitors=frozenset(
            {
                VisitorKind.FINAL_SCALE_CONVERT,
                VisitorKind.SIDE_OUTPUT,
                VisitorKind.PARTIAL_REDUCTION,
            }
        ),
        consumes_launch_config=True,
        schedulable=True,
        supported_plan_kinds=frozenset({"fused"}),
        required_visitors=frozenset({VisitorKind.FINAL_SCALE_CONVERT}),
        required_semantic_inputs=("row_scale",),
        supported_blocks=(32, 64, 128, 256),
        supported_warps=(1, 2, 4, 8),
        supported_stages=(1, 2, 4),
        supported_decompositions=("per_query", "per_route"),
        supported_schedules=("segmented", "full_row"),
    ),
    ExecutionAnchor(
        kind=AnchorKind.SPARSE_DELTA_MEMORY,
        name=SDM_EXTERNAL_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=False,
        commit_capable=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.PAGE_GATHER_UPDATE,
        name="page_gather_update_reserved",
        trusted=False,  # reserved for the SDM/GL-SDM slice
    ),
    ExecutionAnchor(
        kind=AnchorKind.COLLECTIVE_EXCHANGE,
        name="simulated_collective",
        commit_capable=True,
        supported_visitors=frozenset(),
    ),
)


def make_sdm_selector(
    anchor: ExecutionAnchor, support_probe: SDMSupportProbe | None = None
) -> AnchorSelector:
    """Runtime/revision-aware selector for the external original-SDM adapter."""

    def _select(request: AnchorRequest) -> AnchorDecision | None:
        if request.kind is not AnchorKind.SPARSE_DELTA_MEMORY:
            return None
        from urm.compiler.semantic import (
            MergePolicy,
            ScoreNormalization,
            SparseAddressingKind,
            SparseDeltaMemoryAccess,
            SparseReadTiming,
            SparseStatePolicy,
            SparseUpdateRule,
        )

        if not isinstance(request.semantic_op, SparseDeltaMemoryAccess):
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    "SDM anchor requires a typed SparseDeltaMemoryAccess operation",
                ),
            )
        spec = request.semantic_op.spec
        exact = (
            spec.addressing is SparseAddressingKind.PRODUCT_KEY_TOP_K
            and spec.normalization is ScoreNormalization.SOFTMAX
            and spec.update_rule is SparseUpdateRule.DECAYED_DELTA
            and spec.collision_policy is MergePolicy.ORDERED
            and spec.within_token_collision_policy is MergePolicy.REJECT
            and spec.read_timing is SparseReadTiming.AFTER_UPDATE
            and spec.state_policy is SparseStatePolicy.PERSISTENT_IN_PLACE
            and spec.page_size == 1
        )
        if not exact:
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    "SDM semantics do not match the frozen upstream contract",
                ),
            )
        probe = support_probe
        if probe is None:
            try:
                from urm.adapters.sparse_delta_memory import probe_sdm_support

                probe = probe_sdm_support
            except Exception as error:  # noqa: BLE001 - optional runtime must decline
                return AnchorDecision(
                    anchor=None,
                    decline=Decline(
                        DiagnosticCode.DEPENDENCY_MISSING,
                        f"original SDM adapter dependencies unavailable: {error!r}",
                    ),
                )
        support = probe()
        if not support.supported:
            code = {
                "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
                "incompatible_revision": DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
                "modified_upstream_checkout": DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
                "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
                "incompatible_runtime": DiagnosticCode.DEPENDENCY_MISSING,
            }.get(support.code, DiagnosticCode.ANCHOR_DECLINED)
            return AnchorDecision(
                anchor=None,
                decline=Decline(code, support.reason or support.code),
            )
        return AnchorDecision(anchor=anchor, decline=None)

    return _select


def default_registry() -> AnchorRegistry:
    registry = AnchorRegistry()
    sdm_anchor = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.kind is AnchorKind.SPARSE_DELTA_MEMORY
    )
    registry.register(make_sdm_selector(sdm_anchor))
    registry.register(make_selector(TRUSTED_ANCHORS))
    return registry
