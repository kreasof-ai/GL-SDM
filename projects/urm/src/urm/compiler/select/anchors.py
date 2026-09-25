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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from urm.compiler.common.diagnostics import DiagnosticCode
from urm.ir.effects import ORDERED_STATE, PURE, EffectSignature
from urm.compiler.placement.locality import Locality, LocalityConstraint
from urm.ir.program import (
    DType,
    MergePolicy,
    SparseReadTiming,
    SparseStateLayout,
    SparseStateMixerSpec,
    SparseStateOperation,
    SparseStatePolicy,
    SparseUpdateRule,
)


class AnchorKind(StrEnum):
    """Trusted execution-anchor classes."""

    GEMM = "gemm"
    ATTENTION = "attention"
    RECURRENT_SCAN = "recurrent_scan"
    GROUPED_GEMM = "grouped_gemm"
    ROUTED_REDUCTION = "routed_reduction"
    PAGE_GATHER_UPDATE = "page_gather_update"
    SPARSE_ROUTE_SELECTION = "sparse_route_selection"
    SPARSE_STATE_MIXER = "sparse_state_mixer"
    COLLECTIVE_EXCHANGE = "collective_exchange"
    # Ordinary typed operators (cross-call composition merge, A14) — a typed
    # linear combination of producer outputs, unfused by default.
    MERGE = "merge"
    # Strict-causal triangular operand correction (UT) — the data-dependent
    # forward-substitution solve u = (I + diag(β)·strict_tril(P))^{-1}·v.
    TRIANGULAR_SOLVE = "triangular_solve"
    # Banked dyadic hierarchical state (A4) — a bank of K2-matrix states with a
    # carry-cascade promote/reset lifecycle (the log-linear mixer).
    DYADIC_BANKED_STATE = "dyadic_banked_state"


class VisitorKind(StrEnum):
    """Constrained visitor vocabulary exposed by anchors."""

    ELEMENTWISE_MAP = "elementwise_map"
    PAIRWISE_MAP = "pairwise_map"
    VECTOR_LOAD_STORE = "vector_load_store"
    TILE_LOAD_STORE = "tile_load_store"
    PARTIAL_REDUCTION = "partial_reduction"
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
    element_dtype: str
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
    trusted: bool = True
    experimental: bool = False
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
    semantic_contracts: frozenset[str] = frozenset()
    """Equation contracts this anchor implements (e.g.
    ``normalized_softmax_attention_v1``). An anchor with an empty set is
    unconstrained; a non-empty set is matched against the typed node's equation
    before selection, so an incompatible equation (Polar for plain softmax MHA)
    declines rather than silently selecting."""

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
            from urm.compiler.schedule.space import (
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
    equation_contract: str | None = None
    """The equation the typed node computes (e.g.
    ``normalized_softmax_attention_v1``). Anchors declaring a non-empty
    ``semantic_contracts`` set must contain it to be selectable."""


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
            # Semantic legality gate: an anchor that declares the equations it
            # implements must contain this node's equation. This is what makes a
            # forced Polar anchor decline a plain softmax MHA node.
            if (
                anchor.semantic_contracts
                and request.equation_contract is not None
                and request.equation_contract not in anchor.semantic_contracts
            ):
                if first_refusal is None:
                    first_refusal = Decline(
                        reason_code=DiagnosticCode.ANCHOR_DECLINED,
                        message=(
                            f"anchor {anchor.name} does not implement equation "
                            f"{request.equation_contract!r}"
                        ),
                    )
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

    def register_anchors(self, anchors: Sequence[ExecutionAnchor]) -> None:
        """Register additional anchor instances under a generic selector.

        This is the consumer-extension point: the core ships only URM-owned
        anchors; external/architecture-named providers (FLA, ATMA, Mamba, ...)
        are registered here by the consumer that provisions them, alongside
        their executors. The same legality gate applies to registered anchors.
        """
        self._selectors.append(make_selector(tuple(anchors)))

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


# Injectable capability probe for the pinned SDM upstream checkout, used by the
# generic K3 sparse-state fallback. The compiler core never imports the
# comparator package; the consumer that provisions the SDM checkout installs the
# probe via ``set_sdm_support_probe``. When unset, the fallback declines with
# DEPENDENCY_MISSING rather than importing a comparator.
_SDM_SUPPORT_PROBE: Callable[[], object] | None = None


def set_sdm_support_probe(probe: Callable[[], object] | None) -> None:
    """Install (or clear) the pinned-SDM support probe used by the K3 fallback."""
    global _SDM_SUPPORT_PROBE
    _SDM_SUPPORT_PROBE = probe


def _default_sdm_support_probe() -> object:
    if _SDM_SUPPORT_PROBE is None:
        return _SdmProbeUnavailable(
            "missing_dependency",
            "no SDM support probe is installed; the consumer must provision the "
            "pinned checkout and call set_sdm_support_probe",
        )
    return _SDM_SUPPORT_PROBE()


@dataclass(frozen=True, slots=True)
class _SdmProbeUnavailable:
    code: str
    reason: str
    supported: bool = False
NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME = "urm_native_sparse_state_mixer_v0"
NATIVE_SPARSE_ROUTE_ANCHOR_NAME = "urm_native_sparse_route_selection_v0"
NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME = "urm_native_diagonal_recurrence_v1"
NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME = "urm_native_matrix_state_recurrence_v1"
NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME = "urm_native_k1_online_softmax_v1"
NATIVE_DYADIC_BANKED_STATE_ANCHOR_NAME = "urm_native_dyadic_banked_state_v1"
NATIVE_TRIANGULAR_SOLVE_ANCHOR_NAME = "urm_native_triangular_solve_v1"


# ---------------------------------------------------------------------------
# K3 native capability + schedule policy (moved from the retired urm.ir.k3)
#
# This is selection data, not IR: the frozen v0 capability envelope and the
# deterministic launch policy of the native K3 lowering. It is deliberately
# free of GPU-runtime imports so plan-time code can serialize the schedule on
# a GPU-less machine; the backend that honors it lives in
# :mod:`urm.backends.triton.k3`.
# ---------------------------------------------------------------------------


class SparseStateCapabilityEnvelope:
    """Pre-tuning limits of the first native A10G-oriented lowering.

    ``maximum_parallel`` admission note: the v0 bound of 16 was pre-tuning
    conservatism, not a kernel constraint — the parallel dim P is pure data
    parallelism (banks never interact: the read grid is ``(P, T, D-blocks)``, the
    update grid is ``(P, D-blocks)``, no cross-bank data flow). Admitted to 4096 on
    the evidence of ``test_native_mixer_parallel_dim_scaling``: native-vs-reference
    parity is exact (fp32, ~1e-7) at P ∈ {1 … 160} including the training-harness
    point P=80 (B=8 × H=10), and wall time is flat in P (P=4→32 identical, P=80
    within 10%) — the GPU absorbs the banks concurrently. 4096 keeps a declared cap
    against accidental unbounded launches while covering any realistic harness config.
    """

    schema_version: int = 0
    minimum_compute_capability: tuple[int, int] = (8, 0)
    maximum_parallel: int = 4096
    maximum_sequence: int = 2048
    maximum_slots_per_partition: int = 1_048_576
    maximum_value_dim: int = 1024
    maximum_route_width: int = 64
    supported_dtypes: tuple[DType, ...] = (DType.FLOAT32, DType.BFLOAT16)
    supported_index_dtypes: tuple[str, ...] = ("int32", "int64")


FROZEN_V0_ENVELOPE = SparseStateCapabilityEnvelope()


@dataclass(frozen=True, slots=True)
class SparseStateSupportStatus:
    supported: bool
    code: str
    reason: str | None = None
    details: Mapping[str, object] | None = None

    @classmethod
    def yes(cls) -> SparseStateSupportStatus:
        return cls(True, "supported")

    @classmethod
    def no(cls, code: str, reason: str, **details: object) -> SparseStateSupportStatus:
        return cls(False, code, reason, details)

    def require(self) -> None:
        if not self.supported:
            raise ValueError(
                f"{NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME} declined [{self.code}]: "
                f"{self.reason}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "code": self.code,
            "reason": self.reason,
            "details": dict(self.details or {}),
        }


def sparse_state_spec_status(
    spec: SparseStateMixerSpec,
    *,
    device_type: str,
    index_dtype: str = "int64",
    contiguous: bool = True,
    compute_capability: tuple[int, int] | None = None,
) -> SparseStateSupportStatus:
    """Return a structured v0 capability decision without importing Torch."""
    envelope = FROZEN_V0_ENVELOPE
    exact_semantics = (
        spec.update_rule is SparseUpdateRule.DECAYED_DELTA
        and spec.collision_policy is MergePolicy.ORDERED
        and spec.within_token_collision_policy is MergePolicy.REJECT
        and spec.state_policy is SparseStatePolicy.PERSISTENT_IN_PLACE
        and spec.accumulation_dtype is DType.FLOAT32
        and spec.state_layout is SparseStateLayout.PARTITION_SLOT_VALUE
        and spec.page_size == 1
    )
    if not exact_semantics:
        return SparseStateSupportStatus.no(
            "unsupported_semantics", "operation is outside the frozen v0 algebra"
        )
    if spec.operation is SparseStateOperation.READ_ONLY:
        if spec.read_timing is not SparseReadTiming.CURRENT_STATE:
            return SparseStateSupportStatus.no(
                "unsupported_semantics", "read-only mode must read current state"
            )
    elif spec.read_timing not in {
        SparseReadTiming.BEFORE_UPDATE,
        SparseReadTiming.AFTER_UPDATE,
    }:
        return SparseStateSupportStatus.no(
            "unsupported_semantics", "update read timing is unsupported"
        )
    if spec.dtype not in envelope.supported_dtypes:
        return SparseStateSupportStatus.no(
            "unsupported_dtype", f"dtype {spec.dtype.value} is outside v0"
        )
    if device_type != "cuda":
        return SparseStateSupportStatus.no(
            "unsupported_device", "v0 native execution requires CUDA"
        )
    if not contiguous:
        return SparseStateSupportStatus.no(
            "unsupported_layout", "v0 requires contiguous logical tensors"
        )
    if index_dtype not in envelope.supported_index_dtypes:
        return SparseStateSupportStatus.no(
            "unsupported_dtype", f"index dtype {index_dtype} is outside v0"
        )
    if compute_capability is not None and compute_capability < (
        envelope.minimum_compute_capability
    ):
        return SparseStateSupportStatus.no(
            "unsupported_hardware",
            "v0 requires SM80 or newer",
            found_compute_capability=compute_capability,
        )
    limits = {
        "parallel": (spec.parallel, envelope.maximum_parallel),
        "sequence": (spec.sequence, envelope.maximum_sequence),
        "slots_per_partition": (
            spec.slots_per_partition,
            envelope.maximum_slots_per_partition,
        ),
        "value_dim": (spec.value_dim, envelope.maximum_value_dim),
        "writes": (spec.writes, envelope.maximum_route_width),
        "reads": (spec.reads, envelope.maximum_route_width),
    }
    exceeded = {
        name: {"requested": requested, "maximum": maximum}
        for name, (requested, maximum) in limits.items()
        if requested > maximum
    }
    if exceeded:
        return SparseStateSupportStatus.no(
            "unsupported_shape",
            "one or more dimensions exceed the predeclared v0 bounds",
            exceeded=exceeded,
        )
    return SparseStateSupportStatus.yes()


def sparse_state_launch_parameters(value_dim: int) -> tuple[int, int]:
    """Shared production schedule; D=64 retains the reviewed D=4 fragment."""
    if value_dim == 64:
        return 4, 2
    block = max(16, 1 << (min(value_dim, 256) - 1).bit_length())
    return block, 8 if block >= 256 else 4 if block >= 64 else 2


def sparse_state_launch_schedule(spec: SparseStateMixerSpec) -> dict[str, str | int]:
    """Serialize the deterministic v0 schedule without importing a GPU runtime."""
    block_d, warps = sparse_state_launch_parameters(spec.value_dim)
    return {
        "schedule_family": "partition_owned_ordered_token_scan",
        "block_d": block_d,
        "num_warps": warps,
        "num_stages": 3,
        "tokens_per_program": -1,
        "read_timing": spec.read_timing.value,
        "state_layout": spec.state_layout.value,
    }


TRUSTED_ANCHORS: tuple[ExecutionAnchor, ...] = (
    ExecutionAnchor(
        kind=AnchorKind.GEMM,
        name="torch_linear",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset({VisitorKind.FINAL_SCALE_CONVERT}),
    ),
    ExecutionAnchor(
        kind=AnchorKind.MERGE,
        name="torch.merge.linear_combination.v1",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.TRIANGULAR_SOLVE,
        name="urm.unified.triangular_solve.reference.v1",
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    # The native Triton triangular solve runs the exact forward substitution
    # (one program per batch·head × D-block walks the token axis in order) and
    # its adjoint backward substitution, fp32 accumulation throughout; the
    # scalar/row cotangents accumulate across D-blocks through relaxed atomics
    # (the K3 native backward policy). Parity-qualified against the Torch
    # reference, forward AND cotangents.
    ExecutionAnchor(
        kind=AnchorKind.TRIANGULAR_SOLVE,
        name=NATIVE_TRIANGULAR_SOLVE_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32"}),
        deterministic_accumulation=False,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.DYADIC_BANKED_STATE,
        name="urm.unified.dyadic_banked_state.reference.v1",
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    # The native Triton banked dyadic recurrence (decay-forward form, fp32
    # accumulation; forward AND cotangent parity vs the pinned torch reference
    # gated in tests/test_native_dyadic_banked_state.py). The backward reduces
    # per-token operand cotangents across value blocks with relaxed atomics, so
    # cross-program accumulation order is not guaranteed.
    ExecutionAnchor(
        kind=AnchorKind.DYADIC_BANKED_STATE,
        name=NATIVE_DYADIC_BANKED_STATE_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32"}),
        deterministic_accumulation=False,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name="torch.nn.functional.scaled_dot_product_attention",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
        semantic_contracts=frozenset({"normalized_softmax_attention_v1"}),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name="urm.unified.k1.softmax_reference.v1",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
        # The Torch reference K1 executor dispatches on every admitted reducer
        # law (softmax, threshold, squared-sum, map_normalize), so it accepts
        # each non-softmax contract the descriptor can produce. The native and
        # SDPA anchors declare only softmax and decline these.
        semantic_contracts=frozenset(
            {
                "normalized_softmax_attention_v1",
                "k1_threshold_relu_power_v1",
                "k1_squared_sum_v1",
                "k1_map_normalize_v1",
            }
        ),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
        semantic_contracts=frozenset({"normalized_softmax_attention_v1"}),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="urm.unified.k2.state_reference.v1",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
        # The Torch reference K2 executor dispatches every admitted contract:
        # the canonical law plus the normalized, elementwise-gate, and
        # generalized-transition (A8) forms.
        semantic_contracts=frozenset(
            {
                "k2_canonical_linear_delta_v1",
                "k2_normalized_v1",
                "k2_elementwise_gate_v1",
                "k2_generalized_transition_v1",
            }
        ),
    ),
    # The native Triton K2 scan implements the canonical law plus the parity-qualified
    # A8 generalized transitions (dual-gate / multi-rank / retrieval-key / low-rank left
    # transition — measured forward AND cotangent parity vs the pinned laws and the
    # Torch reference: 1.5e-5 / 7.6e-5 / 1.9e-6 / 1.4e-6 forward). It declines the
    # normalized variant (denominator law diverges: y/max(q·z,ε) vs the pinned
    # y/((scale·q·norm)+ε), measured 1.9e7) and the elementwise gate (single-client,
    # reference-tier per the two-client rule) via the equation-contract gate.
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
        semantic_contracts=frozenset(
            {"k2_canonical_linear_delta_v1", "k2_generalized_transition_v1"}
        ),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
        semantic_contracts=frozenset(
            {"k2_canonical_linear_delta_v1", "k2_generalized_transition_v1"}
        ),
    ),
    ExecutionAnchor(
        kind=AnchorKind.GROUPED_GEMM,
        name="grouped_gemm_reserved",
        trusted=False,
    ),
    ExecutionAnchor(
        kind=AnchorKind.ROUTED_REDUCTION,
        name="routed_reduction_v1",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset({VisitorKind.SIDE_OUTPUT}),
        consumes_launch_config=False,
        schedulable=False,
    ),
    ExecutionAnchor(
        kind=AnchorKind.ROUTED_REDUCTION,
        name="routed_reduction_row_scale_epilogue_v0",
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
        kind=AnchorKind.SPARSE_ROUTE_SELECTION,
        name=NATIVE_SPARSE_ROUTE_ANCHOR_NAME,
        effect=PURE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.SPARSE_STATE_MIXER,
        name=NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=False,
        commit_capable=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.SPARSE_STATE_MIXER,
        name="urm.unified.k3.sparse_delta_reference.v1",
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=True,
        commit_capable=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.PAGE_GATHER_UPDATE,
        name="page_gather_update_reserved",
        trusted=False,
    ),
    ExecutionAnchor(
        kind=AnchorKind.COLLECTIVE_EXCHANGE,
        name="simulated_collective",
        commit_capable=True,
        supported_visitors=frozenset(),
    ),)


def make_sparse_state_mixer_selector(
    anchor: ExecutionAnchor,
    support_probe: Callable[[object], object] | None = None,
    fallback_anchor: ExecutionAnchor | None = None,
    fallback_support_probe: SDMSupportProbe | None = None,
) -> AnchorSelector:
    """Prefer native v0, then retain the pinned external route-state fallback."""

    def _select(request: AnchorRequest) -> AnchorDecision | None:
        if request.kind is not AnchorKind.SPARSE_STATE_MIXER:
            return None
        from urm.ir.program import SparseStateMixerAccess

        if not isinstance(request.semantic_op, SparseStateMixerAccess):
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    "native SparseStateMixer requires a typed SparseStateMixerAccess",
                ),
            )
        native_probe = support_probe
        if native_probe is None:
            try:
                from urm.backends.triton.k3 import (
    TritonSparseStateMixerBackend,
)

                native_probe = TritonSparseStateMixerBackend.support_status
            except Exception as error:  # noqa: BLE001 - optional GPU runtime
                return AnchorDecision(
                    anchor=None,
                    decline=Decline(
                        DiagnosticCode.DEPENDENCY_MISSING,
                        f"native SparseStateMixer dependencies unavailable: {error!r}",
                    ),
                )
        spec = request.semantic_op.spec
        preferred = (request.schedule_params or {}).get("anchor_override")
        native_status = native_probe(spec)
        # The pinned external fallback (registered by the comparator consumer) is
        # selected only by explicit override; the native anchor wins otherwise.
        if (
            preferred
            != "facebook_sparse_delta_memory_183e7df_precomputed_route_adapter"
            and native_status.supported
        ):
            return AnchorDecision(anchor=anchor, decline=None)
        if preferred == NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME:
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    {
                        "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
                        "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
                        "unsupported_device": DiagnosticCode.UNSUPPORTED_HARDWARE,
                        "unsupported_semantics": DiagnosticCode.UNSUPPORTED_SEMANTICS,
                        "unsupported_dtype": DiagnosticCode.UNSUPPORTED_SEMANTICS,
                        "unsupported_layout": DiagnosticCode.UNSUPPORTED_SEMANTICS,
                        "unsupported_shape": DiagnosticCode.ANCHOR_DECLINED,
                    }.get(native_status.code, DiagnosticCode.ANCHOR_DECLINED),
                    native_status.reason or native_status.code,
                ),
            )
        from urm.ir.program import SparseReadTiming, SparseStateOperation

        square_root = int(spec.slots_per_partition**0.5)
        fallback_semantics = (
            fallback_anchor is not None
            and spec.slots_per_partition >= 8
            and spec.slots_per_partition % 8 == 0
            and square_root * square_root == spec.slots_per_partition
            and spec.reads <= 128
            and spec.writes <= 128
            and (
                spec.operation is SparseStateOperation.READ_ONLY
                or spec.read_timing is SparseReadTiming.AFTER_UPDATE
            )
            and (spec.mode.value != "training" or spec.sequence >= 16)
        )
        if fallback_semantics:
            upstream_probe = (
                fallback_support_probe
                if fallback_support_probe is not None
                else _default_sdm_support_probe
            )
            upstream_status = upstream_probe()
            if upstream_status.supported:
                return AnchorDecision(anchor=fallback_anchor, decline=None)
            code = {
                "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
                "incompatible_revision": DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
                "modified_upstream_checkout": DiagnosticCode.UPSTREAM_REVISION_MISMATCH,
                "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
                "incompatible_runtime": DiagnosticCode.DEPENDENCY_MISSING,
            }.get(upstream_status.code, DiagnosticCode.ANCHOR_DECLINED)
            return AnchorDecision(
                anchor=None,
                decline=Decline(code, upstream_status.reason or upstream_status.code),
            )
        if not native_status.supported:
            code = {
                "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
                "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
                "unsupported_device": DiagnosticCode.UNSUPPORTED_HARDWARE,
                "unsupported_semantics": DiagnosticCode.UNSUPPORTED_SEMANTICS,
                "unsupported_dtype": DiagnosticCode.UNSUPPORTED_SEMANTICS,
                "unsupported_layout": DiagnosticCode.UNSUPPORTED_SEMANTICS,
                "unsupported_shape": DiagnosticCode.ANCHOR_DECLINED,
            }.get(native_status.code, DiagnosticCode.ANCHOR_DECLINED)
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    code,
                    (native_status.reason or native_status.code)
                    + "; pinned external fallback cannot represent this shape/semantics",
                ),
            )
        return AnchorDecision(
            anchor=None,
            decline=Decline(
                DiagnosticCode.ANCHOR_DECLINED,
                "requested external fallback cannot represent this shape/semantics",
            ),
        )

    return _select


def make_sparse_route_selector(
    anchor: ExecutionAnchor,
    support_probe: Callable[[object], object] | None = None,
) -> AnchorSelector:
    """Select only the independently typed native route-production lowering."""

    def _select(request: AnchorRequest) -> AnchorDecision | None:
        if request.kind is not AnchorKind.SPARSE_ROUTE_SELECTION:
            return None
        from urm.ir.program import SparseRouteGeneration

        if not isinstance(request.semantic_op, SparseRouteGeneration):
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    DiagnosticCode.UNSUPPORTED_SEMANTICS,
                    "native sparse route anchor requires SparseRouteGeneration",
                ),
            )
        preferred = (request.schedule_params or {}).get("anchor_override")
        if preferred not in {None, NATIVE_SPARSE_ROUTE_ANCHOR_NAME}:
            return AnchorDecision(
                anchor=None,
                decline=Decline(
                    DiagnosticCode.ANCHOR_DECLINED,
                    f"override {preferred!r} is not a sparse route anchor",
                ),
            )
        probe = support_probe
        if probe is None:
            try:
                from urm.backends.triton.k3 import TritonSparseRouteBackend

                probe = TritonSparseRouteBackend.support_status
            except Exception as error:  # noqa: BLE001
                return AnchorDecision(
                    anchor=None,
                    decline=Decline(
                        DiagnosticCode.DEPENDENCY_MISSING,
                        f"native sparse route dependencies unavailable: {error!r}",
                    ),
                )
        status = probe(request.semantic_op.spec)
        if status.supported:
            return AnchorDecision(anchor=anchor, decline=None)
        code = {
            "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
            "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
            "unsupported_shape": DiagnosticCode.ANCHOR_DECLINED,
            "unsupported_semantics": DiagnosticCode.UNSUPPORTED_SEMANTICS,
        }.get(status.code, DiagnosticCode.ANCHOR_DECLINED)
        return AnchorDecision(
            anchor=None,
            decline=Decline(code, status.reason or status.code),
        )

    return _select


# Consumer-registered anchor providers. The core ships only URM-owned anchors;
# external/architecture-named providers (FLA, ATMA, Mamba, the frozen SDM
# upstream, ...) are registered here by the consumer that provisions them
# (``benchmarks.comparators.anchors``), alongside their executors. Registration
# is additive and order-preserving; the same legality gate applies.
_PROVIDER_ANCHORS: list[ExecutionAnchor] = []
_PROVIDER_SELECTORS: list[AnchorSelector] = []


def register_anchor_provider(anchors: Sequence[ExecutionAnchor]) -> None:
    """Register external anchor declarations for future default registries."""
    _PROVIDER_ANCHORS.extend(anchors)


def register_anchor_selector(selector: AnchorSelector) -> None:
    """Register an external anchor selector (e.g. a revision-aware SDM probe)."""
    _PROVIDER_SELECTORS.append(selector)


def default_registry() -> AnchorRegistry:
    registry = AnchorRegistry()
    sparse_route_anchor = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.kind is AnchorKind.SPARSE_ROUTE_SELECTION
    )
    registry.register(make_sparse_route_selector(sparse_route_anchor))
    sparse_state_anchor = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.kind is AnchorKind.SPARSE_STATE_MIXER
    )
    registry.register(make_sparse_state_mixer_selector(sparse_state_anchor))
    registry.register(make_selector(TRUSTED_ANCHORS))
    # Consumer-installed selectors (e.g. the SDM revision probe) run before the
    # consumer-registered anchor catalog so pinned-source semantics win.
    for selector in _PROVIDER_SELECTORS:
        registry.register(selector)
    if _PROVIDER_ANCHORS:
        registry.register(make_selector(tuple(_PROVIDER_ANCHORS)))
    return registry
