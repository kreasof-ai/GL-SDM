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
    SPARSE_ROUTE_SELECTION = "sparse_route_selection"
    SPARSE_STATE_MIXER = "sparse_state_mixer"
    COLLECTIVE_EXCHANGE = "collective_exchange"


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


SDM_EXTERNAL_ANCHOR_NAME = "facebook_sparse_delta_memory_183e7df_external_adapter"
NATIVE_SPARSE_MEMORY_ANCHOR_NAME = "urm_native_sparse_memory_e2e_v0"
SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME = (
    "facebook_sparse_delta_memory_183e7df_precomputed_route_adapter"
)
NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME = "urm_native_sparse_state_mixer_v0"
NATIVE_SPARSE_ROUTE_ANCHOR_NAME = "urm_native_sparse_route_selection_v0"
NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME = "urm_native_diagonal_recurrence_v1"
NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME = "urm_native_matrix_state_recurrence_v1"
NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME = "urm_native_k1_online_softmax_v1"
MAMBA_SELECTIVE_SCAN_ANCHOR_NAME = "mamba_selective_scan_adapter"
MAMBA2_SSD_ANCHOR_NAME = "mamba2_ssd_adapter"
FLA_LOG_LINEAR_ANCHOR_NAME = "fla_chunk_log_linear_attention_adapter"
BDH_ATTENTION_ANCHOR_NAME = "bdh_attention_adapter"
FLA_KDA_ANCHOR_NAME = "fla_chunk_kda_adapter"
FLA_GATED_DELTA_PRODUCT_ANCHOR_NAME = "fla_chunk_gated_delta_product_adapter"
FLA_IPLR_ANCHOR_NAME = "fla_fused_recurrent_iplr_adapter"
FLA_DPLR_ANCHOR_NAME = "fla_chunk_dplr_adapter"
FLA_RWKV7_ANCHOR_NAME = "fla_chunk_rwkv7_adapter"
FLA_HGRN_ANCHOR_NAME = "fla_fused_recurrent_hgrn_adapter"
FLA_GDN2_ANCHOR_NAME = "fla_chunk_gdn2_adapter"
FLA_BASED_ANCHOR_NAME = "fla_fused_chunk_based_adapter"
FLA_REBASED_ANCHOR_NAME = "fla_parallel_rebased_adapter"
FLA_RWKV4_ANCHOR_NAME = "fla_fused_recurrent_rwkv4_adapter"
FLA_FORGETTING_ATTENTION_ANCHOR_NAME = "fla_parallel_forgetting_attention_adapter"
FLA_PARALLAX_ANCHOR_NAME = "fla_parallel_parallax_adapter"
FLA_WALL_ANCHOR_NAME = "fla_parallel_wall_attention_adapter"
FLA_MOBA_ANCHOR_NAME = "fla_parallel_moba_adapter"
FLA_ATTNRES_ANCHOR_NAME = "fla_fused_attnres_adapter"
FLA_RWKV6_ANCHOR_NAME = "fla_fused_recurrent_rwkv6_adapter"
FLA_MOMENTUM_DELTA_ANCHOR_NAME = "fla_chunk_momentum_delta_rule_adapter"
FLA_PATH_ATTENTION_ANCHOR_NAME = "fla_parallel_path_attention_adapter"
FLA_GATED_OJA_ANCHOR_NAME = "fla_chunk_gated_oja_adapter"
FLA_COMBA_ANCHOR_NAME = "fla_chunk_comba_adapter"
FLA_PGDN_ANCHOR_NAME = "fla_chunk_precond_gated_delta_adapter"
FLA_PKDA_ANCHOR_NAME = "fla_chunk_precond_kda_adapter"
FLA_ABC_ANCHOR_NAME = "fla_chunk_abc_adapter"
FLA_GSA_ANCHOR_NAME = "fla_chunk_gsa_adapter"
FLA_DELTAFORMER_ANCHOR_NAME = "fla_parallel_deltaformer_adapter"
FLA_MESA_NET_ANCHOR_NAME = "fla_chunk_mesa_net_adapter"
FLA_TITANS_LINEAR_ANCHOR_NAME = "fla_chunk_titans_linear_adapter"
FLA_TTT_LINEAR_ANCHOR_NAME = "fla_chunk_ttt_linear_adapter"
XMA_RNN_ANCHOR_NAME = "xma_rnn_triton_adapter"
XMA_GRU_ANCHOR_NAME = "xma_gru_triton_adapter"
XMA_M2RNN_ANCHOR_NAME = "xma_m2rnn_triton_adapter"
ATMA_POLAR_ANCHOR_NAME = "atma_polar_triton_adapter"
ATMA_POLAR_SPARSE_ANCHOR_NAME = "atma_polar_sparse_triton_adapter"
ATMA_GATED_DELTA_DECODE_ANCHOR_NAME = "atma_gated_delta_decode_adapter"
MAMBA3_SISO_ANCHOR_NAME = "mamba3_siso_combined_adapter"
TDA_TRITON_ANCHOR_NAME = "tda_triton_attention_adapter"
TUCKER_TRITON_ANCHOR_NAME = "tucker_triton_attention_adapter"
LONGFORMER_SLIDING_CHUNKS_ANCHOR_NAME = "longformer_sliding_chunks_adapter"
KATA_PARALLEL_TRITON_ANCHOR_NAME = "kata_parallel_triton_adapter"
FWPKM_SELECTED_SOFTMAX_ANCHOR_NAME = "fwpkm_selected_softmax_triton_adapter"
H3_SSM_FFT_ANCHOR_NAME = "h3_ssm_fft_convolution_adapter"
HYENA_FFT_ANCHOR_NAME = "hyena_fft_convolution_adapter"
HLA_SECOND_ORDER_ANCHOR_NAME = "hla_second_order_triton_adapter"


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
        kind=AnchorKind.ATTENTION,
        name="torch.nn.functional.scaled_dot_product_attention",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_FORGETTING_ATTENTION_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_PARALLAX_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_DELTAFORMER_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_WALL_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_MOBA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_ATTNRES_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=TDA_TRITON_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=TUCKER_TRITON_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=LONGFORMER_SLIDING_CHUNKS_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=KATA_PARALLEL_TRITON_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FWPKM_SELECTED_SOFTMAX_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=H3_SSM_FFT_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=HYENA_FFT_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=HLA_SECOND_ORDER_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_RWKV6_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_MOMENTUM_DELTA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=FLA_PATH_ATTENTION_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_GATED_OJA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_COMBA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_PGDN_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_PKDA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_ABC_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_GSA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_MESA_NET_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_TITANS_LINEAR_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_TTT_LINEAR_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=XMA_RNN_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=XMA_GRU_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=XMA_M2RNN_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=ATMA_POLAR_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=ATMA_POLAR_SPARSE_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=ATMA_GATED_DELTA_DECODE_ANCHOR_NAME,
        forward_only=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=MAMBA3_SISO_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name="urm.unified.k1.softmax_reference.v1",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_gated_delta_rule_adapter",
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_chunk_simple_gla_adapter",
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_chunk_gla_adapter",
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_fused_recurrent_simple_gla_decode_adapter",
        forward_only=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_fused_recurrent_gla_decode_adapter",
        forward_only=True,
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_chunk_linear_attention_adapter",
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="fla_chunk_delta_rule_adapter",
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name="urm.unified.k2.state_reference.v1",
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32", "float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=MAMBA_SELECTIVE_SCAN_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=MAMBA2_SSD_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_LOG_LINEAR_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=BDH_ATTENTION_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_KDA_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_GATED_DELTA_PRODUCT_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_IPLR_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_DPLR_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_RWKV7_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_HGRN_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_GDN2_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_BASED_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_REBASED_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
    ),
    ExecutionAnchor(
        kind=AnchorKind.RECURRENT_SCAN,
        name=FLA_RWKV4_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
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
        kind=AnchorKind.SPARSE_DELTA_MEMORY,
        name=NATIVE_SPARSE_MEMORY_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=False,
        commit_capable=True,
        supported_visitors=frozenset(),
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
        kind=AnchorKind.SPARSE_STATE_MIXER,
        name=SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=False,
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


def make_native_sparse_memory_selector(
    anchor: ExecutionAnchor,
    support_probe: Callable[[object], object] | None = None,
) -> AnchorSelector:
    """Prefer the fully native score-to-state pipeline when v0 can represent it."""

    def _select(request: AnchorRequest) -> AnchorDecision | None:
        if request.kind is not AnchorKind.SPARSE_DELTA_MEMORY:
            return None
        preferred = (request.schedule_params or {}).get("anchor_override")
        if preferred == SDM_EXTERNAL_ANCHOR_NAME:
            return None
        from urm.compiler.semantic import SparseMemoryAccess

        if not isinstance(request.semantic_op, SparseMemoryAccess):
            return None
        probe = support_probe
        if probe is None:
            try:
                from urm.backends.triton.sparse_state.memory import TritonSparseMemoryBackend

                probe = TritonSparseMemoryBackend.support_status
            except Exception as error:  # noqa: BLE001
                if preferred == NATIVE_SPARSE_MEMORY_ANCHOR_NAME:
                    return AnchorDecision(
                        anchor=None,
                        decline=Decline(
                            DiagnosticCode.DEPENDENCY_MISSING,
                            f"native sparse memory dependencies unavailable: {error!r}",
                        ),
                    )
                return None
        status = probe(request.semantic_op.spec)
        if status.supported:
            return AnchorDecision(anchor=anchor, decline=None)
        if preferred != NATIVE_SPARSE_MEMORY_ANCHOR_NAME:
            return None
        code = {
            "missing_dependency": DiagnosticCode.DEPENDENCY_MISSING,
            "unsupported_hardware": DiagnosticCode.UNSUPPORTED_HARDWARE,
            "unsupported_device": DiagnosticCode.UNSUPPORTED_HARDWARE,
            "unsupported_semantics": DiagnosticCode.UNSUPPORTED_SEMANTICS,
            "unsupported_shape": DiagnosticCode.ANCHOR_DECLINED,
        }.get(status.code, DiagnosticCode.ANCHOR_DECLINED)
        return AnchorDecision(
            anchor=None,
            decline=Decline(code, status.reason or status.code),
        )

    return _select


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
        from urm.compiler.semantic import SparseStateMixerAccess, UnifiedMixerAccess

        if isinstance(request.semantic_op, UnifiedMixerAccess):
            return None

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
                from urm.backends.triton.sparse_state.backend import (
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
        if (
            preferred != SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME
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
        from urm.compiler.semantic import SparseReadTiming, SparseStateOperation

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
            upstream_probe = fallback_support_probe
            if upstream_probe is None:
                try:
                    from urm.adapters.sparse_delta_memory import probe_sdm_support

                    upstream_probe = probe_sdm_support
                except Exception as error:  # noqa: BLE001
                    return AnchorDecision(
                        anchor=None,
                        decline=Decline(
                            DiagnosticCode.DEPENDENCY_MISSING,
                            f"pinned SDM fallback dependencies unavailable: {error!r}",
                        ),
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
        from urm.compiler.semantic import SparseRouteGeneration

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
                from urm.backends.triton.sparse_state.route_backend import TritonSparseRouteBackend

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


def default_registry() -> AnchorRegistry:
    registry = AnchorRegistry()
    sparse_route_anchor = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.kind is AnchorKind.SPARSE_ROUTE_SELECTION
    )
    registry.register(make_sparse_route_selector(sparse_route_anchor))
    native_sparse_memory_anchor = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.name == NATIVE_SPARSE_MEMORY_ANCHOR_NAME
    )
    registry.register(make_native_sparse_memory_selector(native_sparse_memory_anchor))
    sdm_anchor = next(
        anchor for anchor in TRUSTED_ANCHORS if anchor.name == SDM_EXTERNAL_ANCHOR_NAME
    )
    registry.register(make_sdm_selector(sdm_anchor))
    sparse_state_anchor = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.kind is AnchorKind.SPARSE_STATE_MIXER
    )
    sparse_state_fallback = next(
        anchor
        for anchor in TRUSTED_ANCHORS
        if anchor.name == SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME
    )
    registry.register(
        make_sparse_state_mixer_selector(
            sparse_state_anchor, fallback_anchor=sparse_state_fallback
        )
    )
    registry.register(make_selector(TRUSTED_ANCHORS))
    return registry
