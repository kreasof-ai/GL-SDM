"""Restricted, declarative operator contract for routed mixers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Domain(StrEnum):
    SEQUENCE = "sequence"
    EXPERT = "expert"
    PARAMETER_BLOCK = "parameter_block"
    RECURRENT_STATE = "recurrent_state"
    MEMORY_PAGE = "memory_page"


class RoutingKind(StrEnum):
    DENSE = "dense"
    BLOCK_SPARSE = "block_sparse"
    TOP_K = "top_k"
    THRESHOLD = "threshold"
    PRODUCT_KEY = "product_key"
    KERNELIZED_RECURRENCE = "kernelized_recurrence"


class Normalization(StrEnum):
    NONE = "none"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    L1 = "l1"


class MutationKind(StrEnum):
    NONE = "none"
    IN_PLACE_RECURRENT = "in_place_recurrent"
    BUFFERED = "buffered"
    TRANSACTIONAL = "transactional"


class Residency(StrEnum):
    DEVICE = "device"
    SRAM_STAGED = "sram_staged"
    HBM = "hbm"
    SHARDED = "sharded"


class CollisionPolicy(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    ORDERED = "ordered"
    SUM = "sum"
    MEAN = "mean"
    LAST_WRITE = "last_write"
    REJECT = "reject"


class ScoreActivation(StrEnum):
    NONE = "none"
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    RELU = "relu"


class ExpertScoreKind(StrEnum):
    LINEAR_ROUTER = "linear_router"
    EXPERT_INTERNAL_NORM = "expert_internal_norm"
    LATENT_LINEAR_ROUTER = "latent_linear_router"


class ExpertSelection(StrEnum):
    TOP_K = "top_k"
    THRESHOLD = "threshold"


class CapacityPolicy(StrEnum):
    DROPLESS = "dropless"
    FIXED_CAPACITY = "fixed_capacity"
    EXPERT_QUOTA = "expert_quota"


class BalanceStrategy(StrEnum):
    NONE = "none"
    AUXILIARY_EXPERT = "auxiliary_expert"
    AUXILIARY_EXPERT_AND_DEVICE = "auxiliary_expert_and_device"
    ADAPTIVE_DENSITY_AUXILIARY = "adaptive_density_auxiliary"
    LOSS_FREE_BIAS = "loss_free_bias"
    QUANTILE_BIAS = "quantile_bias"


class ExpertFunction(StrEnum):
    FFN = "ffn"
    SWIGLU = "swiglu"
    SITU_GLU = "situ_glu"


@dataclass(frozen=True, slots=True)
class ExpertRoutingSpec:
    """MoE choices that cannot be inferred from a bare top-k operation."""

    routed_experts: int
    active_routed_experts: int | None
    shared_experts: int = 0
    score_kind: ExpertScoreKind = ExpertScoreKind.LINEAR_ROUTER
    score_activation: ScoreActivation = ScoreActivation.NONE
    selection: ExpertSelection = ExpertSelection.TOP_K
    capacity_policy: CapacityPolicy = CapacityPolicy.DROPLESS
    balance: BalanceStrategy = BalanceStrategy.NONE
    expert_function: ExpertFunction = ExpertFunction.SWIGLU
    expert_width_ratio: float = 1.0
    latent_width_ratio: float = 1.0
    routing_bias_affects_weights: bool = False
    groups: int = 1
    groups_selected: int | None = None

    def __post_init__(self) -> None:
        if self.routed_experts <= 0:
            raise ValueError("routed_experts must be positive")
        if self.shared_experts < 0:
            raise ValueError("shared_experts must not be negative")
        if self.selection is ExpertSelection.TOP_K:
            if (
                self.active_routed_experts is None
                or self.active_routed_experts <= 0
                or self.active_routed_experts > self.routed_experts
            ):
                raise ValueError(
                    "top-k expert selection requires active_routed_experts in range"
                )
        elif self.active_routed_experts is not None:
            raise ValueError("threshold expert selection has dynamic active width")
        if self.expert_width_ratio <= 0 or self.latent_width_ratio <= 0:
            raise ValueError("expert and latent width ratios must be positive")
        if self.groups <= 0 or self.routed_experts % self.groups:
            raise ValueError("groups must evenly partition routed experts")
        if self.groups_selected is not None and not (
            0 < self.groups_selected <= self.groups
        ):
            raise ValueError("groups_selected must be in [1, groups]")


class RecurrentAlgorithm(StrEnum):
    MAMBA = "mamba"
    MAMBA2 = "mamba2"
    MAMBA3 = "mamba3"
    GATED_DELTANET = "gated_deltanet"
    GATED_DELTANET2 = "gated_deltanet2"
    KIMI_DELTA_ATTENTION = "kimi_delta_attention"


class StateLayout(StrEnum):
    DIAGONAL_SSM = "diagonal_ssm"
    SSD_MATRIX = "ssd_matrix"
    COMPLEX_MIMO_SSM = "complex_mimo_ssm"
    FAST_WEIGHT_MATRIX = "fast_weight_matrix"


class DecayGranularity(StrEnum):
    SCALAR = "scalar"
    HEAD = "head"
    CHANNEL = "channel"


class EditGateKind(StrEnum):
    NONE = "none"
    COUPLED_SCALAR = "coupled_scalar"
    SEPARATE_CHANNEL_ERASE_WRITE = "separate_channel_erase_write"


class ScanMode(StrEnum):
    RECURRENT = "recurrent"
    CHUNKWISE = "chunkwise"
    PARALLEL = "parallel"


@dataclass(frozen=True, slots=True)
class RecurrentSpec:
    """Closed recurrent-family descriptor; the exact equation stays backend-owned."""

    algorithm: RecurrentAlgorithm
    state_layout: StateLayout
    decay_granularity: DecayGranularity
    edit_gate: EditGateKind = EditGateKind.NONE
    scan_modes: tuple[ScanMode, ...] = (ScanMode.RECURRENT,)
    short_convolution: bool = True

    def __post_init__(self) -> None:
        if not self.scan_modes:
            raise ValueError("at least one scan mode is required")
        if len(set(self.scan_modes)) != len(self.scan_modes):
            raise ValueError("scan_modes must not contain duplicates")


class SparseIndexerKind(StrEnum):
    STATIC_MASK = "static_mask"
    LIGHTNING_TOKEN = "lightning_token"
    DOT_PRODUCT_BLOCK = "dot_product_block"


class SelectionGranularity(StrEnum):
    TOKEN = "token"
    BLOCK = "block"


class SelectionScope(StrEnum):
    SHARED_ACROSS_HEADS = "shared_across_heads"
    PER_HEAD = "per_head"
    PER_GQA_GROUP = "per_gqa_group"


@dataclass(frozen=True, slots=True)
class SparseAttentionSpec:
    """Indexer and sparse-layout choices preceding exact main attention."""

    indexer: SparseIndexerKind
    granularity: SelectionGranularity
    scope: SelectionScope
    selected_units: int | None = None
    block_size: int | None = None
    forced_local_units: int = 0
    exact_main_attention: bool = True

    def __post_init__(self) -> None:
        if self.selected_units is not None and self.selected_units <= 0:
            raise ValueError("selected_units must be positive")
        if self.granularity is SelectionGranularity.BLOCK:
            if self.block_size is None or self.block_size <= 0:
                raise ValueError("block selection requires a positive block_size")
        elif self.block_size is not None:
            raise ValueError("token selection must not declare block_size")
        if self.forced_local_units < 0:
            raise ValueError("forced_local_units must not be negative")
        if (
            self.selected_units is not None
            and self.forced_local_units > self.selected_units
        ):
            raise ValueError("forced local units cannot exceed selected units")


@dataclass(frozen=True, slots=True)
class MixerSpec:
    """Compile-time semantic choices exposed to a URM backend.

    The IR intentionally describes routing and state behavior, not arbitrary
    tensor programs. Shapes and tensors remain runtime inputs.
    """

    name: str
    query_domain: Domain
    source_domain: Domain
    routing: RoutingKind
    normalization: Normalization
    mutation: MutationKind = MutationKind.NONE
    residency: Residency = Residency.DEVICE
    collision_policy: CollisionPolicy = CollisionPolicy.NOT_APPLICABLE
    deterministic: bool = True
    top_k: int | None = None
    page_size: int | None = None
    threshold: float | None = None
    expert: ExpertRoutingSpec | None = None
    recurrent: RecurrentSpec | None = None
    sparse_attention: SparseAttentionSpec | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")

        needs_top_k = self.routing in {RoutingKind.TOP_K, RoutingKind.PRODUCT_KEY}
        if needs_top_k and (self.top_k is None or self.top_k <= 0):
            raise ValueError(f"{self.routing} routing requires top_k > 0")
        if not needs_top_k and self.top_k is not None:
            raise ValueError(f"{self.routing} routing must not declare top_k")

        if self.routing is RoutingKind.THRESHOLD:
            if self.threshold is None:
                raise ValueError("threshold routing requires a threshold")
        elif self.threshold is not None:
            raise ValueError(f"{self.routing} routing must not declare threshold")

        if self.page_size is not None and self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.source_domain is Domain.MEMORY_PAGE and self.page_size is None:
            raise ValueError("memory-page sources require page_size")

        needs_collision_policy = self.mutation in {
            MutationKind.BUFFERED,
            MutationKind.TRANSACTIONAL,
        } or (
            self.mutation is MutationKind.IN_PLACE_RECURRENT
            and self.source_domain is Domain.MEMORY_PAGE
        )
        has_collision_policy = (
            self.collision_policy is not CollisionPolicy.NOT_APPLICABLE
        )
        if needs_collision_policy and not has_collision_policy:
            raise ValueError(
                "sparse recurrent, buffered, and transactional operators require "
                "a collision policy"
            )
        if not needs_collision_policy and has_collision_policy:
            raise ValueError(
                "collision handling is only valid for sparse recurrent, buffered, "
                "or transactional operators"
            )

        if (
            self.routing is RoutingKind.KERNELIZED_RECURRENCE
            and self.source_domain is not Domain.RECURRENT_STATE
        ):
            raise ValueError("kernelized recurrence requires recurrent-state sources")

        detail_count = sum(
            detail is not None
            for detail in (self.expert, self.recurrent, self.sparse_attention)
        )
        if detail_count > 1:
            raise ValueError("a mixer may declare only one family detail spec")

        if self.expert is not None:
            if self.source_domain is not Domain.EXPERT:
                raise ValueError("expert details require expert sources")
            expected_routing = (
                RoutingKind.TOP_K
                if self.expert.selection is ExpertSelection.TOP_K
                else RoutingKind.THRESHOLD
            )
            if self.routing is not expected_routing:
                raise ValueError("expert selection and mixer routing disagree")
            if (
                self.expert.active_routed_experts is not None
                and self.top_k != self.expert.active_routed_experts
            ):
                raise ValueError("expert active width and mixer top_k disagree")

        if self.recurrent is not None and (
            self.routing is not RoutingKind.KERNELIZED_RECURRENCE
            or self.source_domain is not Domain.RECURRENT_STATE
        ):
            raise ValueError("recurrent details require recurrent-state routing")

        if self.sparse_attention is not None:
            if self.source_domain is not Domain.SEQUENCE:
                raise ValueError("sparse-attention details require sequence sources")
            if self.sparse_attention.indexer is SparseIndexerKind.STATIC_MASK:
                if self.routing is not RoutingKind.BLOCK_SPARSE:
                    raise ValueError("static sparse attention requires block routing")
            else:
                if self.routing is not RoutingKind.TOP_K:
                    raise ValueError("learned sparse attention requires top-k routing")
                if self.top_k != self.sparse_attention.selected_units:
                    raise ValueError("sparse selected units and mixer top_k disagree")
