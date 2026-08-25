"""Canonical specs used by tests and benchmark adapters."""

from .ir import (
    BalanceStrategy,
    CapacityPolicy,
    CollisionPolicy,
    DecayGranularity,
    Domain,
    EditGateKind,
    ExpertFunction,
    ExpertRoutingSpec,
    ExpertScoreKind,
    ExpertSelection,
    MixerSpec,
    MutationKind,
    Normalization,
    RecurrentAlgorithm,
    RecurrentSpec,
    Residency,
    RoutingKind,
    ScanMode,
    ScoreActivation,
    SelectionGranularity,
    SelectionScope,
    SparseAttentionSpec,
    SparseIndexerKind,
    StateLayout,
)

DENSE_ATTENTION = MixerSpec(
    name="dense_attention",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.SEQUENCE,
    routing=RoutingKind.DENSE,
    normalization=Normalization.SOFTMAX,
)

BLOCK_SPARSE_ATTENTION = MixerSpec(
    name="block_sparse_attention",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.SEQUENCE,
    routing=RoutingKind.BLOCK_SPARSE,
    normalization=Normalization.SOFTMAX,
)

DEEPSEEK_SPARSE_ATTENTION = MixerSpec(
    name="deepseek_sparse_attention",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.SEQUENCE,
    routing=RoutingKind.TOP_K,
    normalization=Normalization.SOFTMAX,
    top_k=2048,
    sparse_attention=SparseAttentionSpec(
        indexer=SparseIndexerKind.LIGHTNING_TOKEN,
        granularity=SelectionGranularity.TOKEN,
        scope=SelectionScope.SHARED_ACROSS_HEADS,
        selected_units=2048,
    ),
)

MINIMAX_SPARSE_ATTENTION = MixerSpec(
    name="minimax_sparse_attention",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.SEQUENCE,
    routing=RoutingKind.TOP_K,
    normalization=Normalization.SOFTMAX,
    top_k=16,
    sparse_attention=SparseAttentionSpec(
        indexer=SparseIndexerKind.DOT_PRODUCT_BLOCK,
        granularity=SelectionGranularity.BLOCK,
        scope=SelectionScope.PER_GQA_GROUP,
        selected_units=16,
        block_size=128,
        forced_local_units=1,
    ),
)

TOP2_MOE = MixerSpec(
    name="top2_moe",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.EXPERT,
    routing=RoutingKind.TOP_K,
    normalization=Normalization.SOFTMAX,
    top_k=2,
)

DEEPSEEK_MOE_16B = MixerSpec(
    name="deepseek_moe_16b",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.EXPERT,
    routing=RoutingKind.TOP_K,
    normalization=Normalization.NONE,
    top_k=6,
    expert=ExpertRoutingSpec(
        routed_experts=64,
        active_routed_experts=6,
        shared_experts=2,
        score_activation=ScoreActivation.SOFTMAX,
        balance=BalanceStrategy.AUXILIARY_EXPERT_AND_DEVICE,
        expert_width_ratio=0.25,
    ),
)

DEEPSEEK_V3_MOE = MixerSpec(
    name="deepseek_v3_moe",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.EXPERT,
    routing=RoutingKind.TOP_K,
    normalization=Normalization.L1,
    top_k=8,
    residency=Residency.SHARDED,
    expert=ExpertRoutingSpec(
        routed_experts=256,
        active_routed_experts=8,
        shared_experts=1,
        score_activation=ScoreActivation.SIGMOID,
        capacity_policy=CapacityPolicy.DROPLESS,
        balance=BalanceStrategy.LOSS_FREE_BIAS,
        groups=8,
        groups_selected=4,
    ),
)

ROUTING_FREE_MOE = MixerSpec(
    name="routing_free_moe",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.EXPERT,
    routing=RoutingKind.THRESHOLD,
    normalization=Normalization.NONE,
    threshold=0.0,
    expert=ExpertRoutingSpec(
        routed_experts=64,
        active_routed_experts=None,
        score_kind=ExpertScoreKind.EXPERT_INTERNAL_NORM,
        score_activation=ScoreActivation.RELU,
        selection=ExpertSelection.THRESHOLD,
        balance=BalanceStrategy.ADAPTIVE_DENSITY_AUXILIARY,
    ),
)

KIMI_K3_LATENT_MOE = MixerSpec(
    name="kimi_k3_stable_latent_moe",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.EXPERT,
    routing=RoutingKind.TOP_K,
    normalization=Normalization.L1,
    top_k=16,
    residency=Residency.SHARDED,
    expert=ExpertRoutingSpec(
        routed_experts=896,
        active_routed_experts=16,
        shared_experts=2,
        score_kind=ExpertScoreKind.LATENT_LINEAR_ROUTER,
        score_activation=ScoreActivation.SIGMOID,
        balance=BalanceStrategy.QUANTILE_BIAS,
        expert_function=ExpertFunction.SITU_GLU,
        latent_width_ratio=0.5,
    ),
)

PARAMETER_TOKEN_MIXER = MixerSpec(
    name="parameter_token_mixer",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.PARAMETER_BLOCK,
    routing=RoutingKind.DENSE,
    normalization=Normalization.SOFTMAX,
)

LINEAR_RECURRENT_MIXER = MixerSpec(
    name="linear_recurrent_mixer",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.RECURRENT_STATE,
    routing=RoutingKind.KERNELIZED_RECURRENCE,
    normalization=Normalization.NONE,
    mutation=MutationKind.IN_PLACE_RECURRENT,
)


def _recurrent_mixer(
    name: str,
    algorithm: RecurrentAlgorithm,
    state_layout: StateLayout,
    decay: DecayGranularity,
    *,
    edit_gate: EditGateKind = EditGateKind.NONE,
    short_convolution: bool = True,
) -> MixerSpec:
    return MixerSpec(
        name=name,
        query_domain=Domain.SEQUENCE,
        source_domain=Domain.RECURRENT_STATE,
        routing=RoutingKind.KERNELIZED_RECURRENCE,
        normalization=Normalization.NONE,
        mutation=MutationKind.IN_PLACE_RECURRENT,
        recurrent=RecurrentSpec(
            algorithm=algorithm,
            state_layout=state_layout,
            decay_granularity=decay,
            edit_gate=edit_gate,
            scan_modes=(ScanMode.RECURRENT, ScanMode.CHUNKWISE),
            short_convolution=short_convolution,
        ),
    )


MAMBA = _recurrent_mixer(
    "mamba",
    RecurrentAlgorithm.MAMBA,
    StateLayout.DIAGONAL_SSM,
    DecayGranularity.CHANNEL,
)

MAMBA2 = _recurrent_mixer(
    "mamba2",
    RecurrentAlgorithm.MAMBA2,
    StateLayout.SSD_MATRIX,
    DecayGranularity.HEAD,
)

MAMBA3 = _recurrent_mixer(
    "mamba3",
    RecurrentAlgorithm.MAMBA3,
    StateLayout.COMPLEX_MIMO_SSM,
    DecayGranularity.HEAD,
    short_convolution=False,
)

GATED_DELTANET = _recurrent_mixer(
    "gated_deltanet",
    RecurrentAlgorithm.GATED_DELTANET,
    StateLayout.FAST_WEIGHT_MATRIX,
    DecayGranularity.HEAD,
    edit_gate=EditGateKind.COUPLED_SCALAR,
)

GATED_DELTANET2 = _recurrent_mixer(
    "gated_deltanet2",
    RecurrentAlgorithm.GATED_DELTANET2,
    StateLayout.FAST_WEIGHT_MATRIX,
    DecayGranularity.CHANNEL,
    edit_gate=EditGateKind.SEPARATE_CHANNEL_ERASE_WRITE,
)

KIMI_DELTA_ATTENTION = _recurrent_mixer(
    "kimi_delta_attention",
    RecurrentAlgorithm.KIMI_DELTA_ATTENTION,
    StateLayout.FAST_WEIGHT_MATRIX,
    DecayGranularity.CHANNEL,
    edit_gate=EditGateKind.COUPLED_SCALAR,
)

SPARSE_DELTA_MEMORY = MixerSpec(
    name="sparse_delta_memory",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.MEMORY_PAGE,
    routing=RoutingKind.PRODUCT_KEY,
    normalization=Normalization.SOFTMAX,
    mutation=MutationKind.IN_PLACE_RECURRENT,
    residency=Residency.HBM,
    collision_policy=CollisionPolicy.ORDERED,
    top_k=64,
    page_size=1,
)

GL_SDM_TRANSACTION = MixerSpec(
    name="gl_sdm_transaction",
    query_domain=Domain.SEQUENCE,
    source_domain=Domain.MEMORY_PAGE,
    routing=RoutingKind.PRODUCT_KEY,
    normalization=Normalization.SOFTMAX,
    mutation=MutationKind.TRANSACTIONAL,
    residency=Residency.HBM,
    collision_policy=CollisionPolicy.SUM,
    top_k=8,
    page_size=128,
)

CATALOG = (
    DENSE_ATTENTION,
    BLOCK_SPARSE_ATTENTION,
    DEEPSEEK_SPARSE_ATTENTION,
    MINIMAX_SPARSE_ATTENTION,
    TOP2_MOE,
    DEEPSEEK_MOE_16B,
    DEEPSEEK_V3_MOE,
    ROUTING_FREE_MOE,
    KIMI_K3_LATENT_MOE,
    PARAMETER_TOKEN_MIXER,
    LINEAR_RECURRENT_MIXER,
    MAMBA,
    MAMBA2,
    MAMBA3,
    GATED_DELTANET,
    GATED_DELTANET2,
    KIMI_DELTA_ATTENTION,
    SPARSE_DELTA_MEMORY,
    GL_SDM_TRANSACTION,
)
