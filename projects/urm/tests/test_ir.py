import pytest

from urm.ir import (
    BalanceStrategy,
    CollisionPolicy,
    Domain,
    ExpertRoutingSpec,
    ExpertScoreKind,
    ExpertSelection,
    MixerSpec,
    MutationKind,
    Normalization,
    RoutingKind,
    SelectionGranularity,
    SelectionScope,
    SparseAttentionSpec,
    SparseIndexerKind,
)
from urm.presets import (
    CATALOG,
    DEEPSEEK_SPARSE_ATTENTION,
    DEEPSEEK_V3_MOE,
    GATED_DELTANET2,
    KIMI_K3_LATENT_MOE,
    MINIMAX_SPARSE_ATTENTION,
    ROUTING_FREE_MOE,
)


def test_canonical_catalog_is_valid_and_named_uniquely() -> None:
    assert len(CATALOG) == 19
    assert len({spec.name for spec in CATALOG}) == len(CATALOG)


def test_advanced_presets_preserve_distinguishing_semantics() -> None:
    assert DEEPSEEK_V3_MOE.expert is not None
    assert DEEPSEEK_V3_MOE.expert.balance is BalanceStrategy.LOSS_FREE_BIAS
    assert KIMI_K3_LATENT_MOE.expert is not None
    assert KIMI_K3_LATENT_MOE.expert.balance is BalanceStrategy.QUANTILE_BIAS
    assert ROUTING_FREE_MOE.expert is not None
    assert ROUTING_FREE_MOE.expert.selection is ExpertSelection.THRESHOLD
    assert GATED_DELTANET2.recurrent is not None


def test_sparse_attention_presets_keep_granularity_and_scope() -> None:
    dsa = DEEPSEEK_SPARSE_ATTENTION.sparse_attention
    msa = MINIMAX_SPARSE_ATTENTION.sparse_attention
    assert dsa is not None and msa is not None
    assert dsa.granularity is SelectionGranularity.TOKEN
    assert dsa.scope is SelectionScope.SHARED_ACROSS_HEADS
    assert msa.granularity is SelectionGranularity.BLOCK
    assert msa.scope is SelectionScope.PER_GQA_GROUP
    assert msa.block_size == 128


def test_top_k_routing_requires_width() -> None:
    with pytest.raises(ValueError, match="top_k"):
        MixerSpec(
            name="invalid",
            query_domain=Domain.SEQUENCE,
            source_domain=Domain.EXPERT,
            routing=RoutingKind.TOP_K,
            normalization=Normalization.SOFTMAX,
        )


def test_memory_pages_require_page_size() -> None:
    with pytest.raises(ValueError, match="page_size"):
        MixerSpec(
            name="invalid",
            query_domain=Domain.SEQUENCE,
            source_domain=Domain.MEMORY_PAGE,
            routing=RoutingKind.DENSE,
            normalization=Normalization.SOFTMAX,
        )


def test_mutation_requires_collision_policy() -> None:
    with pytest.raises(ValueError, match="collision policy"):
        MixerSpec(
            name="invalid",
            query_domain=Domain.SEQUENCE,
            source_domain=Domain.EXPERT,
            routing=RoutingKind.TOP_K,
            normalization=Normalization.NONE,
            mutation=MutationKind.BUFFERED,
            top_k=1,
        )


def test_read_only_operator_rejects_collision_policy() -> None:
    with pytest.raises(ValueError, match="collision handling"):
        MixerSpec(
            name="invalid",
            query_domain=Domain.SEQUENCE,
            source_domain=Domain.SEQUENCE,
            routing=RoutingKind.DENSE,
            normalization=Normalization.SOFTMAX,
            collision_policy=CollisionPolicy.SUM,
        )


def test_threshold_expert_routing_has_dynamic_width() -> None:
    with pytest.raises(ValueError, match="dynamic active width"):
        ExpertRoutingSpec(
            routed_experts=8,
            active_routed_experts=2,
            score_kind=ExpertScoreKind.EXPERT_INTERNAL_NORM,
            selection=ExpertSelection.THRESHOLD,
        )


def test_learned_sparse_attention_width_must_match_mixer() -> None:
    with pytest.raises(ValueError, match="selected units"):
        MixerSpec(
            name="invalid",
            query_domain=Domain.SEQUENCE,
            source_domain=Domain.SEQUENCE,
            routing=RoutingKind.TOP_K,
            normalization=Normalization.SOFTMAX,
            top_k=8,
            sparse_attention=SparseAttentionSpec(
                indexer=SparseIndexerKind.LIGHTNING_TOKEN,
                granularity=SelectionGranularity.TOKEN,
                scope=SelectionScope.SHARED_ACROSS_HEADS,
                selected_units=4,
            ),
        )
