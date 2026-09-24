"""Upstream/architecture-named execution anchors, owned by the comparator suite.

These are pinned source-integration capability declarations (FLA, ATMA, Mamba,
XMA, Tucker, KATA, Longformer, BDH, H3, Hyena, TDA, FwPKM, FlashAttention, and
the frozen SDM upstream). They are NOT part of URM core: the core compiler ships
only URM-owned anchors and a generic provider interface. This module declares the
upstream capabilities and registers them (plus their executors, via
``executors.register_all``) so the compiler can select them when a consumer
provisions the corresponding source checkout.

Importing this module does not import any upstream library; it only builds the
capability declarations. Provisioning and executor registration happen in
``benchmarks.comparators.executors``.
"""

from __future__ import annotations

from urm.compiler.select.anchors import (
    AnchorKind,
    ExecutionAnchor,
    VisitorKind,
)
from urm.compiler.placement.locality import Locality, LocalityConstraint
from urm.ir.effects import ORDERED_STATE

SDM_EXTERNAL_ANCHOR_NAME = "facebook_sparse_delta_memory_183e7df_external_adapter"
SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME = (
    "facebook_sparse_delta_memory_183e7df_precomputed_route_adapter"
)
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


UPSTREAM_ANCHORS: tuple[ExecutionAnchor, ...] = (
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name="flash_attention_adapter",
        backward_verified_dtypes=frozenset({"float16", "bfloat16"}),
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
        semantic_contracts=frozenset({"polar_attention_v1"}),
    ),
    ExecutionAnchor(
        kind=AnchorKind.ATTENTION,
        name=ATMA_POLAR_SPARSE_ANCHOR_NAME,
        backward_verified_dtypes=frozenset({"float32"}),
        supported_visitors=frozenset(),
        semantic_contracts=frozenset({"polar_attention_v1"}),
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
        kind=AnchorKind.SPARSE_STATE_MIXER,
        name=SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME,
        effect=ORDERED_STATE,
        backward_verified_dtypes=frozenset({"float32", "bfloat16"}),
        deterministic_accumulation=False,
        commit_capable=True,
        supported_visitors=frozenset(),
    ),
)


def register_anchor_providers() -> None:
    """Register the upstream anchor declarations with the compiler core.

    Called by ``benchmarks.comparators.executors.register_all`` so the compiler's
    default registry includes these providers when the comparator suite is in use.
    Also installs the pinned-SDM revision-aware selector and the SDM sparse-state
    fallback selector, which reference the consumer-owned SDM anchors.
    """
    from urm.compiler.select.anchors import (
        make_sparse_state_mixer_selector,
        register_anchor_provider,
        register_anchor_selector,
    )

    register_anchor_provider(UPSTREAM_ANCHORS)

    # SDM sparse-state fallback: native anchor stays in core; the fallback is the
    # consumer-owned pinned SDM route adapter.
    from urm.compiler.select.anchors import (
        NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,
        TRUSTED_ANCHORS,
    )

    native_state = next(
        a for a in TRUSTED_ANCHORS if a.name == NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME
    )
    sdm_fallback = next(
        a for a in UPSTREAM_ANCHORS if a.name == SDM_SPARSE_STATE_FALLBACK_ANCHOR_NAME
    )
    register_anchor_selector(
        make_sparse_state_mixer_selector(native_state, fallback_anchor=sdm_fallback)
    )
