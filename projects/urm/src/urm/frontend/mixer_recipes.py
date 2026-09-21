"""Architecture recipes lowered into backend independent mixer semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace

from urm.ir.mixer import (
    DecayGranularity,
    FeatureMap,
    K1Operation,
    MixerKernelFamily,
    PolynomialBasis,
    ReadTiming,
    RecurrenceOperator,
    RecurrentLayout,
    StateEffect,
    StateNormalizer,
    StateTransition,
    StateUpdateRule,
    UnifiedMixerSpec,
)


MIXER_RECIPE_NAMES = (
    "mha",
    "mqa",
    "gqa",
    "polar_attention_core",
    "foveal_sparse_polar_attention_core",
    "fox",
    "wall_attention_core",
    "sparse_attention_core",
    "cat_attention_core",
    "differential_attention_core",
    "tda_attention_core",
    "moba_selected_attention_core",
    "bdh_attention_core",
    "nsa_selected_attention_core",
    "dsa_attention_core",
    "mla_attention_core",
    "path_attention_core",
    "deltaformer_attention_core",
    "parallax_attention_core",
    "foveal_attention_core",
    "attnres_depth_core",
    "pattention_core",
    "tpa_attention_core",
    "tucker_attention_core",
    "longformer_attention_core",
    "kata_attention_core",
    "conformer_attention_core",
    "hopfield_attention_core",
    "fwpkm_memory_read_core",
    "samba_attention_core",
    "h3_ssm_fft_core",
    "hyena_fftconv_core",
    "hla_second_order_core",
    "linear_attention",
    "based_attention_core",
    "rebased_attention_core",
    "retention_core",
    "lightning_attention_core",
    "lightnet_gla_core",
    "simple_gla",
    "gla",
    "rodimus_gla_core",
    "mom_selected_memory_core",
    "hgrn_ssm_core",
    "hgrn2_ssm_core",
    "delta_net",
    "gated_delta_net",
    "atma_gated_delta_decode_core",
    "gdn2_core",
    "gated_oja_core",
    "comba_core",
    "pgdn_core",
    "pkda_core",
    "abc_core",
    "gsa_core",
    "kda_core",
    "gated_delta_product_core",
    "generalized_delta_iplr_core",
    "generalized_delta_dplr_core",
    "rwkv4_memory_core",
    "rwkv6_memory_core",
    "momentum_delta_core",
    "mesa_net_core",
    "rwkv7_transition_core",
    "mamba1_ssm_core",
    "mamba2_ssm_core",
    "mamba3_siso_core",
    "titans_linear_memory_core",
    "ttt_linear_core",
    "rnn_core",
    "gru_core",
    "m2rnn_core",
    "log_linear_attention_core",
    "sparse_delta_memory",
)

@dataclass(frozen=True, slots=True)
class MixerRecipe:
    """A named kernel-level mapping with its model boundary made explicit."""

    architecture_ids: tuple[str, ...]
    spec: UnifiedMixerSpec
    component_scope: str
    required_external_stages: tuple[str, ...] = ()


def softmax_attention_spec(
    name: str = "softmax_attention",
    *,
    causal: bool = True,
    scale: float | None = None,
    score_bias: bool = False,
    requires_attention_mask: bool = False,
    operation: K1Operation = K1Operation.SOFTMAX,
) -> UnifiedMixerSpec:
    return UnifiedMixerSpec(
        name=name,
        family=MixerKernelFamily.SOFTMAX,
        causal=causal,
        attention_scale=scale,
        accepts_score_bias=score_bias,
        requires_attention_mask=requires_attention_mask,
        k1_operation=operation,
    )


def linear_attention_spec(
    name: str = "linear_attention",
    *,
    feature_map: FeatureMap = FeatureMap.ELU_PLUS_ONE,
    normalized: bool = True,
) -> UnifiedMixerSpec:
    return UnifiedMixerSpec(
        name=name,
        family=MixerKernelFamily.RECURRENCE,
        update_rule=StateUpdateRule.ADDITIVE,
        normalizer=(StateNormalizer.QUERY_KEY if normalized else StateNormalizer.NONE),
        feature_map=feature_map,
    )


def delta_rule_spec(
    name: str = "delta_rule",
    *,
    decay: DecayGranularity = DecayGranularity.NONE,
    read_timing: ReadTiming = ReadTiming.AFTER_UPDATE,
) -> UnifiedMixerSpec:
    return UnifiedMixerSpec(
        name=name,
        family=MixerKernelFamily.RECURRENCE,
        update_rule=StateUpdateRule.DELTA,
        decay=decay,
        read_timing=read_timing,
    )


def diagonal_ssm_spec(
    name: str = "diagonal_ssm",
    *,
    step_size_discretization: bool = False,
    hgrn: bool = False,
) -> UnifiedMixerSpec:
    return UnifiedMixerSpec(
        name=name,
        family=MixerKernelFamily.RECURRENCE,
        recurrent_layout=RecurrentLayout.DIAGONAL,
        update_rule=StateUpdateRule.ADDITIVE,
        decay=DecayGranularity.ELEMENTWISE,
        step_size_discretization=step_size_discretization,
        diagonal_hgrn=hgrn,
    )


def sparse_delta_spec(
    name: str = "sparse_delta_memory",
    *,
    read_timing: ReadTiming = ReadTiming.AFTER_UPDATE,
) -> UnifiedMixerSpec:
    return UnifiedMixerSpec(
        name=name,
        family=MixerKernelFamily.SPARSE_DELTA,
        read_timing=read_timing,
    )


def named_mixer_recipe(name: str) -> MixerRecipe:
    """Return a conservative kernel-level recipe for a common mixer family.

    Recipes cover the named mixer equation or state primitive.  Their
    ``required_external_stages`` explicitly lists work outside that primitive;
    a recipe is not full-layer or parity qualification.
    """
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    recipes: dict[str, MixerRecipe] = {}

    attention = softmax_attention_spec("softmax_attention", score_bias=True)
    for alias in ("mha", "mqa", "gqa", "transformer_attention", "fox"):
        ids = {
            "mha": ("arch-001", "arch-014"),
            "mqa": ("arch-002",),
            "gqa": ("arch-003",),
            "transformer_attention": ("arch-001", "arch-002", "arch-003"),
            "fox": ("arch-008",),
        }[alias]
        recipes[alias] = MixerRecipe(
            ids,
            replace(
                attention,
                name=alias,
                k1_operation=(
                    K1Operation.FORGETTING
                    if alias == "fox"
                    else K1Operation.SOFTMAX
                ),
            ),
            (
                "FoX causal softmax attention under per-token log-decay gates"
                if alias == "fox"
                else "causal softmax attention core with explicit additive score bias"
            ),
            (
                ("Q/K/V projections", "FoX forget-gate production")
                if alias == "fox"
                else ("Q/K/V projections", "position/forget-gate bias construction")
            ),
        )
    recipes["sparse_attention_core"] = MixerRecipe(
        ("arch-006", "arch-072"),
        replace(
            attention,
            name="sparse_attention_core",
            requires_attention_mask=True,
        ),
        "exact softmax attention for a caller-supplied boolean/additive mask",
        ("architecture-specific indexer/selection", "sparse traversal kernel"),
    )
    recipes["longformer_attention_core"] = MixerRecipe(
        ("arch-071",),
        softmax_attention_spec(
            "longformer_attention_core", causal=False, score_bias=False,
            operation=K1Operation.LOCAL_WINDOW,
        ),
        "bidirectional local-window softmax attention using Longformer's sliding-chunks K1 operator",
        (
            "Q/K/V projections and output projection",
            "global-token attention and padding-mask routing",
            "Longformer encoder composition and cache integration",
        ),
    )
    recipes["kata_attention_core"] = MixerRecipe(
        ("arch-073",),
        softmax_attention_spec(
            "kata_attention_core", score_bias=False,
            operation=K1Operation.POSITIVE_FEATURE,
        ),
        "causal KATA normalized positive attention with summed squared group dot products",
        (
            "KATA Q/K/V projections and feature normalization",
            "KATA model layer and cache/varlen integration",
        ),
    )
    recipes["conformer_attention_core"] = MixerRecipe(
        ("arch-075",),
        softmax_attention_spec(
            "conformer_attention_core", causal=False, score_bias=False
        ),
        "bidirectional Conformer self-attention K1 core",
        (
            "Conformer convolution module and positionwise feed-forward block",
            "encoder frontend, positional encoding and complete layer residual graph",
        ),
    )
    recipes["hopfield_attention_core"] = MixerRecipe(
        ("arch-078",),
        softmax_attention_spec("hopfield_attention_core", causal=False, scale=1.0),
        "single-update modern Hopfield association core as unscaled softmax attention",
        (
            "Hopfield iterative retrieval beyond the first association update",
            "memory-specific projections, normalization, masks and output composition",
        ),
    )
    recipes["fwpkm_memory_read_core"] = MixerRecipe(
        ("arch-056",),
        softmax_attention_spec(
            "fwpkm_memory_read_core", causal=False, scale=1.0, score_bias=False,
            operation=K1Operation.SELECTED_READ,
        ),
        "FwPKM product-key-selected memory read with normalized learned retrieval weights",
        (
            "product-key factor scoring, Cartesian top-k route generation and selected key/value gather",
            "fast-weight memory write/update, optimizer state and chunk-boundary semantics",
        ),
    )
    recipes["samba_attention_core"] = MixerRecipe(
        ("arch-053",),
        softmax_attention_spec("samba_attention_core", causal=True, score_bias=False),
        "Samba_421M_nope attention branch with source QKV/output projections and causal K1",
        (
            "hybrid Mamba/GLA/Retention layer selection and recurrent state/cache paths",
            "Samba Block normalization, residual, MLP, rotary-enabled and short-convolution branches",
        ),
    )
    recipes["h3_ssm_fft_core"] = MixerRecipe(
        ("arch-076",),
        UnifiedMixerSpec(
            "h3_ssm_fft_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION,
        ),
        "H3 head_dim=1 multiplicative SSM mixer with two causal FFT convolutions and skip paths",
        (
            "SSKernel parameterization and convolution-kernel generation",
            "H3 Q/K/V and output projections, inference state/cache and head_dim>1 mode",
        ),
    )
    recipes["hyena_fftconv_core"] = MixerRecipe(
        ("arch-077",),
        UnifiedMixerSpec(
            "hyena_fftconv_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.FFT_CONVOLUTION,
        ),
        "Hyena implicit-filter causal FFT convolution with a learned direct term",
        (
            "Hyena implicit-filter generation, short convolution, multiplicative order/gating and projections",
            "higher-order filter-bank routing, streaming cache and variable-length filter policy",
        ),
    )
    recipes["hla_second_order_core"] = MixerRecipe(
        ("arch-074",),
        UnifiedMixerSpec(
            "hla_second_order_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.SECOND_ORDER_CUMSUM,
        ),
        "HLA masked second-order causal attention with exact streaming summaries",
        (
            "Higher-order (>2) HLA and asymmetric/decayed variants",
            "Transformer projections, chunk-scan scheduling and streaming cache ABI",
        ),
    )
    recipes["cat_attention_core"] = MixerRecipe(
        ("arch-066",),
        softmax_attention_spec(
            "cat_attention_core", score_bias=False, requires_attention_mask=True
        ),
        "CAT Compress And Attend causal attention over prior compressed tokens and the current local block",
        (
            "chunk compression and compressed-token construction",
            "separator/adaptive tokens, rotary transform and Q/K/V projections",
            "compressor transformer and complete CAT decoder layer",
        ),
    )
    recipes["differential_attention_core"] = MixerRecipe(
        ("arch-067",),
        softmax_attention_spec(
            "differential_attention_core", score_bias=False,
            operation=K1Operation.DIFFERENTIAL,
        ),
        "Differential Transformer V1 paired causal softmax reductions and learned subtraction weight",
        (
            "differential Q/K/V projections and RoPE",
            "lambda-vector parameterization, per-head RMSNorm, output scale and projection",
        ),
    )
    recipes["tda_attention_core"] = MixerRecipe(
        ("arch-068",),
        softmax_attention_spec(
            "tda_attention_core", score_bias=False,
            operation=K1Operation.THRESHOLDED,
        ),
        "Threshold Differential Attention's pair of causal rectified score reductions",
        (
            "TDA threshold beta and lambda production",
            "Q/K normalization and complete projection/output layer",
        ),
    )
    recipes["polar_attention_core"] = MixerRecipe(
        ("arch-064",),
        softmax_attention_spec(
            "polar_attention_core", score_bias=False, operation=K1Operation.POLAR
        ),
        "ATMA Polar causal direction and bounded-magnitude reduction with a learned null sink",
        (
            "Q/K/V projections, GQA expansion, canonical convolution, and output/count projections",
        ),
    )
    recipes["foveal_sparse_polar_attention_core"] = MixerRecipe(
        ("arch-065",),
        softmax_attention_spec(
            "foveal_sparse_polar_attention_core", score_bias=False,
            operation=K1Operation.POLAR_SPARSE,
        ),
        "ATMA Foveal local-window plus selected remote-page Polar reduction",
        (
            "geometric page routing and its gradients, projections, GQA expansion, and full attention layer",
        ),
    )
    recipes["nsa_selected_attention_core"] = MixerRecipe(
        ("arch-005",),
        softmax_attention_spec(
            "nsa_selected_attention_core", requires_attention_mask=True
        ),
        "NSA selected-block causal attention from caller-supplied block routes",
        (
            "NSA compression and indexer routes",
            "multi-branch gate composition and full layer",
        ),
    )
    recipes["moba_selected_attention_core"] = MixerRecipe(
        ("arch-006",),
        softmax_attention_spec(
            "moba_selected_attention_core",
            score_bias=False,
            requires_attention_mask=True,
            operation=K1Operation.BLOCK_ROUTED,
        ),
        "MoBA causal attention over local and selected KV blocks",
        ("block-score route selection", "sparse traversal and full layer"),
    )
    recipes["bdh_attention_core"] = MixerRecipe(
        ("arch-063",),
        UnifiedMixerSpec(
            "bdh_attention_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            normalizer=StateNormalizer.NONE,
            read_timing=ReadTiming.BEFORE_UPDATE,
            recurrence_operator=RecurrenceOperator.EXTERNAL_OPAQUE,
        ),
        "BDH rotary unnormalized causal attention as a strict-past additive K2 matrix recurrence",
        (
            "BDH encoder projections and ReLU",
            "layer normalization",
            "value/MLP projections and gating",
            "dropout and model-level residual composition",
        ),
    )
    recipes["mom_selected_memory_core"] = MixerRecipe(
        ("arch-051",),
        UnifiedMixerSpec(
            "mom_selected_memory_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.HEAD,
            feature_map=FeatureMap.L2_NORMALIZE,
            state_v_first=True,
        ),
        "MoM per-route gated-delta memory update with source Q/K L2 normalization and V-first state",
        (
            "learned memory router and top-k selection",
            "route packing/dispatch and result merge",
            "memory expert projections, convolutions and full layer/cache composition",
        ),
    )
    recipes["dsa_attention_core"] = MixerRecipe(
        ("arch-007",),
            softmax_attention_spec("dsa_attention_core", requires_attention_mask=True),
        "causal softmax attention restricted to caller-supplied DSA token indices",
        (
            "DSA indexer objective and token selection",
            "model projections and full layer",
        ),
    )
    recipes["deltaformer_attention_core"] = MixerRecipe(
        ("arch-013",),
        replace(
            softmax_attention_spec(
                "deltaformer_attention_core", score_bias=False,
                operation=K1Operation.DELTA_TRANSFORM,
            ),
            deltaformer_attention=True,
        ),
        "causal softmax attention after the DeltaFormer lower-triangular value correction",
        ("Q/K/V/beta projections", "rotary embeddings", "output projection"),
    )
    for alias, ids, stages in (
        (
            "mla_attention_core",
            ("arch-004",),
            ("latent-KV projection", "positional branch and cache ABI"),
        ),
        (
            "path_attention_core",
            ("arch-010",),
            ("model-side w/beta/g generation, convolution and projections",),
        ),
        (
            "wall_attention_core",
            ("arch-011",),
            ("per-channel decay scoring and its gate gradients",),
        ),
        (
            "parallax_attention_core",
            ("arch-012",),
            ("architecture-specific positional/state transformation",),
        ),
        (
            "foveal_attention_core",
            ("arch-065",),
            ("geometric selection/index construction", "sparse traversal kernel"),
        ),
        (
            "pattention_core",
            ("arch-057",),
            ("parameter-token construction and complete parameter gradients",),
        ),
        (
            "tpa_attention_core",
            ("arch-069",),
            ("TPA factorized Q/K/V production, RoPE, and compressed-cache ABI",),
        ),
        (
            "tucker_attention_core",
            ("arch-070",),
            (
                "Tucker factor construction, output factor contraction, and complete layer",
            ),
        ),
    ):
        recipes[alias] = MixerRecipe(
            ids,
            softmax_attention_spec(
                alias,
                score_bias=False,
                operation=(
                    K1Operation.POSITIONAL
                    if alias == "parallax_attention_core"
                    else K1Operation.GATED
                ),
            )
            if alias in {"parallax_attention_core", "wall_attention_core"}
            else softmax_attention_spec(
                alias,
                causal=alias != "tucker_attention_core",
                score_bias=False,
                operation=(
                    K1Operation.PROJECTED
                    if alias == "tucker_attention_core"
                    else K1Operation.SOFTMAX
                ),
            )
            if alias in {"tpa_attention_core", "tucker_attention_core"}
            else replace(
                softmax_attention_spec(
                    alias, score_bias=False,
                    operation=K1Operation.PATH_TRANSFORM,
                ),
                path_attention=True,
            )
            if alias == "path_attention_core"
            else softmax_attention_spec(
                alias, causal=False, scale=1.0, score_bias=False
            )
            if alias == "pattention_core"
            else replace(attention, name=alias),
            (
                "causal Parallax attention with a secondary-query local correction"
                if alias == "parallax_attention_core"
                else "causal softmax attention after the differentiable PaTH triangular q/k transform"
                if alias == "path_attention_core"
                else "causal Wall attention with per-channel log-decay score modulation"
                if alias == "wall_attention_core"
                else "softmax attention after caller-supplied projections/transforms"
            ),
            stages,
        )
    recipes["factorized_attention_core"] = MixerRecipe(
        ("arch-069", "arch-070"),
        replace(attention, name="factorized_attention_core"),
        "generic causal softmax attention after caller-supplied factorized projections",
        (
            "architecture-specific factorized projections",
            "architecture-specific cache expansion",
        ),
    )
    recipes["attnres_depth_core"] = MixerRecipe(
        ("arch-054",),
        softmax_attention_spec(
            "attnres_depth_core", causal=False, scale=1.0,
            operation=K1Operation.DEPTH,
        ),
        "softmax reduction along a caller-mapped layer/depth axis",
        ("depth-axis normalization and dependency graph",),
    )

    recipes["linear_attention"] = MixerRecipe(
        ("arch-015",),
        linear_attention_spec("linear_attention"),
        "feature-mapped normalized additive matrix recurrence",
        ("architecture-specific feature-map projection",),
    )
    for name, arch_id, basis in (
        ("based_attention_core", "arch-020", PolynomialBasis.BASED_TAYLOR2),
        ("rebased_attention_core", "arch-021", PolynomialBasis.REBASED_SQUARE),
    ):
        recipes[name] = MixerRecipe(
            (arch_id,),
            UnifiedMixerSpec(
                name,
                MixerKernelFamily.RECURRENCE,
                update_rule=StateUpdateRule.ADDITIVE,
                normalizer=StateNormalizer.QUERY_KEY,
                polynomial_basis=basis,
            ),
            "normalized causal polynomial attention as an additive feature-state recurrence",
            ("Q/K/V projections and architecture-level normalization",),
        )

    for alias, ids, decay in (
        ("delta_net", ("arch-025",), DecayGranularity.NONE),
        ("gated_delta_net", ("arch-026",), DecayGranularity.HEAD),
        ("gdn2_core", ("arch-027",), DecayGranularity.KEY_CHANNEL),
        ("kda_core", ("arch-028",), DecayGranularity.KEY_CHANNEL),
    ):
        spec = (
            UnifiedMixerSpec(
                alias,
                MixerKernelFamily.RECURRENCE,
                update_rule=StateUpdateRule.DELTA,
                decay=decay,
                gdn2_ssm=True,
            )
            if alias == "gdn2_core"
            else (
                UnifiedMixerSpec(
                    alias,
                    MixerKernelFamily.RECURRENCE,
                    update_rule=StateUpdateRule.DELTA,
                    decay=decay,
                    kda_delta=True,
                )
                if alias == "kda_core"
                else delta_rule_spec(alias, decay=decay)
            )
        )
        external_stages = (
            (
                "Q/K/V projections and gate production",
                "key normalization, gated RMS normalization and output projection",
            )
            if alias == "gdn2_core"
            else (
                (
                    "A_log/dt/bias gate generation and optional Q/K normalization",
                    "gated normalization and output projection",
                )
                if alias == "kda_core"
                else (
                    "architecture-specific gate and projection generation",
                    "key normalization where required by the source architecture",
                )
            )
        )
        recipes[alias] = MixerRecipe(
            ids,
            spec,
            (
                "GDN-2 key-decayed matrix state with separate key erase and value write gates"
                if alias == "gdn2_core"
                else "KDA key-channel-decayed delta state with scaled query read"
                if alias == "kda_core"
                else "matrix-state delta update with the declared decay granularity"
            ),
            external_stages,
        )

    recipes["atma_gated_delta_decode_core"] = MixerRecipe(
        ("arch-026",),
        UnifiedMixerSpec(
            "atma_gated_delta_decode_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.HEAD,
            feature_map=FeatureMap.L2_NORMALIZE,
            state_effect=StateEffect.IN_PLACE_SLOT_TABLE,
        ),
        "ATMA's in-place, slot-indexed one-token gated-delta decode update",
        (
            "ATMA gate and Q/K/V projection production",
            "RMSNorm, output gate/projection and full block integration",
        ),
    )

    for alias, arch_id, decay in (
        ("simple_gla", ("arch-018", "arch-052"), DecayGranularity.HEAD),
        ("gla", "arch-019", DecayGranularity.KEY_CHANNEL),
        ("rodimus_gla_core", "arch-036", DecayGranularity.KEY_CHANNEL),
    ):
        architecture_ids = (arch_id,) if isinstance(arch_id, str) else arch_id
        recipes[alias] = MixerRecipe(
            architecture_ids,
            UnifiedMixerSpec(
                alias,
                MixerKernelFamily.RECURRENCE,
                update_rule=StateUpdateRule.ADDITIVE,
                decay=decay,
                read_scale=(64**-0.5 if alias == "rodimus_gla_core" else None),
                state_v_first=alias == "rodimus_gla_core",
            ),
            (
                "Rodimus key-channel-gated GLA with V-first state and the source default 1/sqrt(K) read scale"
                if alias == "rodimus_gla_core"
                else "additive matrix-state attention with the declared gated decay"
            ),
            (
                (
                    "Rodimus input/gate projections, short convolution, normalization and output projection",
                    "full-layer training/prefill/decode and cache integration",
                )
                if alias == "rodimus_gla_core"
                else ("architecture-specific gate and projection generation",)
            ),
        )

    recipes["lightnet_gla_core"] = MixerRecipe(
        ("arch-022",),
        UnifiedMixerSpec(
            "lightnet_gla_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            decay=DecayGranularity.KEY_CHANNEL,
        ),
        "key-channel-decayed GLA recurrence after LightNet feature transforms",
        (
            "Q/K/V projections and optional short convolutions",
            "SiLU query and normalized-key/log-cumulative-decay frontend",
            "gated RMS normalization and output projection",
        ),
    )

    recipes["gated_oja_core"] = MixerRecipe(
        ("arch-033",),
        UnifiedMixerSpec(
            "gated_oja_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.VALUE_CHANNEL,
            gated_oja=True,
            recurrence_operator=RecurrenceOperator.GATED_OJA_VALUE_CHANNEL,
        ),
        "value-channel-decayed Oja matrix update with key residual correction",
        ("gated Oja Q/K/V/gate projections and model-side gate production",),
    )

    recipes["comba_core"] = MixerRecipe(
        ("arch-037",),
        UnifiedMixerSpec(
            "comba_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.HEAD,
            comba_rule=True,
        ),
        "head-decayed delta update with separate prediction and write keys",
        ("COMBA projection and feature-frontend production",),
    )

    recipes["pgdn_core"] = MixerRecipe(
        ("arch-034",),
        UnifiedMixerSpec(
            "pgdn_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.HEAD,
            preconditioned_gated_delta=True,
        ),
        "preconditioned gated delta update with a learned diagonal key metric",
        ("ATK parameter production, projections, and full PGDN layer",),
    )

    recipes["pkda_core"] = MixerRecipe(
        ("arch-035",),
        UnifiedMixerSpec(
            "pkda_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            decay=DecayGranularity.KEY_CHANNEL,
            preconditioned_kda=True,
        ),
        "preconditioned key-decayed delta update with a learned diagonal key metric",
        (
            "ATK parameter production, KDA gate frontend, projections, and full PKDA layer",
        ),
    )

    recipes["abc_core"] = MixerRecipe(
        ("arch-048",),
        UnifiedMixerSpec(
            "abc_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            slot_attention=True,
            recurrence_operator=RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE,
        ),
        "two-stage associative slot attention with cumulative slot normalization",
        ("ABC slot representation and Q/K/V projection frontend",),
    )
    recipes["gsa_core"] = MixerRecipe(
        ("arch-049", "arch-050"),
        UnifiedMixerSpec(
            "gsa_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            slot_attention=True,
            recurrence_operator=RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE,
        ),
        "two-stage gated slot attention with a matrix state per slot",
        ("GSA slot/key/value projections and learned gate production",),
    )

    recipes["gated_delta_product_core"] = MixerRecipe(
        ("arch-029",),
        replace(
            delta_rule_spec("gated_delta_product_core", decay=DecayGranularity.HEAD),
            gated_delta_product=True,
        ),
        "ordered rank-R delta updates within each token",
        ("architecture-specific per-update projections and gates",),
    )
    for alias, arch_id in (
        ("generalized_delta_iplr_core", "arch-031"),
        ("generalized_delta_dplr_core", "arch-032"),
    ):
        recipes[alias] = MixerRecipe(
            (arch_id,),
            UnifiedMixerSpec(
                alias,
                MixerKernelFamily.RECURRENCE,
                update_rule=StateUpdateRule.ADDITIVE,
                transition=StateTransition.FACTORED_MATRIX,
                generalized_delta_iplr=alias == "generalized_delta_iplr_core",
                generalized_delta_dplr=alias == "generalized_delta_dplr_core",
            ),
            "additive key-value write with source-factorized low-rank state transition",
            ("compact low-rank source factorization and rank/cost policy",),
        )

    recipes["retention_core"] = MixerRecipe(
        ("arch-017",),
        UnifiedMixerSpec(
            "retention_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            decay=DecayGranularity.HEAD,
            static_head_decay=True,
        ),
        "per-head statically decayed additive state recurrence",
        ("multiscale head schedule and layer normalization",),
    )
    recipes["lightning_attention_core"] = MixerRecipe(
        ("arch-016",),
        UnifiedMixerSpec(
            "lightning_attention_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            decay=DecayGranularity.HEAD,
            static_head_decay=True,
            static_head_decay_chunk=True,
        ),
        "per-head statically decayed additive state recurrence",
        ("layer-index decay schedule and Q/K/V projections",),
    )
    recipes["rwkv4_memory_core"] = MixerRecipe(
        ("arch-040",),
        UnifiedMixerSpec(
            "rwkv4_memory_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            rwkv4_memory=True,
        ),
        "RWKV-4 time-mix with a stable three-scalar-per-channel state",
        ("time-mix projections and output gate",),
    )
    recipes["rwkv6_memory_core"] = MixerRecipe(
        ("arch-041",),
        UnifiedMixerSpec(
            "rwkv6_memory_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            decay=DecayGranularity.KEY_CHANNEL,
            read_timing=ReadTiming.BEFORE_UPDATE,
            rwkv6_memory=True,
        ),
        "RWKV-6 key-channel-decayed matrix memory with a static bonus read correction",
        ("RWKV-6 time-mix frontend, projections and output gate",),
    )
    recipes["momentum_delta_core"] = MixerRecipe(
        ("arch-030",),
        UnifiedMixerSpec(
            "momentum_delta_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.DELTA,
            momentum_delta=True,
            recurrence_operator=RecurrenceOperator.MOMENTUM_DELTA_STATE,
        ),
        "Momentum DeltaNet with coupled fast-weight and momentum matrix states",
        ("momentum/delta frontend transforms and model projections",),
    )
    recipes["mesa_net_core"] = MixerRecipe(
        ("arch-038",),
        UnifiedMixerSpec(
            "mesa_net_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.REGULARIZED_SOLVE,
        ),
        "MesaNet dual covariance-state recurrence with a regularized per-token linear solve",
        (
            "Q/K L2 normalization and model-side lambda construction",
            "streaming decode and initial-state gradient qualification",
        ),
    )
    recipes["titans_linear_memory_core"] = MixerRecipe(
        ("arch-039",),
        UnifiedMixerSpec(
            "titans_linear_memory_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.MOMENTUM_INNER_STATE,
        ),
        "FLA Titans chunked associative memory update with its learned inner loss and layer-normalized readout",
        (
            "Q/K/V and theta/alpha/eta projections",
            "outer Titans attention, memory hierarchy and complete model block",
            "streaming decode and nonzero initial-state gradients",
        ),
    )
    recipes["ttt_linear_core"] = MixerRecipe(
        ("arch-055",),
        UnifiedMixerSpec(
            "ttt_linear_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.LAYERNORM_INNER_STATE,
        ),
        "TTT-Linear chunkwise inner-loss update with matrix and bias memory states",
        (
            "input/query/key/value projections and learned layer-norm parameters",
            "MLP variant and full TTT layer composition",
            "variable-length training, nonzero-state cache and higher-order gradient qualification",
        ),
    )
    recipes["rnn_core"] = MixerRecipe(
        ("arch-060",),
        UnifiedMixerSpec(
            "rnn_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.TANH_RNN,
        ),
        "XMA tanh RNN nonlinear recurrent state update",
        (
            "multi-head replication and variable-length packed sequences",
            "gradient clipping",
        ),
    )
    recipes["gru_core"] = MixerRecipe(
        ("arch-061",),
        UnifiedMixerSpec(
            "gru_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.GATED_RNN,
        ),
        "XMA reset/update-gated nonlinear GRU state update",
        (
            "multi-head replication and variable-length packed sequences",
            "gradient clipping",
        ),
    )
    recipes["m2rnn_core"] = MixerRecipe(
        ("arch-062",),
        UnifiedMixerSpec(
            "m2rnn_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.MULTIPLICATIVE_RNN,
        ),
        "XMA second-order matrix-memory nonlinear recurrent state update",
        (
            "multi-head replication and variable-length packed sequences",
            "gradient clipping",
        ),
    )
    recipes["mamba1_ssm_core"] = MixerRecipe(
        ("arch-043",),
        diagonal_ssm_spec("mamba1_ssm_core", step_size_discretization=True),
        "input-conditioned diagonal SSM recurrence",
        ("input/output projections", "short convolution", "activation gate"),
    )
    recipes["mamba2_ssm_core"] = MixerRecipe(
        ("arch-044",),
        UnifiedMixerSpec(
            "mamba2_ssm_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            decay=DecayGranularity.HEAD,
            mamba2_ssm=True,
        ),
        "SSD matrix-state recurrence with continuous-time head decay",
        (
            "convolution and SSM projections",
            "dt bias/softplus, skip and output gating",
            "cache ABI and chunk-boundary schedule",
        ),
    )
    recipes["mamba3_siso_core"] = MixerRecipe(
        ("arch-045",),
        UnifiedMixerSpec(
            "mamba3_siso_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            recurrence_operator=RecurrenceOperator.TRAPEZOIDAL_SSM,
        ),
        "Mamba-3 SISO rotary angle accumulator with trapezoidal four-state SSM update",
        (
            "input/output projections and parameterized A/dt/trap frontend",
            "Mamba-3 MIMO/TileLang path, cache integration and full layer",
        ),
    )
    recipes["log_linear_attention_core"] = MixerRecipe(
        ("arch-009", "arch-046"),
        UnifiedMixerSpec(
            "log_linear_attention_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            log_linear_attention=True,
        ),
        "Dyadic level-scaled matrix attention recurrence with 64-token chunk state",
        (
            "Mamba-2 projections and dt/level-scale transformations",
            "incremental partial-chunk cache and variable-length sequence ABI",
            "full layer normalization, skip and output gating",
        ),
    )
    recipes["hgrn2_ssm_core"] = MixerRecipe(
        ("arch-024",),
        UnifiedMixerSpec(
            "hgrn2_ssm_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            decay=DecayGranularity.KEY_CHANNEL,
        ),
        "key-channel-decayed GLA matrix-state recurrent core",
        (
            "Q/K/input projections and gate activation",
            "HGRN2 value-first state layout and layer normalization",
        ),
    )
    recipes["hgrn_ssm_core"] = MixerRecipe(
        ("arch-023",),
        diagonal_ssm_spec("hgrn_ssm_core", hgrn=True),
        "diagonal gated state recurrence core",
        ("architecture-specific gate activation and output projection",),
    )
    recipes["rwkv7_transition_core"] = MixerRecipe(
        ("arch-042",),
        UnifiedMixerSpec(
            "rwkv7_transition_core",
            MixerKernelFamily.RECURRENCE,
            update_rule=StateUpdateRule.ADDITIVE,
            transition=StateTransition.FACTORED_MATRIX,
            generalized_delta_dplr=True,
            read_scale=1.0,
        ),
        "RWKV-7 additive key-value write with source-derived diagonal-plus-low-rank left transition",
        ("model-side RWKV-7 time/value factor production and state/cache integration",),
    )
    recipes["sparse_delta_memory"] = MixerRecipe(
        ("arch-047",),
        sparse_delta_spec(),
        "ordered sparse-slot decayed delta update and weighted read",
        ("product-key route score generation",),
    )
    try:
        return recipes[key]
    except KeyError as error:
        supported = ", ".join(sorted(recipes))
        raise ValueError(
            f"no unified mixer recipe for {name!r}; available recipes: {supported}"
        ) from error
