"""A compiler entry point for the three URM mixer kernel families.

The semantic contract is deliberately closed and backend independent. Plans
flow through URM's semantic validator, candidate enumerator, intent checker,
and trusted-anchor resolver before they bind to a family executor. Native and
library anchors can replace the reference executor without changing these
semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from functools import lru_cache
from typing import Any


class MixerKernelFamily(StrEnum):
    """The three physical mixer families URM composes."""

    SOFTMAX = "K1_softmax_reduction"
    RECURRENCE = "K2_state_recurrence"
    SPARSE_DELTA = "K3_sparse_delta_state"


class RecurrentLayout(StrEnum):
    MATRIX = "matrix_state"
    DIAGONAL = "diagonal_ssm_state"


class StateUpdateRule(StrEnum):
    ADDITIVE = "additive"
    DELTA = "delta"


class StateNormalizer(StrEnum):
    NONE = "none"
    QUERY_KEY = "query_key_denominator"


class FeatureMap(StrEnum):
    IDENTITY = "identity"
    RELU = "relu"
    ELU_PLUS_ONE = "elu_plus_one"
    SOFTPLUS = "softplus"
    L2_NORMALIZE = "l2_normalize"


class PolynomialBasis(StrEnum):
    NONE = "none"
    BASED_TAYLOR2 = "based_taylor2"
    REBASED_SQUARE = "rebased_square"


class DecayGranularity(StrEnum):
    NONE = "none"
    HEAD = "head"
    KEY_CHANNEL = "key_channel"
    ELEMENTWISE = "elementwise"
    VALUE_CHANNEL = "value_channel"


class StateTransition(StrEnum):
    POINTWISE = "pointwise_decay"
    FACTORED_MATRIX = "factored_bilinear"


class ReadTiming(StrEnum):
    BEFORE_UPDATE = "before_update"
    AFTER_UPDATE = "after_update"


class MixerIntent(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"
    FORWARD_ONLY_ANALYSIS = "forward_only_analysis"


class MixerBackend(StrEnum):
    REFERENCE = "reference"
    LIBRARY = "library"
    NATIVE = "native"


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
class UnifiedMixerSpec:
    """Typed equations shared by attention, recurrence, and sparse delta.

    Tensor layouts are fixed at the API boundary and documented in
    :meth:`CompiledMixerPlan.execute`.  Values that change the equation live
    here; physical tiling and launch parameters do not.
    """

    name: str
    family: MixerKernelFamily
    causal: bool = True
    attention_scale: float | None = None
    read_scale: float | None = None
    state_v_first: bool = False
    accepts_score_bias: bool = False
    path_attention: bool = False
    deltaformer_attention: bool = False
    recurrent_layout: RecurrentLayout = RecurrentLayout.MATRIX
    update_rule: StateUpdateRule = StateUpdateRule.ADDITIVE
    normalizer: StateNormalizer = StateNormalizer.NONE
    feature_map: FeatureMap = FeatureMap.IDENTITY
    polynomial_basis: PolynomialBasis = PolynomialBasis.NONE
    decay: DecayGranularity = DecayGranularity.NONE
    transition: StateTransition = StateTransition.POINTWISE
    read_timing: ReadTiming = ReadTiming.AFTER_UPDATE
    static_head_decay: bool = False
    static_head_decay_chunk: bool = False
    mamba2_ssm: bool = False
    log_linear_attention: bool = False
    gdn2_ssm: bool = False
    kda_delta: bool = False
    gated_delta_product: bool = False
    generalized_delta_iplr: bool = False
    generalized_delta_dplr: bool = False
    rwkv4_memory: bool = False
    rwkv6_memory: bool = False
    momentum_delta: bool = False
    gated_oja: bool = False
    comba_rule: bool = False
    preconditioned_gated_delta: bool = False
    preconditioned_kda: bool = False
    slot_attention: bool = False
    step_size_discretization: bool = False
    diagonal_hgrn: bool = False
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("family", MixerKernelFamily),
            ("recurrent_layout", RecurrentLayout),
            ("update_rule", StateUpdateRule),
            ("normalizer", StateNormalizer),
            ("feature_map", FeatureMap),
            ("polynomial_basis", PolynomialBasis),
            ("decay", DecayGranularity),
            ("transition", StateTransition),
            ("read_timing", ReadTiming),
        ):
            try:
                value = enum_type(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid {name}: {getattr(self, name)!r}") from error
            object.__setattr__(self, name, value)
        if not self.name.strip():
            raise ValueError("mixer name must not be empty")
        if not isinstance(self.step_size_discretization, bool):
            raise ValueError("step_size_discretization must be a bool")
        if not isinstance(self.path_attention, bool):
            raise ValueError("path_attention must be a bool")
        if not isinstance(self.deltaformer_attention, bool):
            raise ValueError("deltaformer_attention must be a bool")
        if not isinstance(self.state_v_first, bool):
            raise ValueError("state_v_first must be a bool")
        if self.deltaformer_attention and self.family is not MixerKernelFamily.SOFTMAX:
            raise ValueError("DeltaFormer uses the K1 softmax-attention family")
        if not isinstance(self.static_head_decay, bool):
            raise ValueError("static_head_decay must be a bool")
        if not isinstance(self.static_head_decay_chunk, bool):
            raise ValueError("static_head_decay_chunk must be a bool")
        if self.static_head_decay_chunk and not self.static_head_decay:
            raise ValueError("static_head_decay_chunk requires static_head_decay")
        if not isinstance(self.mamba2_ssm, bool):
            raise ValueError("mamba2_ssm must be a bool")
        if not isinstance(self.log_linear_attention, bool):
            raise ValueError("log_linear_attention must be a bool")
        if not isinstance(self.gdn2_ssm, bool):
            raise ValueError("gdn2_ssm must be a bool")
        if not isinstance(self.kda_delta, bool):
            raise ValueError("kda_delta must be a bool")
        if not isinstance(self.gated_delta_product, bool):
            raise ValueError("gated_delta_product must be a bool")
        if not isinstance(self.generalized_delta_iplr, bool):
            raise ValueError("generalized_delta_iplr must be a bool")
        if not isinstance(self.generalized_delta_dplr, bool):
            raise ValueError("generalized_delta_dplr must be a bool")
        if not isinstance(self.rwkv4_memory, bool):
            raise ValueError("rwkv4_memory must be a bool")
        if not isinstance(self.rwkv6_memory, bool):
            raise ValueError("rwkv6_memory must be a bool")
        if not isinstance(self.momentum_delta, bool):
            raise ValueError("momentum_delta must be a bool")
        if not isinstance(self.gated_oja, bool):
            raise ValueError("gated_oja must be a bool")
        if not isinstance(self.comba_rule, bool):
            raise ValueError("comba_rule must be a bool")
        if not isinstance(self.preconditioned_gated_delta, bool):
            raise ValueError("preconditioned_gated_delta must be a bool")
        if not isinstance(self.preconditioned_kda, bool):
            raise ValueError("preconditioned_kda must be a bool")
        if not isinstance(self.slot_attention, bool):
            raise ValueError("slot_attention must be a bool")
        if self.rwkv4_memory and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "RWKV-4 requires its dedicated stable scalar-state recurrence"
            )
        if self.rwkv4_memory and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv6_memory,
                self.momentum_delta,
                self.gated_oja,
                self.comba_rule,
                self.preconditioned_kda,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError(
                "RWKV-4 semantics cannot be combined with other scan modes"
            )
        if self.rwkv6_memory and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.KEY_CHANNEL
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.BEFORE_UPDATE
        ):
            raise ValueError(
                "RWKV-6 requires a decayed matrix state with bonus read correction"
            )
        if self.rwkv6_memory and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError(
                "RWKV-6 semantics cannot be combined with other scan modes"
            )
        if self.momentum_delta and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "momentum delta requires its dedicated two-matrix-state recurrence"
            )
        if self.momentum_delta and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.rwkv6_memory,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError(
                "momentum delta semantics cannot be combined with other scan modes"
            )
        if self.gated_oja and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.VALUE_CHANNEL
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError("gated Oja requires a value-channel-decayed Oja update")
        if self.decay is DecayGranularity.VALUE_CHANNEL and not self.gated_oja:
            raise ValueError("value-channel state decay is reserved for gated Oja")
        if self.gated_oja and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.rwkv6_memory,
                self.momentum_delta,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
                self.comba_rule,
            )
        ):
            raise ValueError(
                "gated Oja semantics cannot be combined with other scan modes"
            )
        if self.comba_rule and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.HEAD
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError("COMBA requires a head-decayed dual-key delta update")
        if self.comba_rule and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.rwkv6_memory,
                self.momentum_delta,
                self.gated_oja,
                self.preconditioned_gated_delta,
                self.preconditioned_kda,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError("COMBA semantics cannot be combined with other scan modes")
        if self.preconditioned_gated_delta and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.HEAD
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError("PGDN requires a head-gated preconditioned delta state")
        if self.preconditioned_gated_delta and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.rwkv6_memory,
                self.momentum_delta,
                self.gated_oja,
                self.comba_rule,
                self.preconditioned_kda,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError("PGDN semantics cannot be combined with other scan modes")
        if self.preconditioned_kda and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.KEY_CHANNEL
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "PKDA requires key-channel-gated preconditioned delta state"
            )
        if self.preconditioned_kda and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.rwkv6_memory,
                self.momentum_delta,
                self.gated_oja,
                self.comba_rule,
                self.preconditioned_gated_delta,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError("PKDA semantics cannot be combined with other scan modes")
        if self.slot_attention and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "ABC/GSA require their two-stage slot attention recurrence"
            )
        if self.slot_attention and any(
            (
                self.static_head_decay,
                self.static_head_decay_chunk,
                self.mamba2_ssm,
                self.log_linear_attention,
                self.gdn2_ssm,
                self.kda_delta,
                self.gated_delta_product,
                self.generalized_delta_iplr,
                self.generalized_delta_dplr,
                self.rwkv4_memory,
                self.rwkv6_memory,
                self.momentum_delta,
                self.gated_oja,
                self.comba_rule,
                self.preconditioned_gated_delta,
                self.preconditioned_kda,
                self.step_size_discretization,
                self.diagonal_hgrn,
                self.polynomial_basis is not PolynomialBasis.NONE,
            )
        ):
            raise ValueError(
                "ABC/GSA slot attention cannot be combined with other scan modes"
            )
        if self.gdn2_ssm and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.KEY_CHANNEL
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "GDN-2 requires key-channel-decayed gated delta matrix state"
            )
        if self.gdn2_ssm and (
            self.static_head_decay
            or self.static_head_decay_chunk
            or self.mamba2_ssm
            or self.log_linear_attention
            or self.kda_delta
            or self.gated_delta_product
            or self.step_size_discretization
            or self.diagonal_hgrn
        ):
            raise ValueError("GDN-2 semantics cannot be combined with other scan modes")
        if self.polynomial_basis is not PolynomialBasis.NONE and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.QUERY_KEY
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
            or self.static_head_decay
            or self.static_head_decay_chunk
            or self.mamba2_ssm
            or self.log_linear_attention
            or self.kda_delta
            or self.gated_delta_product
            or self.gdn2_ssm
        ):
            raise ValueError(
                "polynomial feature bases require normalized additive matrix recurrence"
            )
        if self.mamba2_ssm and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.HEAD
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "Mamba-2 scan semantics require head-decayed additive matrix state"
            )
        if self.mamba2_ssm and (self.static_head_decay or self.static_head_decay_chunk):
            raise ValueError(
                "Mamba-2 discretized decay cannot be a static head schedule"
            )
        if self.log_linear_attention and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
            or self.static_head_decay
            or self.static_head_decay_chunk
            or self.mamba2_ssm
            or self.gdn2_ssm
            or self.kda_delta
            or self.gated_delta_product
        ):
            raise ValueError("LogLinear attention requires dyadic scaled matrix state")
        if self.kda_delta and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.KEY_CHANNEL
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
            or self.static_head_decay
            or self.static_head_decay_chunk
            or self.mamba2_ssm
            or self.log_linear_attention
            or self.gdn2_ssm
        ):
            raise ValueError("KDA requires key-channel-decayed delta matrix state")
        if self.gated_delta_product and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.DELTA
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.HEAD
            or self.transition is not StateTransition.POINTWISE
            or self.read_timing is not ReadTiming.AFTER_UPDATE
            or self.static_head_decay
            or self.static_head_decay_chunk
            or self.mamba2_ssm
            or self.log_linear_attention
            or self.gdn2_ssm
            or self.kda_delta
        ):
            raise ValueError(
                "Gated DeltaProduct requires head-decayed ordered delta updates"
            )
        if self.generalized_delta_iplr and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.FACTORED_MATRIX
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "IPLR requires additive writes and a factored state transition"
            )
        if self.generalized_delta_dplr and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.NONE
            or self.transition is not StateTransition.FACTORED_MATRIX
            or self.read_timing is not ReadTiming.AFTER_UPDATE
        ):
            raise ValueError(
                "DPLR requires additive writes and a factored state transition"
            )
        if self.generalized_delta_iplr and self.generalized_delta_dplr:
            raise ValueError("IPLR and DPLR are distinct transition contracts")
        if self.static_head_decay and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.MATRIX
            or self.update_rule is not StateUpdateRule.ADDITIVE
            or self.normalizer is not StateNormalizer.NONE
            or self.feature_map is not FeatureMap.IDENTITY
            or self.decay is not DecayGranularity.HEAD
            or self.transition is not StateTransition.POINTWISE
        ):
            raise ValueError(
                "static head decay requires plain additive head-decayed matrix recurrence"
            )
        if self.step_size_discretization and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.DIAGONAL
        ):
            raise ValueError("step-size discretization applies to diagonal K2 only")
        if not isinstance(self.diagonal_hgrn, bool):
            raise ValueError("diagonal_hgrn must be a bool")
        if self.diagonal_hgrn and (
            self.family is not MixerKernelFamily.RECURRENCE
            or self.recurrent_layout is not RecurrentLayout.DIAGONAL
            or self.step_size_discretization
        ):
            raise ValueError("HGRN semantics require direct diagonal K2 recurrence")
        if self.attention_scale is not None and self.attention_scale <= 0:
            raise ValueError("attention_scale must be positive")
        if self.read_scale is not None and self.read_scale <= 0:
            raise ValueError("read_scale must be positive")
        if self.state_v_first and not (
            self.family is MixerKernelFamily.RECURRENCE
            and self.recurrent_layout is RecurrentLayout.MATRIX
        ):
            raise ValueError("state_v_first applies to matrix-state K2 recurrences")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

        if self.family is MixerKernelFamily.SOFTMAX:
            if (
                self.recurrent_layout is not RecurrentLayout.MATRIX
                or self.update_rule is not StateUpdateRule.ADDITIVE
                or self.normalizer is not StateNormalizer.NONE
                or self.feature_map is not FeatureMap.IDENTITY
                or self.decay is not DecayGranularity.NONE
                or self.transition is not StateTransition.POINTWISE
                or self.read_scale is not None
            ):
                raise ValueError("K1 accepts softmax attention semantics only")
            if self.path_attention and self.accepts_score_bias:
                raise ValueError(
                    "PaTH attention uses its own transformed score, not score_bias"
                )
            if self.deltaformer_attention and (
                not self.causal or self.path_attention or self.accepts_score_bias
            ):
                raise ValueError(
                    "DeltaFormer requires its own causal two-stage attention semantics"
                )
        elif self.family is MixerKernelFamily.RECURRENCE:
            if self.accepts_score_bias or self.attention_scale is not None:
                raise ValueError("score bias and attention scale belong to K1")
            if self.recurrent_layout is RecurrentLayout.DIAGONAL:
                if (
                    self.update_rule is not StateUpdateRule.ADDITIVE
                    or self.normalizer is not StateNormalizer.NONE
                    or self.feature_map is not FeatureMap.IDENTITY
                    or self.decay is not DecayGranularity.ELEMENTWISE
                    or self.transition is not StateTransition.POINTWISE
                ):
                    raise ValueError(
                        "diagonal SSM requires additive elementwise state updates"
                    )
            elif self.decay is DecayGranularity.ELEMENTWISE:
                raise ValueError("matrix state decay is head or key-channel scoped")
            if (
                self.transition is StateTransition.FACTORED_MATRIX
                and self.decay is not DecayGranularity.NONE
            ):
                raise ValueError(
                    "factored transitions cannot also declare pointwise decay"
                )
            if (
                self.normalizer is StateNormalizer.QUERY_KEY
                and self.update_rule is not StateUpdateRule.ADDITIVE
            ):
                raise ValueError("query/key denominator is defined for additive state")
            if (
                self.transition is StateTransition.FACTORED_MATRIX
                and self.normalizer is StateNormalizer.QUERY_KEY
            ):
                raise ValueError("factored transitions do not define denominator state")
        elif self.family is MixerKernelFamily.SPARSE_DELTA:
            if (
                self.recurrent_layout is not RecurrentLayout.MATRIX
                or self.update_rule is not StateUpdateRule.ADDITIVE
                or self.normalizer is not StateNormalizer.NONE
                or self.feature_map is not FeatureMap.IDENTITY
                or self.decay is not DecayGranularity.NONE
                or self.transition is not StateTransition.POINTWISE
                or self.accepts_score_bias
                or self.attention_scale is not None
                or self.read_scale is not None
            ):
                raise ValueError("K3 accepts ordered sparse delta semantics only")

    def to_dict(self) -> dict[str, object]:
        return {
            key: value.value if isinstance(value, StrEnum) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True, slots=True)
class MixerResult:
    output: Any
    final_state: Any | None = None
    final_normalizer_state: Any | None = None
    metadata: dict[str, object] | None = None
    auxiliary_output: Any | None = None


@dataclass(frozen=True, slots=True)
class MixerLogLinearState:
    """Complete-chunk hierarchy and retained suffix for LogLinear attention."""

    ht: Any
    offsets: Any
    q_prev: Any
    k_prev: Any
    v_prev: Any
    g_prev: Any
    level_scales_prev: Any


@dataclass(frozen=True, slots=True)
class MixerRecipe:
    """A named kernel-level mapping with its model boundary made explicit."""

    architecture_ids: tuple[str, ...]
    spec: UnifiedMixerSpec
    component_scope: str
    required_external_stages: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompiledMixerPlan:
    """Serializable selection of one of the three executable mixer families."""

    spec: UnifiedMixerSpec
    intent: MixerIntent
    anchor: str
    backend: MixerBackend = MixerBackend.REFERENCE
    recipe: MixerRecipe | None = None
    compiler_result: Any | None = None
    compile_dtype: str = "float32"

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "semantic_spec": self.spec.to_dict(),
            "intent": self.intent.value,
            "physical_kernel_family": self.spec.family.value,
            "anchor": self.anchor,
            "backend": self.backend.value,
            "compile_dtype": self.compile_dtype,
            "implementation": {
                MixerBackend.REFERENCE: "torch_eager_reference_v1",
                MixerBackend.LIBRARY: "trusted_library_anchor",
                MixerBackend.NATIVE: "urm_native_anchor",
            }[self.backend],
            "unified_gpu_fusion": False,
            "native_anchor_launches": (
                1 if self.backend is MixerBackend.NATIVE else None
            ),
            "autograd": self.intent is MixerIntent.TRAINING,
            "compiler_candidate": (
                self.compiler_result.selected_candidate_id
                if self.compiler_result is not None
                else None
            ),
            "compiler_result": (
                self.compiler_result.to_dict()
                if self.compiler_result is not None
                else None
            ),
            "compiler_plan": (
                self.compiler_result.plan.to_dict()
                if self.compiler_result is not None
                else None
            ),
        }
        if self.recipe is not None:
            result["named_coverage"] = {
                "architecture_ids": list(self.recipe.architecture_ids),
                "component_scope": self.recipe.component_scope,
                "required_external_stages": list(self.recipe.required_external_stages),
            }
        return result

    def execute(self, **operands: Any) -> MixerResult:
        """Run this plan using PyTorch tensors without importing Torch at load.

        K1 expects ``query``, ``key``, and ``value`` in BTHD layout. K2 matrix
        recurrence uses the same layout plus ``beta`` and/or ``log_decay``;
        K2 diagonal SSM expects ``x``, ``input_gate``, ``read_gate``, and
        ``log_decay``. RWKV-4 expects ``w``/``u`` vectors, ``k``/``v`` in BTC
        layout and stable ``state`` in B3SC layout. K3 expects ``memory``,
        explicit read/write route indices and weights, and the update operands.
        Returned recurrent state is always available when the equation has
        state. ATMA gated-delta decode takes one-token Q/K/V, per-head
        ``gamma``/``beta``, an fp32
        ``state_table`` in [capacity, H, K, V] layout, and int64 ``slots``. The
        caller must supply valid, distinct active slots; this anchor is
        inference-only.
        """
        torch = _torch()
        primary_name = {
            MixerKernelFamily.SOFTMAX: "query",
            MixerKernelFamily.RECURRENCE: (
                "k"
                if self.spec.rwkv4_memory
                else "x"
                if self.spec.recurrent_layout is RecurrentLayout.DIAGONAL
                or self.spec.mamba2_ssm
                else "query"
            ),
            MixerKernelFamily.SPARSE_DELTA: "memory",
        }[self.spec.family]
        primary = operands.get(primary_name)
        if primary is not None and primary.dtype != getattr(torch, self.compile_dtype):
            raise ValueError(
                f"compiled for {self.compile_dtype}, but {primary_name} uses "
                f"{primary.dtype}"
            )
        if (
            self.backend is MixerBackend.LIBRARY
            and self.spec.name == "atma_gated_delta_decode_core"
        ):
            return _execute_atma_gated_delta_decode(self, torch, **operands)
        if self.spec.name == "h3_ssm_fft_core":
            return _execute_h3_ssm_fft(self, torch, **operands)
        if self.spec.name == "hyena_fftconv_core":
            return _execute_hyena_fftconv(self, torch, **operands)
        if self.spec.name == "hla_second_order_core":
            return _execute_hla_second_order(self, torch, **operands)
        if self.backend is MixerBackend.LIBRARY:
            if self.spec.family is MixerKernelFamily.SOFTMAX:
                if self.spec.name == "fwpkm_memory_read_core":
                    return _execute_fwpkm_selected_read(self, torch, **operands)
                if self.spec.name == "tda_attention_core":
                    return _execute_tda_attention_adapter(self, torch, **operands)
                if self.spec.name == "tucker_attention_core":
                    return _execute_tucker_attention_adapter(self, torch, **operands)
                if self.spec.name == "longformer_attention_core":
                    return _execute_longformer_attention_adapter(
                        self, torch, **operands
                    )
                if self.spec.name == "kata_attention_core":
                    return _execute_kata_attention_adapter(self, torch, **operands)
                if self.spec.name == "differential_attention_core":
                    return _execute_differential_attention(
                        self.spec, torch, library=True, **operands
                    )
                if self.spec.name in {
                    "polar_attention_core",
                    "foveal_sparse_polar_attention_core",
                }:
                    return _execute_atma_polar(self, torch, **operands)
                if self.spec.deltaformer_attention:
                    return _execute_fla_deltaformer(self, torch, **operands)
                if self.spec.path_attention:
                    return _execute_fla_path_attention(self, torch, **operands)
                if self.spec.name == "attnres_depth_core":
                    return _execute_fla_attnres(self, torch, **operands)
                if self.spec.name == "fox":
                    return _execute_fla_forgetting_attention(self, torch, **operands)
                if self.spec.name == "parallax_attention_core":
                    return _execute_fla_parallax_attention(self, torch, **operands)
                if self.spec.name == "wall_attention_core":
                    return _execute_fla_wall_attention(self, torch, **operands)
                if self.spec.name == "moba_selected_attention_core":
                    return _execute_fla_moba_attention(self, torch, **operands)
                return _execute_sdpa(self.spec, torch, **operands)
            if self.spec.family is MixerKernelFamily.RECURRENCE:
                if self.spec.name == "bdh_attention_core":
                    return _execute_bdh_attention(self, torch, **operands)
                if self.spec.name in {"rnn_core", "gru_core", "m2rnn_core"}:
                    return _execute_xma_nonlinear_rnn(self, torch, **operands)
                if self.spec.name == "ttt_linear_core":
                    return _execute_fla_ttt_linear(self, torch, **operands)
                if self.spec.name == "titans_linear_memory_core":
                    return _execute_fla_titans_linear(self, torch, **operands)
                if self.spec.name == "mamba3_siso_core":
                    return _execute_mamba3_siso_adapter(self, torch, **operands)
                if self.spec.name == "mesa_net_core":
                    return _execute_fla_mesa_net(self, torch, **operands)
                if self.spec.rwkv4_memory:
                    return _execute_fla_rwkv4(self, torch, **operands)
                if self.spec.rwkv6_memory:
                    return _execute_fla_rwkv6(self, torch, **operands)
                if self.spec.momentum_delta:
                    return _execute_fla_momentum_delta(self, torch, **operands)
                if self.spec.gated_oja:
                    return _execute_fla_gated_oja(self, torch, **operands)
                if self.spec.comba_rule:
                    return _execute_fla_comba(self, torch, **operands)
                if self.spec.preconditioned_gated_delta:
                    return _execute_fla_pgdn(self, torch, **operands)
                if self.spec.preconditioned_kda:
                    return _execute_fla_pkda(self, torch, **operands)
                if self.spec.slot_attention:
                    return _execute_fla_slot_attention(self, torch, **operands)
                if self.spec.log_linear_attention:
                    return _execute_fla_log_linear_attention(self, torch, **operands)
                if self.spec.generalized_delta_iplr or self.spec.generalized_delta_dplr:
                    return _execute_fla_generalized_delta(self, torch, **operands)
                if self.spec.kda_delta:
                    return _execute_fla_kda(self, torch, **operands)
                if self.spec.mamba2_ssm:
                    return _execute_mamba2_ssm_library(self, torch, **operands)
                if self.spec.gdn2_ssm:
                    return _execute_fla_gdn2(self, torch, **operands)
                if self.spec.polynomial_basis is not PolynomialBasis.NONE:
                    return _execute_fla_polynomial_attention(self, torch, **operands)
                if self.spec.diagonal_hgrn:
                    return _execute_fla_hgrn(self, torch, **operands)
                if self.spec.step_size_discretization:
                    return _execute_mamba_selective_scan(self, torch, **operands)
                return _execute_fla_k2(self, torch, **operands)
        if self.backend is MixerBackend.NATIVE:
            if self.spec.family is MixerKernelFamily.RECURRENCE:
                return _execute_native_diagonal_ssm(self, torch, **operands)
            return _execute_native_sparse_delta(self, torch, **operands)
        if self.spec.name == "atma_gated_delta_decode_core":
            return _execute_atma_gated_delta_reference(self, torch, **operands)
        if self.spec.family is MixerKernelFamily.SOFTMAX:
            if self.spec.name == "kata_attention_core":
                return _execute_kata_attention_reference(self.spec, torch, **operands)
            if self.spec.name == "longformer_attention_core":
                return _execute_longformer_attention_reference(
                    self.spec, torch, **operands
                )
            if self.spec.name == "tucker_attention_core":
                return _execute_tucker_attention_reference(self.spec, torch, **operands)
            if self.spec.name == "differential_attention_core":
                return _execute_differential_attention(
                    self.spec, torch, library=False, **operands
                )
            if self.spec.name == "tda_attention_core":
                return _execute_tda_attention_reference(torch, **operands)
            if self.spec.name in {
                "polar_attention_core",
                "foveal_sparse_polar_attention_core",
            }:
                return _execute_polar_equation(self.spec.name, torch, **operands)
            if self.spec.deltaformer_attention:
                return _execute_deltaformer_reference(torch, **operands)
            return _execute_softmax(self.spec, torch, **operands)
        if self.spec.family is MixerKernelFamily.RECURRENCE:
            if self.spec.name in {"rnn_core", "gru_core", "m2rnn_core"}:
                return _execute_xma_nonlinear_rnn_reference(
                    self.spec.name, torch, **operands
                )
            if self.spec.name == "ttt_linear_core":
                return _execute_ttt_linear_reference(torch, **operands)
            if self.spec.name == "titans_linear_memory_core":
                return _execute_titans_linear_reference(torch, **operands)
            if self.spec.name == "mamba3_siso_core":
                return _execute_mamba3_siso_reference(torch, **operands)
            if self.spec.name == "mesa_net_core":
                return _execute_mesa_net_reference(torch, **operands)
            if self.spec.rwkv4_memory:
                return _execute_rwkv4_reference(torch, **operands)
            if self.spec.rwkv6_memory:
                return _execute_rwkv6_reference(self.spec, torch, **operands)
            if self.spec.momentum_delta:
                return _execute_momentum_delta_reference(self.spec, torch, **operands)
            if self.spec.gated_oja:
                return _execute_gated_oja_reference(self.spec, torch, **operands)
            if self.spec.comba_rule:
                return _execute_comba_reference(self.spec, torch, **operands)
            if self.spec.preconditioned_gated_delta:
                return _execute_pgdn_reference(self.spec, torch, **operands)
            if self.spec.preconditioned_kda:
                return _execute_pgdn_reference(self.spec, torch, **operands)
            if self.spec.slot_attention:
                return _execute_slot_attention_reference(self.spec, torch, **operands)
            if self.spec.log_linear_attention:
                return _execute_log_linear_attention_reference(
                    self.spec, torch, **operands
                )
            if self.spec.mamba2_ssm:
                return _execute_mamba2_ssm_reference(self.spec, torch, **operands)
            if self.spec.gdn2_ssm:
                return _execute_gdn2_reference(self.spec, torch, **operands)
            if self.spec.polynomial_basis is not PolynomialBasis.NONE:
                return _execute_matrix_recurrence(self.spec, torch, **operands)
            if self.spec.recurrent_layout is RecurrentLayout.DIAGONAL:
                return _execute_diagonal_ssm(self.spec, torch, **operands)
            return _execute_matrix_recurrence(self.spec, torch, **operands)
        return _execute_sparse_delta(self.spec, torch, **operands)

    __call__ = execute


def compile_mixer(
    spec: UnifiedMixerSpec | MixerRecipe,
    *,
    intent: MixerIntent | str = MixerIntent.INFERENCE,
    backend: MixerBackend | str = MixerBackend.REFERENCE,
    dtype: str = "float32",
) -> CompiledMixerPlan:
    """Compile a K1/K2/K3 recipe through URM's typed compiler pipeline.

    ``dtype`` is the compile-time tensor dtype used for anchor training checks.
    Runtime tensors are still checked by the selected executor because shapes
    and devices are not part of ``UnifiedMixerSpec``.
    """
    recipe = spec if isinstance(spec, MixerRecipe) else None
    if recipe is not None:
        spec = recipe.spec
    if not isinstance(spec, UnifiedMixerSpec):
        raise TypeError("compile_mixer expects UnifiedMixerSpec or MixerRecipe")
    resolved_intent = MixerIntent(intent)
    resolved_backend = MixerBackend(backend)
    dtype = str(dtype)
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported mixer compile dtype {dtype!r}")
    if resolved_backend is MixerBackend.NATIVE and not (
        spec.family is MixerKernelFamily.SPARSE_DELTA
        or (
            spec.family is MixerKernelFamily.RECURRENCE
            and spec.recurrent_layout is RecurrentLayout.DIAGONAL
        )
    ):
        raise ValueError(
            "URM-native unified anchors support K3 sparse delta or K2 diagonal SSM"
        )
    if (
        resolved_backend is MixerBackend.LIBRARY
        and spec.family is MixerKernelFamily.SPARSE_DELTA
    ):
        raise ValueError("K3 uses the URM-native or reference sparse-state anchor")
    library_k2_anchor = None
    if (
        resolved_backend is MixerBackend.LIBRARY
        and spec.family is MixerKernelFamily.RECURRENCE
    ):
        if spec.name == "atma_gated_delta_decode_core":
            library_k2_anchor = "atma_gated_delta_decode_adapter"
        elif spec.name == "h3_ssm_fft_core":
            library_k2_anchor = "h3_ssm_fft_convolution_adapter"
        elif spec.name == "hyena_fftconv_core":
            library_k2_anchor = "hyena_fft_convolution_adapter"
        elif spec.name == "hla_second_order_core":
            library_k2_anchor = "hla_second_order_triton_adapter"
        elif spec.name == "bdh_attention_core":
            library_k2_anchor = "bdh_attention_adapter"
        elif spec.log_linear_attention:
            library_k2_anchor = "fla_chunk_log_linear_attention_adapter"
        elif spec.name in {"rnn_core", "gru_core", "m2rnn_core"}:
            library_k2_anchor = f"xma_{spec.name.removesuffix('_core')}_triton_adapter"
        elif spec.name == "titans_linear_memory_core":
            library_k2_anchor = "fla_chunk_titans_linear_adapter"
        elif spec.name == "ttt_linear_core":
            library_k2_anchor = "fla_chunk_ttt_linear_adapter"
        elif spec.name == "mamba3_siso_core":
            library_k2_anchor = "mamba3_siso_combined_adapter"
        elif spec.name == "mesa_net_core":
            library_k2_anchor = "fla_chunk_mesa_net_adapter"
        elif spec.rwkv4_memory:
            library_k2_anchor = "fla_fused_recurrent_rwkv4_adapter"
        elif spec.rwkv6_memory:
            library_k2_anchor = "fla_fused_recurrent_rwkv6_adapter"
        elif spec.momentum_delta:
            library_k2_anchor = "fla_chunk_momentum_delta_rule_adapter"
        elif spec.gated_oja:
            library_k2_anchor = "fla_chunk_gated_oja_adapter"
        elif spec.comba_rule:
            library_k2_anchor = "fla_chunk_comba_adapter"
        elif spec.preconditioned_gated_delta:
            library_k2_anchor = "fla_chunk_precond_gated_delta_adapter"
        elif spec.preconditioned_kda:
            library_k2_anchor = "fla_chunk_precond_kda_adapter"
        elif spec.slot_attention:
            library_k2_anchor = (
                "fla_chunk_abc_adapter"
                if spec.name == "abc_core"
                else "fla_chunk_gsa_adapter"
            )
        elif spec.generalized_delta_iplr:
            library_k2_anchor = "fla_fused_recurrent_iplr_adapter"
        elif spec.generalized_delta_dplr:
            library_k2_anchor = (
                "fla_chunk_rwkv7_adapter"
                if spec.name == "rwkv7_transition_core"
                else "fla_chunk_dplr_adapter"
            )
        elif spec.gated_delta_product:
            library_k2_anchor = "fla_chunk_gated_delta_product_adapter"
        elif spec.kda_delta:
            library_k2_anchor = "fla_chunk_kda_adapter"
        elif spec.mamba2_ssm:
            library_k2_anchor = "mamba2_ssd_adapter"
        elif spec.gdn2_ssm:
            library_k2_anchor = "fla_chunk_gdn2_adapter"
        elif spec.diagonal_hgrn:
            library_k2_anchor = "fla_fused_recurrent_hgrn_adapter"
        elif spec.step_size_discretization:
            library_k2_anchor = "mamba_selective_scan_adapter"
        else:
            library_k2_anchor = _fla_k2_anchor_name(
                spec, dtype=dtype, intent=resolved_intent
            )
        if library_k2_anchor is None:
            raise ValueError(
                "the FLA library anchors support gated-delta, un-decayed delta, "
                "additive linear attention, and gated-additive K2 subsets only"
            )
        if library_k2_anchor in {
            "mamba_selective_scan_adapter",
            "mamba2_ssd_adapter",
            "fla_chunk_log_linear_attention_adapter",
            "fla_chunk_titans_linear_adapter",
            "fla_fused_recurrent_rwkv4_adapter",
            "fla_fused_recurrent_rwkv6_adapter",
            "fla_fused_recurrent_iplr_adapter",
            "fla_chunk_kda_adapter",
            "fla_fused_recurrent_hgrn_adapter",
            "fla_chunk_gdn2_adapter",
            "bdh_attention_adapter",
            "h3_ssm_fft_convolution_adapter",
            "hyena_fft_convolution_adapter",
            "hla_second_order_triton_adapter",
            "atma_gated_delta_decode_adapter",
        }:
            if dtype != "float32":
                raise ValueError(
                    "this pinned recurrent source adapter is qualified for float32 only"
                )
        elif library_k2_anchor in {
            "fla_chunk_simple_gla_adapter",
            "fla_fused_chunk_based_adapter",
            "fla_parallel_rebased_adapter",
        }:
            if dtype != "float32":
                raise ValueError(
                    "FLA gated-additive chunk anchors require float32 tensors"
                )
        elif library_k2_anchor == "fla_chunk_gla_adapter":
            if dtype != "float32" and not (
                dtype == "bfloat16" and _is_fla_gla_spec(spec)
            ):
                raise ValueError(
                    "FLA chunk GLA training supports float32 and verified bfloat16"
                )
        elif library_k2_anchor == "fla_chunk_mesa_net_adapter":
            if dtype != "bfloat16":
                raise ValueError("the pinned MesaNet adapter is qualified for bfloat16")
        elif library_k2_anchor == "mamba3_siso_combined_adapter":
            if dtype != "bfloat16":
                raise ValueError(
                    "the pinned Mamba-3 SISO adapter is qualified for bfloat16"
                )
        elif library_k2_anchor in {
            "xma_rnn_triton_adapter",
            "xma_gru_triton_adapter",
            "xma_m2rnn_triton_adapter",
        }:
            if dtype != "float32":
                raise ValueError(
                    "the pinned XMA nonlinear recurrent adapters require float32"
                )
        elif dtype not in {"float16", "bfloat16"}:
            raise ValueError("this FLA K2 library anchor requires float16 or bfloat16")
    anchor = {
        MixerKernelFamily.SOFTMAX: "urm.unified.k1.softmax_reference.v1",
        MixerKernelFamily.RECURRENCE: "urm.unified.k2.state_reference.v1",
        MixerKernelFamily.SPARSE_DELTA: "urm.unified.k3.sparse_delta_reference.v1",
    }[spec.family]
    if resolved_backend is MixerBackend.LIBRARY:
        if spec.family is MixerKernelFamily.SOFTMAX:
            anchor = (
                "atma_polar_triton_adapter"
                if spec.name == "polar_attention_core"
                else "atma_polar_sparse_triton_adapter"
                if spec.name == "foveal_sparse_polar_attention_core"
                else "fla_parallel_deltaformer_adapter"
                if spec.deltaformer_attention
                else "fla_parallel_path_attention_adapter"
                if spec.path_attention
                else "fla_parallel_forgetting_attention_adapter"
                if spec.name == "fox"
                else "fla_parallel_parallax_adapter"
                if spec.name == "parallax_attention_core"
                else "fla_parallel_wall_attention_adapter"
                if spec.name == "wall_attention_core"
                else "fla_parallel_moba_adapter"
                if spec.name == "moba_selected_attention_core"
                else "fla_fused_attnres_adapter"
                if spec.name == "attnres_depth_core"
                else "tda_triton_attention_adapter"
                if spec.name == "tda_attention_core"
                else "tucker_triton_attention_adapter"
                if spec.name == "tucker_attention_core"
                else "longformer_sliding_chunks_adapter"
                if spec.name == "longformer_attention_core"
                else "kata_parallel_triton_adapter"
                if spec.name == "kata_attention_core"
                else "fwpkm_selected_softmax_triton_adapter"
                if spec.name == "fwpkm_memory_read_core"
                else "torch.nn.functional.scaled_dot_product_attention"
            )
        else:
            anchor = library_k2_anchor
            assert anchor is not None
    elif resolved_backend is MixerBackend.NATIVE:
        anchor = (
            "urm_native_sparse_state_mixer_v0"
            if spec.family is MixerKernelFamily.SPARSE_DELTA
            else "urm_native_diagonal_ssm_v0"
        )
    from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler

    program = mixer_semantic_program(spec, dtype=dtype)
    compilation = UrmCompiler().compile(
        program,
        intent=CompilationIntent(resolved_intent.value),
        schedule_params=ScheduleParams(anchor_overrides={"mixer": anchor}),
    )
    selected = tuple(step.anchor for step in compilation.plan.steps if step.anchor)
    if selected != (anchor,):
        raise RuntimeError(
            f"URM selected anchors {selected!r}, expected the bound mixer anchor {anchor!r}"
        )
    return CompiledMixerPlan(
        spec=spec,
        intent=resolved_intent,
        anchor=anchor,
        backend=resolved_backend,
        recipe=recipe,
        compiler_result=compilation,
        compile_dtype=dtype,
    )


def mixer_semantic_program(spec: UnifiedMixerSpec, *, dtype: str = "float32"):
    """Build the backend-independent semantic program for one mixer equation."""
    from urm.compiler.semantic import (
        DType,
        SemanticProgram,
        TensorHandle,
        UnifiedMixerAccess,
    )

    try:
        float_dtype = DType(dtype)
    except ValueError as error:
        raise ValueError(f"unsupported mixer compile dtype {dtype!r}") from error
    if float_dtype not in {DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16}:
        raise ValueError(f"unsupported mixer compile dtype {dtype!r}")

    floating_inputs: tuple[str, ...]
    typed_inputs: dict[str, DType] = {}
    integer_inputs: tuple[str, ...] = ()
    bool_inputs: tuple[str, ...] = ()
    output_names: tuple[str, ...]
    if spec.family is MixerKernelFamily.SOFTMAX:
        if spec.name in {"polar_attention_core", "foveal_sparse_polar_attention_core"}:
            floating_inputs = (
                "query",
                "key",
                "value",
                "n_keys",
                "v_null",
                "null_base",
                "null_slope_raw",
                "len_gain_raw",
                "mag_beta_raw",
            )
            typed_inputs["n_keys"] = DType.FLOAT32
            if spec.name == "foveal_sparse_polar_attention_core":
                integer_inputs = ("page_indices", "page_counts")
            output_names = ("output", "auxiliary_output")
        elif spec.deltaformer_attention:
            floating_inputs = ("query", "key", "value", "beta")
        elif spec.path_attention:
            floating_inputs = ("query", "key", "value", "w", "beta", "g")
        elif spec.name == "attnres_depth_core":
            floating_inputs = ("query", "rms_weight", "residuals")
        else:
            floating_inputs = ("query", "key", "value")
        if spec.name == "parallax_attention_core":
            floating_inputs = ("query", "r", "key", "value")
        elif spec.name == "wall_attention_core":
            floating_inputs = ("query", "key", "value", "g")
        if spec.name not in {
            "polar_attention_core",
            "foveal_sparse_polar_attention_core",
        }:
            bool_inputs = ("attention_mask",)
            if spec.accepts_score_bias:
                floating_inputs += ("score_bias",)
            output_names = ("output",)
    elif spec.family is MixerKernelFamily.RECURRENCE:
        if spec.name == "atma_gated_delta_decode_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "gamma",
                "beta",
                "state_table",
            )
            typed_inputs.update(
                gamma=DType.FLOAT32,
                beta=DType.FLOAT32,
                state_table=DType.FLOAT32,
            )
            integer_inputs = ("slots",)
        elif spec.rwkv4_memory:
            floating_inputs = ("w", "u", "k", "v", "state")
        elif spec.rwkv6_memory:
            floating_inputs = (
                "query",
                "key",
                "value",
                "log_decay",
                "bonus",
                "initial_state",
            )
        elif spec.momentum_delta:
            floating_inputs = (
                "query",
                "key",
                "value",
                "p",
                "log_alpha",
                "log_mu",
                "beta",
                "eta",
                "initial_state",
                "initial_normalizer_state",
            )
        elif spec.gated_oja:
            floating_inputs = (
                "query",
                "key",
                "value",
                "gv",
                "beta",
                "initial_state",
            )
        elif spec.comba_rule:
            floating_inputs = (
                "query",
                "key",
                "value",
                "p",
                "g",
                "beta",
                "initial_state",
            )
        elif spec.preconditioned_gated_delta:
            floating_inputs = (
                "query",
                "key",
                "value",
                "g_atk",
                "g",
                "beta_atk",
                "beta",
                "initial_state",
                "initial_A_state",
            )
        elif spec.preconditioned_kda:
            floating_inputs = (
                "query",
                "key",
                "value",
                "g",
                "g_atk",
                "beta_atk",
                "beta",
                "initial_state",
                "initial_A_state",
            )
        elif spec.slot_attention and spec.name == "abc_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "slot_logits",
                "initial_key_state",
                "initial_value_state",
            )
        elif spec.slot_attention:
            floating_inputs = (
                "query",
                "key",
                "value",
                "slot_weights",
                "log_decay",
                "initial_key_state",
                "initial_value_state",
            )
        elif spec.mamba2_ssm:
            floating_inputs = (
                "x",
                "dt",
                "A",
                "B",
                "C",
                "initial_states",
            )
        elif spec.log_linear_attention:
            floating_inputs = (
                "query",
                "key",
                "value",
                "log_decay",
                "level_scales",
            )
        elif spec.gdn2_ssm:
            floating_inputs = (
                "query",
                "key",
                "value",
                "log_decay",
                "erase_gate",
                "write_gate",
                "initial_state",
            )
        elif spec.name == "mamba3_siso_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "adt",
                "dt",
                "trap",
                "query_bias",
                "key_bias",
                "angles",
            )
            typed_inputs.update(
                adt=DType.FLOAT32,
                dt=DType.FLOAT32,
                angles=DType.FLOAT32,
            )
        elif spec.name == "titans_linear_memory_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "w",
                "b",
                "theta",
                "alpha",
                "eta",
                "initial_state",
            )
        elif spec.name == "ttt_linear_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "w",
                "b",
                "eta",
                "initial_state",
                "initial_state_bias",
            )
            typed_inputs.update(
                initial_state=DType.FLOAT32,
                initial_state_bias=DType.FLOAT32,
            )
        elif spec.name == "rnn_core":
            floating_inputs = ("query", "weight", "initial_state")
        elif spec.name == "gru_core":
            floating_inputs = (
                "query",
                "weight",
                "forget_input",
                "forget_weight",
                "reset_input",
                "reset_weight",
                "initial_state",
            )
        elif spec.name == "m2rnn_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "weight",
                "forget_input",
                "initial_state",
            )
        elif spec.name == "mesa_net_core":
            floating_inputs = (
                "query",
                "key",
                "value",
                "log_decay",
                "beta",
                "lamb",
            )
            typed_inputs.update(
                log_decay=DType.FLOAT32,
                beta=DType.FLOAT32,
                lamb=DType.FLOAT32,
            )
        elif spec.recurrent_layout is RecurrentLayout.DIAGONAL:
            if spec.diagonal_hgrn:
                floating_inputs = ("x", "log_decay", "initial_state")
            else:
                floating_inputs = (
                    "x",
                    "input_gate",
                    "read_gate",
                    "log_decay",
                    "initial_state",
                    "skip",
                )
                if spec.step_size_discretization:
                    floating_inputs += ("step_size",)
        else:
            if spec.generalized_delta_iplr:
                floating_inputs = (
                    "query",
                    "key",
                    "value",
                    "transition_alpha",
                    "transition_beta",
                    "initial_state",
                )
            elif spec.generalized_delta_dplr:
                floating_inputs = (
                    "query",
                    "key",
                    "value",
                    "transition_alpha",
                    "transition_beta",
                    "log_decay",
                    "initial_state",
                )
            else:
                floating_inputs = (
                    "query",
                    "key",
                    "value",
                    "beta",
                    "log_decay",
                    "initial_state",
                    "initial_normalizer_state",
                    "update_keys",
                    "update_values",
                    "left_transition",
                    "right_transition",
                )
        output_names = (
            ("output", "final_state", "final_normalizer_state")
            if spec.momentum_delta
            or spec.preconditioned_gated_delta
            or spec.preconditioned_kda
            or spec.name == "ttt_linear_core"
            else ("output", "final_state")
            if spec.gated_oja
            else ("output", "final_state")
            if spec.log_linear_attention or spec.rwkv4_memory or spec.rwkv6_memory
            else (
                ("output", "final_state", "final_normalizer_state")
                if spec.normalizer is StateNormalizer.QUERY_KEY
                else ("output", "final_state")
            )
        )
    else:
        floating_inputs = (
            "memory",
            "read_weights",
            "write_weights",
            "values",
            "beta",
            "log_decay",
        )
        integer_inputs = ("read_indices", "write_indices")
        output_names = ("output", "final_state")

    handles = (
        tuple(
            TensorHandle(name, typed_inputs.get(name, float_dtype), ("...",))
            for name in floating_inputs
        )
        + tuple(TensorHandle(name, DType.INT64, ("...",)) for name in integer_inputs)
        + tuple(TensorHandle(name, DType.BOOL, ("...",)) for name in bool_inputs)
    )
    op_inputs = tuple(handle.name for handle in handles)
    op = UnifiedMixerAccess(
        name="mixer",
        inputs=op_inputs,
        outputs=output_names,
        spec=spec,
    )
    return SemanticProgram.build(
        name=f"unified_mixer:{spec.name}",
        inputs=handles,
        ops=(op,),
        outputs=output_names,
    )


def compile_frontend_mixer(
    spec: Any,
    *,
    intent: MixerIntent | str = MixerIntent.INFERENCE,
    backend: MixerBackend | str = MixerBackend.REFERENCE,
    dtype: str = "float32",
) -> CompiledMixerPlan:
    """Lower the existing declarative ``MixerSpec`` into a kernel recipe.

    Only frontend contracts with a supported kernel equation are accepted.
    Missing gate/projection inputs remain external runtime operands; unsupported
    routing or state semantics decline with a concrete reason.
    """
    from urm.frontend.spec import MixerSpec

    if not isinstance(spec, MixerSpec):
        raise TypeError("compile_frontend_mixer expects urm.frontend.MixerSpec")
    if spec.expert is not None or spec.source_domain.value == "expert":
        raise ValueError(
            "expert routing requires coordinated dispatch and grouped expert kernels"
        )

    recipe_name: str | None = None
    if spec.recurrent is not None:
        algorithm = spec.recurrent.algorithm.value
        recurrent_recipes = {
            "mamba": "mamba1_ssm_core",
            "mamba2": "mamba2_ssm_core",
            "gated_deltanet": "gated_delta_net",
            "gated_deltanet2": "gdn2_core",
            "kimi_delta_attention": "kda_core",
        }
        recipe_name = recurrent_recipes.get(algorithm)
        if recipe_name is None:
            raise ValueError(
                f"recurrent algorithm {algorithm!r} has no supported K2 core recipe"
            )
    elif spec.source_domain.value == "memory_page":
        if (
            spec.mutation.value != "in_place_recurrent"
            or spec.collision_policy.value != "ordered"
            or spec.page_size != 1
        ):
            raise ValueError(
                "K3 requires page_size=1, ordered collisions and in-place recurrence"
            )
        recipe_name = "sparse_delta_memory"
    elif spec.source_domain.value == "parameter_block":
        if spec.normalization.value != "softmax":
            raise ValueError("K1 parameter contraction currently requires softmax")
        recipe_name = "pattention_core"
    elif spec.source_domain.value == "sequence":
        if spec.normalization.value != "softmax":
            raise ValueError("K1 sequence attention currently requires softmax")
        if spec.sparse_attention is not None:
            if not spec.sparse_attention.exact_main_attention:
                raise ValueError("approximate main attention has no K1 recipe")
            recipe = named_mixer_recipe("sparse_attention_core")
            recipe = replace(
                recipe,
                architecture_ids=(),
                spec=replace(recipe.spec, name=spec.name),
                component_scope=(f"{spec.name}: exact masked softmax attention core"),
            )
            return compile_mixer(recipe, intent=intent, backend=backend, dtype=dtype)
        if spec.routing.value != "dense":
            raise ValueError(
                "sparse sequence attention requires a typed SparseAttentionSpec"
            )
        recipe = MixerRecipe(
            architecture_ids=(),
            spec=softmax_attention_spec(spec.name),
            component_scope="dense softmax attention core from frontend MixerSpec",
            required_external_stages=("Q/K/V projections and positional transforms",),
        )
        return compile_mixer(recipe, intent=intent, backend=backend, dtype=dtype)
    else:
        raise ValueError(
            f"source domain {spec.source_domain.value!r} has no K1/K2/K3 recipe"
        )

    recipe = named_mixer_recipe(recipe_name)
    recipe = replace(
        recipe,
        spec=replace(recipe.spec, name=spec.name),
        component_scope=f"{spec.name}: {recipe.component_scope}",
    )
    return compile_mixer(recipe, intent=intent, backend=backend, dtype=dtype)


def _is_fla_gated_delta_spec(spec: UnifiedMixerSpec) -> bool:
    return (
        spec.family is MixerKernelFamily.RECURRENCE
        and spec.recurrent_layout is RecurrentLayout.MATRIX
        and spec.update_rule is StateUpdateRule.DELTA
        and spec.normalizer is StateNormalizer.NONE
        and spec.feature_map in {FeatureMap.IDENTITY, FeatureMap.L2_NORMALIZE}
        and spec.decay is DecayGranularity.HEAD
        and spec.transition is StateTransition.POINTWISE
        and spec.read_timing is ReadTiming.AFTER_UPDATE
    )


def _is_fla_gdn2_spec(spec: UnifiedMixerSpec) -> bool:
    return spec.gdn2_ssm


def _is_fla_linear_attention_spec(spec: UnifiedMixerSpec) -> bool:
    return (
        spec.family is MixerKernelFamily.RECURRENCE
        and spec.recurrent_layout is RecurrentLayout.MATRIX
        and spec.update_rule is StateUpdateRule.ADDITIVE
        and spec.normalizer in {StateNormalizer.NONE, StateNormalizer.QUERY_KEY}
        and spec.decay is DecayGranularity.NONE
        and spec.transition is StateTransition.POINTWISE
        and spec.read_timing is ReadTiming.AFTER_UPDATE
    )


def _is_fla_simple_gla_spec(spec: UnifiedMixerSpec) -> bool:
    return (
        spec.family is MixerKernelFamily.RECURRENCE
        and spec.recurrent_layout is RecurrentLayout.MATRIX
        and spec.update_rule is StateUpdateRule.ADDITIVE
        and spec.normalizer is StateNormalizer.NONE
        and spec.decay is DecayGranularity.HEAD
        and spec.transition is StateTransition.POINTWISE
        and spec.read_timing is ReadTiming.AFTER_UPDATE
    )


def _is_fla_gla_spec(spec: UnifiedMixerSpec) -> bool:
    return (
        spec.family is MixerKernelFamily.RECURRENCE
        and spec.recurrent_layout is RecurrentLayout.MATRIX
        and spec.update_rule is StateUpdateRule.ADDITIVE
        and spec.normalizer is StateNormalizer.NONE
        and spec.decay is DecayGranularity.KEY_CHANNEL
        and spec.transition is StateTransition.POINTWISE
        and spec.read_timing is ReadTiming.AFTER_UPDATE
    )


def _is_fla_delta_rule_spec(spec: UnifiedMixerSpec) -> bool:
    return (
        spec.family is MixerKernelFamily.RECURRENCE
        and spec.recurrent_layout is RecurrentLayout.MATRIX
        and spec.update_rule is StateUpdateRule.DELTA
        and spec.normalizer is StateNormalizer.NONE
        and spec.feature_map is FeatureMap.IDENTITY
        and spec.decay is DecayGranularity.NONE
        and spec.transition is StateTransition.POINTWISE
        and spec.read_timing is ReadTiming.AFTER_UPDATE
    )


def _fla_k2_anchor_name(
    spec: UnifiedMixerSpec,
    *,
    dtype: str = "float16",
    intent: MixerIntent = MixerIntent.INFERENCE,
) -> str | None:
    if spec.log_linear_attention:
        return "fla_chunk_log_linear_attention_adapter"
    if spec.generalized_delta_iplr:
        return "fla_fused_recurrent_iplr_adapter"
    if spec.generalized_delta_dplr:
        return "fla_chunk_dplr_adapter"
    if spec.gated_delta_product:
        return "fla_chunk_gated_delta_product_adapter"
    if spec.kda_delta:
        return "fla_chunk_kda_adapter"
    if spec.polynomial_basis is PolynomialBasis.BASED_TAYLOR2:
        return "fla_fused_chunk_based_adapter"
    if spec.polynomial_basis is PolynomialBasis.REBASED_SQUARE:
        return "fla_parallel_rebased_adapter"
    if _is_fla_gdn2_spec(spec):
        return "fla_chunk_gdn2_adapter"
    if spec.diagonal_hgrn:
        return "fla_fused_recurrent_hgrn_adapter"
    if _is_fla_gated_delta_spec(spec):
        return "fla_gated_delta_rule_adapter"
    if _is_fla_gla_spec(spec):
        if dtype == "float32" or (
            dtype == "bfloat16" and intent is MixerIntent.TRAINING
        ):
            return "fla_chunk_gla_adapter"
        return "fla_fused_recurrent_gla_decode_adapter"
    if _is_fla_simple_gla_spec(spec):
        return (
            "fla_chunk_simple_gla_adapter"
            if dtype == "float32"
            else "fla_fused_recurrent_simple_gla_decode_adapter"
        )
    if _is_fla_linear_attention_spec(spec):
        return "fla_chunk_linear_attention_adapter"
    if _is_fla_delta_rule_spec(spec):
        return "fla_chunk_delta_rule_adapter"
    return None


def softmax_attention_spec(
    name: str = "softmax_attention",
    *,
    causal: bool = True,
    scale: float | None = None,
    score_bias: bool = False,
) -> UnifiedMixerSpec:
    return UnifiedMixerSpec(
        name=name,
        family=MixerKernelFamily.SOFTMAX,
        causal=causal,
        attention_scale=scale,
        accepts_score_bias=score_bias,
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
            replace(attention, name=alias),
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
        replace(attention, name="sparse_attention_core"),
        "exact softmax attention for a caller-supplied boolean/additive mask",
        ("architecture-specific indexer/selection", "sparse traversal kernel"),
    )
    recipes["longformer_attention_core"] = MixerRecipe(
        ("arch-071",),
        softmax_attention_spec(
            "longformer_attention_core", causal=False, score_bias=False
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
        softmax_attention_spec("kata_attention_core", score_bias=False),
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
            "fwpkm_memory_read_core", causal=False, scale=1.0, score_bias=False
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
        ),
        "HLA masked second-order causal attention with exact streaming summaries",
        (
            "Higher-order (>2) HLA and asymmetric/decayed variants",
            "Transformer projections, chunk-scan scheduling and streaming cache ABI",
        ),
    )
    recipes["cat_attention_core"] = MixerRecipe(
        ("arch-066",),
        softmax_attention_spec("cat_attention_core", score_bias=False),
        "CAT Compress And Attend causal attention over prior compressed tokens and the current local block",
        (
            "chunk compression and compressed-token construction",
            "separator/adaptive tokens, rotary transform and Q/K/V projections",
            "compressor transformer and complete CAT decoder layer",
        ),
    )
    recipes["differential_attention_core"] = MixerRecipe(
        ("arch-067",),
        softmax_attention_spec("differential_attention_core", score_bias=False),
        "Differential Transformer V1 paired causal softmax reductions and learned subtraction weight",
        (
            "differential Q/K/V projections and RoPE",
            "lambda-vector parameterization, per-head RMSNorm, output scale and projection",
        ),
    )
    recipes["tda_attention_core"] = MixerRecipe(
        ("arch-068",),
        softmax_attention_spec("tda_attention_core", score_bias=False),
        "Threshold Differential Attention's pair of causal rectified score reductions",
        (
            "TDA threshold beta and lambda production",
            "Q/K normalization and complete projection/output layer",
        ),
    )
    recipes["polar_attention_core"] = MixerRecipe(
        ("arch-064",),
        softmax_attention_spec("polar_attention_core", score_bias=False),
        "ATMA Polar causal direction and bounded-magnitude reduction with a learned null sink",
        (
            "Q/K/V projections, GQA expansion, canonical convolution, and output/count projections",
        ),
    )
    recipes["foveal_sparse_polar_attention_core"] = MixerRecipe(
        ("arch-065",),
        softmax_attention_spec("foveal_sparse_polar_attention_core", score_bias=False),
        "ATMA Foveal local-window plus selected remote-page Polar reduction",
        (
            "geometric page routing and its gradients, projections, GQA expansion, and full attention layer",
        ),
    )
    recipes["nsa_selected_attention_core"] = MixerRecipe(
        ("arch-005",),
        softmax_attention_spec("nsa_selected_attention_core"),
        "NSA selected-block causal attention from caller-supplied block routes",
        (
            "NSA compression and indexer routes",
            "multi-branch gate composition and full layer",
        ),
    )
    recipes["moba_selected_attention_core"] = MixerRecipe(
        ("arch-006",),
        softmax_attention_spec("moba_selected_attention_core", score_bias=False),
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
        softmax_attention_spec("dsa_attention_core"),
        "causal softmax attention restricted to caller-supplied DSA token indices",
        (
            "DSA indexer objective and token selection",
            "model projections and full layer",
        ),
    )
    recipes["deltaformer_attention_core"] = MixerRecipe(
        ("arch-013",),
        replace(
            softmax_attention_spec("deltaformer_attention_core", score_bias=False),
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
            softmax_attention_spec(alias, score_bias=False)
            if alias in {"parallax_attention_core", "wall_attention_core"}
            else softmax_attention_spec(
                alias, causal=alias != "tucker_attention_core", score_bias=False
            )
            if alias in {"tpa_attention_core", "tucker_attention_core"}
            else replace(
                softmax_attention_spec(alias, score_bias=False),
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
        softmax_attention_spec("attnres_depth_core", causal=False, scale=1.0),
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


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "unified mixer execution requires PyTorch; install urm-kernel-lab[torch]"
        ) from error
    return torch


def _require_shape(tensor: Any, expected: tuple[int | None, ...], name: str) -> None:
    shape = tuple(tensor.shape)
    if len(shape) != len(expected) or any(
        want is not None and got != want for got, want in zip(shape, expected)
    ):
        raise ValueError(f"{name} must have shape {expected}, got {shape}")


def _feature(torch: Any, value: Any, kind: FeatureMap):
    input_dtype = value.dtype
    value = value.float()
    if kind is FeatureMap.IDENTITY:
        return value
    if kind is FeatureMap.L2_NORMALIZE:
        normalized = value * torch.rsqrt(
            (value * value).sum(dim=-1, keepdim=True) + 1e-6
        )
        return normalized.to(input_dtype).float()
    if kind is FeatureMap.RELU:
        return torch.relu(value)
    if kind is FeatureMap.ELU_PLUS_ONE:
        return torch.where(value > 0, value, torch.expm1(value)) + 1.0
    return torch.nn.functional.softplus(value)


def _expand_attention_operand(value: Any, name: str):
    if value.ndim == 2:
        # [Q,K], broadcast over batch and heads.
        return value
    if value.ndim == 3:
        # [B,Q,K], shared across heads.
        return value.unsqueeze(1)
    if value.ndim == 4:
        return value
    raise ValueError(f"{name} must have rank 2, 3, or 4")


def _deltaformer_operands(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    beta = operands.pop("beta")
    if operands:
        raise TypeError(
            f"unexpected DeltaFormer operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("DeltaFormer query/key/value must share BTHD shape")
    if beta.shape != query.shape[:3]:
        raise ValueError("DeltaFormer beta must use BTH layout")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("DeltaFormer query/key/value must share a dtype")
    if not (query.device == key.device == value.device == beta.device):
        raise ValueError("DeltaFormer inputs must share one device")
    if not (query.is_floating_point() and beta.is_floating_point()):
        raise ValueError("DeltaFormer inputs must be floating point")
    return query, key, value, beta


def _execute_deltaformer_reference(torch: Any, **operands: Any):
    query, key, value, beta = _deltaformer_operands(torch, **operands)
    batch, sequence, heads, key_dim = query.shape
    q = query.float().transpose(1, 2)
    k = key.float().transpose(1, 2)
    v = value.float().transpose(1, 2)
    beta = beta.float().transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-1, -2)) * (key_dim**-0.5)
    positions = torch.arange(sequence, device=query.device)
    strict_causal = positions[None, :] < positions[:, None]
    masked_scores = scores.masked_fill(~strict_causal, -float("inf"))
    row_max = masked_scores.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, 0.0)
    unnormalized = torch.where(
        strict_causal,
        torch.exp(scores - row_max),
        0.0,
    )
    probabilities = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(
        1e-20
    )
    system = torch.eye(sequence, device=query.device, dtype=torch.float32)
    system = system.view(1, 1, sequence, sequence) + beta.unsqueeze(-1) * probabilities
    transformed_value = torch.linalg.solve_triangular(system, v, upper=False)

    causal = positions[None, :] <= positions[:, None]
    causal_scores = scores.masked_fill(~causal, -float("inf"))
    attention = torch.softmax(causal_scores, dim=-1)
    output = torch.matmul(attention, transformed_value.to(query.dtype).float())
    return MixerResult(
        output.transpose(1, 2).to(query.dtype),
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "execution": "torch_eager_deltaformer_triangular_value_solve",
            "backward_supported": True,
        },
    )


def _execute_fla_deltaformer(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query, key, value, beta = _deltaformer_operands(torch, **operands)
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned DeltaFormer adapter requires CUDA FP16 or BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("DeltaFormer requires the exact recorded FLA source")
    from fla.ops.deltaformer import deltaformer_attn

    output = deltaformer_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        beta.contiguous(),
        C=32,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_fla_parallel_deltaformer",
            "backward_supported": True,
        },
    )


def _execute_sdpa(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    mask = operands.pop("attention_mask", None)
    score_bias = operands.pop("score_bias", None)
    if operands:
        raise TypeError(f"unexpected K1 operands: {', '.join(sorted(operands))}")
    if (
        spec.name
        in {
            "dsa_attention_core",
            "nsa_selected_attention_core",
            "moba_selected_attention_core",
        }
        and mask is None
    ):
        raise ValueError(
            "sparse K1 attention requires precomputed selected-token attention_mask"
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("K1 query/key/value use BTHD rank-4 layout")
    batch, q_len, q_heads, key_dim = query.shape
    b_key, k_len, kv_heads, key_dim_k = key.shape
    if min(batch, q_len, q_heads, key_dim, k_len, kv_heads, value.shape[-1]) <= 0:
        raise ValueError(
            "K1 batch, sequence, head, and feature dimensions must be positive"
        )
    if (batch, key_dim) != (b_key, key_dim_k) or value.shape[:3] != (
        batch,
        k_len,
        kv_heads,
    ):
        raise ValueError("K1 query/key/value dimensions do not agree")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if not (
        query.is_floating_point()
        and key.is_floating_point()
        and value.is_floating_point()
    ):
        raise ValueError("K1 query/key/value must be floating point")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("K1 query/key/value must use the same dtype")
    if not (query.device == key.device == value.device):
        raise ValueError("K1 query/key/value must share a device")
    if score_bias is not None and not spec.accepts_score_bias:
        raise ValueError("this K1 recipe does not accept score_bias")
    if mask is not None and mask.device != query.device:
        raise ValueError("attention_mask must share the query device")
    if score_bias is not None and score_bias.device != query.device:
        raise ValueError("score_bias must share the query device")

    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    enable_gqa = q_heads != kv_heads

    combined_mask = None
    causal_strategy = "none"
    if mask is not None:
        if mask.dtype is not torch.bool and not mask.is_floating_point():
            raise ValueError("attention_mask must be boolean or floating point")
        combined_mask = _expand_attention_operand(mask, "attention_mask")
    if score_bias is not None:
        if not score_bias.is_floating_point():
            raise ValueError("score_bias must be floating point")
        bias = _expand_attention_operand(score_bias, "score_bias")
        if combined_mask is None:
            combined_mask = bias
        elif combined_mask.dtype is torch.bool:
            # Convert allowed/disallowed positions to SDPA's additive mask.
            combined_mask = bias.masked_fill(~combined_mask, float("-inf"))
        else:
            combined_mask = combined_mask + bias
    if (
        combined_mask is not None
        and combined_mask.is_floating_point()
        and combined_mask.dtype != query.dtype
    ):
        # PyTorch SDPA requires additive masks to use the query dtype. Keep the
        # public K1 contract flexible and cast only at this backend boundary.
        combined_mask = combined_mask.to(dtype=query.dtype)
    is_causal = False
    if spec.causal and q_len == k_len and combined_mask is None:
        # Preserve the fused SDPA causal path when query and key positions
        # share the same origin.
        is_causal = True
        causal_strategy = "sdpa_is_causal"
    elif spec.causal and q_len == 1 and combined_mask is None:
        # A one-token cached decode query is aligned to the final key position,
        # so every cached key is visible without a materialized mask.
        causal_strategy = "single_query_cached_decode"
    elif spec.causal:
        q_positions = torch.arange(q_len, device=query.device) + (k_len - q_len)
        k_positions = torch.arange(k_len, device=query.device)
        causal_mask = k_positions[None, :] <= q_positions[:, None]
        causal_mask = causal_mask[None, None]
        causal_strategy = "explicit_bottom_right_mask"
        if combined_mask is None:
            combined_mask = causal_mask
        elif combined_mask.dtype is torch.bool:
            combined_mask = combined_mask & causal_mask
        else:
            combined_mask = combined_mask.masked_fill(~causal_mask, float("-inf"))
    output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=combined_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=spec.attention_scale or (key_dim**-0.5),
        enable_gqa=enable_gqa,
    )
    return MixerResult(
        output.transpose(1, 2),
        metadata={
            "anchor": "torch.nn.functional.scaled_dot_product_attention",
            "library_backend": "PyTorch SDPA dispatcher",
            "enable_gqa": enable_gqa,
            "causal_strategy": causal_strategy,
            "execution": "trusted_library_anchor",
        },
    )


def _execute_fwpkm_selected_read(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(f"unexpected K1 operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FwPKM K1 inputs use BTHD rank-4 layout")
    rows, q_len, q_heads, q_dim = query.shape
    k_rows, k_len, kv_heads, k_dim = key.shape
    if (
        q_len != 1
        or q_heads != 1
        or q_dim != 1
        or (rows, k_len, kv_heads, k_dim) != (k_rows, value.shape[1], 1, 1)
        or value.shape[0] != rows
        or value.shape[2] != 1
    ):
        raise ValueError(
            "FwPKM K1 expects unit queries [rows,1,1,1], scalar selected logits "
            "[rows,selected,1,1], and values [rows,selected,1,width]"
        )
    if not (query.dtype == key.dtype == value.dtype == torch.float32):
        raise TypeError("FwPKM selected-read K1 currently supports float32")
    if not (query.device == key.device == value.device):
        raise ValueError("FwPKM K1 inputs must share a device")
    from urm.backends.selected_softmax import selected_softmax_read

    scores = key[:, :, 0, 0] * query[:, 0, 0, 0].unsqueeze(-1)
    output = selected_softmax_read(scores, value[:, :, 0, :])
    return MixerResult(
        output[:, None, None, :],
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_triton_k1_selected_softmax_value_reduction",
            "backward_supported": True,
        },
    )


def _execute_h3_ssm_fft(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    ssm_kernel = operands.pop("ssm_kernel")
    ssm_k_kernel = operands.pop("ssm_k_kernel")
    ssm_k_direct = operands.pop("ssm_k_direct")
    skip = operands.pop("skip")
    if operands:
        raise TypeError(f"unexpected H3 K2 operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("H3 K2 query/key/value use matching BTHD rank-4 tensors")
    if query.shape[-1] != 1:
        raise ValueError("H3 K2 currently supports the source head_dim=1 path")
    if not all(
        tensor.dtype == torch.float32
        for tensor in (query, key, value, ssm_kernel, ssm_k_kernel, ssm_k_direct, skip)
    ):
        raise TypeError("H3 FFT K2 currently supports float32 operands")
    if not all(
        tensor.device == query.device
        for tensor in (key, value, ssm_kernel, ssm_k_kernel, ssm_k_direct, skip)
    ):
        raise ValueError("H3 K2 operands must share a device")

    def fft_convolution(x, kernel, direct):
        sequence = x.shape[-1]
        fft_size = kernel.shape[-1] + sequence
        kernel_spectrum = torch.fft.rfft(kernel, n=fft_size)
        input_spectrum = torch.fft.rfft(x.to(dtype=kernel.dtype), n=fft_size)
        channel_shape = (1,) * (x.ndim - 2) + (kernel.shape[0], -1)
        kernel_spectrum = kernel_spectrum.reshape(channel_shape)
        direct_shape = (1,) * (x.ndim - 2) + (direct.shape[0], 1)
        output = torch.fft.irfft(kernel_spectrum * input_spectrum, n=fft_size)[
            ..., :sequence
        ]
        return output + direct.reshape(direct_shape) * x

    q = query[..., 0].transpose(1, 2).contiguous()
    k = key[..., 0].transpose(1, 2).contiguous()
    v = value[..., 0].transpose(1, 2).contiguous()
    shifted_key = fft_convolution(k, ssm_k_kernel, ssm_k_direct)
    read = fft_convolution(shifted_key * v, ssm_kernel, skip)
    output = (read * q).transpose(1, 2).unsqueeze(-1)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": (
                "h3_two_stage_causal_fft_k2_library_adapter"
                if plan.backend is MixerBackend.LIBRARY
                else "h3_two_stage_causal_fft_k2_reference"
            ),
            "backward_supported": True,
        },
    )


def _execute_hyena_fftconv(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    kernel = operands.pop("kernel")
    direct = operands.pop("direct")
    if operands:
        raise TypeError(f"unexpected Hyena K2 operands: {', '.join(sorted(operands))}")
    if query.ndim != 3 or kernel.ndim != 2 or direct.ndim != 1:
        raise ValueError(
            "Hyena FFT K2 expects query [batch,time,channels], "
            "kernel [channels,time], direct [channels]"
        )
    batch, sequence, channels = query.shape
    if kernel.shape != (channels, sequence) or direct.shape != (channels,):
        raise ValueError(
            "Hyena filter and direct term must match query channel/sequence"
        )
    if not (query.dtype == kernel.dtype == direct.dtype == torch.float32):
        raise TypeError("Hyena FFT K2 currently supports float32 operands")
    if not (query.device == kernel.device == direct.device):
        raise ValueError("Hyena K2 operands must share a device")
    x = query.transpose(1, 2).contiguous()
    fft_size = 2 * sequence
    kernel_spectrum = torch.fft.rfft(kernel, n=fft_size) / fft_size
    input_spectrum = torch.fft.rfft(x.to(dtype=kernel.dtype), n=fft_size)
    output = torch.fft.irfft(
        input_spectrum * kernel_spectrum, n=fft_size, norm="forward"
    )[..., :sequence]
    output = output + x * direct.unsqueeze(-1)
    return MixerResult(
        output.transpose(1, 2),
        metadata={
            "anchor": plan.anchor,
            "execution": (
                "hyena_torch_fft_k2_library_adapter"
                if plan.backend is MixerBackend.LIBRARY
                else "hyena_torch_fft_k2_reference"
            ),
            "backward_supported": True,
        },
    )


def _execute_hla_second_order(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    """Vectorized causal recurrence for paper Eq. (3.3), masked second-order HLA."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(f"unexpected HLA operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError(
            "HLA expects query/key [batch,time,heads,key_dim] and value [batch,time,heads,value_dim]"
        )
    if value.shape[:3] != query.shape[:3]:
        raise ValueError(
            "HLA query, key and value batch/time/head dimensions must match"
        )
    if not (query.dtype == key.dtype == value.dtype == torch.float32):
        raise TypeError("HLA second-order K2 currently supports float32 operands")
    if not (query.device == key.device == value.device):
        raise ValueError("HLA operands must share a device")

    if plan.backend is MixerBackend.LIBRARY and query.is_cuda:
        from urm.backends.hla_triton import hla_second_order_triton

        output = hla_second_order_triton(query, key, value)
        return MixerResult(
            output,
            metadata={
                "anchor": plan.anchor,
                "execution": "hla_masked_second_order_triton_forward_reverse_scan",
                "backward_supported": True,
            },
        )

    # Inclusive prefixes S_t=Σk_i k_iᵀ and C_t=Σq_i v_iᵀ.
    delta_s = key.unsqueeze(-1) * key.unsqueeze(-2)
    delta_c = query.unsqueeze(-1) * value.unsqueeze(-2)
    state_s = delta_s.cumsum(dim=1)
    state_c = delta_c.cumsum(dim=1)
    previous_c = state_c - delta_c

    # The masked correction G_t=Σ_i k_i(k_iᵀ C_{i-1}) removes terms
    # whose value index is after the inner key index.
    key_previous_c = torch.matmul(key.unsqueeze(-2), previous_c).squeeze(-2)
    delta_g = key.unsqueeze(-1) * key_previous_c.unsqueeze(-2)
    masked_correction = delta_g.cumsum(dim=1)
    # Evaluate qᵀ(SC) as (qᵀS)C, matching the paper's O(D² + D·Dv)
    # per-token read rather than materializing the full D×Dv state product.
    query_state = torch.matmul(query.unsqueeze(-2), state_s).squeeze(-2)
    output = torch.matmul(query_state.unsqueeze(-2), state_c).squeeze(-2)
    output = output - torch.matmul(query.unsqueeze(-2), masked_correction).squeeze(-2)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": (
                "hla_masked_second_order_cumulative_k2_library_adapter"
                if plan.backend is MixerBackend.LIBRARY
                else "hla_masked_second_order_cumulative_k2_reference"
            ),
            "backward_supported": True,
        },
    )


def _execute_fla_forgetting_attention(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    if operands:
        raise TypeError(f"unexpected FoX operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FoX Q/K/V use BTHD rank-4 layout")
    if log_decay.shape != query.shape[:3]:
        raise ValueError("FoX log_decay must use BTH query-head layout")
    if not (query.device == key.device == value.device == log_decay.device):
        raise ValueError("FoX Q/K/V/log_decay must share a device")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("FoX Q/K/V must use the same dtype")
    from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn

    output = parallel_forgetting_attn(
        query,
        key,
        value,
        log_decay,
        scale=plan.spec.attention_scale or key.shape[-1] ** -0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": "fla_parallel_forgetting_attention_adapter",
            "execution": "pinned_fla_parallel_forgetting_attention",
        },
    )


def _execute_fla_parallax_attention(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    query = operands.pop("query")
    secondary_query = operands.pop("r")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(f"unexpected Parallax operands: {', '.join(sorted(operands))}")
    from fla.ops.parallax.parallel import parallel_parallax

    output = parallel_parallax(
        query,
        secondary_query,
        key,
        value,
        scale=plan.spec.attention_scale or key.shape[-1] ** -0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": "fla_parallel_parallax_adapter",
            "execution": "pinned_fla_parallel_parallax",
        },
    )


def _execute_fla_wall_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("g")
    if operands:
        raise TypeError(f"unexpected Wall operands: {', '.join(sorted(operands))}")
    import os

    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    from fla.ops.wall_attn.parallel import parallel_wall_attn

    output = parallel_wall_attn(
        query,
        key,
        value,
        log_decay,
        scale=plan.spec.attention_scale or key.shape[-1] ** -0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_fla_parallel_wall_attention",
        },
    )


def _execute_fla_path_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query, key, value, weight, beta, gate = _path_attention_operands(torch, **operands)
    if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("the pinned FLA PaTH adapter requires CUDA FP16 or BF16")
    if query.shape[-1] not in {16, 32, 64, 128} or value.shape[-1] not in {
        16,
        32,
        64,
        128,
    }:
        raise RuntimeError(
            "pinned PaTH kernels support key/value widths 16, 32, 64 or 128"
        )
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("the PaTH adapter requires the exact recorded FLA source")
    from fla.ops.path_attn.parallel import parallel_path_attn

    output, _ = parallel_path_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        weight.contiguous(),
        beta.contiguous(),
        gate.contiguous(),
        scale=query.shape[-1] ** -0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_fla_parallel_path_attention",
            "upstream": identity,
            "backward_supported": True,
        },
    )


def _execute_fla_moba_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    operands.pop("attention_mask", None)
    cu_seqlens = operands.pop("cu_seqlens")
    max_seqlen = operands.pop("max_seqlen")
    chunk_size = operands.pop("chunk_size")
    topk = operands.pop("topk")
    if operands:
        raise TypeError(f"unexpected MoBA operands: {', '.join(sorted(operands))}")
    from fla.ops.moba.parallel import parallel_moba

    output = parallel_moba(
        query,
        key,
        value,
        cu_seqlens,
        max_seqlen=max_seqlen,
        chunk_size=chunk_size,
        topk=topk,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": "fla_parallel_moba_adapter",
            "execution": "pinned_fla_parallel_moba",
        },
    )


def _execute_fla_attnres(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    rms_weight = operands.pop("rms_weight")
    residuals = tuple(operands.pop("residuals"))
    output_rms_weight = operands.pop("output_rms_weight", None)
    rms_eps = float(operands.pop("rms_eps", 1e-6))
    scale = float(operands.pop("scale", 1.0))
    checkpoint_level = int(operands.pop("checkpoint_level", 1))
    return_weights = bool(operands.pop("return_weights", False))
    if operands:
        raise TypeError(f"unexpected AttnRes operands: {', '.join(sorted(operands))}")
    if return_weights:
        raise ValueError("the AttnRes adapter returns the mixed residual only")
    if not residuals or query.ndim != 1 or rms_weight.shape != query.shape:
        raise ValueError("AttnRes requires a query and RMS weight shaped [width]")
    if any(residual.shape[-1] != query.shape[0] for residual in residuals):
        raise ValueError("AttnRes residual widths must match the query width")
    if any(
        residual.shape != residuals[0].shape
        or residual.device != query.device
        or residual.dtype != query.dtype
        for residual in residuals
    ):
        raise ValueError(
            "AttnRes residuals must have matching shapes, device and dtype"
        )
    if rms_weight.device != query.device or rms_weight.dtype != query.dtype:
        raise ValueError("AttnRes query and RMS weight must share device and dtype")
    if not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("the pinned FLA AttnRes adapter requires CUDA FP16 or BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("the AttnRes adapter requires the exact recorded FLA source")
    from fla.ops.attnres import fused_attnres

    output = fused_attnres(
        query=query,
        residuals=residuals,
        rms_weight=rms_weight,
        output_rms_weight=output_rms_weight,
        rms_eps=rms_eps,
        scale=scale,
        return_weights=False,
        checkpoint_level=checkpoint_level,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": "fla_fused_attnres_adapter",
            "execution": "pinned_fla_fused_attnres",
            "upstream": identity,
            "backward_supported": True,
        },
    )


def _path_attention_operands(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    weight = operands.pop("w")
    beta = operands.pop("beta")
    gate = operands.pop("g")
    if operands:
        raise TypeError(f"unexpected PaTH operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("PaTH query/key/value must use BTHD layout")
    batch, sequence, query_heads, key_dim = query.shape
    key_heads = key.shape[2]
    if key.shape != (batch, sequence, key_heads, key_dim):
        raise ValueError("PaTH key must align with query batch/time/key width")
    if value.shape[:3] != (batch, sequence, key_heads):
        raise ValueError("PaTH value must align with key batch/time/head axes")
    if weight.shape != key.shape or beta.shape != (batch, sequence, key_heads):
        raise ValueError("PaTH w and beta must use BTHK and BTH layouts")
    if gate.shape != (batch, sequence, query_heads):
        raise ValueError("PaTH forget gate must use BTHQ layout")
    if query_heads % key_heads:
        raise ValueError("PaTH query heads must be divisible by key/value heads")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("PaTH query/key/value must use one dtype")
    if (
        weight.dtype != torch.float32
        or beta.dtype != torch.float32
        or gate.dtype != torch.float32
    ):
        raise ValueError("PaTH w, beta and g must be float32")
    if not (
        query.device
        == key.device
        == value.device
        == weight.device
        == beta.device
        == gate.device
    ):
        raise ValueError("PaTH operands must share one device")
    return query, key, value, weight, beta, gate


def _execute_path_attention_reference(torch: Any, **operands: Any):
    query, key, value, weight, beta, gate = _path_attention_operands(torch, **operands)
    original_dtype = query.dtype
    batch, sequence, query_heads, key_dim = query.shape
    key_heads = key.shape[2]
    chunk_size = 64
    pad = (-sequence) % chunk_size
    q = query.float().transpose(1, 2)
    k = key.float().transpose(1, 2).repeat_interleave(query_heads // key_heads, dim=1)
    v = value.float().transpose(1, 2).repeat_interleave(query_heads // key_heads, dim=1)
    w = (
        weight.float()
        .transpose(1, 2)
        .repeat_interleave(query_heads // key_heads, dim=1)
    )
    beta = (
        beta.float().transpose(1, 2).repeat_interleave(query_heads // key_heads, dim=1)
    )
    gate = gate.float().transpose(1, 2)
    gate_cumsum = gate.cumsum(dim=-1)
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        w = torch.nn.functional.pad(w, (0, 0, 0, pad))
        beta = torch.nn.functional.pad(beta, (0, pad))
    num_chunks = q.shape[2] // chunk_size
    q = q.reshape(batch, query_heads, num_chunks, chunk_size, key_dim)
    k = k.reshape(batch, query_heads, num_chunks, chunk_size, key_dim)
    w = w.reshape(batch, query_heads, num_chunks, chunk_size, key_dim)
    w_beta = w * beta.unsqueeze(-1).reshape(
        batch, query_heads, num_chunks, chunk_size, 1
    )
    upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )
    triangular = -(w_beta @ w.transpose(-1, -2)).masked_fill(upper, 0.0)
    for row in range(1, chunk_size):
        correction = (triangular[..., row, :, None] * triangular[..., :, :row]).sum(
            dim=-2
        )
        triangular = triangular.clone()
        triangular[..., row, :row] = triangular[..., row, :row].clone() + correction
    eye = torch.eye(chunk_size, dtype=torch.float32, device=query.device)
    triangular = triangular + eye
    transformed_k = triangular @ (w_beta @ k.transpose(-1, -2)).masked_fill(upper, 0.0)
    qw = (q @ w.transpose(-1, -2)).tril()
    transformed_w = triangular @ w_beta
    local_scores = (q @ k.transpose(-1, -2)).tril() - qw @ transformed_k
    q = q - qw @ transformed_w
    k = k - transformed_k.transpose(-1, -2) @ w
    transition = w.transpose(-1, -2) @ transformed_w
    q = q.reshape(batch, query_heads, num_chunks * chunk_size, key_dim)
    k = k.reshape(batch, query_heads, num_chunks * chunk_size, key_dim)

    score_rows = []
    for chunk in range(num_chunks):
        q_chunk = q[:, :, chunk * chunk_size : (chunk + 1) * chunk_size]
        row_parts: dict[int, Any] = {}
        for previous in range(chunk - 1, -1, -1):
            k_chunk = k[:, :, previous * chunk_size : (previous + 1) * chunk_size]
            row_parts[previous] = q_chunk @ k_chunk.transpose(-1, -2)
            q_chunk = q_chunk - q_chunk @ transition[:, :, previous]
        row_parts[chunk] = local_scores[:, :, chunk]
        for following in range(chunk + 1, num_chunks):
            row_parts[following] = torch.zeros(
                batch,
                query_heads,
                chunk_size,
                chunk_size,
                dtype=torch.float32,
                device=query.device,
            )
        score_rows.append(
            torch.cat([row_parts[index] for index in range(num_chunks)], dim=-1)
        )
    scores = torch.cat(score_rows, dim=-2)[..., :sequence, :sequence]
    causal = torch.tril(
        torch.ones(sequence, sequence, dtype=torch.bool, device=query.device)
    )
    scores = scores.masked_fill(~causal, float("-inf"))
    gate_cumsum = gate_cumsum[..., :sequence]
    scores = scores + gate_cumsum.unsqueeze(-1) - gate_cumsum.unsqueeze(-2)
    attention = torch.softmax(scores * (key_dim**-0.5), dim=-1)
    output = attention @ v[..., :sequence, :]
    return MixerResult(
        output.transpose(1, 2).to(original_dtype),
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "execution": "torch_eager_path_triangular_qk_transform",
            "backward_supported": True,
        },
    )


def _polar_allowed_mask(torch: Any, name: str, query: Any, operands: dict[str, Any]):
    batch, heads, sequence, _ = query.shape
    positions = torch.arange(sequence, device=query.device)
    allowed = positions[None, :] <= positions[:, None]
    if name == "foveal_sparse_polar_attention_core":
        page_indices = operands.pop("page_indices")
        page_counts = operands.pop("page_counts")
        page_size = int(operands.pop("page_size", 16))
        local_window = int(operands.pop("local_window", 16))
        if sequence % page_size or local_window <= 0 or local_window % page_size:
            raise ValueError(
                "Foveal page size and local window must divide the sequence and each other"
            )
        pages = sequence // page_size
        if page_indices.ndim != 3 or page_indices.shape[:2] != (batch, pages):
            raise ValueError(
                "Foveal page_indices must use B,(T/page_size),capacity layout"
            )
        if page_counts.shape != (batch, pages):
            raise ValueError("Foveal page_counts must use B,(T/page_size) layout")
        local = positions[None, :] > (positions[:, None] - local_window)
        allowed = allowed & local
        allowed = (
            allowed.view(1, 1, sequence, sequence).expand(batch, heads, -1, -1).clone()
        )
        for batch_idx in range(batch):
            for query_page in range(pages):
                start, stop = query_page * page_size, (query_page + 1) * page_size
                count = int(page_counts[batch_idx, query_page].item())
                if count < 0 or count > page_indices.shape[-1]:
                    raise ValueError(
                        "Foveal page count exceeds the supplied route capacity"
                    )
                for slot in range(count):
                    key_page = int(page_indices[batch_idx, query_page, slot].item())
                    if key_page < 0 or key_page >= query_page:
                        raise ValueError(
                            "Foveal remote routes must select completed earlier pages"
                        )
                    key_start, key_stop = (
                        key_page * page_size,
                        (key_page + 1) * page_size,
                    )
                    allowed[batch_idx, :, start:stop, key_start:key_stop] = True
        return allowed, positions
    return allowed.view(1, 1, sequence, sequence).expand(
        batch, heads, -1, -1
    ), positions


def _execute_polar_equation(name: str, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    n_keys = operands.pop("n_keys")
    v_null = operands.pop("v_null")
    null_base = operands.pop("null_base")
    null_slope_raw = operands.pop("null_slope_raw")
    len_gain_raw = operands.pop("len_gain_raw")
    mag_beta_raw = operands.pop("mag_beta_raw")
    allowed, positions = _polar_allowed_mask(torch, name, query, operands)
    if operands:
        raise TypeError(
            f"unexpected Polar attention operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape:
        raise ValueError("Polar Q/K/V must share B,H,T,D layout and shape")
    batch, heads, sequence, dim = query.shape
    if n_keys.shape != (sequence,):
        raise ValueError(
            "Polar n_keys must contain one valid-key count per query token"
        )
    if any(
        param.shape != (heads,)
        for param in (null_base, null_slope_raw, len_gain_raw, mag_beta_raw)
    ):
        raise ValueError("Polar scalar parameters must have one value per head")
    if v_null.shape != (heads, dim):
        raise ValueError("Polar v_null must use H,D layout")
    scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) / (dim**0.5)
    scores = scores.masked_fill(~allowed, float("-inf"))
    n = n_keys.float().clamp(min=1.0)
    temperature = 1.0 + torch.nn.functional.softplus(len_gain_raw.float()).view(
        1, heads, 1, 1
    ) * torch.log(n).view(1, 1, sequence, 1)
    null = null_base.float().view(1, heads, 1, 1) + torch.nn.functional.softplus(
        null_slope_raw.float()
    ).view(1, heads, 1, 1) * torch.sqrt(torch.log(n + 1.0)).view(1, 1, sequence, 1)
    masked_scores = torch.isneginf(scores)
    safe_scores = torch.where(masked_scores, torch.zeros_like(scores), scores)
    real_logits = (safe_scores * temperature).masked_fill(masked_scores, float("-inf"))
    logits = torch.cat(
        (real_logits, null.expand(batch, heads, sequence, 1) * temperature), dim=-1
    )
    weights = torch.softmax(logits, dim=-1)
    real_weights, null_weights = weights[..., :-1], weights[..., -1:]
    direction_sum = torch.matmul(
        real_weights, value.float()
    ) + null_weights * v_null.float().view(1, heads, 1, dim)
    direction = torch.nn.functional.normalize(direction_sum, p=2, dim=-1, eps=1e-6)
    normalized = real_weights / real_weights.sum(-1, keepdim=True).clamp_min(1e-6)
    effective = 1.0 / normalized.square().sum(-1).clamp_min(1e-6)
    magnitude = torch.tanh(
        torch.nn.functional.softplus(mag_beta_raw.float()).view(1, heads, 1)
        * torch.log1p(effective * (1.0 - null_weights.squeeze(-1)))
    )
    return MixerResult(
        direction.to(value.dtype),
        auxiliary_output=magnitude.to(value.dtype),
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "state_layout": "polar_direction_and_magnitude",
            "execution": "torch_eager_materialized_equation",
            "valid_key_counts": positions.numel(),
        },
    )


def _execute_atma_polar(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    from kernel.polar_triton import polar_attention, polar_attention_sparse

    name = plan.spec.name
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    n_keys = operands.pop("n_keys")
    v_null = operands.pop("v_null")
    null_base = operands.pop("null_base")
    null_slope_raw = operands.pop("null_slope_raw")
    len_gain_raw = operands.pop("len_gain_raw")
    mag_beta_raw = operands.pop("mag_beta_raw")
    if name == "polar_attention_core":
        output, auxiliary = polar_attention(
            query,
            key,
            value,
            n_keys,
            v_null=v_null,
            null_base=null_base,
            null_slope_raw=null_slope_raw,
            len_gain_raw=len_gain_raw,
            mag_beta_raw=mag_beta_raw,
        )
    else:
        page_indices = operands.pop("page_indices")
        page_counts = operands.pop("page_counts")
        page_size = int(operands.pop("page_size", 16))
        local_window = int(operands.pop("local_window", 16))
        if operands:
            raise TypeError(
                f"unexpected Foveal operands: {', '.join(sorted(operands))}"
            )
        output, auxiliary = polar_attention_sparse(
            query,
            key,
            value,
            page_indices,
            page_counts,
            page_size=page_size,
            local_window=local_window,
            v_null=v_null,
            null_base=null_base,
            null_slope_raw=null_slope_raw,
            len_gain_raw=len_gain_raw,
            mag_beta_raw=mag_beta_raw,
        )
    if operands:
        raise TypeError(
            f"unexpected Polar attention operands: {', '.join(sorted(operands))}"
        )
    return MixerResult(
        output,
        auxiliary_output=auxiliary,
        metadata={
            "anchor": plan.anchor,
            "upstream": "ATMA Triton Polar reduction",
            "backward_supported": True,
        },
    )


def _execute_softmax(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    if spec.path_attention:
        return _execute_path_attention_reference(torch, **operands)
    if spec.name == "parallax_attention_core":
        return _execute_parallax_reference(spec, torch, **operands)
    if spec.name == "wall_attention_core":
        return _execute_wall_reference(spec, torch, **operands)
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    mask = operands.pop("attention_mask", None)
    score_bias = operands.pop("score_bias", None)
    if operands:
        raise TypeError(f"unexpected K1 operands: {', '.join(sorted(operands))}")
    if (
        spec.name
        in {
            "dsa_attention_core",
            "nsa_selected_attention_core",
            "cat_attention_core",
        }
        and mask is None
    ):
        raise ValueError(
            "sparse K1 attention requires precomputed selected-token attention_mask"
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("K1 query/key/value use BTHD rank-4 layout")
    batch, q_len, q_heads, key_dim = query.shape
    b_key, k_len, kv_heads, key_dim_k = key.shape
    if min(batch, q_len, q_heads, key_dim, k_len, kv_heads, value.shape[-1]) <= 0:
        raise ValueError(
            "K1 batch, sequence, head, and feature dimensions must be positive"
        )
    if (batch, key_dim) != (b_key, key_dim_k) or value.shape[:3] != (
        batch,
        k_len,
        kv_heads,
    ):
        raise ValueError("K1 query/key/value dimensions do not agree")
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if not (
        query.is_floating_point()
        and key.is_floating_point()
        and value.is_floating_point()
    ):
        raise ValueError("K1 query/key/value must be floating point")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("K1 query/key/value must use the same dtype")
    if not (query.device == key.device == value.device):
        raise ValueError("K1 query/key/value must share a device")
    if (
        mask is not None
        and mask.dtype is not torch.bool
        and not mask.is_floating_point()
    ):
        raise ValueError("attention_mask must be boolean or floating point")
    if score_bias is not None and not spec.accepts_score_bias:
        raise ValueError("this K1 recipe does not accept score_bias")
    if mask is not None and mask.device != query.device:
        raise ValueError("attention_mask must share the query device")
    if score_bias is not None and score_bias.device != query.device:
        raise ValueError("score_bias must share the query device")

    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    if q_heads != kv_heads:
        groups = q_heads // kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scale = spec.attention_scale or (key_dim**-0.5)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if score_bias is not None:
        if not score_bias.is_floating_point():
            raise ValueError("score_bias must be floating point")
        score_bias = _expand_attention_operand(score_bias, "score_bias")
        scores = scores + score_bias.float()
    if spec.causal:
        q_positions = torch.arange(q_len, device=query.device) + (k_len - q_len)
        k_positions = torch.arange(k_len, device=query.device)
        causal_mask = k_positions[None, :] <= q_positions[:, None]
        scores = scores.masked_fill(~causal_mask[None, None], float("-inf"))
    if mask is not None:
        expanded_mask = _expand_attention_operand(mask, "attention_mask")
        if expanded_mask.dtype is torch.bool:
            scores = scores.masked_fill(~expanded_mask, float("-inf"))
        else:
            scores = scores + expanded_mask.float()
    probabilities = torch.softmax(scores, dim=-1)
    probabilities = torch.nan_to_num(probabilities, nan=0.0)
    output = torch.matmul(probabilities, v).transpose(1, 2).to(value.dtype)
    return MixerResult(
        output,
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "head_mode": "mha" if q_heads == kv_heads else "gqa_or_mqa",
            "execution": "torch_eager_reference",
        },
    )


def _execute_differential_attention(
    spec: UnifiedMixerSpec, torch: Any, *, library: bool, **operands: Any
) -> MixerResult:
    query_a = operands.pop("query_a")
    query_b = operands.pop("query_b")
    key_a = operands.pop("key_a")
    key_b = operands.pop("key_b")
    value = operands.pop("value")
    lambda_weight = operands.pop("lambda_weight")
    if operands:
        raise TypeError(
            f"unexpected differential K1 operands: {', '.join(sorted(operands))}"
        )
    if query_a.ndim != 4 or any(
        item.shape != query_a.shape for item in (query_b, key_a, key_b)
    ):
        raise ValueError("differential Q/K operands must share [B,T,H,K] layout")
    batch, sequence, heads, key_dim = query_a.shape
    if value.ndim != 4 or value.shape[:3] != (batch, sequence, heads):
        raise ValueError("differential values must use [B,T,H,V] layout")
    if lambda_weight.ndim == 0:
        scale = lambda_weight
    elif lambda_weight.shape == (heads,):
        scale = lambda_weight.view(1, 1, heads, 1)
    else:
        raise ValueError("differential lambda_weight must be scalar or [H]")
    attention_scale = spec.attention_scale or key_dim**-0.5
    if library:
        q_a, q_b, k_a, k_b, v = (
            item.transpose(1, 2) for item in (query_a, query_b, key_a, key_b, value)
        )
        output_a = torch.nn.functional.scaled_dot_product_attention(
            q_a, k_a, v, is_causal=spec.causal, scale=attention_scale
        )
        output_b = torch.nn.functional.scaled_dot_product_attention(
            q_b, k_b, v, is_causal=spec.causal, scale=attention_scale
        )
        output = output_a.transpose(1, 2) - scale * output_b.transpose(1, 2)
        implementation = "two_sdpa_reductions_and_differential_combine"
    else:

        def attend(query, key):
            scores = torch.einsum("bthd,bshd->bhts", query, key) * attention_scale
            if spec.causal:
                causal = torch.ones(
                    (sequence, sequence), dtype=torch.bool, device=query.device
                ).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
            return torch.einsum("bhts,bshv->bthv", weights, value)

        output = attend(query_a, key_a) - scale * attend(query_b, key_b)
        implementation = "two_materialized_softmax_equations_and_differential_combine"
    return MixerResult(
        output,
        metadata={
            "anchor": (
                "torch.nn.functional.scaled_dot_product_attention"
                if library
                else "urm.unified.k1.softmax_reference.v1"
            ),
            "execution": implementation,
            "backward_supported": True,
        },
    )


def _execute_tda_attention_reference(torch: Any, **operands: Any) -> MixerResult:
    query_a = operands.pop("query_a")
    query_b = operands.pop("query_b")
    key_a = operands.pop("key_a")
    key_b = operands.pop("key_b")
    value = operands.pop("value")
    beta = operands.pop("beta")
    lambda_weight = operands.pop("lambda_weight")
    if operands:
        raise TypeError(f"unexpected TDA K1 operands: {', '.join(sorted(operands))}")
    if query_a.ndim != 4 or any(
        item.shape != query_a.shape for item in (query_b, key_a, key_b, value)
    ):
        raise ValueError("TDA Q/K/V operands must share [B,T,H,D] layout")
    batch, sequence, heads, dim = query_a.shape
    del batch
    # The pinned TDA implementation normalizes each Q/K branch before its
    # thresholded, unnormalized causal score reduction.
    query_a, query_b, key_a, key_b = (
        torch.nn.functional.normalize(item, p=2, dim=-1)
        for item in (query_a, query_b, key_a, key_b)
    )
    positions = torch.arange(
        1, sequence + 1, device=query_a.device, dtype=torch.float32
    )
    threshold = beta.float() * torch.sqrt(2.0 * torch.log(positions) / dim)
    causal = torch.ones(
        (sequence, sequence), dtype=torch.bool, device=query_a.device
    ).tril()

    def attend(query, key):
        scores = torch.einsum("bthd,bshd->bhts", query.float(), key.float())
        scores = torch.where(causal, scores, torch.zeros_like(scores))
        rectified = torch.relu(scores - threshold.view(1, 1, sequence, 1))
        weights = rectified.square()
        return torch.einsum("bhts,bshv->bthv", weights, value.float()).to(value.dtype)

    coefficient = lambda_weight.clamp(0.0, 1.0)
    output = attend(query_a, key_a) - coefficient * attend(query_b, key_b)
    return MixerResult(
        output,
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "execution": "thresholded_rectified_differential_attention_equation",
            "backward_supported": True,
        },
    )


def _execute_tda_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from urm.adapters.tda import tda_attention_adapter

    output, identity = tda_attention_adapter(**operands)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "backward_supported": True,
        },
    )


def _execute_tucker_attention_reference(
    spec: UnifiedMixerSpec, torch: Any, **operands: Any
) -> MixerResult:
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    B_pre = operands.pop("B_pre")
    if operands:
        raise TypeError(f"unexpected Tucker K1 operands: {', '.join(sorted(operands))}")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3 or B_pre.ndim != 3:
        raise ValueError(
            "Tucker Q/K/V/B_pre operands must use BTR/HDR rank-three layouts"
        )
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ValueError("Tucker Q/K/V batch and sequence dimensions must match")
    if B_pre.shape[1:] != (query.shape[-1], key.shape[-1]):
        raise ValueError("Tucker B_pre dimensions must map query rank to key rank")
    expanded_query = torch.einsum("btr,hrk->bthk", query.float(), B_pre.float()).to(
        query.dtype
    )
    result = _execute_softmax(
        spec,
        torch,
        query=expanded_query,
        key=key.unsqueeze(2),
        value=value.unsqueeze(2),
    )
    result.metadata.update(
        {
            "equation": "softmax((query @ B_pre) @ key^T / sqrt(key_rank)) @ value",
            "execution": "factorized_query_tucker_attention_reference",
        }
    )
    return result


def _execute_tucker_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from urm.adapters.tucker import tucker_attention_adapter

    output, identity = tucker_attention_adapter(**operands)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "backward_supported": True,
        },
    )


def _execute_longformer_attention_reference(
    spec: UnifiedMixerSpec, torch: Any, **operands: Any
) -> MixerResult:
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    attention_window = operands.pop("attention_window")
    if operands:
        raise TypeError(
            f"unexpected Longformer K1 operands: {', '.join(sorted(operands))}"
        )
    if (
        query.ndim != 4
        or query.shape != key.shape
        or query.shape[:3] != value.shape[:3]
    ):
        raise ValueError("Longformer Q/K/V operands must use compatible BTHD layouts")
    sequence = query.shape[1]
    if not isinstance(attention_window, int) or attention_window <= 0:
        raise ValueError("attention_window must be a positive integer")
    positions = torch.arange(sequence, device=query.device)
    local_mask = (positions[:, None] - positions[None, :]).abs() <= attention_window
    result = _execute_softmax(
        spec,
        torch,
        query=query,
        key=key,
        value=value,
        attention_mask=local_mask.view(1, 1, sequence, sequence),
    )
    result.metadata.update(
        {
            "equation": "softmax(Q K^T / sqrt(D), |query_index-key_index|<=window) V",
            "execution": "dense_mask_reference_for_local_sliding_chunks_attention",
        }
    )
    return result


def _execute_longformer_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from urm.adapters.longformer import longformer_attention_adapter

    output, identity = longformer_attention_adapter(**operands)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_longformer_sliding_chunks_adapter",
            "upstream": identity,
            "backward_supported": True,
        },
    )


def _execute_kata_attention_reference(
    spec: UnifiedMixerSpec, torch: Any, **operands: Any
) -> MixerResult:
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    num_groups = operands.pop("num_groups")
    if operands:
        raise TypeError(f"unexpected KATA K1 operands: {', '.join(sorted(operands))}")
    if (
        query.ndim != 4
        or query.shape != key.shape
        or query.shape[:3] != value.shape[:3]
    ):
        raise ValueError("KATA Q/K/V operands must use matching BTHD layouts")
    if not isinstance(num_groups, int) or num_groups not in (1, 2, 4):
        raise ValueError("KATA num_groups must be one of 1, 2, or 4")
    batch, sequence, heads, dim = query.shape
    if dim % num_groups:
        raise ValueError("KATA head dimension must be divisible by num_groups")
    group_dim = dim // num_groups
    q_grouped = query.float().view(batch, sequence, heads, num_groups, group_dim)
    k_grouped = key.float().view(batch, sequence, heads, num_groups, group_dim)
    group_scores = torch.einsum("bthme,bshme->bhtsm", q_grouped, k_grouped) * (
        group_dim**-0.5
    )
    scores = group_scores.square().sum(dim=-1)
    causal = torch.ones(
        sequence, sequence, dtype=torch.bool, device=query.device
    ).tril()
    scores = scores.masked_fill(~causal.view(1, 1, sequence, sequence), 0.0)
    denominator = scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probabilities = scores / denominator
    output = torch.einsum("bhts,bshv->bthv", probabilities, value.float()).to(
        value.dtype
    )
    return MixerResult(
        output,
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "execution": "causal_spd_normalized_positive_attention_equation",
            "attention_family": spec.family.value,
            "num_groups": num_groups,
            "backward_supported": True,
        },
    )


def _execute_kata_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from urm.adapters.kata import kata_attention_adapter

    output, identity = kata_attention_adapter(**operands)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_kata_parallel_triton_attention",
            "upstream": identity,
            "backward_supported": True,
        },
    )


def _execute_parallax_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    secondary_query = operands.pop("r")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(f"unexpected Parallax operands: {', '.join(sorted(operands))}")
    if any(tensor.ndim != 4 for tensor in (query, secondary_query, key, value)):
        raise ValueError("Parallax Q/R/K/V use BTHD rank-4 layout")
    if secondary_query.shape != query.shape:
        raise ValueError("Parallax r must match query shape")
    batch, q_len, q_heads, key_dim = query.shape
    b_key, k_len, kv_heads, key_dim_k = key.shape
    if (batch, key_dim) != (b_key, key_dim_k) or value.shape != (
        batch,
        k_len,
        kv_heads,
        key_dim,
    ):
        raise ValueError("Parallax requires matching Q/R/K/V feature widths")
    if q_heads % kv_heads:
        raise ValueError("Parallax query heads must be divisible by KV heads")
    if not (query.dtype == secondary_query.dtype == key.dtype == value.dtype):
        raise ValueError("Parallax Q/R/K/V must use the same dtype")
    if not (query.device == secondary_query.device == key.device == value.device):
        raise ValueError("Parallax Q/R/K/V must share a device")

    q = query.transpose(1, 2).float()
    r = secondary_query.transpose(1, 2).float()
    k = key.transpose(1, 2).float()
    v = value.transpose(1, 2).float()
    if q_heads != kv_heads:
        groups = q_heads // kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    scale = spec.attention_scale or key_dim**-0.5
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    secondary_scores = torch.matmul(r, k.transpose(-1, -2))
    if spec.causal:
        q_positions = torch.arange(q_len, device=query.device) + (k_len - q_len)
        k_positions = torch.arange(k_len, device=query.device)
        causal_mask = k_positions[None, :] <= q_positions[:, None]
        logits = logits.masked_fill(~causal_mask[None, None], float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    correction_probabilities = probabilities * secondary_scores
    ordinary_output = torch.matmul(probabilities, v)
    correction_mean = correction_probabilities.sum(dim=-1, keepdim=True)
    correction_output = torch.matmul(correction_probabilities, v)
    output = (
        (ordinary_output * (1.0 + correction_mean) - correction_output)
        .transpose(1, 2)
        .to(value.dtype)
    )
    return MixerResult(
        output,
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "execution": "torch_eager_parallax_reference",
        },
    )


def _execute_wall_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("g")
    if operands:
        raise TypeError(f"unexpected Wall operands: {', '.join(sorted(operands))}")
    if any(tensor.ndim != 4 for tensor in (query, key, value, log_decay)):
        raise ValueError("Wall Q/K/V/g use BTHD rank-4 layout")
    if log_decay.shape != query.shape:
        raise ValueError("Wall g must match query shape")
    batch, q_len, q_heads, key_dim = query.shape
    b_key, k_len, kv_heads, key_dim_k = key.shape
    if (batch, key_dim) != (b_key, key_dim_k) or value.shape[:3] != (
        batch,
        k_len,
        kv_heads,
    ):
        raise ValueError("Wall Q/K/V dimensions do not agree")
    if q_heads % kv_heads:
        raise ValueError("Wall query heads must be divisible by KV heads")
    if not (query.dtype == key.dtype == value.dtype == log_decay.dtype):
        raise ValueError("Wall Q/K/V/g must use the same dtype")
    if not (query.device == key.device == value.device == log_decay.device):
        raise ValueError("Wall Q/K/V/g must share a device")

    groups = q_heads // kv_heads
    expanded_key = key.repeat_interleave(groups, dim=2).float()
    expanded_value = value.repeat_interleave(groups, dim=2).transpose(1, 2).float()
    prefix = log_decay.float().cumsum(dim=1)
    decay = torch.exp(prefix[:, :, None, :, :] - prefix[:, None, :, :, :])
    scores = (
        (
            query.float()[:, :, None, :, :]
            * expanded_key.float()[:, None, :, :, :]
            * decay
        )
        .sum(dim=-1)
        .permute(0, 3, 1, 2)
    )
    scores = scores * (spec.attention_scale or key_dim**-0.5)
    if spec.causal:
        q_positions = torch.arange(q_len, device=query.device) + (k_len - q_len)
        k_positions = torch.arange(k_len, device=query.device)
        causal_mask = k_positions[None, :] <= q_positions[:, None]
        scores = scores.masked_fill(~causal_mask[None, None], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, expanded_value).transpose(1, 2).to(value.dtype)
    return MixerResult(
        output,
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "execution": "torch_eager_wall_reference",
        },
    )


def _matrix_decay(
    torch: Any, spec: UnifiedMixerSpec, log_decay: Any, t: int, state: Any
):
    if spec.decay is DecayGranularity.NONE:
        if log_decay is not None:
            raise ValueError("log_decay was supplied to a recipe without decay")
        return state
    if log_decay is None:
        raise ValueError("this recurrent recipe requires log_decay")
    if spec.static_head_decay:
        if log_decay.ndim != 1 or log_decay.shape[0] not in (
            1,
            state.shape[1],
        ):
            raise ValueError("static head log_decay must be [H] or [1]")
        return state * torch.exp(log_decay.float())[None, :, None, None]
    value = log_decay[:, t].float()
    if spec.decay is DecayGranularity.HEAD:
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[1] not in (1, state.shape[1]):
            raise ValueError("head log_decay must be [B], [B,1], or [B,Hv]")
        return state * torch.exp(value)[:, :, None, None]
    if value.ndim == 2:
        value = value.unsqueeze(1)
    if value.ndim != 3 or value.shape[1] not in (1, state.shape[1]):
        raise ValueError("key-channel log_decay must be [B,K] or [B,Hv,K]")
    if value.shape[-1] != state.shape[2]:
        raise ValueError("key-channel log_decay width must equal state key width")
    return state * torch.exp(value)[:, :, :, None]


def _bdh_rotary(torch: Any, value: Any):
    if value.ndim != 4 or value.shape[-1] % 2:
        raise ValueError("BDH rotary query/key must use even-width BTHD tensors")
    _, sequence, _, dim = value.shape
    positions = torch.arange(sequence, device=value.device, dtype=torch.float32)
    channels = torch.arange(dim, device=value.device, dtype=torch.float32)
    quantized = torch.floor(channels / 2.0) * 2.0
    frequencies = 1.0 / (65536.0 ** (quantized / dim)) / (2.0 * torch.pi)
    phases = torch.remainder(
        positions.view(1, sequence, 1, 1) * frequencies.view(1, 1, 1, dim),
        1.0,
    ) * (2.0 * torch.pi)
    rotated = torch.stack((-value[..., 1::2], value[..., ::2]), dim=-1).view_as(value)
    return (value * torch.cos(phases)).to(value.dtype) + (
        rotated * torch.sin(phases)
    ).to(value.dtype)


def _execute_bdh_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    from urm.adapters.bdh import bdh_attention_adapter

    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(
            f"unexpected BDH attention operands: {', '.join(sorted(operands))}"
        )
    if key is not query:
        raise ValueError("the pinned BDH source requires K to alias Q")
    output, identity = bdh_attention_adapter(query, value)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_bdh_attention_forward",
            "upstream_revision": identity["revision"],
            "backward_supported": True,
        },
    )


def _beta_for_update(
    torch: Any, beta: Any, t: int, rank: int, ranks: int, heads: int, value_dim: int
):
    if beta is None:
        raise ValueError("delta update requires beta")
    selected = beta[:, t]
    if beta.ndim in (4, 5):
        ranked = beta.ndim == 5 or (beta.shape[2] == ranks and beta.shape[3] == heads)
        if ranked:
            selected = selected[:, rank]
    if selected.ndim == 1:
        selected = selected[:, None]
    if selected.ndim == 2:
        if selected.shape[1] not in (1, heads):
            raise ValueError("beta head width must be one or match value heads")
        selected = selected[:, :, None]
    elif selected.ndim == 3:
        if selected.shape[1] not in (1, heads) or selected.shape[2] not in (
            1,
            value_dim,
        ):
            raise ValueError("beta must be [B,Hv] or [B,Hv,V]")
    else:
        raise ValueError("beta must use [B,T,Hv,(V)] or [B,T,R,Hv,(V)] layout")
    return selected.float()


def _execute_matrix_recurrence(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    initial_state = operands.pop("initial_state", None)
    initial_normalizer_state = operands.pop("initial_normalizer_state", None)
    beta = operands.pop("beta", None)
    log_decay = operands.pop("log_decay", None)
    update_keys = operands.pop("update_keys", None)
    update_values = operands.pop("update_values", None)
    left_transition = operands.pop("left_transition", None)
    right_transition = operands.pop("right_transition", None)
    transition_alpha = operands.pop("transition_alpha", None)
    transition_beta = operands.pop("transition_beta", None)
    if operands:
        raise TypeError(f"unexpected K2 operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("matrix K2 query/key/value use BTHD rank-4 layout")
    if not (
        query.is_floating_point()
        and key.is_floating_point()
        and value.is_floating_point()
    ):
        raise ValueError("matrix K2 query/key/value must be floating point")
    if not (query.device == key.device == value.device):
        raise ValueError("matrix K2 query/key/value must share a device")
    optional_tensors = (
        initial_state,
        initial_normalizer_state,
        beta,
        log_decay,
        update_keys,
        update_values,
        left_transition,
        right_transition,
        transition_alpha,
        transition_beta,
    )
    if any(
        tensor is not None and tensor.device != query.device
        for tensor in optional_tensors
    ):
        raise ValueError("matrix K2 operands must share the query device")
    if any(
        tensor is not None and not tensor.is_floating_point()
        for tensor in optional_tensors
    ):
        raise ValueError(
            "matrix K2 state, gates, updates, and transitions must be floating point"
        )
    batch, sequence, q_heads, key_dim = query.shape
    if min(batch, sequence, q_heads, key_dim) <= 0:
        raise ValueError("matrix K2 dimensions must be positive")
    if key.shape != query.shape:
        raise ValueError("matrix K2 query and key must have identical shapes")
    if spec.name == "bdh_attention_core":
        if key is not query:
            raise ValueError("BDH attention requires key to alias query")
        if initial_state is not None:
            raise ValueError(
                "the pinned BDH attention callable has no initial-state input"
            )
        query = _bdh_rotary(torch, query)
        key = query
    if value.ndim != 4 or value.shape[:2] != (batch, sequence):
        raise ValueError("matrix K2 values must use [B,T,Hv,V]")
    value_heads, value_dim = value.shape[2:]
    if min(value_heads, value_dim) <= 0:
        raise ValueError("matrix K2 value dimensions must be positive")
    if value_heads < q_heads or value_heads % q_heads:
        raise ValueError("value heads must be a positive multiple of query heads")
    polynomial_basis = spec.polynomial_basis
    if polynomial_basis is PolynomialBasis.NONE:
        q_features = _feature(torch, query, spec.feature_map)
        k_features = _feature(torch, key, spec.feature_map)
    else:
        scale = spec.read_scale or key_dim**-0.5
        q_scaled = query.float() * scale
        k_unscaled = key.float()
        q_quadratic = torch.einsum("...i,...j->...ij", q_scaled, q_scaled)
        k_quadratic = torch.einsum("...i,...j->...ij", k_unscaled, k_unscaled)
        if polynomial_basis is PolynomialBasis.BASED_TAYLOR2:
            q_features = torch.cat(
                (
                    torch.ones_like(q_scaled[..., :1]),
                    q_scaled,
                    q_quadratic.flatten(start_dim=-2) / (2.0**0.5),
                ),
                dim=-1,
            )
            k_features = torch.cat(
                (
                    torch.ones_like(k_unscaled[..., :1]),
                    k_unscaled,
                    k_quadratic.flatten(start_dim=-2) / (2.0**0.5),
                ),
                dim=-1,
            )
        elif polynomial_basis is PolynomialBasis.REBASED_SQUARE:
            q_features = q_quadratic.flatten(start_dim=-2)
            k_features = k_quadratic.flatten(start_dim=-2)
        else:
            raise ValueError(f"unsupported polynomial basis {polynomial_basis}")
        key_dim = q_features.shape[-1]
    q = q_features.transpose(1, 2)
    k = k_features.transpose(1, 2)
    if value_heads != q_heads:
        q = q.repeat_interleave(value_heads // q_heads, dim=1)
        k = k.repeat_interleave(value_heads // q_heads, dim=1)

    expected_state = (
        (batch, value_heads, value_dim, key_dim)
        if spec.state_v_first
        else (batch, value_heads, key_dim, value_dim)
    )
    if initial_state is None:
        state = torch.zeros(expected_state, dtype=torch.float32, device=query.device)
        if spec.state_v_first:
            state = state.transpose(-1, -2).contiguous()
    else:
        _require_shape(initial_state, expected_state, "initial_state")
        state = initial_state.float()
        if spec.state_v_first:
            state = state.transpose(-1, -2).contiguous()
    if spec.transition is StateTransition.FACTORED_MATRIX:
        if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
            if left_transition is not None or right_transition is not None:
                raise ValueError(
                    "low-rank transition operands cannot also pass dense matrices"
                )
            if transition_alpha is None or transition_beta is None:
                raise ValueError(
                    "low-rank transitions require transition_alpha and transition_beta"
                )
            _require_shape(
                transition_alpha,
                (batch, sequence, value_heads, key_dim),
                "transition_alpha",
            )
            _require_shape(
                transition_beta,
                (batch, sequence, value_heads, key_dim),
                "transition_beta",
            )
            if spec.generalized_delta_dplr:
                _require_shape(
                    log_decay, (batch, sequence, value_heads, key_dim), "log_decay"
                )
            elif log_decay is not None:
                raise ValueError("IPLR does not accept a diagonal decay input")
        else:
            if left_transition is None or right_transition is None:
                raise ValueError("factored transition requires left and right matrices")
            if log_decay is not None:
                raise ValueError("factored and pointwise decay cannot be combined")
            _require_shape(
                left_transition,
                (batch, sequence, value_heads, key_dim, key_dim),
                "left_transition",
            )
            _require_shape(
                right_transition,
                (batch, sequence, value_heads, value_dim, value_dim),
                "right_transition",
            )
    elif (
        left_transition is not None
        or right_transition is not None
        or transition_alpha is not None
        or transition_beta is not None
    ):
        raise ValueError("left/right transitions require factored_bilinear semantics")

    normalizer_state = None
    if spec.normalizer is StateNormalizer.QUERY_KEY:
        if initial_normalizer_state is None:
            normalizer_state = torch.zeros(
                (batch, value_heads, key_dim), dtype=torch.float32, device=query.device
            )
        else:
            _require_shape(
                initial_normalizer_state,
                (batch, value_heads, key_dim),
                "initial_normalizer_state",
            )
            normalizer_state = initial_normalizer_state.float()
    elif initial_normalizer_state is not None:
        raise ValueError("initial_normalizer_state requires query_key normalization")

    if update_keys is None:
        update_keys = (
            k.transpose(1, 2).unsqueeze(2)
            if polynomial_basis is not PolynomialBasis.NONE
            else key.unsqueeze(2)
        )
    elif update_keys.ndim != 5 or update_keys.shape[:2] != (batch, sequence):
        raise ValueError("update_keys must use [B,T,R,H,K]")
    ranks = update_keys.shape[2]
    if update_values is None:
        update_values = value.unsqueeze(2)
    if update_values.ndim != 5 or update_values.shape[:3] != (
        batch,
        sequence,
        ranks,
    ):
        raise ValueError("update_values must use [B,T,R,Hv,V]")
    if update_values.shape[3:] != (value_heads, value_dim):
        raise ValueError("update_values head/value dimensions must match value")
    if update_keys.shape[3:] != (q_heads, key_dim):
        raise ValueError("update_keys head/key dimensions must match query")
    if spec.update_rule is StateUpdateRule.DELTA and beta is None:
        raise ValueError("delta update requires beta")
    if spec.update_rule is StateUpdateRule.ADDITIVE and beta is not None:
        raise ValueError("beta is only valid for delta update semantics")

    outputs = []
    for token in range(sequence):
        if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
            alpha_t = transition_alpha[:, token].float()
            beta_t = transition_beta[:, token].float()
            identity = torch.eye(key_dim, dtype=torch.float32, device=query.device)
            identity = identity.expand(batch, value_heads, key_dim, key_dim)
            rank_one = torch.einsum("bhk,bhj->bhkj", beta_t, alpha_t)
            if spec.generalized_delta_dplr:
                decay_t = log_decay[:, token].float().exp()
                diagonal = torch.diag_embed(decay_t)
                left = diagonal + rank_one
            else:
                left = identity + rank_one
            state_for_token = torch.matmul(left, state)
            if normalizer_state is not None:
                normalizer_for_token = torch.matmul(
                    left, normalizer_state.unsqueeze(-1)
                ).squeeze(-1)
        elif spec.transition is StateTransition.FACTORED_MATRIX:
            left = left_transition[:, token].float()
            right = right_transition[:, token].float()
            state_for_token = torch.matmul(
                torch.matmul(left, state), right.transpose(-1, -2)
            )
            if normalizer_state is not None:
                normalizer_for_token = torch.matmul(
                    left, normalizer_state.unsqueeze(-1)
                ).squeeze(-1)
        else:
            state_for_token = _matrix_decay(torch, spec, log_decay, token, state)
            if normalizer_state is not None:
                # The denominator follows the same head/key-channel decay.
                if spec.decay is DecayGranularity.HEAD:
                    decay_t = log_decay[:, token].float()
                    if decay_t.ndim == 2 and decay_t.shape[-1] == 1:
                        decay_t = decay_t.squeeze(-1)
                    if decay_t.ndim == 1:
                        decay_t = decay_t[:, None]
                    normalizer_for_token = (
                        normalizer_state * torch.exp(decay_t)[:, :, None]
                    )
                elif spec.decay is DecayGranularity.KEY_CHANNEL:
                    decay_t = log_decay[:, token].float()
                    if decay_t.ndim == 2:
                        decay_t = decay_t.unsqueeze(1)
                    normalizer_for_token = normalizer_state * torch.exp(decay_t)
                else:
                    normalizer_for_token = normalizer_state
        query_scale = (
            1.0
            if polynomial_basis is not PolynomialBasis.NONE
            else (
                spec.read_scale
                or (
                    key_dim**-0.5
                    if (
                        spec.kda_delta
                        or spec.gated_delta_product
                        or spec.transition is StateTransition.FACTORED_MATRIX
                    )
                    else 1.0
                )
            )
        )
        query_t = q[:, :, token] * query_scale

        def read(current_state: Any, current_normalizer: Any | None):
            result = torch.einsum("bhk,bhkv->bhv", query_t, current_state)
            if current_normalizer is not None:
                denominator = torch.einsum("bhk,bhk->bh", query_t, current_normalizer)
                result = result / denominator.clamp_min(spec.epsilon).unsqueeze(-1)
            return result

        if spec.read_timing is ReadTiming.BEFORE_UPDATE:
            outputs.append(
                read(
                    state_for_token,
                    normalizer_for_token if normalizer_state is not None else None,
                )
            )

        for rank in range(ranks):
            update_key = _feature(torch, update_keys[:, token, rank], spec.feature_map)
            if value_heads != q_heads:
                update_key = update_key.repeat_interleave(value_heads // q_heads, dim=1)
            update_value = update_values[:, token, rank].float()
            if spec.update_rule is StateUpdateRule.DELTA:
                retrieved = torch.einsum("bhk,bhkv->bhv", update_key, state_for_token)
                beta_t = _beta_for_update(
                    torch, beta, token, rank, ranks, value_heads, value_dim
                )
                update_value = beta_t * (update_value - retrieved)
            state_for_token = state_for_token + torch.einsum(
                "bhk,bhv->bhkv", update_key, update_value
            )
            if normalizer_state is not None:
                normalizer_for_token = normalizer_for_token + update_key

        state = state_for_token
        if normalizer_state is not None:
            normalizer_state = normalizer_for_token
        if spec.read_timing is ReadTiming.AFTER_UPDATE:
            outputs.append(read(state, normalizer_state))
    output = torch.stack(outputs, dim=2).transpose(1, 2).to(value.dtype)
    return MixerResult(
        output,
        final_state=(
            state.transpose(-1, -2).contiguous() if spec.state_v_first else state
        ),
        final_normalizer_state=normalizer_state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": spec.recurrent_layout.value,
            "ordered_updates_per_token": ranks,
            "execution": "torch_eager_reference",
        },
    )


def _execute_xma_nonlinear_rnn_reference(name: str, torch: Any, **operands: Any):
    """Equation-level fixed-length references for XMA nonlinear recurrences."""
    query = operands.pop("query")
    state = operands.pop("initial_state")
    if query.ndim != 4 or state.ndim not in {3, 4}:
        raise ValueError(
            "XMA nonlinear recurrences require batched B,T,N,H inputs and state"
        )
    batch, sequence, heads = query.shape[:3]
    if state.shape[0:2] != (batch, heads):
        raise ValueError(
            "XMA recurrent initial state must match batch and head dimensions"
        )
    outputs = []
    if name == "rnn_core":
        weight = operands.pop("weight")
        if operands or state.ndim != 3 or query.shape[-1] != state.shape[-1]:
            raise ValueError(
                "RNN requires query/initial_state B,T,N,H / B,N,H and weight N,H,H"
            )
        for token in range(sequence):
            state = torch.tanh(
                torch.matmul(state.unsqueeze(-2), weight.unsqueeze(0)).squeeze(-2)
                + query[:, token]
            )
            outputs.append(state)
    elif name == "gru_core":
        weight = operands.pop("weight")
        forget_input = operands.pop("forget_input")
        forget_weight = operands.pop("forget_weight")
        reset_input = operands.pop("reset_input")
        reset_weight = operands.pop("reset_weight")
        if operands or state.ndim != 3 or query.shape[-1] != state.shape[-1]:
            raise ValueError(
                "GRU requires matching B,T,N,H inputs, B,N,H state and N,H,H gate weights"
            )
        for token in range(sequence):
            forget = torch.sigmoid(
                torch.matmul(state.unsqueeze(-2), forget_weight.unsqueeze(0)).squeeze(
                    -2
                )
                + forget_input[:, token]
            )
            reset = torch.sigmoid(
                torch.matmul(state.unsqueeze(-2), reset_weight.unsqueeze(0)).squeeze(-2)
                + reset_input[:, token]
            )
            candidate = torch.tanh(
                torch.matmul(
                    (state * reset).unsqueeze(-2), weight.unsqueeze(0)
                ).squeeze(-2)
                + query[:, token]
            )
            state = forget * state + (1.0 - forget) * candidate
            outputs.append(state)
    elif name == "m2rnn_core":
        key = operands.pop("key")
        value = operands.pop("value")
        weight = operands.pop("weight")
        forget_input = operands.pop("forget_input")
        if operands or state.ndim != 4 or query.shape[:3] != key.shape[:3]:
            raise ValueError(
                "M2RNN requires B,T,N,K Q/K, B,T,N,V values and B,N,K,V state"
            )
        if query.shape[-1] != key.shape[-1] or value.shape[-1] != state.shape[-1]:
            raise ValueError("M2RNN query/key and value/state dimensions must match")
        for token in range(sequence):
            update = key[:, token].unsqueeze(-1) * value[:, token].unsqueeze(-2)
            candidate = torch.tanh(torch.matmul(state, weight.unsqueeze(0)) + update)
            forget = forget_input[:, token].unsqueeze(-1).unsqueeze(-1)
            state = forget * state + (1.0 - forget) * candidate
            read = torch.matmul(query[:, token].unsqueeze(-2), state).squeeze(-2)
            outputs.append(read)
    else:
        raise ValueError(f"unsupported XMA nonlinear recurrence {name!r}")
    output = torch.stack(outputs, dim=1)
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": "xma_nonlinear_recurrence",
            "execution": "torch_eager_tokenwise_equation",
        },
    )


def _execute_xma_nonlinear_rnn(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    """Call the selected pinned XMA Triton recurrent operator."""
    from xma import KernelBackend

    name = plan.spec.name
    if name == "rnn_core":
        from xma.layers.rnn import rnn

        output, state = rnn(
            operands.pop("query"),
            operands.pop("weight"),
            input_state=operands.pop("initial_state"),
            kernel_backend=KernelBackend.triton,
        )
    elif name == "gru_core":
        from xma.layers.gru import gru

        output, state = gru(
            operands.pop("query"),
            operands.pop("weight"),
            operands.pop("forget_input"),
            operands.pop("forget_weight"),
            operands.pop("reset_input"),
            operands.pop("reset_weight"),
            input_state=operands.pop("initial_state"),
            kernel_backend=KernelBackend.triton,
        )
    elif name == "m2rnn_core":
        from xma.layers.m2rnn import m2rnn

        output, state = m2rnn(
            operands.pop("query"),
            operands.pop("key"),
            operands.pop("value"),
            operands.pop("weight"),
            operands.pop("forget_input"),
            input_state=operands.pop("initial_state"),
            kernel_backend=KernelBackend.triton,
        )
    else:
        raise ValueError(f"unsupported XMA recurrent operator {name!r}")
    if operands:
        raise TypeError(
            f"unexpected XMA recurrent operands: {', '.join(sorted(operands))}"
        )
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": plan.anchor,
            "upstream": "XMA Triton nonlinear recurrence",
            "backward_supported": True,
        },
    )


def _execute_ttt_linear_reference(torch: Any, **operands: Any):
    """Chunkwise TTT-Linear reference with its full matrix/bias memory."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    weight = operands.pop("w")
    bias = operands.pop("b")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    initial_state_bias = operands.pop("initial_state_bias", None)
    chunk_size = int(operands.pop("chunk_size", 16))
    eps = float(operands.pop("eps", 1e-6))
    if operands:
        raise TypeError(
            f"unexpected TTT-Linear operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("TTT-Linear requires matching BTHD Q/K/V and K=V")
    batch, sequence, heads, dim = query.shape
    if sequence % chunk_size:
        raise ValueError("TTT-Linear sequence length must be divisible by chunk_size")
    if weight.shape != (heads, dim) or bias.shape != (heads, dim):
        raise ValueError("TTT-Linear norm parameters must use H,D layout")
    if eta.shape != (batch, sequence, heads, 1):
        raise ValueError("TTT-Linear eta must use B,T,H,1 layout")
    state_shape = (batch, heads, dim, dim)
    bias_state_shape = (batch, heads, 1, dim)
    memory = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    memory_bias = (
        torch.zeros(bias_state_shape, device=query.device, dtype=torch.float32)
        if initial_state_bias is None
        else initial_state_bias.float()
    )
    if memory.shape != state_shape or memory_bias.shape != bias_state_shape:
        raise ValueError(
            "TTT-Linear initial states must use B,H,D,D and B,H,1,D layouts"
        )

    q = query.float().transpose(1, 2) * (dim**-0.5)
    k = key.float().transpose(1, 2)
    v = value.float().transpose(1, 2)
    eta = eta.float().transpose(1, 2)
    weight = weight.float().reshape(heads, 1, dim)
    bias = bias.float().reshape(heads, 1, dim)
    outputs = []
    for start in range(0, sequence, chunk_size):
        stop = start + chunk_size
        q_chunk = q[:, :, start:stop]
        k_chunk = k[:, :, start:stop]
        v_chunk = v[:, :, start:stop]
        eta_chunk = eta[:, :, start:stop]
        kh = k_chunk @ memory + memory_bias
        target = v_chunk - k_chunk
        mean = kh.mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(kh.var(dim=-1, unbiased=False, keepdim=True) + eps)
        kh_hat = (kh - mean) * rstd
        grad = (weight * kh_hat + bias - target) * weight
        grad = (
            (
                dim * grad
                - grad.sum(dim=-1, keepdim=True)
                - kh_hat * (grad * kh_hat).sum(dim=-1, keepdim=True)
            )
            * rstd
            / dim
        )
        attention = torch.tril(q_chunk @ k_chunk.transpose(-1, -2))
        output_chunk = (
            q_chunk @ memory
            - (eta_chunk * attention) @ grad
            + memory_bias
            - torch.tril(eta_chunk.expand_as(attention)) @ grad
        )
        eta_last = eta_chunk[:, :, -1, :, None]
        memory = memory - (eta_last * k_chunk).transpose(-1, -2) @ grad
        memory_bias = memory_bias - (eta_last * grad).sum(dim=-2, keepdim=True)
        out_mean = output_chunk.mean(dim=-1, keepdim=True)
        out_rstd = torch.rsqrt(
            output_chunk.var(dim=-1, unbiased=False, keepdim=True) + eps
        )
        outputs.append(
            output_chunk + (output_chunk - out_mean) * out_rstd * weight + bias
        )
    output = torch.cat(outputs, dim=-2).transpose(1, 2).to(value.dtype)
    return MixerResult(
        output,
        final_state=memory,
        final_normalizer_state=memory_bias,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": "ttt_linear_matrix_and_bias",
            "chunk_size": chunk_size,
            "execution": "torch_eager_ttt_linear_chunkwise_inner_update",
        },
    )


def _execute_titans_linear_reference(torch: Any, **operands: Any):
    """Tokenwise Titans memory reference with a chunk-frozen learning target."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    weight = operands.pop("w")
    bias = operands.pop("b")
    theta = operands.pop("theta")
    alpha = operands.pop("alpha")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    chunk_size = int(operands.pop("chunk_size", 16))
    eps = float(operands.pop("eps", 1e-6))
    if operands:
        raise TypeError(f"unexpected Titans operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("Titans linear memory requires matching BTHD Q/K/V")
    batch, sequence, heads, dim = query.shape
    if sequence % chunk_size:
        raise ValueError("Titans sequence length must be divisible by chunk_size")
    if weight.shape != (heads, dim) or bias.shape != (heads, dim):
        raise ValueError("Titans layer norm weight and bias must use H,D layout")
    gate_shape = (batch, sequence, heads, 1)
    if (
        theta.shape != gate_shape
        or alpha.shape != gate_shape
        or eta.shape != gate_shape
    ):
        raise ValueError("Titans theta, alpha and eta must use B,T,H,1 layout")
    state_shape = (batch, heads, dim, dim)
    memory = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    if memory.shape != state_shape:
        raise ValueError("Titans initial_state must use B,H,D,D layout")
    momentum = torch.zeros_like(memory)
    update_base = memory
    q, k, v = query.float(), key.float(), value.float()
    theta, alpha, eta = theta.float(), alpha.float(), eta.float()
    weight, bias = weight.float(), bias.float()
    outputs = []
    for token in range(sequence):
        q_t = q[:, token].unsqueeze(-2)
        k_t = k[:, token].unsqueeze(-2)
        v_t = v[:, token].unsqueeze(-2)
        km = k_t @ update_base
        reconstruction_target = v_t - k_t
        mean = km.mean(dim=-1, keepdim=True)
        rstd = torch.sqrt(km.var(dim=-1, unbiased=False, keepdim=True) + eps)
        km_hat = (km - mean) / rstd
        grad = (
            weight[None, :, None, :] * km_hat
            + bias[None, :, None, :]
            - reconstruction_target
        ) * weight[None, :, None, :]
        v_new = dim * grad - grad.sum(dim=-1, keepdim=True) / (rstd * dim)
        v_new = v_new - km_hat * (grad * km_hat).sum(dim=-1, keepdim=True) / (
            rstd * dim
        )
        theta_t = theta[:, token].unsqueeze(-2)
        alpha_t = alpha[:, token].unsqueeze(-2)
        eta_t = eta[:, token].unsqueeze(-2)
        momentum = eta_t * momentum - 2.0 * theta_t * (k_t.transpose(-1, -2) @ v_new)
        memory = (1.0 - alpha_t[..., :1]) * memory + momentum
        output = q_t @ memory
        output_mean = output.mean(dim=-1, keepdim=True)
        output_rstd = torch.sqrt(output.var(dim=-1, unbiased=False, keepdim=True) + eps)
        output = (
            output
            + (output - output_mean) / output_rstd * weight[None, :, None, :]
            + bias[None, :, None, :]
        )
        outputs.append(output.squeeze(-2))
        if (token + 1) % chunk_size == 0:
            update_base = memory
    return MixerResult(
        torch.stack(outputs, dim=1).to(value.dtype),
        final_state=memory,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": "titans_matrix_memory",
            "chunk_size": chunk_size,
            "execution": "torch_eager_titans_chunk_frozen_inner_update",
        },
    )


def _execute_mamba3_siso_reference(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    adt = operands.pop("adt")
    dt = operands.pop("dt")
    trap = operands.pop("trap")
    query_bias = operands.pop("query_bias")
    key_bias = operands.pop("key_bias")
    angles = operands.pop("angles")
    d_skip = operands.pop("d_skip", None)
    gate = operands.pop("gate", None)
    initial_states = operands.pop("initial_states", None)
    if operands:
        raise TypeError(f"unexpected Mamba-3 operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("Mamba-3 SISO Q/K/V must use BTHD layouts")
    batch, sequence, query_heads, key_dim = query.shape
    if key.shape != (batch, sequence, query_heads, key_dim):
        raise ValueError("Mamba-3 SISO query and key shapes must match")
    value_heads, value_dim = value.shape[2], value.shape[3]
    if value.shape[:2] != (batch, sequence) or value_heads % query_heads:
        raise ValueError("Mamba-3 SISO value heads must be divisible by Q/K heads")
    if (
        key_dim % 2
        or angles.ndim != 4
        or angles.shape[:3] != (batch, sequence, value_heads)
    ):
        raise ValueError("Mamba-3 SISO requires even Q/K width and B,T,H,A angles")
    angle_dim = angles.shape[-1]
    if 2 * angle_dim > key_dim:
        raise ValueError("Mamba-3 angle width cannot exceed half the Q/K width")
    if (
        adt.shape != (batch, value_heads, sequence)
        or dt.shape != adt.shape
        or trap.shape != adt.shape
    ):
        raise ValueError("Mamba-3 ADT, DT and trap must use BHT layout")
    if query_bias.shape != (value_heads, key_dim) or key_bias.shape != query_bias.shape:
        raise ValueError("Mamba-3 Q/K biases must use H,D layout")
    if initial_states is not None and (
        not isinstance(initial_states, (tuple, list)) or len(initial_states) != 4
    ):
        raise ValueError(
            "Mamba-3 initial_states must contain angle, SSM, K and V state"
        )

    if query_heads != value_heads:
        repeat = value_heads // query_heads
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)
    if initial_states is None:
        angle_state = torch.zeros(
            batch, value_heads, angle_dim, device=query.device, dtype=torch.float32
        )
        ssm_state = torch.zeros(
            batch,
            value_heads,
            value_dim,
            key_dim,
            device=query.device,
            dtype=torch.float32,
        )
        key_state = torch.zeros(
            batch, value_heads, key_dim, device=query.device, dtype=query.dtype
        )
        value_state = torch.zeros(
            batch, value_heads, value_dim, device=query.device, dtype=value.dtype
        )
    else:
        angle_state, ssm_state, key_state, value_state = initial_states
        angle_state = angle_state.float()
        ssm_state = ssm_state.float()

    angles = angles.to(query.dtype)
    query_bias = query_bias.to(query.dtype)
    key_bias = key_bias.to(query.dtype)
    trap = trap.to(query.dtype)

    def rotary(tensor: Any, cosine: Any, sine: Any):
        paired = tensor.float().reshape(batch, value_heads, key_dim // 2, 2)
        first, second = paired.unbind(dim=-1)
        if cosine.shape[-1] < key_dim // 2:
            padding = key_dim // 2 - cosine.shape[-1]
            cosine = torch.nn.functional.pad(cosine, (0, padding), value=1.0)
            sine = torch.nn.functional.pad(sine, (0, padding), value=0.0)
        return torch.stack(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        ).reshape(batch, value_heads, key_dim)

    outputs = []
    for token in range(sequence):
        angle_state = angle_state + (
            torch.tanh(angles[:, token].float()) * 3.141592653589793
        ) * dt[:, :, token].float().unsqueeze(-1)
        angle_state = angle_state - (2.0 * 3.141592653589793) * torch.floor(
            angle_state / (2.0 * 3.141592653589793)
        )
        cosine, sine = angle_state.cos(), angle_state.sin()
        q_t = rotary(query[:, token] + query_bias.unsqueeze(0), cosine, sine)
        k_t = rotary(key[:, token] + key_bias.unsqueeze(0), cosine, sine)
        v_t = value[:, token]
        trap_t = torch.sigmoid(trap[:, :, token].float())
        dt_t = dt[:, :, token].float()
        alpha = adt[:, :, token].float().exp()
        beta = (1.0 - trap_t) * dt_t * alpha
        gamma = trap_t * dt_t
        ssm_state = (
            alpha[..., None, None] * ssm_state
            + beta[..., None, None]
            * (key_state.unsqueeze(-2).float() * value_state.unsqueeze(-1).float())
            + gamma[..., None, None] * (k_t.unsqueeze(-2) * v_t.float().unsqueeze(-1))
        )
        output = torch.einsum("bhvd,bhd->bhv", ssm_state, q_t)
        if d_skip is not None:
            output = output + d_skip.float()[None, :, None] * v_t.float()
        if gate is not None:
            output = output * torch.nn.functional.silu(gate[:, token].float())
        outputs.append(output)
        key_state, value_state = k_t, v_t

    return MixerResult(
        torch.stack(outputs, dim=1).to(value.dtype),
        final_state=(angle_state, ssm_state, key_state, value_state),
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": "mamba3_angle_ssm_k_v",
            "execution": "torch_eager_mamba3_siso_trapezoidal_recurrence",
        },
    )


@lru_cache(maxsize=1)
def _mamba3_pinned_source_revision() -> str:
    import inspect
    import subprocess
    from pathlib import Path

    import mamba_ssm

    source_file = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source_file.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        raise RuntimeError("could not locate the pinned Mamba source checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "e9594ce1c732d97440f0332fdc43170a2294dbfa":
        raise RuntimeError(f"Mamba-3 requires the exact pinned source, got {revision}")
    return revision


def _execute_mamba3_siso_adapter(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    _mamba3_pinned_source_revision()

    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    adt = operands.pop("adt")
    dt = operands.pop("dt")
    trap = operands.pop("trap")
    query_bias = operands.pop("query_bias")
    key_bias = operands.pop("key_bias")
    angles = operands.pop("angles")
    d_skip = operands.pop("d_skip", None)
    gate = operands.pop("gate", None)
    initial_states = operands.pop("initial_states", None)
    if operands:
        raise TypeError(f"unexpected Mamba-3 operands: {', '.join(sorted(operands))}")
    if not query.is_cuda or query.dtype != torch.bfloat16:
        raise RuntimeError("the pinned Mamba-3 SISO adapter requires CUDA BF16 Q/K/V")
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined

    output, angle_state, ssm_state, key_state, value_state = mamba3_siso_combined(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        adt.contiguous(),
        dt.contiguous(),
        trap.contiguous(),
        query_bias.contiguous(),
        key_bias.contiguous(),
        angles.contiguous(),
        D=d_skip,
        Z=gate,
        Input_States=initial_states,
        chunk_size=64,
        return_final_states=True,
    )
    return MixerResult(
        output,
        final_state=(angle_state, ssm_state, key_state, value_state),
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_mamba3_siso_combined",
            "chunk_size": 64,
            "backward_supported": True,
        },
    )


def _execute_mesa_net_reference(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    beta = operands.pop("beta")
    lamb = operands.pop("lamb")
    h_kk_init = operands.pop("h_kk_init", None)
    h_kv_init = operands.pop("h_kv_init", None)
    if operands:
        raise TypeError(f"unexpected MesaNet operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("MesaNet core requires matching BTHK query/key/value tensors")
    batch, sequence, heads, key_dim = query.shape
    if log_decay.shape != (batch, sequence, heads) or beta.shape != (
        batch,
        sequence,
        heads,
    ):
        raise ValueError("MesaNet log_decay and beta must use BTH layout")
    if lamb.shape != (heads, key_dim):
        raise ValueError("MesaNet lamb must use HK layout")
    state_shape = (batch, heads, key_dim, key_dim)
    h_kk = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if h_kk_init is None
        else h_kk_init.float()
    )
    h_kv = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if h_kv_init is None
        else h_kv_init.float()
    )
    if h_kk.shape != state_shape or h_kv.shape != state_shape:
        raise ValueError("MesaNet initial states must use B,H,K,K layout")

    outputs = []
    final_h_kk, final_h_kv = [], []
    regularizer = torch.diag_embed(lamb.float()).unsqueeze(0)
    for token in range(sequence):
        q_t = query[:, token].float()
        k_t = key[:, token].float()
        v_t = value[:, token].float()
        beta_t = beta[:, token].float()
        decay_t = log_decay[:, token].float().exp()
        k_beta = k_t * beta_t.unsqueeze(-1)
        h_kk = decay_t[..., None, None] * h_kk + (
            k_beta.unsqueeze(-1) * k_t.unsqueeze(-2)
        )
        h_kv = decay_t[..., None, None] * h_kv + (
            k_beta.unsqueeze(-1) * v_t.unsqueeze(-2)
        )
        q_star = torch.linalg.solve(
            h_kk + regularizer,
            q_t.unsqueeze(-1),
        ).squeeze(-1)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_star, h_kv))
    output = torch.stack(outputs, dim=1).to(value.dtype)
    final_h_kk = h_kk
    final_h_kv = h_kv
    return MixerResult(
        output,
        final_state=(final_h_kk, final_h_kv),
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": "mesa_net_h_kk_h_kv",
            "execution": "torch_eager_mesa_net_regularized_recurrence_and_solve",
        },
    )


def _execute_rwkv4_reference(torch: Any, **operands: Any):
    w = operands.pop("w")
    u = operands.pop("u")
    key = operands.pop("k")
    value = operands.pop("v")
    state_input = operands.pop("state")
    if operands:
        raise TypeError(f"unexpected RWKV-4 operands: {', '.join(sorted(operands))}")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("RWKV-4 k and v must use matching [B,T,C] layouts")
    batch, sequence, channels = key.shape
    if w.shape != (channels,) or u.shape != (channels,):
        raise ValueError("RWKV-4 w and u must use [C]")
    if state_input.shape != (batch, 3, 1, channels):
        raise ValueError("RWKV-4 state must use [B,3,1,C] as alpha, beta, eps")
    tensors = (w, u, value, state_input)
    if any(
        tensor.device != key.device or tensor.dtype != key.dtype for tensor in tensors
    ):
        raise ValueError("RWKV-4 operands must share one device and dtype")
    if not key.is_floating_point():
        raise ValueError("RWKV-4 operands must use a floating-point dtype")

    decay = -torch.exp(w.float())
    bonus = u.float()
    alpha, denominator_state, log_scale = (
        state_input.float()[:, index, 0, :] for index in range(3)
    )
    outputs = []
    for token in range(sequence):
        key_t = key[:, token].float()
        value_t = value[:, token].float()
        bonus_key = bonus + key_t
        read_scale = torch.maximum(log_scale, bonus_key)
        read_state_scale = torch.exp(log_scale - read_scale)
        read_value_scale = torch.exp(bonus_key - read_scale)
        output_t = (read_state_scale * alpha + read_value_scale * value_t) / (
            read_state_scale * denominator_state + read_value_scale
        )
        outputs.append(output_t.to(value.dtype))

        decayed_scale = decay + log_scale
        log_scale_next = torch.maximum(decayed_scale, key_t)
        old_scale = torch.exp(decayed_scale - log_scale_next)
        new_scale = torch.exp(key_t - log_scale_next)
        alpha = old_scale * alpha + new_scale * value_t
        denominator_state = old_scale * denominator_state + new_scale
        log_scale = log_scale_next

    final_state = torch.stack((alpha, denominator_state, log_scale), dim=1).unsqueeze(2)
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_reference_v1",
            "backward_supported": True,
        },
    )


def _execute_rwkv6_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    bonus = operands.pop("bonus")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected RWKV-6 operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("RWKV-6 query/key/value use BTHD layout")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("RWKV-6 value batch, sequence and heads must match query")
    if log_decay.shape != query.shape or bonus.shape != (heads, key_dim):
        raise ValueError("RWKV-6 log_decay is BTHK and bonus is HK")
    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            device=query.device,
            dtype=torch.float32,
        )
    else:
        if initial_state.shape != (batch, heads, key_dim, value_dim):
            raise ValueError("RWKV-6 initial_state must use [B,H,K,V]")
        state = initial_state.float()
    scale = key_dim**-0.5
    outputs = []
    for token in range(sequence):
        q_t = query[:, token].float()
        k_t = key[:, token].float()
        v_t = value[:, token].float()
        decay_t = torch.exp(log_decay[:, token].float())
        decayed_state = state * decay_t.unsqueeze(-1)
        bonus_write = (k_t * bonus.float().unsqueeze(0)).unsqueeze(-1) * v_t.unsqueeze(
            -2
        )
        read_state = state + bonus_write
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q_t * scale, read_state).to(value.dtype)
        )
        state = decayed_state + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_rwkv6_bonus_corrected_state_recurrence",
            "backward_supported": True,
        },
    )


def _momentum_delta_operands(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    p = operands.pop("p")
    log_alpha = operands.pop("log_alpha")
    log_mu = operands.pop("log_mu")
    beta = operands.pop("beta")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    initial_normalizer_state = operands.pop("initial_normalizer_state", None)
    if operands:
        raise TypeError(
            f"unexpected Momentum DeltaNet operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or key.shape != query.shape or p.shape != query.shape:
        raise ValueError("Momentum DeltaNet query/key/p must share BTHK layout")
    batch, sequence, heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:3] != (batch, sequence, heads):
        raise ValueError("Momentum DeltaNet value must use matching BTHV axes")
    scalar_shape = (batch, sequence, heads)
    if any(t.shape != scalar_shape for t in (log_alpha, log_mu, beta, eta)):
        raise ValueError("Momentum DeltaNet log_alpha/log_mu/beta/eta must use BTH")
    state_shape = (batch, heads, key_dim, value.shape[-1])
    for name, state in (
        ("initial_state", initial_state),
        ("initial_normalizer_state", initial_normalizer_state),
    ):
        if state is not None and state.shape != state_shape:
            raise ValueError(f"Momentum DeltaNet {name} must use [B,H,K,V]")
    tensors = (key, value, p, log_alpha, log_mu, beta, eta) + tuple(
        state
        for state in (initial_state, initial_normalizer_state)
        if state is not None
    )
    if any(t.device != query.device or t.dtype != query.dtype for t in tensors):
        raise ValueError("Momentum DeltaNet operands must share device and dtype")
    return (
        query,
        key,
        value,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        initial_state,
        initial_normalizer_state,
        state_shape,
    )


def _execute_momentum_delta_reference(
    spec: UnifiedMixerSpec, torch: Any, **operands: Any
):
    if not spec.momentum_delta:
        raise RuntimeError("Momentum DeltaNet semantics are not selected")
    (
        query,
        key,
        value,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        initial_state,
        initial_normalizer_state,
        state_shape,
    ) = _momentum_delta_operands(torch, **operands)
    state = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    momentum = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_normalizer_state is None
        else initial_normalizer_state.float()
    )
    scale = query.shape[-1] ** -0.5
    outputs = []
    for token in range(query.shape[1]):
        q_t = query[:, token].float()
        k_t = key[:, token].float()
        v_t = value[:, token].float()
        p_t = p[:, token].float()
        alpha_t = log_alpha[:, token].float().exp()[..., None, None]
        mu_t = log_mu[:, token].float().exp()[..., None, None]
        beta_t = beta[:, token].float()[..., None, None]
        eta_t = eta[:, token].float()[..., None]
        prediction = torch.einsum("bhk,bhkv->bhv", p_t, state)
        residual = v_t - prediction
        momentum = mu_t * momentum - (eta_t * k_t).unsqueeze(-1) * residual.unsqueeze(
            -2
        )
        state = alpha_t * state - beta_t * momentum
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q_t * scale, state).to(query.dtype)
        )
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return MixerResult(
        output,
        final_state=state,
        final_normalizer_state=momentum,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_momentum_delta_two_matrix_state_recurrence",
            "backward_supported": True,
        },
    )


def _gated_oja_operands(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    gate = operands.pop("gv")
    beta = operands.pop("beta")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected gated Oja operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("gated Oja query/key/value must use BTHD layout")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("gated Oja value must align with query batch/time/head axes")
    if gate.shape != (batch, sequence, heads, value_dim):
        raise ValueError("gated Oja gv must use BTHV value-channel layout")
    if beta.shape != (batch, sequence, heads):
        raise ValueError("the current gated Oja recipe requires headwise beta [B,T,H]")
    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is not None and initial_state.shape != state_shape:
        raise ValueError("gated Oja initial_state must use [B,H,K,V]")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("gated Oja query/key/value must use one dtype")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("gated Oja gv and beta must be float32")
    tensors = (key, value, gate, beta) + (
        () if initial_state is None else (initial_state,)
    )
    if any(t.device != query.device for t in tensors):
        raise ValueError("gated Oja operands must share one device")
    return query, key, value, gate, beta, initial_state, state_shape


def _execute_gated_oja_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    if not spec.gated_oja:
        raise RuntimeError("gated Oja semantics are not selected")
    query, key, value, gate, beta, initial_state, state_shape = _gated_oja_operands(
        torch, **operands
    )
    state = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    scale = query.shape[-1] ** -0.5
    outputs = []
    for token in range(query.shape[1]):
        q_t = query[:, token].float()
        k_t = key[:, token].float()
        v_t = value[:, token].float()
        state = state * gate[:, token].float().exp().unsqueeze(-2)
        prediction = (state * v_t.unsqueeze(-2)).sum(dim=-1)
        correction = beta[:, token].float().unsqueeze(-1) * (k_t - prediction)
        state = state + correction.unsqueeze(-1) * v_t.unsqueeze(-2)
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q_t * scale, state).to(query.dtype)
        )
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_gated_oja_value_channel_recurrence",
            "backward_supported": True,
        },
    )


def _comba_operands(torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    prediction_key = operands.pop("p")
    log_decay = operands.pop("g")
    beta = operands.pop("beta")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected COMBA operands: {', '.join(sorted(operands))}")
    if (
        query.ndim != 4
        or key.shape != query.shape
        or prediction_key.shape != query.shape
    ):
        raise ValueError("COMBA query/key/p must share BTHK layout")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("COMBA value must use matching batch/time/head axes")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if log_decay.shape != (batch, sequence, heads):
        raise ValueError("COMBA log decay must use BTH layout")
    if beta.shape != (batch, sequence, heads):
        raise ValueError("COMBA beta must use BTH layout")
    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is not None and initial_state.shape != state_shape:
        raise ValueError("COMBA initial_state must use [B,H,K,V]")
    if not (query.dtype == key.dtype == prediction_key.dtype == value.dtype):
        raise ValueError("COMBA query/key/p/value must use one dtype")
    if log_decay.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("COMBA log_decay and beta must be float32")
    tensors = (key, value, prediction_key, log_decay, beta) + (
        () if initial_state is None else (initial_state,)
    )
    if any(t.device != query.device for t in tensors):
        raise ValueError("COMBA operands must share one device")
    return (
        query,
        key,
        value,
        prediction_key,
        log_decay,
        beta,
        initial_state,
        state_shape,
    )


def _execute_comba_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    if not spec.comba_rule:
        raise RuntimeError("COMBA semantics are not selected")
    (
        query,
        key,
        value,
        prediction_key,
        log_decay,
        beta,
        initial_state,
        state_shape,
    ) = _comba_operands(torch, **operands)
    state = (
        torch.zeros(state_shape, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    scale = query.shape[-1] ** -0.5
    outputs = []
    for token in range(query.shape[1]):
        q_t = query[:, token].float()
        k_t = key[:, token].float()
        p_t = prediction_key[:, token].float()
        v_t = value[:, token].float()
        state = state * log_decay[:, token].float().exp()[..., None, None]
        residual = v_t - (state * p_t.unsqueeze(-1)).sum(dim=-2)
        state = state + k_t.unsqueeze(-1) * (
            beta[:, token].float().unsqueeze(-1) * residual
        ).unsqueeze(-2)
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", q_t * scale, state).to(query.dtype)
        )
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_comba_dual_key_delta_recurrence",
            "backward_supported": True,
        },
    )


def _pgdn_operands(torch: Any, *, kda: bool = False, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    g_atk = operands.pop("g_atk")
    gate = operands.pop("g")
    beta_atk = operands.pop("beta_atk")
    beta = operands.pop("beta")
    initial_state = operands.pop("initial_state", None)
    initial_A_state = operands.pop("initial_A_state", None)
    if operands:
        raise TypeError(f"unexpected PGDN operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("PGDN query/key/value must use BTHD layout")
    batch, sequence, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("the PGDN core currently requires matching query/value heads")
    scalar_shape = (batch, sequence, heads)
    gate_shape = (*scalar_shape, key_dim) if kda else scalar_shape
    if (
        g_atk.shape != scalar_shape
        or beta_atk.shape != scalar_shape
        or beta.shape != scalar_shape
        or gate.shape != gate_shape
    ):
        raise ValueError(
            "preconditioned delta gates must use their declared BTH or BTHK layouts"
        )
    state_shape = (batch, heads, key_dim, value_dim)
    auxiliary_shape = (batch, heads, key_dim)
    if initial_state is not None and initial_state.shape != state_shape:
        raise ValueError("PGDN initial_state must use [B,H,K,V]")
    if initial_A_state is not None and initial_A_state.shape != auxiliary_shape:
        raise ValueError("PGDN initial_A_state must use [B,H,K]")
    if kda and initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("PKDA initial_state must be float32")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("PGDN query/key/value must use one dtype")
    if any(t.dtype != torch.float32 for t in (g_atk, gate, beta_atk, beta)):
        raise ValueError("PGDN gates and betas must use float32")
    tensors = (key, value, g_atk, gate, beta_atk, beta) + tuple(
        item for item in (initial_state, initial_A_state) if item is not None
    )
    if any(t.device != query.device for t in tensors):
        raise ValueError("PGDN operands must share one device")
    return (
        query,
        key,
        value,
        g_atk,
        gate,
        beta_atk,
        beta,
        initial_state,
        initial_A_state,
    )


def _execute_pgdn_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    is_pkda = spec.preconditioned_kda
    if not (spec.preconditioned_gated_delta or is_pkda):
        raise RuntimeError("preconditioned delta semantics are not selected")
    (
        query,
        key,
        value,
        g_atk,
        gate,
        beta_atk,
        beta,
        initial_state,
        initial_A_state,
    ) = _pgdn_operands(torch, kda=is_pkda, **operands)
    batch, _, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    state = (
        torch.zeros(
            (batch, heads, key_dim, value_dim), device=query.device, dtype=torch.float32
        )
        if initial_state is None
        else initial_state.float()
    )
    metric = (
        torch.zeros((batch, heads, key_dim), device=query.device, dtype=torch.float32)
        if initial_A_state is None
        else initial_A_state.float()
    )
    scale = key_dim**-0.5
    outputs = []
    for token in range(query.shape[1]):
        q_t = (
            torch.nn.functional.normalize(query[:, token].float(), p=2, dim=-1) * scale
        )
        k_t = torch.nn.functional.normalize(key[:, token].float(), p=2, dim=-1)
        v_t = value[:, token].float()
        metric = metric * g_atk[:, token].float().exp().unsqueeze(-1)
        metric = metric + beta_atk[:, token].float().unsqueeze(-1) * k_t.square()
        squash_input = torch.log(metric + 1e-6) + 0.2
        squash = squash_input / (1.0 + squash_input.abs())
        preconditioner = torch.exp(
            -torch.log(torch.tensor(1.5, device=query.device)) * squash
        )
        preconditioned_key = k_t * preconditioner
        if is_pkda:
            state = state * gate[:, token].float().exp().unsqueeze(-1)
        else:
            state = state * gate[:, token].float().exp()[..., None, None]
        residual = v_t - (state * k_t.unsqueeze(-1)).sum(dim=-2)
        state = state + preconditioned_key.unsqueeze(-1) * (
            beta[:, token].float().unsqueeze(-1) * residual
        ).unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state).to(query.dtype))
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    return MixerResult(
        output,
        final_state=state,
        final_normalizer_state=metric.to(query.dtype),
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_pgdn_atk_preconditioned_delta_recurrence",
            "backward_supported": True,
        },
    )


def _slot_attention_operands(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    if spec.name == "abc_core":
        slot_logits = operands.pop("slot_logits")
        cumulative = torch.logcumsumexp(slot_logits.float(), dim=1)
        log_decay = (
            torch.cat((cumulative[:, :1], cumulative[:, :-1]), dim=1) - cumulative
        )
        slot_weights = torch.exp(slot_logits.float() - cumulative).to(key.dtype)
    else:
        slot_weights = operands.pop("slot_weights")
        log_decay = operands.pop("log_decay")
    initial_key_state = operands.pop("initial_key_state", None)
    initial_value_state = operands.pop("initial_value_state", None)
    if operands:
        raise TypeError(
            f"unexpected slot-attention operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("ABC/GSA query/key/value must use BTHD layouts")
    batch, sequence, query_heads, key_dim = query.shape
    if key.shape[:2] != (batch, sequence) or value.shape[:3] != key.shape[:3]:
        raise ValueError("ABC/GSA key and value must share batch/time/head axes")
    key_heads = key.shape[2]
    if query_heads % key_heads:
        raise ValueError("ABC/GSA query heads must be divisible by key/value heads")
    if key.shape[-1] != key_dim:
        raise ValueError("ABC/GSA key dimension must match query dimension")
    if slot_weights.ndim != 4 or slot_weights.shape[:3] != (batch, sequence, key_heads):
        raise ValueError("ABC/GSA slot weights must use [B,T,Hkv,M]")
    slots = slot_weights.shape[-1]
    if log_decay.shape != (batch, sequence, key_heads, slots):
        raise ValueError("ABC/GSA log decay must align with the slot-weight layout")
    key_state_shape = (batch, key_heads, key_dim, slots)
    value_state_shape = (batch, key_heads, slots, value.shape[-1])
    if initial_key_state is not None and initial_key_state.shape != key_state_shape:
        raise ValueError("ABC/GSA initial key state must use [B,Hkv,K,M]")
    if (
        initial_value_state is not None
        and initial_value_state.shape != value_state_shape
    ):
        raise ValueError("ABC/GSA initial value state must use [B,Hkv,M,V]")
    if not (query.dtype == key.dtype == value.dtype == slot_weights.dtype):
        raise ValueError("ABC/GSA query/key/value/slot weights must share a dtype")
    tensors = (key, value, slot_weights, log_decay) + tuple(
        item for item in (initial_key_state, initial_value_state) if item is not None
    )
    if any(item.device != query.device for item in tensors):
        raise ValueError("ABC/GSA operands must share one device")
    return (
        query,
        key,
        value,
        slot_weights,
        log_decay,
        initial_key_state,
        initial_value_state,
        query_heads // key_heads,
    )


def _execute_slot_attention_reference(
    spec: UnifiedMixerSpec, torch: Any, **operands: Any
):
    if not spec.slot_attention or spec.name not in {"abc_core", "gsa_core"}:
        raise RuntimeError("ABC/GSA slot-attention semantics are not selected")
    (
        query,
        key,
        value,
        slot_weights,
        log_decay,
        initial_key_state,
        initial_value_state,
        group_size,
    ) = _slot_attention_operands(spec, torch, **operands)
    batch, sequence, query_heads, key_dim = query.shape
    key_heads = key.shape[2]
    slots = slot_weights.shape[-1]
    value_dim = value.shape[-1]
    repeated_key = key.float().repeat_interleave(group_size, dim=2)
    repeated_value = value.float().repeat_interleave(group_size, dim=2)
    repeated_slots = slot_weights.float().repeat_interleave(group_size, dim=2)
    repeated_decay = log_decay.float().repeat_interleave(group_size, dim=2)
    key_state = (
        torch.zeros(
            (batch, query_heads, key_dim, slots),
            device=query.device,
            dtype=torch.float32,
        )
        if initial_key_state is None
        else initial_key_state.float().repeat_interleave(group_size, dim=1)
    )
    value_state = (
        torch.zeros(
            (batch, query_heads, slots, value_dim),
            device=query.device,
            dtype=torch.float32,
        )
        if initial_value_state is None
        else initial_value_state.float().repeat_interleave(group_size, dim=1)
    )
    scale = key_dim**-0.5
    slot_scores = []
    for token in range(sequence):
        decay_t = repeated_decay[:, token].exp()
        weights_t = repeated_slots[:, token]
        key_state = key_state * decay_t.unsqueeze(-2)
        key_state = key_state + repeated_key[:, token].unsqueeze(
            -1
        ) * weights_t.unsqueeze(-2)
        slot_scores.append(
            (query[:, token].float() * scale).unsqueeze(-1).mul(key_state).sum(dim=-2)
        )
    slot_probability = torch.stack(slot_scores, dim=1).softmax(dim=-1)
    outputs = []
    for token in range(sequence):
        weights_t = repeated_slots[:, token]
        decay_t = repeated_decay[:, token].exp()
        value_state = value_state * decay_t.unsqueeze(-1) + (
            weights_t.unsqueeze(-1) * repeated_value[:, token].unsqueeze(-2)
        )
        outputs.append(
            (slot_probability[:, token].unsqueeze(-1) * value_state)
            .sum(dim=-2)
            .to(value.dtype)
        )
    output = torch.stack(outputs, dim=1) if outputs else value[:, :0]
    final_key_state = key_state.view(batch, key_heads, group_size, key_dim, slots)[
        :, :, 0
    ]
    final_value_state = value_state.view(
        batch, key_heads, group_size, slots, value_dim
    )[:, :, 0]
    return MixerResult(
        output,
        final_state=(final_key_state, final_value_state),
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": f"torch_eager_{spec.name}_two_stage_slot_recurrence",
            "backward_supported": True,
        },
    )


def _execute_diagonal_ssm(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    x = operands.pop("x")
    if spec.diagonal_hgrn:
        input_gate = read_gate = None
    else:
        input_gate = operands.pop("input_gate")
        read_gate = operands.pop("read_gate")
    log_decay = operands.pop("log_decay")
    step_size = operands.pop("step_size", None)
    initial_state = operands.pop("initial_state", None)
    skip = operands.pop("skip", 0.0)
    if operands:
        raise TypeError(
            f"unexpected diagonal K2 operands: {', '.join(sorted(operands))}"
        )
    if spec.diagonal_hgrn:
        if x.ndim != 3 or log_decay.shape != x.shape:
            raise ValueError("HGRN expects x and log_decay with shape [B,T,C]")
        batch, sequence, channels = x.shape
        input_gate = torch.ones(
            (batch, sequence, 1), device=x.device, dtype=torch.float32
        )
        read_gate = input_gate
        log_decay = log_decay.unsqueeze(-1)
        if initial_state is not None:
            if initial_state.shape == (batch, channels):
                initial_state = initial_state.unsqueeze(-1)
            elif initial_state.shape != (batch, channels, 1):
                raise ValueError("HGRN initial_state must have shape [B,C] or [B,C,1]")
        skip = 0.0
    assert input_gate is not None and read_gate is not None
    if spec.step_size_discretization and step_size is None:
        raise ValueError("step-size diagonal SSM requires step_size [B,T,C]")
    if not spec.step_size_discretization and step_size is not None:
        raise ValueError("step_size requires step-size diagonal SSM semantics")
    if x.ndim != 3 or input_gate.ndim not in (3, 4) or read_gate.ndim not in (3, 4):
        raise ValueError(
            "diagonal SSM expects x [B,T,C] and gates [B,T,N] or [B,T,C,N]"
        )
    if not x.is_floating_point():
        raise ValueError("diagonal SSM x must be floating point")
    if any(tensor.device != x.device for tensor in (input_gate, read_gate, log_decay)):
        raise ValueError("diagonal SSM inputs must share a device")
    if any(
        not tensor.is_floating_point() for tensor in (input_gate, read_gate, log_decay)
    ):
        raise ValueError("diagonal SSM gates and transition must be floating point")
    batch, sequence, channels = x.shape
    if min(batch, sequence, channels) <= 0:
        raise ValueError(
            "diagonal SSM batch, sequence, and channel dimensions must be positive"
        )
    if input_gate.shape[:2] != (batch, sequence) or read_gate.shape[:2] != (
        batch,
        sequence,
    ):
        raise ValueError("diagonal SSM gate batch/sequence dimensions must match x")
    state_width = input_gate.shape[-1]
    if state_width <= 0:
        raise ValueError("diagonal SSM state width must be positive")
    if read_gate.shape[-1] != state_width:
        raise ValueError("input_gate and read_gate state widths must match")
    expected_state = (batch, channels, state_width)
    if initial_state is None:
        state = torch.zeros(expected_state, dtype=torch.float32, device=x.device)
    else:
        _require_shape(initial_state, expected_state, "initial_state")
        if initial_state.device != x.device:
            raise ValueError("initial_state must share the x device")
        state = initial_state.float()
    if log_decay.shape[:2] != (batch, sequence) or log_decay.shape[-1] != state_width:
        raise ValueError("diagonal log_decay must use [B,T,N] or [B,T,C,N]")
    for gate_name, gate in (("input_gate", input_gate), ("read_gate", read_gate)):
        if gate.ndim == 4 and gate.shape[2] not in (1, channels):
            raise ValueError(f"{gate_name} channel width must be one or match x")
    if log_decay.ndim == 4 and log_decay.shape[2] not in (1, channels):
        raise ValueError("log_decay channel width must be one or match x")
    if step_size is not None:
        if step_size.shape not in ((batch, sequence), (batch, sequence, channels)):
            raise ValueError("step_size must use [B,T] or [B,T,C]")
        if step_size.device != x.device or not step_size.is_floating_point():
            raise ValueError("step_size must be floating point and share the x device")

    skip_t = torch.as_tensor(skip, dtype=torch.float32, device=x.device)
    if skip_t.ndim == 1 and skip_t.shape[0] != channels:
        raise ValueError("skip vector width must match x channels")
    outputs = []
    for token in range(sequence):
        decay_t = log_decay[:, token].float()
        if decay_t.ndim == 2:
            decay_t = decay_t[:, None, :]
        step_t = None if step_size is None else step_size[:, token].float()
        if step_t is not None:
            if step_t.ndim == 1:
                step_t = step_t[:, None]
            decay_t = decay_t * step_t.unsqueeze(-1)
        transition = torch.exp(decay_t)
        if transition.shape[1] not in (1, channels):
            raise ValueError("log_decay channel width must be one or match x")
        in_gate = input_gate[:, token].float()
        out_gate = read_gate[:, token].float()
        if in_gate.ndim == 2:
            in_gate = in_gate[:, None, :]
        if out_gate.ndim == 2:
            out_gate = out_gate[:, None, :]
        if spec.read_timing is ReadTiming.BEFORE_UPDATE:
            out = (state * out_gate).sum(dim=-1) + x[:, token].float() * skip_t
            outputs.append(out)
        update = x[:, token].float().unsqueeze(-1) * in_gate
        if step_t is not None:
            update = update * step_t.unsqueeze(-1)
        state = transition * state + update
        if spec.read_timing is ReadTiming.AFTER_UPDATE:
            out = (state * out_gate).sum(dim=-1) + x[:, token].float() * skip_t
            outputs.append(out)
    return MixerResult(
        torch.stack(outputs, dim=1).to(x.dtype),
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "state_layout": RecurrentLayout.DIAGONAL.value,
            "execution": "torch_eager_reference",
        },
    )


def _validate_routes(
    torch: Any,
    addresses: Any,
    weights: Any,
    batch: int,
    sequence: int,
    slots: int,
    name: str,
):
    if addresses.ndim != 3 or addresses.shape[:2] != (batch, sequence):
        raise ValueError(f"{name}_indices must use [B,T,R]")
    if weights.shape != addresses.shape:
        raise ValueError(f"{name}_weights must match {name}_indices")
    if addresses.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name}_indices must be integer")
    if addresses.shape[-1] <= 0:
        raise ValueError(f"{name}_indices must contain at least one route")
    if addresses.device != weights.device:
        raise ValueError(f"{name}_indices and weights must share a device")
    if not weights.is_floating_point():
        raise ValueError(f"{name}_weights must be floating point")
    if bool(((addresses < 0) | (addresses >= slots)).any().item()):
        raise ValueError(f"{name}_indices are outside [0, slots)")
    if addresses.shape[-1] > 1:
        sorted_addresses = addresses.sort(dim=-1).values
        if bool((sorted_addresses[..., 1:] == sorted_addresses[..., :-1]).any().item()):
            raise ValueError(f"{name}_indices must be unique within each token")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError(f"{name}_weights must be finite")
    if bool((weights < 0).any().item()):
        raise ValueError(f"{name}_weights must be nonnegative")
    sums = weights.float().sum(dim=-1)
    atol = 4e-3 if weights.dtype in (torch.bfloat16, torch.float16) else 2e-5
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0):
        raise ValueError(f"{name}_weights must be normalized")


def _execute_sparse_delta(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    memory = operands.pop("memory")
    read_indices = operands.pop("read_indices")
    read_weights = operands.pop("read_weights")
    write_indices = operands.pop("write_indices", None)
    write_weights = operands.pop("write_weights", None)
    values = operands.pop("values", None)
    beta = operands.pop("beta", None)
    log_decay = operands.pop("log_decay", None)
    if operands:
        raise TypeError(f"unexpected K3 operands: {', '.join(sorted(operands))}")
    if memory.ndim != 3:
        raise ValueError("K3 memory must use [B,S,D]")
    if not memory.is_floating_point():
        raise ValueError("K3 memory must be floating point")
    batch, slots, value_dim = memory.shape
    if min(batch, slots, value_dim) <= 0:
        raise ValueError("K3 memory dimensions must be positive")
    if read_indices.ndim != 3:
        raise ValueError("read_indices must use [B,T,R]")
    sequence = read_indices.shape[1]
    if sequence <= 0:
        raise ValueError("K3 sequence length must be positive")
    _validate_routes(torch, read_indices, read_weights, batch, sequence, slots, "read")
    if write_indices is None or write_weights is None or values is None:
        raise ValueError("K3 update requires write routes and values")
    _validate_routes(
        torch, write_indices, write_weights, batch, sequence, slots, "write"
    )
    if values.shape != (batch, sequence, value_dim):
        raise ValueError("K3 values must use [B,T,D]")
    if beta is None or log_decay is None:
        raise ValueError("K3 delta update requires beta and log_decay")
    if beta.shape not in ((batch, sequence), (batch, sequence, 1)):
        raise ValueError("K3 beta must use [B,T] or [B,T,1]")
    if log_decay.shape not in ((batch, sequence), (batch, sequence, 1)):
        raise ValueError("K3 log_decay must use [B,T] or [B,T,1]")
    if not all(tensor.is_floating_point() for tensor in (values, beta, log_decay)):
        raise ValueError("K3 values, beta, and log_decay must be floating point")
    if any(
        tensor.device != memory.device
        for tensor in (
            read_indices,
            read_weights,
            write_indices,
            write_weights,
            values,
            beta,
            log_decay,
        )
    ):
        raise ValueError("K3 memory and operands must share a device")

    storage_dtype = memory.dtype
    state = memory.float()
    outputs = []
    for token in range(sequence):
        decay_t = log_decay[:, token].reshape(batch, 1, 1).float().exp()
        if spec.read_timing is ReadTiming.BEFORE_UPDATE:
            selected = state.gather(
                1, read_indices[:, token, :, None].expand(-1, -1, value_dim).long()
            )
            outputs.append((selected * read_weights[:, token, :, None].float()).sum(1))
        update_index = write_indices[:, token, :, None].expand(-1, -1, value_dim).long()
        selected_write = state.gather(1, update_index)
        decayed_write = selected_write * decay_t
        write_weight = write_weights[:, token, :, None].float()
        retrieved = (decayed_write * write_weight).sum(1)
        beta_t = beta[:, token].reshape(batch, 1).float()
        delta = beta_t * (values[:, token].float() - retrieved)
        updated_write = decayed_write + write_weight * delta[:, None, :]
        state = state.scatter(1, update_index, updated_write)
        # K3's semantic contract commits state storage precision once per token.
        state = state.to(storage_dtype).float()
        if spec.read_timing is ReadTiming.AFTER_UPDATE:
            selected = state.gather(
                1, read_indices[:, token, :, None].expand(-1, -1, value_dim).long()
            )
            outputs.append((selected * read_weights[:, token, :, None].float()).sum(1))
    return MixerResult(
        torch.stack(outputs, dim=1).to(memory.dtype),
        final_state=state.to(memory.dtype),
        metadata={
            "anchor": "urm.unified.k3.sparse_delta_reference.v1",
            "collision_order": "token_ordered_unique_within_token",
            "execution": "torch_eager_reference",
        },
    )


def _execute_mamba_selective_scan(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    x = operands.pop("x")
    input_gate = operands.pop("input_gate")
    read_gate = operands.pop("read_gate")
    log_decay = operands.pop("log_decay")
    step_size = operands.pop("step_size")
    initial_state = operands.pop("initial_state", None)
    skip = operands.pop("skip", 0.0)
    if operands:
        raise TypeError(
            f"unexpected Mamba selective-scan operands: {', '.join(sorted(operands))}"
        )
    if initial_state is not None:
        raise ValueError(
            "the pinned Mamba selective-scan operator has no initial-state input"
        )
    if isinstance(skip, torch.Tensor):
        if skip.ndim == 0:
            skip = skip.expand(x.shape[1])
    elif skip is None or float(skip) == 0.0:
        skip = None
    else:
        skip = torch.full((x.shape[1],), float(skip), device=x.device, dtype=x.dtype)
    from urm.compiler.execution import MAMBA_SELECTIVE_SCAN_ANCHOR_NAME

    output, final_state = _pinned_mamba_selective_scan()(
        x,
        step_size,
        log_decay,
        input_gate,
        read_gate,
        D=skip,
        delta_softplus=False,
        return_last_state=True,
    )
    return MixerResult(
        output.transpose(1, 2),
        final_state=final_state,
        metadata={
            "anchor": MAMBA_SELECTIVE_SCAN_ANCHOR_NAME,
            "execution": "pinned_mamba_selective_scan",
            "compiler_plan": plan.anchor,
            "upstream_revision": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
        },
    )


def _mamba2_shapes(x: Any, dt: Any, A: Any, B: Any, C: Any, initial_states: Any):
    if x.ndim != 4:
        raise ValueError("Mamba-2 x must use [B,T,H,P]")
    batch, sequence, heads, head_dim = x.shape
    if dt.shape != (batch, sequence, heads) or A.shape != (heads,):
        raise ValueError("Mamba-2 dt/A must use [B,T,H] and [H]")
    if B.ndim != 4 or B.shape[:2] != (batch, sequence) or C.shape != B.shape:
        raise ValueError("Mamba-2 B/C must use matching [B,T,G,N]")
    groups, state_dim = B.shape[2:]
    if heads % groups:
        raise ValueError("Mamba-2 head count must be divisible by B/C groups")
    if initial_states is not None and initial_states.shape != (
        batch,
        heads,
        head_dim,
        state_dim,
    ):
        raise ValueError("Mamba-2 initial_states must use [B,H,P,N]")
    tensors = (dt, A, B, C) + (() if initial_states is None else (initial_states,))
    if any(t.device != x.device or not t.is_floating_point() for t in tensors):
        raise ValueError("Mamba-2 operands must be floating point and share a device")
    if not x.is_floating_point():
        raise ValueError("Mamba-2 x must be floating point")
    if any(t.dtype != x.dtype for t in tensors):
        raise ValueError("Mamba-2 operands must use the same dtype")
    return batch, sequence, heads, head_dim, groups, state_dim


def _log_linear_shapes(
    torch: Any,
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any,
    level_scales: Any,
    initial_state: Any,
):
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("LogLinear query/key must use matching [B,T,1,K]")
    batch, sequence, groups, key_dim = query.shape
    if groups != 1:
        raise ValueError(
            "pinned LogLinear attention requires one shared query/key group"
        )
    if value.ndim != 4 or value.shape[:2] != (batch, sequence):
        raise ValueError("LogLinear values must use [B,T,H,V]")
    heads, value_dim = value.shape[2:]
    if log_decay.shape != (batch, sequence, heads):
        raise ValueError("LogLinear log_decay must use [B,T,H]")
    if level_scales.ndim != 4 or level_scales.shape[:3] != (batch, sequence, heads):
        raise ValueError("LogLinear level_scales must use [B,T,H,L]")
    if sequence < 64:
        raise ValueError("LogLinear prototype requires at least one 64-token chunk")
    if key_dim % 64 or value_dim & (value_dim - 1):
        raise ValueError(
            "LogLinear requires key width divisible by 64 and power-of-two value width"
        )
    if initial_state is not None:
        raise ValueError(
            "LogLinear prototype currently requires an empty initial state"
        )
    tensors = (key, value, log_decay, level_scales)
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError(
            "LogLinear operands must be floating point and share one device"
        )
    if any(t.dtype != torch.float32 for t in tensors + (query,)):
        raise ValueError("LogLinear prototype is currently qualified for float32")
    chunks = (sequence + 63) // 64
    largest_level = 6 if chunks == 1 else 7 + (chunks - 1).bit_length() - 1
    if level_scales.shape[-1] <= largest_level:
        raise ValueError(f"LogLinear needs at least {largest_level + 1} level scales")
    return batch, sequence, heads, key_dim, value_dim, chunks


def _log_linear_final_state(
    torch: Any, query: Any, key: Any, value: Any, log_decay: Any, level_scales: Any
) -> MixerLogLinearState:
    batch, sequence, _, key_dim = query.shape
    heads, value_dim = value.shape[2:]
    chunks = sequence // 64
    total_chunks = (sequence + 63) // 64
    level_count = (total_chunks - 1).bit_length() + 1
    levels = [
        torch.zeros(
            (batch, heads, key_dim, value_dim), device=query.device, dtype=torch.float32
        )
        for _ in range(level_count)
    ]
    for chunk in range(chunks):
        start, end = chunk * 64, (chunk + 1) * 64
        local_gate = log_decay[:, start:end].float().cumsum(dim=1)
        total_gate = local_gate[:, -1]
        levels = [state * torch.exp(total_gate)[:, :, None, None] for state in levels]
        weighted_value = (
            value[:, start:end].float()
            * torch.exp(total_gate[:, None, :] - local_gate)[..., None]
        )
        expanded_key = key[:, start:end].expand(-1, -1, heads, -1).float()
        levels[0] = levels[0] + torch.einsum(
            "bthk,bthv->bhkv", expanded_key, weighted_value
        )
        completed = chunk + 1
        level = 0
        while completed % (1 << (level + 1)) == 0:
            levels[level + 1] = levels[level + 1] + levels[level]
            levels[level] = torch.zeros_like(levels[level])
            level += 1
    tail_start = chunks * 64
    tail_length = sequence - tail_start

    def padded_tail(tensor: Any):
        suffix = tensor[:, tail_start:]
        padding = torch.zeros(
            (batch, 64 - tail_length, *tensor.shape[2:]),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return torch.cat((suffix, padding), dim=1).contiguous()

    gate_prev = padded_tail(log_decay)
    return MixerLogLinearState(
        ht=torch.stack(levels, dim=1),
        offsets=torch.full((batch,), sequence, device=query.device, dtype=torch.int32),
        q_prev=padded_tail(query),
        k_prev=padded_tail(key),
        v_prev=padded_tail(value),
        g_prev=gate_prev.contiguous(),
        level_scales_prev=padded_tail(level_scales),
    )


def _execute_log_linear_attention_reference(
    spec: UnifiedMixerSpec, torch: Any, **operands: Any
):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    level_scales = operands.pop("level_scales")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected LogLinear operands: {', '.join(sorted(operands))}")
    batch, sequence, heads, _, _, _ = _log_linear_shapes(
        torch, query, key, value, log_decay, level_scales, initial_state
    )
    prefix_gate = log_decay.float().cumsum(dim=1)
    outputs = []
    for token in range(sequence):
        source_levels = []
        query_chunk = token // 64
        for source in range(token + 1):
            source_chunk = source // 64
            if source_chunk == query_chunk:
                source_levels.append(((token % 64) ^ (source % 64)).bit_length())
            else:
                source_levels.append(
                    7 + ((query_chunk ^ source_chunk).bit_length() - 1)
                )
        indices = torch.tensor(source_levels, device=query.device, dtype=torch.long)
        scale = level_scales[:, token].index_select(-1, indices).transpose(1, 2)
        decay = torch.exp(
            prefix_gate[:, token : token + 1] - prefix_gate[:, : token + 1]
        )
        similarity = (
            (key[:, : token + 1] * query[:, token : token + 1]).sum(dim=-1).squeeze(-1)
        )
        weights = similarity.unsqueeze(-1) * scale * decay
        outputs.append((weights[..., None] * value[:, : token + 1].float()).sum(dim=1))
    output = torch.stack(outputs, dim=1).to(value.dtype)
    state = _log_linear_final_state(torch, query, key, value, log_decay, level_scales)
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_log_linear_dyadic_attention",
            "state_layout": "FLA dyadic chunk hierarchy plus retained 64-token suffix",
        },
    )


def _execute_fla_log_linear_attention(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    if not plan.spec.log_linear_attention:
        raise RuntimeError("selected FLA anchor does not match LogLinear semantics")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    level_scales = operands.pop("level_scales")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected LogLinear operands: {', '.join(sorted(operands))}")
    _log_linear_shapes(torch, query, key, value, log_decay, level_scales, initial_state)
    identity = _check_fla_k2_runtime(query, key, value, torch)
    from fla.ops.log_linear_attn import chunk_log_linear_attn

    output, state = chunk_log_linear_attn(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        log_decay.contiguous(),
        level_scales.contiguous(),
        initial_state=None,
        output_final_state=True,
    )
    return MixerResult(
        output,
        final_state=state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "64-token chunked prefill",
            "backward_supported": True,
        },
    )


def _execute_mamba2_ssm_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    x = operands.pop("x")
    dt = operands.pop("dt")
    A = operands.pop("A")
    B = operands.pop("B")
    C = operands.pop("C")
    initial_states = operands.pop("initial_states", None)
    if operands:
        raise TypeError(f"unexpected Mamba-2 operands: {', '.join(sorted(operands))}")
    batch, sequence, heads, head_dim, groups, state_dim = _mamba2_shapes(
        x, dt, A, B, C, initial_states
    )
    input_dtype = x.dtype
    b_heads = B.float().repeat_interleave(heads // groups, dim=2)
    c_heads = C.float().repeat_interleave(heads // groups, dim=2)
    state = (
        torch.zeros(
            (batch, heads, head_dim, state_dim), device=x.device, dtype=torch.float32
        )
        if initial_states is None
        else initial_states.float()
    )
    outputs = []
    for token in range(sequence):
        step = dt[:, token].float()
        decay = torch.exp(step * A.float()[None, :])
        state = state * decay[:, :, None, None]
        state = state + (
            x[:, token].float().unsqueeze(-1)
            * b_heads[:, token].unsqueeze(-2)
            * step[:, :, None, None]
        )
        outputs.append((state * c_heads[:, token].unsqueeze(-2)).sum(dim=-1))
    return MixerResult(
        torch.stack(outputs, dim=1).to(input_dtype),
        final_state=state.to(C.dtype),
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_mamba2_recurrence",
            "state_layout": "[B,H,P,N]",
        },
    )


def _execute_mamba2_ssm_library(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    x = operands.pop("x")
    dt = operands.pop("dt")
    A = operands.pop("A")
    B = operands.pop("B")
    C = operands.pop("C")
    initial_states = operands.pop("initial_states", None)
    if operands:
        raise TypeError(f"unexpected Mamba-2 operands: {', '.join(sorted(operands))}")
    _mamba2_shapes(x, dt, A, B, C, initial_states)
    if not all(t.is_cuda for t in (x, dt, A, B, C)):
        raise RuntimeError("the pinned Mamba-2 SSD adapter requires CUDA tensors")
    if x.dtype != torch.float32:
        raise RuntimeError("the pinned Mamba-2 SSD adapter is qualified for float32")
    from urm.compiler.execution import MAMBA2_SSD_ANCHOR_NAME

    output, final_state = _pinned_mamba2_scan()(
        x,
        dt,
        A,
        B,
        C,
        chunk_size=64,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=initial_states,
        dt_softplus=False,
        return_final_states=True,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": MAMBA2_SSD_ANCHOR_NAME,
            "execution": "pinned_mamba2_chunk_scan",
            "upstream_revision": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
            "chunk_size": 64,
        },
    )


def _execute_fla_hgrn(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    x = operands.pop("x")
    log_decay = operands.pop("log_decay")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected FLA HGRN operands: {', '.join(sorted(operands))}")
    if initial_state is not None and initial_state.ndim == 3:
        if initial_state.shape[-1] != 1:
            raise ValueError("HGRN initial_state must have shape [B,C] or [B,C,1]")
        initial_state = initial_state.squeeze(-1)
    from urm.compiler.execution import FLA_HGRN_ANCHOR_NAME

    output, final_state = _pinned_fla_hgrn()(
        x,
        log_decay,
        initial_state=initial_state,
        output_final_state=True,
    )
    return MixerResult(
        output,
        final_state=final_state.unsqueeze(-1),
        metadata={
            "anchor": FLA_HGRN_ANCHOR_NAME,
            "execution": "pinned_fla_fused_recurrent_hgrn",
            "compiler_plan": plan.anchor,
            "upstream_revision": "864a87f6ce5be8828bef81eb22baafd41937cdf2",
        },
    )


@lru_cache(maxsize=1)
def _pinned_fla_hgrn():
    import inspect
    import subprocess
    from pathlib import Path

    import fla
    from fla.ops.hgrn import fused_recurrent_hgrn

    source = Path(inspect.getfile(fla)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError("FLA HGRN anchor requires its pinned source checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
    if revision != expected:
        raise RuntimeError(
            f"FLA HGRN anchor requires revision {expected}, loaded {revision}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("FLA HGRN anchor requires a clean pinned source tree")
    return fused_recurrent_hgrn


@lru_cache(maxsize=1)
def _pinned_mamba_selective_scan():
    import inspect
    import subprocess
    from pathlib import Path

    import mamba_ssm
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    source = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(
            "Mamba library anchor requires its pinned source checkout in the Python path"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
    if revision != expected:
        raise RuntimeError(
            f"Mamba library anchor requires revision {expected}, loaded {revision} from {source}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("Mamba library anchor requires a clean pinned source tree")
    return selective_scan_fn


@lru_cache(maxsize=1)
def _pinned_mamba2_scan():
    import inspect
    import subprocess
    from pathlib import Path

    import mamba_ssm
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    source = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError("Mamba-2 adapter requires its pinned source checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
    if revision != expected:
        raise RuntimeError(
            f"Mamba-2 adapter requires revision {expected}, loaded {revision}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("Mamba-2 adapter requires a clean pinned source tree")
    return mamba_chunk_scan_combined


def _execute_fla_rwkv4(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.rwkv4_memory:
        raise RuntimeError("selected FLA anchor does not match RWKV-4 semantics")
    w = operands.pop("w")
    u = operands.pop("u")
    key = operands.pop("k")
    value = operands.pop("v")
    state = operands.pop("state")
    if operands:
        raise TypeError(f"unexpected RWKV-4 operands: {', '.join(sorted(operands))}")
    if key.ndim != 3 or value.shape != key.shape:
        raise ValueError("RWKV-4 k and v must use matching [B,T,C] layouts")
    batch, _, channels = key.shape
    if w.shape != (channels,) or u.shape != (channels,):
        raise ValueError("RWKV-4 w and u must use [C]")
    if state.shape != (batch, 3, 1, channels):
        raise ValueError("RWKV-4 state must use [B,3,1,C] as alpha, beta, eps")
    tensors = (w, u, value, state)
    if any(t.device != key.device or t.dtype != key.dtype for t in tensors):
        raise ValueError("RWKV-4 operands must share one device and dtype")
    if not key.is_cuda:
        raise RuntimeError("pinned FLA RWKV-4 requires CUDA tensors")
    if key.dtype != torch.float32:
        raise RuntimeError("the pinned FLA RWKV-4 adapter is qualified for float32")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError(
            "the pinned FLA RWKV-4 adapter requires the recorded FLA source"
        )
    from fla.ops.rwkv4 import fused_recurrent_rwkv4

    output, final_state = fused_recurrent_rwkv4(
        w.contiguous(),
        u.contiguous(),
        key.contiguous(),
        value.contiguous(),
        state.contiguous(),
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "recurrent",
            "backward_supported": True,
        },
    )


def _execute_fla_rwkv6(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.rwkv6_memory:
        raise RuntimeError("selected FLA anchor does not match RWKV-6 semantics")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    bonus = operands.pop("bonus")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected RWKV-6 operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("RWKV-6 query/key/value use BTHD layout")
    batch, _, heads, key_dim = query.shape
    if value.shape[:3] != query.shape[:3] or log_decay.shape != query.shape:
        raise ValueError("RWKV-6 value and log_decay must align with query axes")
    if bonus.shape != (heads, key_dim):
        raise ValueError("RWKV-6 bonus must use [H,K]")
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value.shape[-1],
    ):
        raise ValueError("RWKV-6 initial_state must use [B,H,K,V]")
    tensors = (key, value, log_decay, bonus) + (
        () if initial_state is None else (initial_state,)
    )
    if any(t.device != query.device or t.dtype != query.dtype for t in tensors):
        raise ValueError("RWKV-6 inputs must share device and dtype")
    if not query.is_cuda:
        raise RuntimeError("pinned FLA RWKV-6 requires CUDA tensors")
    if query.dtype != torch.float32:
        raise RuntimeError("the pinned FLA RWKV-6 adapter is qualified for float32")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("the RWKV-6 adapter requires the exact recorded FLA source")
    from fla.ops.rwkv6 import fused_recurrent_rwkv6

    output, final_state = fused_recurrent_rwkv6(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        log_decay.contiguous(),
        bonus.contiguous(),
        scale=key_dim**-0.5,
        initial_state=None if initial_state is None else initial_state.contiguous(),
        output_final_state=True,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "recurrent",
            "backward_supported": True,
        },
    )


def _execute_fla_mesa_net(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    beta = operands.pop("beta")
    lamb = operands.pop("lamb")
    h_kk_init = operands.pop("h_kk_init", None)
    h_kv_init = operands.pop("h_kv_init", None)
    if operands:
        raise TypeError(f"unexpected MesaNet operands: {', '.join(sorted(operands))}")
    if not query.is_cuda or query.dtype != torch.bfloat16:
        raise RuntimeError("the pinned MesaNet adapter requires CUDA BF16 Q/K/V")
    if any(tensor.dtype != torch.float32 for tensor in (log_decay, beta, lamb)):
        raise ValueError("MesaNet log_decay, beta and lamb must be float32")
    from urm.adapters.gated_delta_rule import fla_version

    if fla_version().get("comparison_compatible") is not True:
        raise RuntimeError("MesaNet requires the exact recorded FLA source")
    from fla.ops.mesa_net.chunk import chunk_mesa_net

    output, h_kk_final, h_kv_final = chunk_mesa_net(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        log_decay.contiguous(),
        beta.contiguous(),
        lamb.contiguous(),
        h_kk_init=h_kk_init,
        h_kv_init=h_kv_init,
        output_final_state=True,
        max_CG_iteration=30,
        use_qk_l2norm_in_kernel=False,
    )
    return MixerResult(
        output,
        final_state=(h_kk_final, h_kv_final),
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_fla_chunk_mesa_net",
            "max_CG_iteration": 30,
            "backward_supported": True,
        },
    )


def _execute_fla_titans_linear(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    weight = operands.pop("w")
    bias = operands.pop("b")
    theta = operands.pop("theta")
    alpha = operands.pop("alpha")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    chunk_size = int(operands.pop("chunk_size", 16))
    eps = float(operands.pop("eps", 1e-6))
    if operands:
        raise TypeError(f"unexpected Titans operands: {', '.join(sorted(operands))}")
    if not query.is_cuda or query.dtype != torch.float32:
        raise RuntimeError("the pinned FLA Titans adapter requires CUDA float32 inputs")
    if query.shape[1] % chunk_size:
        raise ValueError("Titans sequence length must be divisible by chunk_size")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("source_revision") != "864a87f6ce5be8828bef81eb22baafd41937cdf2":
        raise RuntimeError("Titans requires the exact recorded FLA source revision")
    from fla.ops.titans.naive import chunk_titans_linear_ref

    output, final_state = chunk_titans_linear_ref(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        theta.contiguous(),
        alpha.contiguous(),
        eta.contiguous(),
        eps=eps,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=True,
        use_chunk=True,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_fla_chunk_titans_linear",
            "upstream": identity,
            "chunk_size": chunk_size,
            "backward_supported": True,
        },
    )


def _execute_fla_ttt_linear(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    weight = operands.pop("w")
    bias = operands.pop("b")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    initial_state_bias = operands.pop("initial_state_bias", None)
    chunk_size = int(operands.pop("chunk_size", 16))
    eps = float(operands.pop("eps", 1e-6))
    if operands:
        raise TypeError(
            f"unexpected TTT-Linear operands: {', '.join(sorted(operands))}"
        )
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned FLA TTT-Linear adapter requires CUDA FP16/BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("source_revision") != "864a87f6ce5be8828bef81eb22baafd41937cdf2":
        raise RuntimeError("TTT-Linear requires the exact recorded FLA source revision")
    from fla.ops.ttt.chunk import chunk_ttt_linear

    output, final_state, final_bias_state = chunk_ttt_linear(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        eta.contiguous(),
        scale=query.shape[-1] ** -0.5,
        eps=eps,
        chunk_size=chunk_size,
        initial_state=initial_state,
        initial_state_bias=initial_state_bias,
        output_final_state=True,
    )
    return MixerResult(
        output,
        final_state=final_state,
        final_normalizer_state=final_bias_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_fla_chunk_ttt_linear",
            "upstream": identity,
            "chunk_size": chunk_size,
            "backward_supported": True,
        },
    )


def _execute_fla_momentum_delta(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.momentum_delta:
        raise RuntimeError("selected FLA anchor does not match Momentum DeltaNet")
    (
        query,
        key,
        value,
        p,
        log_alpha,
        log_mu,
        beta,
        eta,
        initial_state,
        initial_normalizer_state,
        _,
    ) = _momentum_delta_operands(torch, **operands)
    if not query.is_cuda:
        raise RuntimeError("pinned FLA Momentum DeltaNet requires CUDA tensors")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("pinned FLA Momentum DeltaNet requires float16 or bfloat16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("Momentum DeltaNet requires the exact recorded FLA source")
    from fla.ops.momentum_delta_rule.chunk import chunk_momentum_delta_rule

    initial = None
    if initial_state is not None or initial_normalizer_state is not None:
        if initial_state is None or initial_normalizer_state is None:
            raise ValueError(
                "Momentum DeltaNet initial matrix states must be supplied together"
            )
        initial = torch.stack(
            (initial_state.contiguous(), initial_normalizer_state.contiguous()), dim=0
        )
    output, final_state = chunk_momentum_delta_rule(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        log_alpha.contiguous(),
        log_mu.contiguous(),
        p=p.contiguous(),
        beta=beta.contiguous(),
        eta=eta.contiguous(),
        scale=query.shape[-1] ** -0.5,
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_p_times_alpha=False,
        chunk_size=64,
    )
    return MixerResult(
        output,
        final_state=final_state[0],
        final_normalizer_state=final_state[1],
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "chunked_recurrent",
            "backward_supported": True,
        },
    )


def _execute_fla_gated_oja(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.gated_oja:
        raise RuntimeError("selected FLA anchor does not match gated Oja semantics")
    query, key, value, gate, beta, initial_state, _ = _gated_oja_operands(
        torch, **operands
    )
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError(
            "the pinned FLA gated Oja adapter requires CUDA FP16 or BF16"
        )
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("gated Oja requires the exact recorded FLA source")
    from fla.ops.gated_oja_rule import chunk_gated_oja_rule

    output, final_state = chunk_gated_oja_rule(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        beta.contiguous(),
        scale=query.shape[-1] ** -0.5,
        initial_state=None if initial_state is None else initial_state.contiguous(),
        output_final_state=True,
        use_q_l2norm=False,
        use_k_l2norm=False,
        chunk_size=64,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "chunked_recurrent",
            "backward_supported": True,
        },
    )


def _execute_fla_comba(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.comba_rule:
        raise RuntimeError("selected FLA anchor does not match COMBA semantics")
    (
        query,
        key,
        value,
        prediction_key,
        log_decay,
        beta,
        initial_state,
        _,
    ) = _comba_operands(torch, **operands)
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned FLA COMBA adapter requires CUDA FP16 or BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("COMBA requires the exact recorded FLA source")
    from fla.ops.comba import chunk_comba

    output, final_state = chunk_comba(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        prediction_key.contiguous(),
        log_decay.contiguous(),
        beta=beta.contiguous(),
        scale=query.shape[-1] ** -0.5,
        initial_state=None if initial_state is None else initial_state.contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
        },
    )


def _execute_fla_pgdn(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.preconditioned_gated_delta:
        raise RuntimeError("selected FLA anchor does not match PGDN semantics")
    (
        query,
        key,
        value,
        g_atk,
        gate,
        beta_atk,
        beta,
        initial_state,
        initial_A_state,
    ) = _pgdn_operands(torch, **operands)
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned FLA PGDN adapter requires CUDA FP16 or BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("PGDN requires the exact recorded FLA source")
    from fla.ops.precond_gated_delta_rule.chunk import chunk_precond_gated_delta_rule

    output, final_state, final_A_state = chunk_precond_gated_delta_rule(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g_atk.contiguous(),
        gate.contiguous(),
        beta_atk.contiguous(),
        beta.contiguous(),
        scale=query.shape[-1] ** -0.5,
        initial_state=None if initial_state is None else initial_state.contiguous(),
        initial_A_state=None
        if initial_A_state is None
        else initial_A_state.contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        x=1.5,
        eps=1e-6,
        log_atk_scale=None,
    )
    return MixerResult(
        output,
        final_state=final_state,
        final_normalizer_state=final_A_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
        },
    )


def _execute_fla_pkda(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.preconditioned_kda:
        raise RuntimeError("selected FLA anchor does not match PKDA semantics")
    (
        query,
        key,
        value,
        g_atk,
        gate,
        beta_atk,
        beta,
        initial_state,
        initial_A_state,
    ) = _pgdn_operands(torch, kda=True, **operands)
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned FLA PKDA adapter requires CUDA FP16 or BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("PKDA requires the exact recorded FLA source")
    from fla.ops.precond_kda.chunk import chunk_precond_kda

    output, final_state, final_A_state = chunk_precond_kda(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        gate.contiguous(),
        g_atk.contiguous(),
        beta_atk.contiguous(),
        beta.contiguous(),
        scale=query.shape[-1] ** -0.5,
        initial_state=None if initial_state is None else initial_state.contiguous(),
        initial_A_state=None
        if initial_A_state is None
        else initial_A_state.contiguous(),
        output_final_state=True,
        use_gate_in_kernel=False,
        safe_gate=False,
        x=1.5,
        eps=1e-6,
        log_atk_scale=None,
    )
    return MixerResult(
        output,
        final_state=final_state,
        final_normalizer_state=final_A_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
        },
    )


def _execute_fla_slot_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.slot_attention:
        raise RuntimeError(
            "selected FLA anchor does not match slot-attention semantics"
        )
    if plan.spec.name == "abc_core":
        query = operands["query"]
        key = operands["key"]
        value = operands["value"]
        slot_logits = operands["slot_logits"]
        initial_key_state = operands.get("initial_key_state")
        initial_value_state = operands.get("initial_value_state")
        if slot_logits.ndim != 4:
            raise ValueError("ABC slot logits must use [B,T,H,M]")
        expected_slot_shape = (*key.shape[:3], slot_logits.shape[-1])
        if slot_logits.shape != expected_slot_shape:
            raise ValueError("ABC slot logits must use [B,T,H,M]")
        if not (query.dtype == key.dtype == value.dtype == slot_logits.dtype):
            raise ValueError("ABC query/key/value/slot logits must share a dtype")
        if query.shape[:2] != key.shape[:2] or value.shape[:3] != key.shape[:3]:
            raise ValueError("ABC query/key/value batch and time axes must align")
        if query.shape[2] != key.shape[2] or query.shape[-1] != key.shape[-1]:
            raise ValueError("ABC requires matching query/key heads and key dimensions")
        if value.device != query.device or slot_logits.device != query.device:
            raise ValueError("ABC operands must share one device")
        if initial_key_state is not None and initial_key_state.shape != (
            key.shape[0],
            key.shape[2],
            key.shape[3],
            slot_logits.shape[-1],
        ):
            raise ValueError("ABC initial key state must use [B,H,K,M]")
        if initial_value_state is not None and initial_value_state.shape != (
            key.shape[0],
            key.shape[2],
            slot_logits.shape[-1],
            value.shape[-1],
        ):
            raise ValueError("ABC initial value state must use [B,H,M,V]")
        if any(
            state is not None and state.device != query.device
            for state in (initial_key_state, initial_value_state)
        ):
            raise ValueError("ABC initial states must share the input device")
        slot_weights = log_decay = None
    else:
        (
            query,
            key,
            value,
            slot_weights,
            log_decay,
            initial_key_state,
            initial_value_state,
            _,
        ) = _slot_attention_operands(plan.spec, torch, **operands)
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned FLA ABC/GSA adapters require CUDA FP16 or BF16")
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError("ABC/GSA require the exact recorded FLA source")
    initial_state = (initial_key_state, initial_value_state)
    if plan.spec.name == "abc_core":
        from fla.ops.abc import chunk_abc

        output, final_state = chunk_abc(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            slot_logits.contiguous(),
            initial_state=initial_state,
            output_final_state=True,
        )
    else:
        from fla.ops.gsa import chunk_gsa

        output, final_states = chunk_gsa(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            slot_weights.contiguous(),
            log_decay.contiguous(),
            scale=query.shape[-1] ** -0.5,
            initial_state=initial_state,
            output_final_state=True,
            checkpoint_level=0,
        )
        final_state = tuple(final_states)
    return MixerResult(
        output,
        final_state=tuple(final_state),
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
        },
    )


def _execute_fla_k2(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if plan.anchor == "fla_chunk_gdn2_adapter":
        return _execute_fla_gdn2(plan, torch, **operands)
    if plan.anchor in {"fla_fused_recurrent_iplr_adapter", "fla_chunk_dplr_adapter"}:
        return _execute_fla_generalized_delta(plan, torch, **operands)
    if plan.anchor == "fla_chunk_gated_delta_product_adapter":
        return _execute_fla_gated_delta_product(plan, torch, **operands)
    if plan.anchor == "fla_chunk_kda_adapter":
        return _execute_fla_kda(plan, torch, **operands)
    if plan.anchor == "fla_gated_delta_rule_adapter":
        return _execute_fla_gated_delta(plan, torch, **operands)
    if plan.anchor in {
        "fla_chunk_gla_adapter",
        "fla_chunk_simple_gla_adapter",
        "fla_fused_recurrent_gla_decode_adapter",
        "fla_fused_recurrent_simple_gla_decode_adapter",
    }:
        return _execute_fla_gated_additive(plan, torch, **operands)
    if plan.anchor == "fla_chunk_linear_attention_adapter":
        return _execute_fla_linear_attention(plan, torch, **operands)
    if plan.anchor == "fla_chunk_delta_rule_adapter":
        return _execute_fla_delta_rule(plan, torch, **operands)
    raise RuntimeError(f"no K2 library executor is registered for {plan.anchor!r}")


def _gdn2_shapes(
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any,
    erase_gate: Any,
    write_gate: Any,
    initial_state: Any,
):
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("GDN-2 query/key must use matching [B,T,H,K]")
    batch, sequence, heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:2] != (batch, sequence):
        raise ValueError("GDN-2 values must use [B,T,Hv,V]")
    value_heads, value_dim = value.shape[2:]
    if value_heads < heads or value_heads % heads:
        raise ValueError("GDN-2 value heads must be a positive multiple of query heads")
    gate_shape = (batch, sequence, value_heads, key_dim)
    if log_decay.shape != gate_shape or erase_gate.shape != gate_shape:
        raise ValueError("GDN-2 decay and erase gates must use [B,T,Hv,K]")
    if write_gate.shape != (batch, sequence, value_heads, value_dim):
        raise ValueError("GDN-2 write gate must use [B,T,Hv,V]")
    if initial_state is not None and initial_state.shape != (
        batch,
        value_heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("GDN-2 initial_state must use [B,Hv,K,V]")
    tensors = (key, value, log_decay, erase_gate, write_gate) + (
        () if initial_state is None else (initial_state,)
    )
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError("GDN-2 operands must be floating point and share a device")
    if any(t.dtype != query.dtype for t in tensors):
        raise ValueError("GDN-2 operands must use the same dtype")
    return batch, sequence, heads, value_heads, key_dim, value_dim


def _execute_gdn2_reference(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    erase_gate = operands.pop("erase_gate")
    write_gate = operands.pop("write_gate")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected GDN-2 operands: {', '.join(sorted(operands))}")
    batch, sequence, heads, value_heads, key_dim, value_dim = _gdn2_shapes(
        query, key, value, log_decay, erase_gate, write_gate, initial_state
    )
    input_dtype = value.dtype
    query_f = query.float()
    key_f = key.float()
    if value_heads != heads:
        repeats = value_heads // heads
        query_f = query_f.repeat_interleave(repeats, dim=2)
        key_f = key_f.repeat_interleave(repeats, dim=2)
    state = (
        torch.zeros(
            (batch, value_heads, key_dim, value_dim),
            dtype=torch.float32,
            device=query.device,
        )
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    scale = spec.read_scale or key_dim**-0.5
    for token in range(sequence):
        decay_t = torch.exp(log_decay[:, token].float())
        erase_t = erase_gate[:, token].float()
        write_t = write_gate[:, token].float()
        value_t = value[:, token].float()
        key_t = key_f[:, token]
        query_t = query_f[:, token]
        decayed = state * decay_t.unsqueeze(-1)
        correction = torch.einsum("bhk,bhkv->bhv", erase_t * key_t, decayed)
        update = write_t * value_t - correction
        state = decayed + torch.einsum("bhk,bhv->bhkv", key_t, update)
        outputs.append(torch.einsum("bhk,bhkv->bhv", query_t * scale, state))
    return MixerResult(
        torch.stack(outputs, dim=1).to(input_dtype),
        final_state=state,
        metadata={
            "anchor": "urm.unified.k2.state_reference.v1",
            "execution": "torch_eager_gdn2_recurrence",
            "state_layout": "[B,Hv,K,V]",
        },
    )


def _execute_fla_gdn2(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.gdn2_ssm:
        raise RuntimeError("selected FLA anchor does not match GDN-2 semantics")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    erase_gate = operands.pop("erase_gate")
    write_gate = operands.pop("write_gate")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected GDN-2 operands: {', '.join(sorted(operands))}")
    _gdn2_shapes(query, key, value, log_decay, erase_gate, write_gate, initial_state)
    identity = _check_fla_k2_runtime(query, key, value, torch)
    if query.dtype != torch.float32:
        raise RuntimeError(
            "the pinned FLA GDN-2 chunk adapter is qualified for float32"
        )
    from fla.ops.gdn2 import chunk_gdn2

    output, final_state = chunk_gdn2(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        log_decay.contiguous(),
        erase_gate.contiguous(),
        write_gate.contiguous(),
        scale=plan.spec.read_scale or query.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "chunk",
            "backward_supported": True,
        },
    )


def _execute_fla_kda(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not plan.spec.kda_delta:
        raise RuntimeError("selected FLA anchor does not match KDA semantics")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    beta = operands.pop("beta")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected KDA operands: {', '.join(sorted(operands))}")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("KDA query/key must use matching [B,T,H,K]")
    batch, sequence, heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("KDA value must use [B,T,H,V] with matching heads")
    value_dim = value.shape[-1]
    if log_decay.shape != (batch, sequence, heads, key_dim):
        raise ValueError("KDA log_decay must use [B,T,H,K]")
    if beta.shape not in ((batch, sequence, heads), (batch, sequence, heads, 1)):
        raise ValueError("KDA beta must use [B,T,H]")
    if beta.ndim == 4:
        beta = beta.squeeze(-1)
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("KDA initial_state must use [B,H,K,V]")
    identity = _check_fla_k2_runtime(query, key, value, torch)
    if query.dtype != torch.float32:
        raise RuntimeError("the pinned FLA KDA chunk adapter is qualified for float32")
    if any(t.device != query.device for t in (log_decay, beta)):
        raise ValueError("KDA gates must share the query device")
    from fla.ops.kda import chunk_kda

    output, final_state = chunk_kda(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        log_decay.contiguous(),
        beta.contiguous(),
        scale=plan.spec.read_scale or key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        state_v_first=False,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "chunk",
            "backward_supported": True,
        },
    )


def _execute_fla_gated_delta_product(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    if not plan.spec.gated_delta_product:
        raise RuntimeError(
            "selected FLA anchor does not match Gated DeltaProduct semantics"
        )
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    beta = operands.pop("beta")
    update_keys = operands.pop("update_keys")
    update_values = operands.pop("update_values")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(
            f"unexpected Gated DeltaProduct operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("Gated DeltaProduct query/key must use matching [B,T,H,K]")
    batch, sequence, heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("Gated DeltaProduct value must use [B,T,H,V]")
    if update_keys.ndim != 5 or update_keys.shape[:2] != (batch, sequence):
        raise ValueError("update_keys must use [B,T,R,H,K]")
    ranks = update_keys.shape[2]
    value_dim = value.shape[-1]
    if ranks <= 0 or update_keys.shape[3:] != (heads, key_dim):
        raise ValueError("Gated DeltaProduct update_keys must use [B,T,R,H,K]")
    if update_values.shape != (batch, sequence, ranks, heads, value_dim):
        raise ValueError("update_values must use [B,T,R,H,V]")
    if beta.shape != (batch, sequence, ranks, heads):
        raise ValueError("Gated DeltaProduct beta must use [B,T,R,H]")
    if log_decay.shape not in ((batch, sequence, heads), (batch, sequence, heads, 1)):
        raise ValueError("Gated DeltaProduct log_decay must use [B,T,H]")
    if log_decay.ndim == 4:
        log_decay = log_decay.squeeze(-1)
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("Gated DeltaProduct initial_state must use [B,H,K,V]")
    tensors = (key, value, log_decay, beta, update_keys, update_values) + (
        () if initial_state is None else (initial_state,)
    )
    if any(
        tensor.device != query.device or not tensor.is_floating_point()
        for tensor in tensors
    ):
        raise ValueError(
            "Gated DeltaProduct operands must be floating point and share a device"
        )
    if not (
        query.dtype
        == key.dtype
        == value.dtype
        == update_keys.dtype
        == update_values.dtype
    ):
        raise ValueError(
            "Gated DeltaProduct query, key and value tensors must share a dtype"
        )
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            "the pinned FLA Gated DeltaProduct chunk requires float16 or bfloat16"
        )
    identity = _check_fla_k2_runtime(query, key, value, torch)
    from fla.ops.gated_delta_product import chunk_gated_delta_product

    output, final_state = chunk_gated_delta_product(
        query.contiguous(),
        update_keys.reshape(batch, sequence * ranks, heads, key_dim).contiguous(),
        update_values.reshape(batch, sequence * ranks, heads, value_dim).contiguous(),
        log_decay.contiguous(),
        beta.reshape(batch, sequence * ranks, heads).contiguous(),
        num_householder=ranks,
        scale=plan.spec.read_scale or key_dim**-0.5,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "chunk",
            "ordered_updates_per_token": ranks,
            "backward_supported": True,
        },
    )


def _execute_fla_generalized_delta(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    if not (plan.spec.generalized_delta_iplr or plan.spec.generalized_delta_dplr):
        raise RuntimeError(
            "selected FLA anchor does not match generalized-delta semantics"
        )
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    transition_alpha = operands.pop("transition_alpha")
    transition_beta = operands.pop("transition_beta")
    log_decay = operands.pop("log_decay", None)
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(
            f"unexpected generalized-delta operands: {', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("generalized-delta query/key must use matching [B,T,H,K]")
    batch, sequence, heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("generalized-delta value must use [B,T,H,V]")
    if transition_alpha.shape != query.shape or transition_beta.shape != query.shape:
        raise ValueError("generalized-delta factors must use [B,T,H,K]")
    if plan.spec.generalized_delta_dplr:
        if log_decay is None or log_decay.shape != query.shape:
            raise ValueError("DPLR log_decay must use [B,T,H,K]")
    elif log_decay is not None:
        raise ValueError("IPLR does not accept log_decay")
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value.shape[-1],
    ):
        raise ValueError("generalized-delta initial_state must use [B,H,K,V]")
    tensors = (
        (key, value, transition_alpha, transition_beta)
        + (() if log_decay is None else (log_decay,))
        + (() if initial_state is None else (initial_state,))
    )
    if any(t.device != query.device or not t.is_floating_point() for t in tensors):
        raise ValueError(
            "generalized-delta operands must be floating point and share a device"
        )
    if any(
        t.dtype != query.dtype for t in (key, value, transition_alpha, transition_beta)
    ):
        raise ValueError("generalized-delta query/key/value/factors must share a dtype")
    identity = _check_fla_k2_runtime(query, key, value, torch)
    if plan.spec.generalized_delta_iplr:
        if query.dtype != torch.float32:
            raise RuntimeError(
                "the pinned IPLR recurrent adapter is qualified for float32"
            )
        from fla.ops.generalized_delta_rule.iplr import fused_recurrent_iplr_delta_rule

        output, final_state = fused_recurrent_iplr_delta_rule(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            transition_alpha.contiguous(),
            transition_beta.contiguous(),
            scale=plan.spec.read_scale or key_dim**-0.5,
            initial_state=initial_state,
            output_final_state=True,
        )
    else:
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                "the pinned DPLR chunk adapter requires float16 or bfloat16"
            )
        if plan.spec.name == "rwkv7_transition_core":
            from fla.ops.rwkv7 import chunk_rwkv7

            output, final_state = chunk_rwkv7(
                r=query.contiguous(),
                w=log_decay.contiguous(),
                k=key.contiguous(),
                v=value.contiguous(),
                a=transition_alpha.contiguous(),
                b=transition_beta.contiguous(),
                scale=plan.spec.read_scale or 1.0,
                initial_state=initial_state,
                output_final_state=True,
                safe_gate=True,
                lower_bound=-0.6065306597126334,
                chunk_size=64,
            )
        else:
            from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

            output, final_state = chunk_dplr_delta_rule(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                transition_alpha.contiguous(),
                transition_beta.contiguous(),
                log_decay.contiguous(),
                scale=plan.spec.read_scale or key_dim**-0.5,
                initial_state=initial_state,
                output_final_state=True,
                chunk_size=16,
            )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "recurrent"
            if plan.spec.generalized_delta_iplr
            else "chunk",
            "backward_supported": True,
        },
    )


def _execute_fla_polynomial_attention(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    initial_state = operands.pop("initial_state", None)
    initial_normalizer_state = operands.pop("initial_normalizer_state", None)
    if operands:
        raise TypeError(
            f"unexpected polynomial-attention operands: {', '.join(sorted(operands))}"
        )
    if initial_state is not None or initial_normalizer_state is not None:
        raise ValueError(
            "pinned FLA polynomial attention does not accept initial state"
        )
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("polynomial attention query/key must use matching [B,T,H,K]")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("polynomial attention values must use [B,T,H,V]")
    identity = _check_fla_k2_runtime(query, key, value, torch)
    if query.dtype != torch.float32:
        raise RuntimeError("pinned FLA polynomial-attention anchors require float32")
    scale = spec.read_scale or query.shape[-1] ** -0.5
    if spec.polynomial_basis is PolynomialBasis.BASED_TAYLOR2:
        if query.shape[-1] > 16:
            raise ValueError(
                "pinned fused-chunk Based supports key dimensions up to 16"
            )
        from fla.ops.based import fused_chunk_based

        output = fused_chunk_based(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            scale=scale,
            use_norm=True,
        )
    elif spec.polynomial_basis is PolynomialBasis.REBASED_SQUARE:
        from fla.ops.rebased import parallel_rebased

        output = parallel_rebased(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            eps=spec.epsilon,
            use_scale=True,
            use_normalize=True,
        )
    else:
        raise RuntimeError("unknown polynomial-attention basis")
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "causal_sequence",
            "backward_supported": True,
        },
    )


def _execute_fla_gated_additive(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    spec = plan.spec
    is_simple = plan.anchor in {
        "fla_chunk_simple_gla_adapter",
        "fla_fused_recurrent_simple_gla_decode_adapter",
    }
    decode_only = "decode_adapter" in plan.anchor
    matches = _is_fla_simple_gla_spec(spec) if is_simple else _is_fla_gla_spec(spec)
    if not matches:
        raise RuntimeError("selected FLA anchor does not match K2 additive semantics")

    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay", None)
    initial_state = operands.pop("initial_state", None)
    update_keys = operands.pop("update_keys", None)
    update_values = operands.pop("update_values", None)
    beta = operands.pop("beta", None)
    initial_normalizer = operands.pop("initial_normalizer_state", None)
    left_transition = operands.pop("left_transition", None)
    right_transition = operands.pop("right_transition", None)
    if operands:
        raise TypeError(
            f"unexpected FLA gated-additive operands: {', '.join(sorted(operands))}"
        )
    if any(
        item is not None
        for item in (
            update_keys,
            update_values,
            beta,
            initial_normalizer,
            left_transition,
            right_transition,
        )
    ):
        raise ValueError(
            "FLA gated-additive anchors accept one additive update and no denominator state"
        )
    if log_decay is None:
        raise ValueError("FLA gated-additive anchors require log_decay")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("FLA gated-additive query/key use matching [B,T,H,K]")
    batch, sequence, heads, key_dim = query.shape
    if decode_only and (sequence != 1 or plan.intent is MixerIntent.TRAINING):
        raise RuntimeError(
            "the pinned FLA simple-GLA/GLA anchor is qualified only for "
            "forward-only one-token decode"
        )
    verified_bf16_gla = (
        not is_simple
        and plan.anchor == "fla_chunk_gla_adapter"
        and query.dtype == torch.bfloat16
    )
    if not decode_only and query.dtype != torch.float32 and not verified_bf16_gla:
        raise RuntimeError(
            "FLA gated-additive chunk anchors require float32, except for verified BF16 GLA"
        )
    if value.ndim != 4 or value.shape[:3] != (batch, sequence, heads):
        raise ValueError("FLA gated-additive values must use [B,T,H,V] with Hq=Hv")
    if is_simple:
        if spec.static_head_decay:
            if log_decay.shape not in ((heads,), (1,)):
                raise ValueError("static head log_decay must be [H] or [1]")
        elif log_decay.shape == (batch, sequence, heads, 1):
            log_decay = log_decay.squeeze(-1)
        elif log_decay.shape != (batch, sequence, heads):
            raise ValueError("head log_decay must use [B,T,H]")
    elif log_decay.shape != (batch, sequence, heads, key_dim):
        raise ValueError("key-channel log_decay must use [B,T,H,K]")
    if not log_decay.is_floating_point() or log_decay.device != query.device:
        raise ValueError("log_decay must be floating point and share the query device")
    if spec.static_head_decay and log_decay.requires_grad:
        raise ValueError(
            "static head log_decay is a fixed schedule, not a trainable input"
        )
    expected_state = (
        (batch, heads, value.shape[-1], key_dim)
        if spec.state_v_first
        else (batch, heads, key_dim, value.shape[-1])
    )
    if initial_state is not None:
        _require_shape(initial_state, expected_state, "initial_state")
        if initial_state.device != query.device:
            raise ValueError("initial_state must share the query device")
        if initial_state.dtype != torch.float32:
            raise ValueError("FLA initial_state must use float32")

    identity = _check_fla_k2_runtime(query, key, value, torch)
    if (
        is_simple
        and plan.intent is MixerIntent.TRAINING
        and query.dtype in {torch.float16, torch.bfloat16}
        and (query.shape[-1] % 2 or value.shape[-1] % 2)
    ):
        raise ValueError(
            "the pinned FLA simple-GLA chunk backward requires even key and "
            "value dimensions for float16/bfloat16 tensors"
        )
    if verified_bf16_gla and spec.feature_map is FeatureMap.IDENTITY:
        q = query.contiguous()
        k = key.contiguous()
    else:
        q = _feature(torch, query, spec.feature_map).contiguous()
        k = _feature(torch, key, spec.feature_map).contiguous()
    v = value.contiguous()
    gate = (
        log_decay.contiguous() if verified_bf16_gla else log_decay.float().contiguous()
    )
    use_decode = decode_only
    if is_simple:
        if use_decode:
            from fla.ops.simple_gla import fused_recurrent_simple_gla

            if spec.static_head_decay:
                output, final_state = fused_recurrent_simple_gla(
                    q,
                    k,
                    v,
                    g_gamma=gate,
                    scale=spec.read_scale or 1.0,
                    initial_state=initial_state,
                    output_final_state=True,
                )
            else:
                output, final_state = fused_recurrent_simple_gla(
                    q,
                    k,
                    v,
                    g=gate,
                    scale=spec.read_scale or 1.0,
                    initial_state=initial_state,
                    output_final_state=True,
                )
        else:
            if spec.static_head_decay:
                if spec.static_head_decay_chunk:
                    from fla.ops.simple_gla import chunk_simple_gla

                    output, final_state = chunk_simple_gla(
                        q,
                        k,
                        v,
                        g_gamma=gate,
                        scale=spec.read_scale or 1.0,
                        initial_state=initial_state,
                        output_final_state=True,
                    )
                else:
                    from fla.ops.simple_gla import fused_chunk_simple_gla

                    output, final_state = fused_chunk_simple_gla(
                        q,
                        k,
                        v,
                        g_gamma=gate,
                        scale=spec.read_scale or 1.0,
                        initial_state=initial_state,
                        output_final_state=True,
                    )
            else:
                from fla.ops.simple_gla import chunk_simple_gla

                output, final_state = chunk_simple_gla(
                    q,
                    k,
                    v,
                    g=gate,
                    scale=spec.read_scale or 1.0,
                    initial_state=initial_state,
                    output_final_state=True,
                )
    else:
        if use_decode:
            from fla.ops.gla import fused_recurrent_gla

            output, final_state = fused_recurrent_gla(
                q,
                k,
                v,
                gk=gate,
                scale=spec.read_scale or 1.0,
                initial_state=initial_state,
                output_final_state=True,
                state_v_first=spec.state_v_first,
            )
        else:
            from fla.ops.gla import chunk_gla

            output, final_state = chunk_gla(
                q,
                k,
                v,
                g=gate,
                scale=spec.read_scale or 1.0,
                initial_state=initial_state,
                output_final_state=True,
                state_v_first=spec.state_v_first,
            )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "decode" if use_decode else "prefill",
            "backward_supported": not decode_only,
        },
    )


def _check_fla_k2_runtime(query: Any, key: Any, value: Any, torch: Any):
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("FLA K2 library anchors require CUDA tensors")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("FLA K2 query/key/value must use the same dtype")
    if query.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(
            "FLA K2 library anchors support float32, float16, and bfloat16"
        )
    from urm.adapters.gated_delta_rule import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError(
            "FLA K2 library anchors require flash-linear-attention==0.5.2 or "
            "the source revision recorded in the architecture coverage register; "
            f"identity={identity}"
        )
    return identity


def _execute_fla_linear_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    spec = plan.spec
    if not _is_fla_linear_attention_spec(spec):
        raise RuntimeError("selected FLA anchor does not match K2 linear semantics")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    initial_state = operands.pop("initial_state", None)
    initial_normalizer = operands.pop("initial_normalizer_state", None)
    beta = operands.pop("beta", None)
    log_decay = operands.pop("log_decay", None)
    update_keys = operands.pop("update_keys", None)
    update_values = operands.pop("update_values", None)
    left_transition = operands.pop("left_transition", None)
    right_transition = operands.pop("right_transition", None)
    if operands:
        raise TypeError(
            f"unexpected FLA linear-attention operands: {', '.join(sorted(operands))}"
        )
    if any(
        item is not None
        for item in (
            beta,
            log_decay,
            update_keys,
            update_values,
            left_transition,
            right_transition,
        )
    ):
        raise ValueError("FLA linear attention accepts one additive update per token")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("FLA linear attention query/key use matching [B,T,H,K]")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("FLA linear attention values must use [B,T,H,V] with H=Hq")
    identity = _check_fla_k2_runtime(query, key, value, torch)
    from fla.ops.linear_attn import chunk_linear_attn, fused_recurrent_linear_attn

    if (
        plan.intent is MixerIntent.TRAINING
        and query.dtype in {torch.float16, torch.bfloat16}
        and (query.shape[-1] % 2 or value.shape[-1] % 2)
    ):
        raise ValueError(
            "the pinned FLA linear-attention chunk backward requires even key "
            "and value dimensions for float16/bfloat16 tensors"
        )

    q = _feature(torch, query, spec.feature_map).to(query.dtype)
    k = _feature(torch, key, spec.feature_map).to(key.dtype)
    normalized = spec.normalizer is StateNormalizer.QUERY_KEY
    initial = initial_state
    if normalized:
        batch, _, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        if initial_state is not None:
            _require_shape(
                initial_state,
                (batch, heads, key_dim, value_dim),
                "initial_state",
            )
        if initial_normalizer is not None:
            _require_shape(
                initial_normalizer,
                (batch, heads, key_dim),
                "initial_normalizer_state",
            )
        if initial_state is not None or initial_normalizer is not None:
            if initial_state is None:
                initial_state = torch.zeros(
                    (batch, heads, key_dim, value_dim),
                    dtype=torch.float32,
                    device=query.device,
                )
            if initial_normalizer is None:
                initial_normalizer = torch.zeros(
                    (batch, heads, key_dim),
                    dtype=torch.float32,
                    device=query.device,
                )
            initial = (initial_state, initial_normalizer.unsqueeze(1))
    elif initial_normalizer is not None:
        raise ValueError("initial_normalizer_state requires query_key normalization")

    use_decode = query.shape[1] == 1 and plan.intent is not MixerIntent.TRAINING
    function = fused_recurrent_linear_attn if use_decode else chunk_linear_attn
    output, final = function(
        q,
        k,
        value,
        scale=spec.read_scale or 1.0,
        initial_state=initial,
        output_final_state=True,
        normalize=normalized,
    )
    if normalized:
        if not isinstance(final, tuple) or len(final) != 2:
            raise RuntimeError(
                "FLA normalized linear attention omitted its state tuple"
            )
        final_state, final_normalizer = final
        if final_normalizer.ndim == 4 and final_normalizer.shape[1] == 1:
            final_normalizer = final_normalizer.squeeze(1)
    else:
        final_state, final_normalizer = final, None
    return MixerResult(
        output,
        final_state=final_state,
        final_normalizer_state=final_normalizer,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "decode" if use_decode else "prefill",
            "backward_supported": not use_decode,
        },
    )


def _execute_fla_delta_rule(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    spec = plan.spec
    if not _is_fla_delta_rule_spec(spec):
        raise RuntimeError("selected FLA anchor does not match K2 delta semantics")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    beta = operands.pop("beta", None)
    initial_state = operands.pop("initial_state", None)
    log_decay = operands.pop("log_decay", None)
    update_keys = operands.pop("update_keys", None)
    update_values = operands.pop("update_values", None)
    left_transition = operands.pop("left_transition", None)
    right_transition = operands.pop("right_transition", None)
    if operands:
        raise TypeError(
            f"unexpected FLA delta-rule operands: {', '.join(sorted(operands))}"
        )
    if any(
        item is not None
        for item in (
            log_decay,
            update_keys,
            update_values,
            left_transition,
            right_transition,
        )
    ):
        raise ValueError("FLA delta rule accepts one un-decayed update per token")
    if beta is None:
        raise ValueError("FLA delta rule requires beta")
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("FLA delta-rule query/key use matching [B,T,H,K]")
    if value.ndim != 4 or value.shape[:3] != query.shape[:3]:
        raise ValueError("FLA delta-rule values must use [B,T,H,V] with H=Hq")
    identity = _check_fla_k2_runtime(query, key, value, torch)
    from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule

    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)
    if beta.shape != query.shape[:3]:
        raise ValueError("FLA delta-rule beta must use [B,T,H]")
    use_decode = query.shape[1] == 1 and plan.intent is not MixerIntent.TRAINING
    function = fused_recurrent_delta_rule if use_decode else chunk_delta_rule
    output, final_state = function(
        query,
        key,
        value,
        beta,
        scale=spec.read_scale or 1.0,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "execution_mode": "decode" if use_decode else "prefill",
            "backward_supported": not use_decode,
        },
    )


def _execute_atma_gated_delta_decode(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    """Run ATMA's one-token gated-delta step against a slot-indexed state table."""
    if plan.spec.name != "atma_gated_delta_decode_core":
        raise RuntimeError("ATMA gated-delta adapter is bound to its decode recipe")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    gamma = operands.pop("gamma")
    beta = operands.pop("beta")
    state_table = operands.pop("state_table")
    slots = operands.pop("slots")
    if operands:
        raise TypeError(
            f"unexpected ATMA gated-delta operands: {', '.join(sorted(operands))}"
        )
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("ATMA gated-delta query/key/value use [B,1,H,D] layout")
    batch, sequence, heads, key_dim = query.shape
    if sequence != 1 or key.shape != query.shape:
        raise ValueError("ATMA gated-delta decode needs matching one-token query/key")
    if value.shape[:3] != (batch, 1, heads):
        raise ValueError("ATMA gated-delta value must use [B,1,H,Dv] layout")
    value_dim = value.shape[-1]
    if state_table.ndim != 4 or state_table.shape[1:] != (
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError("ATMA state_table must use [capacity,H,K,V] layout")
    if beta.shape != (batch, 1, heads) or gamma.shape != beta.shape:
        raise ValueError("ATMA gamma and beta must use [B,1,H] layout")
    if (
        slots.shape != (batch,)
        or slots.dtype != torch.int64
        or not slots.is_contiguous()
    ):
        raise ValueError("ATMA slots must be contiguous int64 [B] indices")
    if not all(
        tensor.is_cuda
        for tensor in (query, key, value, gamma, beta, state_table, slots)
    ):
        raise ValueError("ATMA gated-delta decode requires CUDA tensors")
    if not all(
        tensor.dtype == torch.float32
        for tensor in (query, key, value, gamma, beta, state_table)
    ):
        raise TypeError("ATMA gated-delta decode is qualified for float32 tensors")
    if not all(
        tensor.device == query.device
        for tensor in (key, value, gamma, beta, state_table, slots)
    ):
        raise ValueError("ATMA gated-delta inputs must share one CUDA device")

    from urm.adapters.atma_gated_delta import AtmaGatedDeltaDecodeAdapter

    with torch.no_grad():
        output = AtmaGatedDeltaDecodeAdapter()(
            query[:, 0].contiguous(),
            key[:, 0].contiguous(),
            value[:, 0].contiguous(),
            gamma[:, 0].contiguous(),
            beta[:, 0].contiguous(),
            state_table,
            slots,
        )
    return MixerResult(
        output.unsqueeze(1),
        final_state=state_table,
        metadata={
            "anchor": plan.anchor,
            "execution": "trusted_library_anchor",
            "upstream": {
                "repository": "kreasof-ai/atma",
                "revision": "28bb3de8afbe7c0b00115e0fbff36afc9ad49c11",
                "callable": "kernel.gated_delta_triton.gated_delta_decode_step",
            },
            "execution_mode": "decode",
            "state_update": "in_place_slot_table",
            "backward_supported": False,
        },
    )


def _execute_atma_gated_delta_reference(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    gamma = operands.pop("gamma")
    beta = operands.pop("beta")
    state_table = operands.pop("state_table")
    slots = operands.pop("slots")
    if operands:
        raise TypeError(
            "unexpected ATMA gated-delta reference operands: "
            f"{', '.join(sorted(operands))}"
        )
    if query.ndim != 4 or query.shape[1] != 1 or key.shape != query.shape:
        raise ValueError("ATMA gated-delta reference expects matching [B,1,H,K] Q/K")
    batch, _, heads, key_dim = query.shape
    if value.ndim != 4 or value.shape[:3] != (batch, 1, heads):
        raise ValueError("ATMA gated-delta reference value must use [B,1,H,V]")
    value_dim = value.shape[-1]
    if state_table.shape[1:] != (heads, key_dim, value_dim):
        raise ValueError("ATMA reference state_table must use [capacity,H,K,V]")
    if beta.shape != (batch, 1, heads) or gamma.shape != beta.shape:
        raise ValueError("ATMA reference gamma and beta must use [B,1,H]")
    if slots.shape != (batch,) or slots.dtype != torch.int64:
        raise ValueError("ATMA reference slots must use int64 [B]")

    q = torch.nn.functional.normalize(query[:, 0].float(), dim=-1)
    k = torch.nn.functional.normalize(key[:, 0].float(), dim=-1)
    v = value[:, 0].float()
    gamma = gamma[:, 0].float()
    write = beta[:, 0].float()
    selected = state_table.index_select(0, slots)
    decayed = gamma[..., None, None] * selected
    prediction = torch.einsum("bhkv,bhk->bhv", decayed, k)
    update = write[..., None] * (v - prediction)
    updated = decayed + k[..., None] * update[..., None, :]
    output = torch.einsum("bhkv,bhk->bhv", updated, q).unsqueeze(1)
    final_state = state_table.index_copy(0, slots, updated)
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_reference_atma_gated_delta_state_table",
            "execution_mode": "decode",
            "state_update": "functional_slot_table",
            "backward_supported": True,
        },
    )


def _execute_fla_gated_delta(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    if not _is_fla_gated_delta_spec(plan.spec):
        raise RuntimeError("selected FLA anchor does not match the K2 semantic spec")
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    beta = operands.pop("beta")
    gate = operands.pop("log_decay")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected FLA K2 operands: {', '.join(sorted(operands))}")
    if beta.ndim == 4 and beta.shape[-1] == 1:
        beta = beta.squeeze(-1)
    if gate.ndim == 4 and gate.shape[-1] == 1:
        gate = gate.squeeze(-1)
    if beta.ndim != 3 or gate.ndim != 3:
        raise ValueError("FLA gated delta beta/log_decay must use [B,T,Hv]")
    identity = _check_fla_k2_runtime(query, key, value, torch)
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )

    mode = (
        "decode"
        if query.shape[1] == 1 and plan.intent is not MixerIntent.TRAINING
        else "prefill"
    )
    if mode == "prefill":
        output, final_state = chunk_gated_delta_rule(
            query,
            key,
            value,
            gate,
            beta,
            scale=plan.spec.read_scale or 1.0,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=plan.spec.feature_map is FeatureMap.L2_NORMALIZE,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=plan.spec.state_v_first,
        )
    else:
        output, final_state = fused_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=gate,
            beta=beta,
            scale=plan.spec.read_scale or 1.0,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=plan.spec.feature_map is FeatureMap.L2_NORMALIZE,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=plan.spec.state_v_first,
        )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": "fla_gated_delta_rule_adapter",
            "execution": "trusted_library_anchor",
            "upstream": identity,
            "backward_supported": mode != "decode",
            "execution_mode": mode,
        },
    )


def _execute_native_sparse_delta(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    spec = plan.spec
    if spec.family is not MixerKernelFamily.SPARSE_DELTA:
        raise RuntimeError("the current native unified anchor implements K3 only")
    memory = operands.pop("memory")
    read_indices = operands.pop("read_indices")
    read_weights = operands.pop("read_weights")
    write_indices = operands.pop("write_indices")
    write_weights = operands.pop("write_weights")
    values = operands.pop("values")
    beta = operands.pop("beta")
    log_decay = operands.pop("log_decay")
    if operands:
        raise TypeError(f"unexpected native K3 operands: {', '.join(sorted(operands))}")
    if memory.ndim != 3 or values.ndim != 3:
        raise ValueError("native K3 memory and values use [B,S,D] and [B,T,D]")
    batch, slots, value_dim = memory.shape
    sequence = values.shape[1]
    if values.shape != (batch, sequence, value_dim):
        raise ValueError("native K3 values must use [B,T,D]")
    if beta.shape not in ((batch, sequence), (batch, sequence, 1)):
        raise ValueError("native K3 beta must use [B,T] or [B,T,1]")
    if log_decay.shape not in ((batch, sequence), (batch, sequence, 1)):
        raise ValueError("native K3 log_decay must use [B,T] or [B,T,1]")
    try:
        dtype = {
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
        }[memory.dtype]
    except KeyError as error:
        raise ValueError("native K3 supports float32 or bfloat16 state") from error

    from urm.backends.sparse_state_mixer import (
        CertifiedSparseStateRoutes,
        SparseState,
        TritonSparseStateMixerBackend,
    )
    from urm.compiler.execution import NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME
    from urm.compiler.semantic import (
        DType,
        SparseReadTiming,
        SparseStateExecutionMode,
        SparseStateMixerSpec,
        SparseStateOperation,
    )

    read_width = read_indices.shape[-1]
    write_width = write_indices.shape[-1]
    read_timing = (
        SparseReadTiming.BEFORE_UPDATE
        if spec.read_timing is ReadTiming.BEFORE_UPDATE
        else SparseReadTiming.AFTER_UPDATE
    )
    state_mode = (
        SparseStateExecutionMode.TRAINING
        if plan.intent is MixerIntent.TRAINING
        else SparseStateExecutionMode.INFERENCE
    )
    sparse_spec = SparseStateMixerSpec(
        parallel=batch,
        sequence=sequence,
        slots_per_partition=slots,
        value_dim=value_dim,
        writes=write_width,
        reads=read_width,
        dtype=DType(dtype),
        operation=SparseStateOperation.UPDATE,
        read_timing=read_timing,
        mode=state_mode,
    )
    bound_anchor, launch_items = _compile_native_k3_binding(sparse_spec)
    if bound_anchor != NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME:
        raise RuntimeError(
            f"UrmCompiler selected {bound_anchor!r} for the native K3 plan"
        )
    launch_config = dict(launch_items)
    routes = CertifiedSparseStateRoutes.certify(
        sparse_spec,
        read_indices,
        read_weights,
        write_indices=write_indices,
        write_weights=write_weights,
    )
    backend = TritonSparseStateMixerBackend(sparse_spec)
    prepared = backend.prepare(
        routes,
        values=values,
        beta=beta.reshape(batch, sequence, 1),
        log_decay=log_decay.reshape(batch, sequence, 1),
    )
    readings, updated_state = backend.execute(SparseState(memory), prepared)
    return MixerResult(
        readings,
        final_state=updated_state.memory,
        metadata={
            "anchor": NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME,
            "execution": "urm_native_triton",
            "compiler_plan": plan.anchor,
            "urm_compiler_verified": True,
            "runtime_compiler_binding": "cached_semantic_shape",
            "runtime_binding_cache_size": _compile_native_k3_binding.cache_info().currsize,
            "launch_config": launch_config,
        },
    )


def _execute_native_diagonal_ssm(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    spec = plan.spec
    if (
        spec.family is not MixerKernelFamily.RECURRENCE
        or spec.recurrent_layout is not RecurrentLayout.DIAGONAL
    ):
        raise RuntimeError(
            "the native diagonal SSM anchor implements K2 diagonal state only"
        )
    x = operands.pop("x")
    if spec.diagonal_hgrn:
        input_gate = read_gate = None
    else:
        input_gate = operands.pop("input_gate")
        read_gate = operands.pop("read_gate")
    log_decay = operands.pop("log_decay")
    step_size = operands.pop("step_size", None)
    initial_state = operands.pop("initial_state", None)
    skip = operands.pop("skip", 0.0)
    if operands:
        raise TypeError(
            f"unexpected native diagonal SSM operands: {', '.join(sorted(operands))}"
        )
    if spec.diagonal_hgrn:
        if x.ndim != 3 or log_decay.shape != x.shape:
            raise ValueError("HGRN expects x and log_decay with shape [B,T,C]")
        batch_hgrn, sequence_hgrn, channels_hgrn = x.shape
        input_gate = torch.ones(
            (batch_hgrn, sequence_hgrn, 1), device=x.device, dtype=torch.float32
        )
        read_gate = input_gate
        log_decay = log_decay.unsqueeze(-1)
        if initial_state is not None:
            if initial_state.shape == (batch_hgrn, channels_hgrn):
                initial_state = initial_state.unsqueeze(-1)
            elif initial_state.shape != (batch_hgrn, channels_hgrn, 1):
                raise ValueError("HGRN initial_state must have shape [B,C] or [B,C,1]")
        skip = 0.0
    assert input_gate is not None and read_gate is not None
    if spec.step_size_discretization and step_size is None:
        raise ValueError("step-size diagonal SSM requires step_size [B,T,C]")
    if not spec.step_size_discretization and step_size is not None:
        raise ValueError("step_size requires step-size diagonal SSM semantics")
    if x.ndim != 3:
        raise ValueError("native diagonal SSM x must use [B,T,C]")
    batch, sequence, channels = x.shape
    if sequence <= 0 or channels <= 0 or batch <= 0:
        raise ValueError("native diagonal SSM dimensions must be positive")
    state_width = input_gate.shape[-1]
    if state_width <= 0 or read_gate.shape[-1] != state_width:
        raise ValueError("native diagonal SSM gates must share a positive state width")
    for name, gate in (
        ("input_gate", input_gate),
        ("read_gate", read_gate),
        ("log_decay", log_decay),
    ):
        if gate.ndim not in (3, 4) or gate.shape[:2] != (batch, sequence):
            raise ValueError(f"{name} must use [B,T,N] or [B,T,C,N]")
        if gate.shape[-1] != state_width:
            raise ValueError(f"{name} state width must match input_gate")
        if gate.ndim == 4 and gate.shape[2] not in (1, channels):
            raise ValueError(f"{name} channel width must be one or match x")
    if log_decay.shape[:2] != (batch, sequence):
        raise ValueError("log_decay batch/sequence dimensions must match x")
    if initial_state is not None and tuple(initial_state.shape) != (
        batch,
        channels,
        state_width,
    ):
        raise ValueError("initial_state must use [B,C,N]")
    from urm.backends.diagonal_ssm import execute_diagonal_ssm
    from urm.compiler.execution import NATIVE_DIAGONAL_SSM_ANCHOR_NAME

    dtype = str(x.dtype).removeprefix("torch.")
    bound_anchor = _compile_native_diagonal_binding(
        spec, dtype=dtype, intent=plan.intent.value
    )
    if bound_anchor != NATIVE_DIAGONAL_SSM_ANCHOR_NAME:
        raise RuntimeError(
            f"UrmCompiler selected {bound_anchor!r} for the native diagonal SSM plan"
        )
    output, final_state = execute_diagonal_ssm(
        x=x,
        input_gate=input_gate,
        read_gate=read_gate,
        log_decay=log_decay,
        initial_state=initial_state,
        step_size=step_size,
        skip=skip,
        read_before=spec.read_timing is ReadTiming.BEFORE_UPDATE,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": NATIVE_DIAGONAL_SSM_ANCHOR_NAME,
            "execution": "urm_native_triton",
            "compiler_plan": plan.anchor,
            "urm_compiler_verified": True,
            "runtime_compiler_binding": "cached_semantic_shape",
            "runtime_binding_cache_size": _compile_native_diagonal_binding.cache_info().currsize,
        },
    )


@lru_cache(maxsize=128)
def _compile_native_diagonal_binding(
    spec: UnifiedMixerSpec, *, dtype: str, intent: str
) -> str:
    from urm.compiler.planner import CompilationIntent, ScheduleParams, UrmCompiler
    from urm.compiler.execution import NATIVE_DIAGONAL_SSM_ANCHOR_NAME

    compilation = UrmCompiler().compile(
        mixer_semantic_program(spec, dtype=dtype),
        intent=CompilationIntent(intent),
        schedule_params=ScheduleParams(
            anchor_overrides={"mixer": NATIVE_DIAGONAL_SSM_ANCHOR_NAME}
        ),
    )
    selected = tuple(step.anchor for step in compilation.plan.steps if step.anchor)
    if selected != (NATIVE_DIAGONAL_SSM_ANCHOR_NAME,):
        raise RuntimeError(
            "UrmCompiler produced an invalid native diagonal SSM plan: "
            f"anchors={selected}"
        )
    return selected[0]


@lru_cache(maxsize=128)
def _compile_native_k3_binding(
    sparse_spec: Any,
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Compile each concrete K3 semantic shape once, then reuse its verified launch."""
    from urm.compiler.planner import CompilationIntent, UrmCompiler
    from urm.compiler.semantic import sparse_state_mixer_program
    from urm.sparse_state_mixer import sparse_state_launch_schedule

    semantic_program = sparse_state_mixer_program(
        name="unified_k3_sparse_delta",
        parallel=sparse_spec.parallel,
        sequence=sparse_spec.sequence,
        slots_per_partition=sparse_spec.slots_per_partition,
        value_dim=sparse_spec.value_dim,
        writes=sparse_spec.writes,
        reads=sparse_spec.reads,
        dtype=sparse_spec.dtype,
        operation=sparse_spec.operation,
        read_timing=sparse_spec.read_timing,
        mode=sparse_spec.mode,
    )
    compilation_intent = (
        CompilationIntent.TRAINING
        if sparse_spec.mode.value == "training"
        else CompilationIntent.INFERENCE
    )
    compilation = UrmCompiler().compile(semantic_program, intent=compilation_intent)
    dispatch_steps = [
        step for step in compilation.plan.steps if step.kind == "anchor_dispatch"
    ]
    if len(dispatch_steps) != 1:
        raise RuntimeError(
            "UrmCompiler produced an invalid K3 native plan: "
            f"steps={len(dispatch_steps)}"
        )
    dispatch = dispatch_steps[0]
    launch_config = dispatch.launch_config
    if launch_config is None:
        raise RuntimeError("UrmCompiler omitted the K3 native launch configuration")
    expected_launch = sparse_state_launch_schedule(sparse_spec)
    if launch_config != expected_launch:
        raise RuntimeError(
            "verified K3 launch config differs from the native runtime: "
            f"serialized={launch_config}, runtime={expected_launch}"
        )
    return dispatch.anchor, tuple(sorted(launch_config.items()))


__all__ = [
    "CompiledMixerPlan",
    "DecayGranularity",
    "FeatureMap",
    "MixerBackend",
    "MixerIntent",
    "MixerKernelFamily",
    "MIXER_RECIPE_NAMES",
    "MixerRecipe",
    "MixerResult",
    "ReadTiming",
    "RecurrentLayout",
    "StateNormalizer",
    "StateTransition",
    "StateUpdateRule",
    "UnifiedMixerSpec",
    "compile_mixer",
    "compile_frontend_mixer",
    "mixer_semantic_program",
    "delta_rule_spec",
    "diagonal_ssm_spec",
    "linear_attention_spec",
    "named_mixer_recipe",
    "softmax_attention_spec",
    "sparse_delta_spec",
]
