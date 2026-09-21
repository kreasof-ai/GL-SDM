"""Backend independent mixer operation semantics shared by the compiler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class MixerKernelFamily(StrEnum):
    """The three physical mixer families URM composes."""

    SOFTMAX = "K1_softmax_reduction"
    RECURRENCE = "K2_state_recurrence"
    SPARSE_DELTA = "K3_sparse_delta_state"


class K1Operation(StrEnum):
    """K1 equations expressed without frontend architecture identities."""

    SOFTMAX = "normalized_softmax_attention"
    FORGETTING = "forgetting_gated_softmax_attention"
    POLAR = "polar_attention"
    POLAR_SPARSE = "sparse_polar_attention"
    DIFFERENTIAL = "difference_of_softmax_attention"
    THRESHOLDED = "thresholded_softmax_attention"
    PROJECTED = "projected_softmax_attention"
    LOCAL_WINDOW = "local_window_softmax_attention"
    POSITIVE_FEATURE = "positive_feature_attention"
    SELECTED_READ = "selected_memory_read"
    BLOCK_ROUTED = "block_routed_softmax_attention"
    POSITIONAL = "position_indexed_attention"
    GATED = "gated_attention"
    DEPTH = "depth_weighted_attention"
    PATH_TRANSFORM = "path_transform_attention"
    DELTA_TRANSFORM = "delta_transform_attention"


class RecurrentLayout(StrEnum):
    MATRIX = "matrix_state"
    DIAGONAL = "diagonal_ssm_state"


class RecurrenceOperator(StrEnum):
    """The explicit equation a K2 recurrence computes, without architecture identity.

    The additive/no-decay semantic signature is shared by equations that differ
    fundamentally (a GRU's gated tanh interpolation, an FFT long convolution, a
    second-order cumsum correction, a per-token linear solve, and similar). This
    field carries the equation explicitly so different equations have
    distinguishable semantic representations (acceptance-contract section 4) and
    the compiler never dispatches on a recipe name.

    PLAIN is the default linear matrix/diagonal recurrence the canonical cores
    cover. The others name the distinct equation structures.
    """

    PLAIN = "plain_linear_recurrence"
    TANH_RNN = "tanh_rnn"
    GATED_RNN = "gated_rnn_gru"
    MULTIPLICATIVE_RNN = "multiplicative_rnn_second_order"
    LAYERNORM_INNER_STATE = "layernorm_inner_loss_state"
    MOMENTUM_INNER_STATE = "momentum_inner_loss_state"
    MOMENTUM_DELTA_STATE = "momentum_delta_two_matrix_state"
    GATED_OJA_VALUE_CHANNEL = "gated_oja_value_channel"
    SLOT_ATTENTION_TWO_STAGE = "slot_attention_two_stage"
    RWKV4_SCALAR_STATE = "rwkv4_scalar_state"
    RWKV6_BONUS_CORRECTED = "rwkv6_bonus_corrected"
    TRAPEZOIDAL_SSM = "trapezoidal_ssm_rotary"
    REGULARIZED_SOLVE = "regularized_solve_state"
    SECOND_ORDER_CUMSUM = "second_order_cumsum"
    FFT_CONVOLUTION = "fft_long_convolution"
    TWO_STAGE_FFT_CONVOLUTION = "two_stage_fft_convolution"
    EXTERNAL_OPAQUE = "external_opaque"


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


class StateEffect(StrEnum):
    FUNCTIONAL = "functional"
    IN_PLACE_SLOT_TABLE = "in_place_slot_table"


class MixerIntent(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"
    FORWARD_ONLY_ANALYSIS = "forward_only_analysis"


class MixerBackend(StrEnum):
    REFERENCE = "reference"
    LIBRARY = "library"
    NATIVE = "native"


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
    requires_attention_mask: bool = False
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
    state_effect: StateEffect = StateEffect.FUNCTIONAL
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
    k1_operation: K1Operation = K1Operation.SOFTMAX
    recurrence_operator: RecurrenceOperator = RecurrenceOperator.PLAIN

    def __post_init__(self) -> None:
        for name, enum_type in (
            ("family", MixerKernelFamily),
            ("k1_operation", K1Operation),
            ("recurrent_layout", RecurrentLayout),
            ("recurrence_operator", RecurrenceOperator),
            ("update_rule", StateUpdateRule),
            ("normalizer", StateNormalizer),
            ("feature_map", FeatureMap),
            ("polynomial_basis", PolynomialBasis),
            ("decay", DecayGranularity),
            ("transition", StateTransition),
            ("read_timing", ReadTiming),
            ("state_effect", StateEffect),
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
        if not isinstance(self.requires_attention_mask, bool):
            raise ValueError("requires_attention_mask must be a bool")
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
                or self.state_effect is not StateEffect.FUNCTIONAL
            ):
                raise ValueError("K1 accepts softmax attention semantics only")
            if (
                self.path_attention
                and self.k1_operation
                not in {K1Operation.SOFTMAX, K1Operation.PATH_TRANSFORM}
            ) or (
                self.deltaformer_attention
                and self.k1_operation
                not in {K1Operation.SOFTMAX, K1Operation.DELTA_TRANSFORM}
            ):
                raise ValueError("K1 operation conflicts with its semantic flags")
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
            if self.k1_operation is not K1Operation.SOFTMAX:
                raise ValueError("k1_operation is valid only for K1 semantics")
            if self.requires_attention_mask:
                raise ValueError("attention masks belong to K1")
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
                self.k1_operation is not K1Operation.SOFTMAX
                or self.recurrent_layout is not RecurrentLayout.MATRIX
                or self.update_rule is not StateUpdateRule.ADDITIVE
                or self.normalizer is not StateNormalizer.NONE
                or self.feature_map is not FeatureMap.IDENTITY
                or self.decay is not DecayGranularity.NONE
                or self.transition is not StateTransition.POINTWISE
                or self.accepts_score_bias
                or self.attention_scale is not None
                or self.read_scale is not None
                or self.requires_attention_mask
                or self.state_effect is not StateEffect.FUNCTIONAL
            ):
                raise ValueError("K3 accepts ordered sparse delta semantics only")

    def to_dict(self) -> dict[str, object]:
        return {
            key: value.value if isinstance(value, StrEnum) else value
            for key, value in asdict(self).items()
        }

    def semantic_signature(self) -> tuple[tuple[str, object], ...]:
        """Canonical equation identity, excluding frontend recipe metadata."""
        return tuple(
            (key, value)
            for key, value in sorted(self.to_dict().items())
            if key != "name"
        )

    def is_normalized_softmax_attention(self) -> bool:
        """Whether this contract is exactly the reusable K1 softmax equation."""
        if self.family is not MixerKernelFamily.SOFTMAX:
            return False
        expected = UnifiedMixerSpec(
            name="normalized_softmax_attention",
            family=MixerKernelFamily.SOFTMAX,
            causal=self.causal,
            attention_scale=self.attention_scale,
            accepts_score_bias=self.accepts_score_bias,
            requires_attention_mask=self.requires_attention_mask,
        )
        return self.semantic_signature() == expected.semantic_signature()
