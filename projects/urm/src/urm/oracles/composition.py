"""Spec-driven canonical NumPy execution: the declarative-composition path.

This module is the representation-coverage boundary. A
:class:`~urm.ir.mixer.UnifiedMixerSpec` is a declarative composition of reusable
operations - normalization, routing, state, decay, read/write timing - and this
module executes that composition through the single canonical NumPy path for its
core (K1/K2/K3). Every architecture whose spec lowers to the same canonical form
executes identically here, so recipe renaming cannot change execution, and two
specs that differ in an equation-affecting field produce distinguishable results.

This is an oracle for representation and composition checks in float64, never a
performance backend. It deliberately declines equations the canonical cores do
not represent (for example the additive/no-decay collision group, whose spec
does not distinguish a plain additive recurrence from a GRU's nonlinearity) so
an under-specified composition never silently computes the wrong equation.
"""

from __future__ import annotations

import numpy as np

from urm.ir.mixer import (
    DecayGranularity,
    FeatureMap,
    K1Operation,
    MixerKernelFamily,
    PolynomialBasis,
    RecurrenceOperator,
    RecurrentLayout,
    StateEffect,
    StateNormalizer,
    StateTransition,
    StateUpdateRule,
    UnifiedMixerSpec,
)

from . import matrix_state, softmax_attention, sparse_slot


class UnderspecifiedComposition(ValueError):
    """Raised when a spec does not distinguish its equation within the core."""


# K1 operations the canonical path covers, each a reparameterization or
# composition of the normalized routed reduction:
# - SOFTMAX: the plain reduction.
# - LOCAL_WINDOW: a sliding-window mask composed with the reduction.
# - DIFFERENTIAL: two reductions combined by a learned per-head scalar.
# - PROJECTED: a low-rank query projection (query @ B_pre) feeding the reduction.
_COVERED_K1_OPERATIONS = frozenset(
    {
        K1Operation.SOFTMAX,
        K1Operation.LOCAL_WINDOW,
        K1Operation.DIFFERENTIAL,
        K1Operation.PROJECTED,
        K1Operation.POSITIONAL,
        K1Operation.POSITIVE_FEATURE,
    }
)


def _require_plain_k1(spec: UnifiedMixerSpec) -> None:
    if spec.family is not MixerKernelFamily.SOFTMAX:
        raise UnderspecifiedComposition("not a K1 spec")
    if spec.k1_operation not in _COVERED_K1_OPERATIONS:
        raise UnderspecifiedComposition(
            f"K1 canonical path does not yet cover {spec.k1_operation.value}"
        )


def _require_canonical_k2(spec: UnifiedMixerSpec) -> None:
    """Decline K2 specs the canonical matrix-state path does not distinguish."""
    if spec.family is not MixerKernelFamily.RECURRENCE:
        raise UnderspecifiedComposition("not a K2 spec")
    if spec.recurrent_layout is RecurrentLayout.DIAGONAL:
        # The diagonal (SSM) layout is a separate canonical core; only the HGRN
        # and step-size (Mamba-1) variants are currently distinguished.
        if not (spec.diagonal_hgrn or spec.step_size_discretization):
            raise UnderspecifiedComposition(
                "diagonal layout covers HGRN / step-size SSM only"
            )
        return
    if spec.recurrent_layout is not RecurrentLayout.MATRIX:
        raise UnderspecifiedComposition(
            "the canonical matrix-state path covers the matrix layout only"
        )
    if spec.update_rule not in (StateUpdateRule.ADDITIVE, StateUpdateRule.DELTA):
        raise UnderspecifiedComposition(f"unsupported update rule {spec.update_rule}")
    # The recurrence_operator field carries the equation explicitly. The plain
    # linear matrix-state recurrence (PLAIN) is the canonical path; the distinct
    # nonlinear/convolution/solve operators route to their own canonical
    # executors below. A PLAIN additive/no-decay spec with no distinguishing axis
    # is the genuinely plain linear recurrence (no hidden nonlinearity).
    if spec.recurrence_operator is not RecurrenceOperator.PLAIN:
        raise UnderspecifiedComposition(
            f"non-plain recurrence operator {spec.recurrence_operator.value} routes "
            "to its own canonical executor"
        )
    if spec.normalizer not in (StateNormalizer.NONE, StateNormalizer.QUERY_KEY):
        raise UnderspecifiedComposition(f"normalizer {spec.normalizer} not canonical")
    if spec.feature_map not in (
        FeatureMap.IDENTITY, FeatureMap.L2_NORMALIZE, FeatureMap.ELU_PLUS_ONE,
        FeatureMap.RELU,
    ):
        raise UnderspecifiedComposition(f"feature map {spec.feature_map} not canonical")
    if spec.polynomial_basis not in (
        PolynomialBasis.NONE, PolynomialBasis.BASED_TAYLOR2, PolynomialBasis.REBASED_SQUARE,
    ):
        raise UnderspecifiedComposition(f"polynomial basis {spec.polynomial_basis} not canonical")
    if spec.transition is not StateTransition.POINTWISE:
        # Factored (low-rank) transitions are canonical for the generalized-delta
        # recipes (IPLR/DPLR): left = I + beta⊗alpha or diag(decay) + beta⊗alpha.
        if not (spec.transition is StateTransition.FACTORED_MATRIX and (
            spec.generalized_delta_iplr or spec.generalized_delta_dplr
        )):
            raise UnderspecifiedComposition(
                "only pointwise decay and generalized-delta factored transitions are canonical"
            )
    if spec.state_effect is not StateEffect.FUNCTIONAL:
        raise UnderspecifiedComposition("only functional state is canonical")
    exotic = [
        name for name in (
            "mamba2_ssm", "log_linear_attention",
            "rwkv4_memory", "rwkv6_memory",
            "momentum_delta", "gated_oja", "preconditioned_gated_delta",
            "preconditioned_kda", "slot_attention", "step_size_discretization",
            "diagonal_hgrn", "path_attention", "deltaformer_attention",
        )
        if getattr(spec, name, False)
    ]
    if exotic:
        raise UnderspecifiedComposition(f"exotic composition flags: {exotic}")
    # comba is the dual-key delta rule (separate prediction/write keys).
    if spec.comba_rule and spec.update_rule is not StateUpdateRule.DELTA:
        raise UnderspecifiedComposition("comba requires the delta update rule")
    # kda is the plain delta rule with key-channel decay and a key_dim**-0.5 read
    # scale; it requires the delta update rule and key-channel decay.
    if spec.kda_delta and not (
        spec.update_rule is StateUpdateRule.DELTA
        and spec.decay is DecayGranularity.KEY_CHANNEL
    ):
        raise UnderspecifiedComposition("kda requires delta + key-channel decay")
    # The dual-gate delta (gdn2) is the delta rule with independent erase/write
    # gates; it requires key-channel decay and the delta update rule.
    if spec.gdn2_ssm and not (
        spec.update_rule is StateUpdateRule.DELTA
        and spec.decay is DecayGranularity.KEY_CHANNEL
    ):
        raise UnderspecifiedComposition("gdn2 requires delta + key-channel decay")
    # Static head decay is a reparameterization of head decay with a
    # time-constant schedule; it requires head-granularity decay.
    if spec.static_head_decay and spec.decay is not DecayGranularity.HEAD:
        raise UnderspecifiedComposition("static_head_decay requires head decay")


def _feature_map(x, kind):
    if kind is FeatureMap.IDENTITY:
        return x
    if kind is FeatureMap.L2_NORMALIZE:
        return x * np.reciprocal(np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-6))
    if kind is FeatureMap.RELU:
        return np.maximum(x, 0.0)
    if kind is FeatureMap.ELU_PLUS_ONE:
        return np.where(x > 0, x, np.expm1(x)) + 1.0
    raise UnderspecifiedComposition(f"feature map {kind} not canonical")


def _polynomial_features(x, basis, scale, *, is_query):
    """Quadratic feature expansion for the based/rebased polynomial bases.

    The query is scaled by ``scale``; the key is left unscaled (matching the
    reference construction).
    """
    z = x * scale if is_query else x
    quad = np.einsum("...i,...j->...ij", z, z)
    if basis is PolynomialBasis.BASED_TAYLOR2:
        return np.concatenate(
            [np.ones_like(z[..., :1]), z,
             quad.reshape(*z.shape[:-1], -1) / np.sqrt(2.0)],
            axis=-1,
        )
    if basis is PolynomialBasis.REBASED_SQUARE:
        return quad.reshape(*z.shape[:-1], -1)
    raise UnderspecifiedComposition(f"polynomial basis {basis} not canonical")


def execute_canonical(spec: UnifiedMixerSpec, **operands):
    """Execute a spec's declarative composition through the canonical NumPy core.

    Operands use the compiler's BTHD rank-4 convention for K1/K2 and the K3
    route/index convention. Returns a dict with ``output`` and, for stateful
    cores, ``final_state``. Raises :class:`UnderspecifiedComposition` when the
    spec does not map onto a canonical core equation.
    """
    if spec.family is MixerKernelFamily.SOFTMAX:
        return _execute_k1(spec, **operands)
    if spec.family is MixerKernelFamily.RECURRENCE:
        return _execute_k2(spec, **operands)
    if spec.family is MixerKernelFamily.SPARSE_DELTA:
        return _execute_k3(spec, **operands)
    raise UnderspecifiedComposition(f"unknown family {spec.family}")


def _k1_reduction(query, key, value, spec, mask=None, bias=None):
    """One canonical K1 normalized reduction over a BTHD rank-4 batch."""
    outputs = []
    for b in range(query.shape[0]):
        out = softmax_attention.attention(
            query[b].transpose(1, 0, 2),
            key[b].transpose(1, 0, 2),
            value[b].transpose(1, 0, 2),
            scale=spec.attention_scale,
            causal=spec.causal,
            score_bias=None if bias is None else bias[b],
            attention_mask=None if mask is None else mask[b],
        )
        outputs.append(out.transpose(1, 0, 2))
    return np.stack(outputs, axis=0)


def _execute_k1(spec: UnifiedMixerSpec, **operands):
    _require_plain_k1(spec)
    if spec.k1_operation is K1Operation.DIFFERENTIAL:
        # Two normalized reductions combined by a learned per-head scalar:
        # softmax(q_a k_a) v - lambda * softmax(q_b k_b) v.
        query_a = np.asarray(operands.pop("query_a"), dtype=np.float64)
        query_b = np.asarray(operands.pop("query_b"), dtype=np.float64)
        key_a = np.asarray(operands.pop("key_a"), dtype=np.float64)
        key_b = np.asarray(operands.pop("key_b"), dtype=np.float64)
        value = np.asarray(operands.pop("value"), dtype=np.float64)
        lambda_weight = np.asarray(operands.pop("lambda_weight"), dtype=np.float64)
        if operands:
            raise TypeError(f"unexpected differential K1 operands: {sorted(operands)}")
        heads = query_a.shape[2]
        combine = (
            lambda_weight.reshape(1, 1, heads, 1)
            if lambda_weight.ndim == 1
            else lambda_weight
        )
        out_a = _k1_reduction(query_a, key_a, value, spec)
        out_b = _k1_reduction(query_b, key_b, value, spec)
        return {"output": out_a - combine * out_b}

    if spec.k1_operation is K1Operation.POSITIONAL:
        # Parallax: softmax probabilities P, secondary scores S = r.k, output
        # (P V)(1 + sum(P*S)) - (P*S) V. A composition on the canonical reduction.
        query = np.asarray(operands.pop("query"), dtype=np.float64)
        secondary = np.asarray(operands.pop("r"), dtype=np.float64)
        key = np.asarray(operands.pop("key"), dtype=np.float64)
        value = np.asarray(operands.pop("value"), dtype=np.float64)
        if operands:
            raise TypeError(f"unexpected positional K1 operands: {sorted(operands)}")
        batch = query.shape[0]
        outputs = []
        for b in range(batch):
            qb = query[b].transpose(1, 0, 2)
            kb = key[b].transpose(1, 0, 2)
            vb = value[b].transpose(1, 0, 2)
            rb = secondary[b].transpose(1, 0, 2)
            probs = softmax_attention.attention_probs(
                qb, kb, vb, scale=spec.attention_scale, causal=spec.causal
            )
            # Secondary scores with the same GQA head sharing.
            group = qb.shape[0] // kb.shape[0]
            k_b = np.repeat(kb, group, axis=0) if group != 1 else kb
            v_b = np.repeat(vb, group, axis=0) if group != 1 else vb
            secondary_scores = np.einsum("htk,hsk->hts", rb, k_b)
            correction = probs * secondary_scores
            ordinary = np.einsum("hts,hsv->htv", probs, v_b)
            correction_mean = correction.sum(axis=-1, keepdims=True)
            correction_out = np.einsum("hts,hsv->htv", correction, v_b)
            out = ordinary * (1.0 + correction_mean) - correction_out
            outputs.append(out.transpose(1, 0, 2))
        return {"output": np.stack(outputs, axis=0)}

    if spec.k1_operation is K1Operation.POSITIVE_FEATURE:
        # KATA: grouped SPD positive-feature scores with L1 normalization (not
        # softmax). scores = sum_groups (q.k / sqrt(group_dim))^2, causal-masked,
        # normalized by their row sum.
        query = np.asarray(operands.pop("query"), dtype=np.float64)
        key = np.asarray(operands.pop("key"), dtype=np.float64)
        value = np.asarray(operands.pop("value"), dtype=np.float64)
        num_groups = operands.pop("num_groups")
        if operands:
            raise TypeError(f"unexpected positive-feature K1 operands: {sorted(operands)}")
        batch, sequence, heads, dim = query.shape
        group_dim = dim // num_groups
        qg = query.reshape(batch, sequence, heads, num_groups, group_dim)
        kg = key.reshape(batch, sequence, heads, num_groups, group_dim)
        group_scores = np.einsum("bthme,bshme->bhtsm", qg, kg) * (group_dim ** -0.5)
        scores = np.square(group_scores).sum(axis=-1)
        causal = np.tril(np.ones((sequence, sequence), dtype=bool))
        scores = np.where(causal[None, None], scores, 0.0)
        denom = scores.sum(axis=-1, keepdims=True).clip(min=1e-12)
        probs = scores / denom
        output = np.einsum("bhts,bshv->bthv", probs, value)
        return {"output": output}

    query = np.asarray(operands.pop("query"), dtype=np.float64)
    key = np.asarray(operands.pop("key"), dtype=np.float64)
    value = np.asarray(operands.pop("value"), dtype=np.float64)
    mask = operands.pop("attention_mask", None)
    bias = operands.pop("score_bias", None)
    if spec.k1_operation is K1Operation.PROJECTED:
        # Reparameterize: expand the low-rank query by B_pre, then plain softmax.
        # query [B,T,R], B_pre [H,R,K] -> expanded query [B,T,H,K].
        b_pre = np.asarray(operands.pop("B_pre"), dtype=np.float64)
        query = np.einsum("btr,hrk->bthk", query, b_pre)
        if key.ndim == 3:
            key = key[:, :, None, :]
        if value.ndim == 3:
            value = value[:, :, None, :]
    if spec.k1_operation is K1Operation.LOCAL_WINDOW:
        window = operands.pop("attention_window", None)
        if not isinstance(window, int) or window <= 0:
            raise ValueError("local-window attention requires a positive attention_window")
        # Compose the sliding-window mask: |query_index - key_index| <= window.
        t = query.shape[1]
        positions = np.arange(t)
        mask = (np.abs(positions[:, None] - positions[None, :]) <= window)[
            None, None
        ]
    if operands:
        raise TypeError(f"unexpected K1 operands: {sorted(operands)}")
    if query.ndim != 4:
        raise ValueError("K1 query/key/value use BTHD rank-4 layout")
    return {"output": _k1_reduction(query, key, value, spec, mask=mask, bias=bias)}


def _diagonal_recurrent(x, input_gate, read_gate, log_decay, step_size,
                        initial_state, skip, read_before_update):
    """Canonical diagonal (SSM) recurrence for one batch, in float64.

    State ``S`` has shape ``[C, N]`` (channels x state width). Per token:
    ``transition = exp(log_decay (* step_size))``; ``update = x ⊗ input_gate``
    (``* step_size`` for the step-size-discretized form); ``state = transition *
    state + update``; ``out = sum_N(state * read_gate) + x * skip``.
    """
    batch, sequence, channels = x.shape
    state_width = input_gate.shape[-1]
    state = (
        np.zeros((batch, channels, state_width))
        if initial_state is None
        else initial_state.astype(np.float64).copy()
    )
    skip_t = np.asarray(skip, dtype=np.float64)
    outputs = []
    for t in range(sequence):
        decay = log_decay[:, t]
        if decay.ndim == 2:
            decay = decay[:, None, :]
        step = None if step_size is None else step_size[:, t]
        if step is not None:
            if step.ndim == 1:
                step = step[:, None]
            decay = decay * step[..., None]
        transition = np.exp(decay)
        in_gate = input_gate[:, t]
        out_gate = read_gate[:, t]
        if in_gate.ndim == 2:
            in_gate = in_gate[:, None, :]
        if out_gate.ndim == 2:
            out_gate = out_gate[:, None, :]
        if read_before_update:
            outputs.append((state * out_gate).sum(-1) + x[:, t] * skip_t)
        update = x[:, t][..., None] * in_gate
        if step is not None:
            update = update * step[..., None]
        state = transition * state + update
        if not read_before_update:
            outputs.append((state * out_gate).sum(-1) + x[:, t] * skip_t)
    return np.stack(outputs, axis=1), state


def _execute_k2_operator(spec: UnifiedMixerSpec, **operands):
    """Execute a distinct (non-plain) K2 recurrence operator via its canonical path."""
    from . import nonlinear_recurrence as nl

    op = spec.recurrence_operator
    if op is RecurrenceOperator.TANH_RNN:
        out, state = nl.tanh_rnn(
            operands.pop("query"), operands.pop("weight"),
            operands.pop("initial_state"),
        )
    elif op is RecurrenceOperator.GATED_RNN:
        out, state = nl.gated_rnn(
            operands.pop("query"), operands.pop("weight"),
            operands.pop("forget_input"), operands.pop("forget_weight"),
            operands.pop("reset_input"), operands.pop("reset_weight"),
            operands.pop("initial_state"),
        )
    elif op is RecurrenceOperator.MULTIPLICATIVE_RNN:
        out, state = nl.multiplicative_rnn(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("weight"), operands.pop("forget_input"),
            operands.pop("initial_state"),
        )
    elif op is RecurrenceOperator.FFT_CONVOLUTION:
        out, state = nl.hyena_fft_convolution(
            operands.pop("query"), operands.pop("kernel"), operands.pop("direct"),
        )
    elif op is RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION:
        out, state = nl.two_stage_fft_convolution(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("ssm_kernel"), operands.pop("ssm_k_kernel"),
            operands.pop("ssm_k_direct"), operands.pop("skip"),
        )
    elif op is RecurrenceOperator.SECOND_ORDER_CUMSUM:
        out, state = nl.second_order_cumsum(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
        )
    elif op is RecurrenceOperator.REGULARIZED_SOLVE:
        out, state = nl.regularized_solve(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("log_decay"), operands.pop("beta"), operands.pop("lamb"),
            h_kk_init=operands.pop("h_kk_init", None),
            h_kv_init=operands.pop("h_kv_init", None),
        )
    elif op is RecurrenceOperator.LAYERNORM_INNER_STATE:
        out, state = nl.layernorm_inner_state(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("w"), operands.pop("b"), operands.pop("eta"),
            initial_state=operands.pop("initial_state", None),
            initial_state_bias=operands.pop("initial_state_bias", None),
            chunk_size=int(operands.pop("chunk_size", 16)),
            eps=float(operands.pop("eps", 1e-6)),
        )
    elif op is RecurrenceOperator.MOMENTUM_INNER_STATE:
        out, state = nl.momentum_inner_state(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("w"), operands.pop("b"), operands.pop("theta"),
            operands.pop("alpha"), operands.pop("eta"),
            initial_state=operands.pop("initial_state", None),
            chunk_size=int(operands.pop("chunk_size", 16)),
            eps=float(operands.pop("eps", 1e-6)),
        )
    elif op is RecurrenceOperator.TRAPEZOIDAL_SSM:
        out, state = nl.trapezoidal_ssm(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("adt"), operands.pop("dt"), operands.pop("trap"),
            operands.pop("query_bias"), operands.pop("key_bias"), operands.pop("angles"),
            d_skip=operands.pop("d_skip", None), gate=operands.pop("gate", None),
            initial_states=operands.pop("initial_states", None),
        )
    elif op is RecurrenceOperator.MOMENTUM_DELTA_STATE:
        out, state = nl.momentum_delta(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("p"), operands.pop("log_alpha"), operands.pop("log_mu"),
            operands.pop("beta"), operands.pop("eta"),
            initial_state=operands.pop("initial_state", None),
            initial_momentum=operands.pop("initial_normalizer_state", None),
            scale=spec.read_scale,
        )
    elif op is RecurrenceOperator.GATED_OJA_VALUE_CHANNEL:
        out, state = nl.gated_oja(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("gv"), operands.pop("beta"),
            initial_state=operands.pop("initial_state", None),
            scale=spec.read_scale,
        )
    elif op is RecurrenceOperator.RWKV4_SCALAR_STATE:
        out, state = nl.rwkv4_scalar_state(
            operands.pop("w"), operands.pop("u"), operands.pop("k"),
            operands.pop("v"), operands.pop("state"),
        )
    elif op is RecurrenceOperator.RWKV6_BONUS_CORRECTED:
        out, state = nl.rwkv6_bonus_corrected(
            operands.pop("query"), operands.pop("key"), operands.pop("value"),
            operands.pop("log_decay"), operands.pop("bonus"),
            initial_state=operands.pop("initial_state", None),
        )
    elif op is RecurrenceOperator.MAMBA2_STRUCTURED_SSM:
        out, state = nl.mamba2_structured_ssm(
            operands.pop("x"), operands.pop("dt"), operands.pop("A"),
            operands.pop("B"), operands.pop("C"),
            initial_states=operands.pop("initial_states", None),
        )
    elif op is RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE:
        query = operands.pop("query")
        key = operands.pop("key")
        value = operands.pop("value")
        if spec.name == "abc_core":
            # ABC derives slot_weights and log_decay from slot_logits via a
            # cumulative log-sum-exp over time.
            slot_logits = np.asarray(operands.pop("slot_logits"), dtype=np.float64)
            cumulative = np.apply_along_axis(
                lambda x: np.logaddexp.accumulate(x), 1, slot_logits
            )
            log_decay = (
                np.concatenate((cumulative[:, :1], cumulative[:, :-1]), axis=1)
                - cumulative
            )
            slot_weights = np.exp(slot_logits - cumulative)
        else:
            slot_weights = operands.pop("slot_weights")
            log_decay = operands.pop("log_decay")
        # group_size is derived from the head counts, not passed as an operand.
        group_size = np.asarray(query).shape[2] // np.asarray(key).shape[2]
        out, state = nl.slot_attention_two_stage(
            query, key, value, slot_weights, log_decay,
            initial_key_state=operands.pop("initial_key_state", None),
            initial_value_state=operands.pop("initial_value_state", None),
            group_size=group_size,
        )
    else:
        raise UnderspecifiedComposition(
            f"no canonical executor yet for recurrence operator {op.value}"
        )
    if operands:
        raise TypeError(f"unexpected {op.value} operands: {sorted(operands)}")
    result = {"output": out}
    if state is not None:
        result["final_state"] = state
    return result


def _execute_k2(spec: UnifiedMixerSpec, **operands):
    # Route the distinct recurrence operators (the IR-distinguished equations) to
    # their own canonical executors before the plain matrix-state path.
    if spec.recurrence_operator is not RecurrenceOperator.PLAIN:
        return _execute_k2_operator(spec, **operands)
    _require_canonical_k2(spec)
    if spec.recurrent_layout is RecurrentLayout.DIAGONAL:
        x = np.asarray(operands.pop("x"), dtype=np.float64)
        log_decay = np.asarray(operands.pop("log_decay"), dtype=np.float64)
        step_size = operands.pop("step_size", None)
        initial_state = operands.pop("initial_state", None)
        skip = operands.pop("skip", 0.0)
        if spec.diagonal_hgrn:
            batch, sequence, channels = x.shape
            input_gate = np.ones((batch, sequence, 1))
            read_gate = input_gate
            log_decay = log_decay[..., None]
            skip = 0.0
        else:
            input_gate = np.asarray(operands.pop("input_gate"), dtype=np.float64)
            read_gate = np.asarray(operands.pop("read_gate"), dtype=np.float64)
        if operands:
            raise TypeError(f"unexpected diagonal K2 operands: {sorted(operands)}")
        out, state = _diagonal_recurrent(
            x, input_gate, read_gate, log_decay,
            None if step_size is None else np.asarray(step_size, dtype=np.float64),
            initial_state, skip,
            read_before_update=spec.read_timing.name == "BEFORE_UPDATE",
        )
        return {"output": out, "final_state": state}
    query = np.asarray(operands.pop("query"), dtype=np.float64)
    key = np.asarray(operands.pop("key"), dtype=np.float64)
    value = np.asarray(operands.pop("value"), dtype=np.float64)
    beta = operands.pop("beta", None)
    log_decay = operands.pop("log_decay", None)
    initial_state = operands.pop("initial_state", None)
    erase_gate = operands.pop("erase_gate", None)
    write_gate = operands.pop("write_gate", None)
    # comba names its prediction key "p" and its log decay "g".
    prediction_key = operands.pop("prediction_key", operands.pop("p", None))
    if spec.comba_rule and log_decay is None:
        log_decay = operands.pop("g", None)
    # generalized-delta factored transitions.
    transition_alpha = operands.pop("transition_alpha", None)
    transition_beta = operands.pop("transition_beta", None)
    # gated-delta-product multi-rank updates.
    update_keys = operands.pop("update_keys", None)
    update_values = operands.pop("update_values", None)
    if operands:
        raise TypeError(f"unexpected K2 operands: {sorted(operands)}")
    if spec.gdn2_ssm and (erase_gate is None or write_gate is None):
        raise ValueError("gdn2 dual-gate delta requires erase_gate and write_gate")
    if spec.comba_rule and prediction_key is None:
        raise ValueError("comba dual-key delta requires prediction_key (p)")
    if spec.update_rule is StateUpdateRule.DELTA and beta is None and not spec.gdn2_ssm:
        raise ValueError("delta update requires beta")
    if spec.decay is not DecayGranularity.NONE and log_decay is None:
        raise ValueError("decay requires log_decay")

    batch, sequence, q_heads, key_dim = query.shape
    value_heads, value_dim = value.shape[2], value.shape[3]
    # Read-scale convention: the plain matrix-state reference defaults to 1.0,
    # but the dual-gate (gdn2), kda, comba, and generalized-delta references
    # default to key_dim**-0.5.
    if spec.read_scale is not None:
        scale = spec.read_scale
    elif (
        spec.gdn2_ssm or spec.kda_delta or spec.comba_rule
        or spec.generalized_delta_iplr or spec.generalized_delta_dplr
        or spec.gated_delta_product
    ):
        scale = key_dim ** -0.5
    else:
        scale = 1.0
    # Feature construction: polynomial basis (quadratic expansion) or a feature
    # map applied to identity features. The polynomial basis changes the feature
    # dimension, so the state width follows the expanded key dimension.
    if spec.polynomial_basis is not PolynomialBasis.NONE:
        poly_scale = spec.read_scale or key_dim ** -0.5
        qf = _polynomial_features(query, spec.polynomial_basis, poly_scale, is_query=True)
        kf = _polynomial_features(key, spec.polynomial_basis, poly_scale, is_query=False)
        scale = 1.0  # the scale is folded into the polynomial features
    else:
        qf = _feature_map(query, spec.feature_map)
        kf = _feature_map(key, spec.feature_map)
    feat_dim = kf.shape[-1]
    normalizer = spec.normalizer is StateNormalizer.QUERY_KEY

    outputs = np.empty((batch, sequence, value_heads, value_dim))
    final_states = np.empty((batch, value_heads, feat_dim, value_dim))
    group = value_heads // q_heads
    for b in range(batch):
        for vh in range(value_heads):
            qh = vh // group
            m0 = (
                np.zeros((feat_dim, value_dim))
                if initial_state is None
                else np.asarray(initial_state[b, vh], dtype=np.float64)
            )
            # Decay schedule for this (batch, head): head scalar or key-channel.
            # Static head decay is a time-constant head schedule ([H] or [1]).
            if spec.decay is DecayGranularity.NONE:
                g = np.zeros(sequence)
            elif spec.static_head_decay:
                static = np.asarray(log_decay, dtype=np.float64).reshape(-1)
                g = np.full(sequence, static[vh] if static.size > 1 else static[0])
            elif spec.decay is DecayGranularity.HEAD:
                g = np.asarray(log_decay[b, :, vh], dtype=np.float64).reshape(sequence)
            elif spec.decay is DecayGranularity.KEY_CHANNEL:
                g = np.asarray(log_decay[b, :, vh], dtype=np.float64)
            else:
                raise UnderspecifiedComposition(f"decay {spec.decay} not canonical")
            is_delta = spec.update_rule is StateUpdateRule.DELTA
            beta_col = (
                np.asarray(beta[b, :, vh], dtype=np.float64).reshape(sequence)
                if is_delta and not spec.gdn2_ssm and not spec.gated_delta_product
                else np.ones(sequence)
            )
            erase_col = write_col = None
            if spec.gdn2_ssm:
                # erase_gate [B,T,Hv,K], write_gate [B,T,Hv,V]
                erase_col = np.asarray(erase_gate[b, :, vh], dtype=np.float64)
                write_col = np.asarray(write_gate[b, :, vh], dtype=np.float64)
            retr_col = None
            if spec.comba_rule:
                # prediction_key [B,T,H,K] is the retrieval key.
                retr_col = np.asarray(prediction_key[b, :, qh], dtype=np.float64)
            left_col = None
            if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
                # left_t = I + beta_t⊗alpha_t (IPLR) or diag(exp(log_decay)) +
                # beta_t⊗alpha_t (DPLR), applied as Z = left_t @ M.
                ta = np.asarray(transition_alpha[b, :, vh], dtype=np.float64)  # [T,K]
                tb = np.asarray(transition_beta[b, :, vh], dtype=np.float64)   # [T,K]
                eye = np.eye(feat_dim)
                left_col = np.empty((sequence, feat_dim, feat_dim))
                # DPLR's diagonal decay comes from log_decay [B,T,H,K].
                dplr_decay = (
                    np.asarray(log_decay[b, :, vh], dtype=np.float64)
                    if spec.generalized_delta_dplr
                    else None
                )
                for ti in range(sequence):
                    rank_one = np.outer(tb[ti], ta[ti])
                    if spec.generalized_delta_dplr:
                        left_col[ti] = np.diag(np.exp(dplr_decay[ti])) + rank_one
                    else:
                        left_col[ti] = eye + rank_one
            uk_col = uv_col = rb_col = None
            if spec.gated_delta_product:
                # update_keys [B,T,R,H,K], update_values [B,T,R,H,V], beta [B,T,R,H].
                uk_col = np.asarray(update_keys[b, :, :, qh], dtype=np.float64)
                uv_col = np.asarray(update_values[b, :, :, vh], dtype=np.float64)
                rb_col = np.asarray(beta[b, :, :, vh], dtype=np.float64)
            out, m = matrix_state.recurrent(
                m0,
                kf[b, :, qh],
                qf[b, :, qh],
                value[b, :, vh],
                beta_col,
                g,
                scale=scale,
                read_before_update=spec.read_timing.name == "BEFORE_UPDATE",
                is_delta=is_delta,
                normalizer=normalizer,
                epsilon=spec.epsilon,
                erase_gate=erase_col,
                write_gate=write_col,
                retrieval_keys=retr_col,
                left_transitions=left_col,
                update_keys=uk_col,
                update_values=uv_col,
                rank_beta=rb_col,
            )
            outputs[b, :, vh] = out
            final_states[b, vh] = m[0] if normalizer else m
    if spec.state_v_first:
        final_states = final_states.transpose(0, 1, 3, 2)
    return {"output": outputs, "final_state": final_states}


def _execute_k3(spec: UnifiedMixerSpec, **operands):
    if spec.family is not MixerKernelFamily.SPARSE_DELTA:
        raise UnderspecifiedComposition("not a K3 spec")
    memory = np.asarray(operands.pop("memory"), dtype=np.float64)
    read_indices = np.asarray(operands.pop("read_indices"))
    read_weights = np.asarray(operands.pop("read_weights"), dtype=np.float64)
    write_indices = np.asarray(operands.pop("write_indices"))
    write_weights = np.asarray(operands.pop("write_weights"), dtype=np.float64)
    values = np.asarray(operands.pop("values"), dtype=np.float64)
    beta = np.asarray(operands.pop("beta"), dtype=np.float64)
    log_decay = np.asarray(operands.pop("log_decay"), dtype=np.float64)
    if operands:
        raise TypeError(f"unexpected K3 operands: {sorted(operands)}")

    batch, slots, value_dim = memory.shape
    sequence = read_indices.shape[1]
    outputs = np.empty((batch, sequence, value_dim))
    final = np.empty_like(memory)
    for b in range(batch):
        # Densify the sparse routes into [T, S] weight vectors + selection mask.
        w = np.zeros((sequence, slots))
        q = np.zeros((sequence, slots))
        selected = np.zeros((sequence, slots), dtype=bool)
        for t in range(sequence):
            w[t, write_indices[b, t]] = write_weights[b, t]
            q[t, read_indices[b, t]] = read_weights[b, t]
            selected[t, write_indices[b, t]] = True
        out, m = sparse_slot.recurrent(
            memory[b], w, q, values[b],
            beta[b].reshape(sequence), log_decay[b].reshape(sequence), selected,
        )
        outputs[b] = out
        final[b] = m
    return {"output": outputs, "final_state": final}
