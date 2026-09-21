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
    # The additive/no-decay configuration is under-specified only when no other
    # axis distinguishes the equation: a feature map, a normalizer, or a
    # polynomial basis makes an additive recurrence distinguishable from the
    # collision group (GRU tanh/gate, FFT convolution, second-order correction),
    # which all present as identity-feature / no-normalizer / no-polynomial.
    distinguishes = (
        spec.feature_map is not FeatureMap.IDENTITY
        or spec.normalizer is not StateNormalizer.NONE
        or spec.polynomial_basis is not PolynomialBasis.NONE
    )
    if (
        spec.update_rule is StateUpdateRule.ADDITIVE
        and spec.decay is DecayGranularity.NONE
        and not distinguishes
    ):
        raise UnderspecifiedComposition(
            "additive/no-decay is under-specified in the IR (collision group)"
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
        raise UnderspecifiedComposition("only pointwise decay transitions are canonical")
    if spec.state_effect is not StateEffect.FUNCTIONAL:
        raise UnderspecifiedComposition("only functional state is canonical")
    exotic = [
        name for name in (
            "mamba2_ssm", "log_linear_attention",
            "gated_delta_product", "generalized_delta_iplr",
            "generalized_delta_dplr", "rwkv4_memory", "rwkv6_memory",
            "momentum_delta", "gated_oja", "comba_rule", "preconditioned_gated_delta",
            "preconditioned_kda", "slot_attention", "step_size_discretization",
            "diagonal_hgrn", "path_attention", "deltaformer_attention",
        )
        if getattr(spec, name, False)
    ]
    if exotic:
        raise UnderspecifiedComposition(f"exotic composition flags: {exotic}")
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


def _execute_k2(spec: UnifiedMixerSpec, **operands):
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
    if operands:
        raise TypeError(f"unexpected K2 operands: {sorted(operands)}")
    if spec.gdn2_ssm and (erase_gate is None or write_gate is None):
        raise ValueError("gdn2 dual-gate delta requires erase_gate and write_gate")
    if spec.update_rule is StateUpdateRule.DELTA and beta is None and not spec.gdn2_ssm:
        raise ValueError("delta update requires beta")
    if spec.decay is not DecayGranularity.NONE and log_decay is None:
        raise ValueError("decay requires log_decay")

    batch, sequence, q_heads, key_dim = query.shape
    value_heads, value_dim = value.shape[2], value.shape[3]
    # Read-scale convention: the plain matrix-state reference defaults to 1.0,
    # but the dual-gate (gdn2) and kda references default to key_dim**-0.5.
    if spec.read_scale is not None:
        scale = spec.read_scale
    elif spec.gdn2_ssm or spec.kda_delta:
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
                if is_delta and not spec.gdn2_ssm
                else np.ones(sequence)
            )
            erase_col = write_col = None
            if spec.gdn2_ssm:
                # erase_gate [B,T,Hv,K], write_gate [B,T,Hv,V]
                erase_col = np.asarray(erase_gate[b, :, vh], dtype=np.float64)
                write_col = np.asarray(write_gate[b, :, vh], dtype=np.float64)
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
