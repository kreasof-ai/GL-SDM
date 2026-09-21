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


def _require_plain_k1(spec: UnifiedMixerSpec) -> None:
    if spec.family is not MixerKernelFamily.SOFTMAX:
        raise UnderspecifiedComposition("not a K1 spec")
    if spec.k1_operation is not K1Operation.SOFTMAX:
        raise UnderspecifiedComposition(
            f"K1 canonical path covers the normalized softmax reduction only; "
            f"{spec.k1_operation.value} is a distinct equation"
        )


def _require_canonical_k2(spec: UnifiedMixerSpec) -> None:
    """Decline K2 specs the canonical matrix-state path does not distinguish."""
    if spec.family is not MixerKernelFamily.RECURRENCE:
        raise UnderspecifiedComposition("not a K2 spec")
    if spec.recurrent_layout is not RecurrentLayout.MATRIX:
        raise UnderspecifiedComposition(
            "the canonical matrix-state path covers the matrix layout only"
        )
    if spec.update_rule not in (StateUpdateRule.ADDITIVE, StateUpdateRule.DELTA):
        raise UnderspecifiedComposition(f"unsupported update rule {spec.update_rule}")
    # The additive/no-decay configuration is under-specified: the IR does not
    # distinguish a plain additive recurrence from the exotic additive equations
    # (GRU tanh/gate, FFT convolution, second-order correction) that share its
    # fields. Decline it rather than silently compute the wrong equation.
    if spec.update_rule is StateUpdateRule.ADDITIVE and spec.decay is DecayGranularity.NONE:
        raise UnderspecifiedComposition(
            "additive/no-decay is under-specified in the IR (collision group)"
        )
    if spec.normalizer is not StateNormalizer.NONE:
        raise UnderspecifiedComposition("normalized state is not canonical")
    if spec.feature_map not in (FeatureMap.IDENTITY, FeatureMap.L2_NORMALIZE):
        raise UnderspecifiedComposition(f"feature map {spec.feature_map} not canonical")
    if spec.polynomial_basis is not PolynomialBasis.NONE:
        raise UnderspecifiedComposition("polynomial basis is not canonical")
    if spec.transition is not StateTransition.POINTWISE:
        raise UnderspecifiedComposition("only pointwise decay transitions are canonical")
    if spec.state_effect is not StateEffect.FUNCTIONAL:
        raise UnderspecifiedComposition("only functional state is canonical")
    exotic = [
        name for name in (
            "static_head_decay", "mamba2_ssm", "log_linear_attention", "gdn2_ssm",
            "kda_delta", "gated_delta_product", "generalized_delta_iplr",
            "generalized_delta_dplr", "rwkv4_memory", "rwkv6_memory",
            "momentum_delta", "gated_oja", "comba_rule", "preconditioned_gated_delta",
            "preconditioned_kda", "slot_attention", "step_size_discretization",
            "diagonal_hgrn", "path_attention", "deltaformer_attention",
        )
        if getattr(spec, name, False)
    ]
    if exotic:
        raise UnderspecifiedComposition(f"exotic composition flags: {exotic}")


def _feature_map(x, kind):
    if kind is FeatureMap.IDENTITY:
        return x
    if kind is FeatureMap.L2_NORMALIZE:
        return x * np.reciprocal(np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-6))
    raise UnderspecifiedComposition(f"feature map {kind} not canonical")


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


def _execute_k1(spec: UnifiedMixerSpec, **operands):
    _require_plain_k1(spec)
    query = np.asarray(operands.pop("query"), dtype=np.float64)
    key = np.asarray(operands.pop("key"), dtype=np.float64)
    value = np.asarray(operands.pop("value"), dtype=np.float64)
    mask = operands.pop("attention_mask", None)
    bias = operands.pop("score_bias", None)
    if operands:
        raise TypeError(f"unexpected K1 operands: {sorted(operands)}")
    if query.ndim != 4:
        raise ValueError("K1 query/key/value use BTHD rank-4 layout")
    # BTHD -> per-batch [H, T, d] stacks; execute each batch through the core.
    # BTHD [B, T, H, D] -> per-batch [H, T, D] stacks; run each through the core.
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
    return {"output": np.stack(outputs, axis=0)}


def _execute_k2(spec: UnifiedMixerSpec, **operands):
    _require_canonical_k2(spec)
    query = np.asarray(operands.pop("query"), dtype=np.float64)
    key = np.asarray(operands.pop("key"), dtype=np.float64)
    value = np.asarray(operands.pop("value"), dtype=np.float64)
    beta = operands.pop("beta", None)
    log_decay = operands.pop("log_decay", None)
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected K2 operands: {sorted(operands)}")
    if spec.update_rule is StateUpdateRule.DELTA and beta is None:
        raise ValueError("delta update requires beta")
    if spec.decay is not DecayGranularity.NONE and log_decay is None:
        raise ValueError("decay requires log_decay")

    batch, sequence, q_heads, key_dim = query.shape
    value_heads, value_dim = value.shape[2], value.shape[3]
    scale = spec.read_scale if spec.read_scale is not None else 1.0
    qf = _feature_map(query, spec.feature_map)
    kf = _feature_map(key, spec.feature_map)

    outputs = np.empty((batch, sequence, value_heads, value_dim))
    final_states = np.empty((batch, value_heads, key_dim, value_dim))
    group = value_heads // q_heads
    for b in range(batch):
        for vh in range(value_heads):
            qh = vh // group
            m0 = (
                np.zeros((key_dim, value_dim))
                if initial_state is None
                else np.asarray(initial_state[b, vh], dtype=np.float64)
            )
            # Decay schedule for this (batch, head): head scalar or key-channel.
            if spec.decay is DecayGranularity.NONE:
                g = np.zeros(sequence)
            elif spec.decay is DecayGranularity.HEAD:
                g = np.asarray(log_decay[b, :, vh], dtype=np.float64).reshape(sequence)
            elif spec.decay is DecayGranularity.KEY_CHANNEL:
                g = np.asarray(log_decay[b, :, vh], dtype=np.float64)
            else:
                raise UnderspecifiedComposition(f"decay {spec.decay} not canonical")
            is_delta = spec.update_rule is StateUpdateRule.DELTA
            beta_col = (
                np.asarray(beta[b, :, vh], dtype=np.float64).reshape(sequence)
                if is_delta
                else np.ones(sequence)
            )
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
            )
            outputs[b, :, vh] = out
            final_states[b, vh] = m
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
