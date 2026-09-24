"""External (upstream/comparator) executors for URM mixer plans.

These executors run the pinned upstream/library implementations behind the
external anchors. They are registered by anchor name into the core registry
(:func:`urm.compiler.pipeline.register_external_executor`) so that the core
compiler and runtime hold no comparator or upstream imports. Importing this
module has the side effect of registering every external executor it provides;
benchmarks import it before compiling a plan with ``backend=MixerBackend.LIBRARY``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from urm.compiler import pipeline as _core
from urm.compiler.pipeline import (
    CompiledMixerPlan,
    MixerResult,
    register_external_executor,
)
from urm.ir.graph import (
    DecayGranularity,
    FeatureMap,
    K1Operation,
    MixerBackend,
    MixerIntent,
    MixerKernelFamily,
    PolynomialBasis,
    RecurrenceOperator,
    RecurrentLayout,
    StateEffect,
    StateNormalizer,
    StateUpdateRule,
    UnifiedMixerSpec,
)

# Shared private helpers owned by the core compiler. The executors use them for
# operand assembly and pinned-revision checks; importing them here keeps the
# dependency direction consumer -> core (the core never imports this module).
_comba_operands = _core._comba_operands
_deltaformer_operands = _core._deltaformer_operands
_feature = _core._feature
_gated_oja_operands = _core._gated_oja_operands
_gdn2_shapes = _core._gdn2_shapes
_is_atma_gated_delta_decode_spec = _core._is_atma_gated_delta_decode_spec
_is_fla_delta_rule_spec = _core._is_fla_delta_rule_spec
_is_fla_gated_delta_spec = _core._is_fla_gated_delta_spec
_is_fla_gla_spec = _core._is_fla_gla_spec
_is_fla_linear_attention_spec = _core._is_fla_linear_attention_spec
_is_fla_simple_gla_spec = _core._is_fla_simple_gla_spec
_log_linear_shapes = _core._log_linear_shapes
_mamba2_shapes = _core._mamba2_shapes
_momentum_delta_operands = _core._momentum_delta_operands
_path_attention_operands = _core._path_attention_operands
_pgdn_operands = _core._pgdn_operands
_require_shape = _core._require_shape
_slot_attention_operands = _core._slot_attention_operands
_validate_atma_decode_dimensions = _core._validate_atma_decode_dimensions

def _execute_atma_gated_delta_decode(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    """Run ATMA's one-token gated-delta step against a slot-indexed state table."""
    if not _is_atma_gated_delta_decode_spec(plan.spec):
        raise RuntimeError("ATMA adapter requires its exact K2 semantic contract")
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
    _validate_atma_decode_dimensions(batch, key_dim, value_dim)
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

    from benchmarks.comparators.atma_gated_delta import AtmaGatedDeltaDecodeAdapter

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


def _execute_atma_polar(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    from kernel.polar_triton import polar_attention, polar_attention_sparse

    operation = plan.spec.k1_operation
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    n_keys = operands.pop("n_keys")
    v_null = operands.pop("v_null")
    null_base = operands.pop("null_base")
    null_slope_raw = operands.pop("null_slope_raw")
    len_gain_raw = operands.pop("len_gain_raw")
    mag_beta_raw = operands.pop("mag_beta_raw")
    if operation is K1Operation.POLAR:
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


def _execute_bdh_attention(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    from benchmarks.comparators.bdh import bdh_attention_adapter

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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


def _execute_fla_deltaformer(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    query, key, value, beta = _deltaformer_operands(torch, **operands)
    if not query.is_cuda or query.dtype not in {torch.float16, torch.bfloat16}:
        raise RuntimeError("the pinned DeltaFormer adapter requires CUDA FP16 or BF16")
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from urm.compiler.select.anchors import FLA_HGRN_ANCHOR_NAME

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from benchmarks.comparators.fla_gated_delta import fla_version

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
    from urm.backends.triton.k1.selected import selected_softmax_read

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
        from urm.backends.triton.k2.second_order import hla_second_order_triton

        output = hla_second_order_triton(query, key, value)
        return MixerResult(
            output,
            metadata={
                "anchor": plan.anchor,
                "execution": "hla_masked_second_order_triton_forward_reverse_scan",
                "backward_supported": True,
            },
        )

    delta_s = key.unsqueeze(-1) * key.unsqueeze(-2)
    delta_c = query.unsqueeze(-1) * value.unsqueeze(-2)
    state_s = delta_s.cumsum(dim=1)
    state_c = delta_c.cumsum(dim=1)
    previous_c = state_c - delta_c

    key_previous_c = torch.matmul(key.unsqueeze(-2), previous_c).squeeze(-2)
    delta_g = key.unsqueeze(-1) * key_previous_c.unsqueeze(-2)
    masked_correction = delta_g.cumsum(dim=1)
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


def _execute_kata_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from benchmarks.comparators.kata import kata_attention_adapter

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


def _execute_longformer_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from benchmarks.comparators.longformer import longformer_attention_adapter

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
    from urm.compiler.select.anchors import MAMBA2_SSD_ANCHOR_NAME

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
    batch, sequence, channels = x.shape
    state_width = log_decay.shape[-1]
    u = x.transpose(1, 2).contiguous()
    delta = step_size.transpose(1, 2).contiguous()
    a_static = (
        log_decay.transpose(1, 2)
        .unsqueeze(1)
        .expand(batch, channels, state_width, sequence)
        .mean(dim=(0, 3))
        .contiguous()
    )
    b_mat = input_gate.transpose(1, 2).contiguous()
    c_mat = read_gate.transpose(1, 2).contiguous()
    if isinstance(skip, torch.Tensor):
        if skip.ndim == 0:
            skip = skip.expand(channels)
    elif skip is None or float(skip) == 0.0:
        skip = None
    else:
        skip = torch.full((channels,), float(skip), device=x.device, dtype=x.dtype)
    from urm.compiler.select.anchors import MAMBA_SELECTIVE_SCAN_ANCHOR_NAME

    output, final_state = _pinned_mamba_selective_scan()(
        u,
        delta,
        a_static,
        b_mat,
        c_mat,
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


def _execute_tda_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from benchmarks.comparators.tda import tda_attention_adapter

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


def _execute_tucker_attention_adapter(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    from benchmarks.comparators.tucker import tucker_attention_adapter

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


def _execute_upstream_sparse_delta(plan: CompiledMixerPlan, torch: Any, **operands: Any):
    """Pinned Facebook sparse-delta-memory comparator for the K3 update.

    Routes the mixer's K3 operands (memory [B,S,D], integer routes [B,T,R],
    softmax weights, values [B,T,D], beta/log_decay [B,T]) through the pinned
    ``GatedSparseMemoryWriteRead`` autograd kernel. The SDM kernel is a
    post-update read (decay -> retrieve -> delta-scatter -> read), matching the
    recipe's ``read_timing=after_update`` semantics. The kernel is verified
    against the URM K3 reference on the frozen runtime; the SDM adapter's
    conservative torch/triton version pin is bypassed only via the sanctioned
    ``URM_SDM_ALLOW_UNPINNED_RUNTIME`` override (see the SDM adapter).
    """
    spec = plan.spec
    if spec.family is not MixerKernelFamily.SPARSE_DELTA:
        raise RuntimeError("the upstream K3 comparator implements SPARSE_DELTA only")
    memory = operands.pop("memory")
    read_indices = operands.pop("read_indices")
    read_weights = operands.pop("read_weights")
    write_indices = operands.pop("write_indices")
    write_weights = operands.pop("write_weights")
    values = operands.pop("values")
    beta = operands.pop("beta")
    log_decay = operands.pop("log_decay")
    if operands:
        raise TypeError(f"unexpected upstream K3 operands: {', '.join(sorted(operands))}")
    batch, slots, value_dim = memory.shape
    sequence = values.shape[1]
    from benchmarks.comparators.sdm.upstream import probe_sdm_support

    support = probe_sdm_support()
    if not support.supported:
        raise RuntimeError(
            f"pinned SDM comparator unavailable [{support.code}]: {support.reason}"
        )
    from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead

    flat_memory = memory.reshape(batch * slots, value_dim).contiguous().clone()
    offsets = (
        torch.arange(batch, device=memory.device, dtype=torch.int64).view(batch, 1, 1)
        * slots
    )
    write_global = (write_indices.to(torch.int64) + offsets).contiguous()
    read_global = (read_indices.to(torch.int64) + offsets).contiguous()
    readings, _unused = GatedSparseMemoryWriteRead.apply(
        flat_memory,
        write_global,
        write_weights.contiguous(),
        values.contiguous(),
        beta.reshape(batch, sequence, 1).contiguous(),
        log_decay.reshape(batch, sequence, 1).contiguous(),
        read_global,
        read_weights.contiguous(),
        min(64, sequence),
        True,
        slots,
        batch,
        False,
        "none",
        None,
    )
    readings = readings.view(batch, sequence, value_dim)
    return MixerResult(
        readings,
        final_state=flat_memory.view(batch, slots, value_dim),
        metadata={
            "anchor": plan.anchor,
            "execution": "pinned_facebook_sparse_delta_memory_gated_write_read",
            "upstream": "facebookresearch/sparse-delta-memory@183e7df",
            "backward_supported": True,
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


def _check_fla_k2_runtime(query: Any, key: Any, value: Any, torch: Any):
    if not (query.is_cuda and key.is_cuda and value.is_cuda):
        raise RuntimeError("FLA K2 library anchors require CUDA tensors")
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError("FLA K2 query/key/value must use the same dtype")
    if query.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError(
            "FLA K2 library anchors support float32, float16, and bfloat16"
        )
    from benchmarks.comparators.fla_gated_delta import fla_version

    identity = fla_version()
    if identity.get("comparison_compatible") is not True:
        raise RuntimeError(
            "FLA K2 library anchors require flash-linear-attention==0.5.2 or "
            "the source revision recorded in the architecture coverage register; "
            f"identity={identity}"
        )
    return identity



def register_all() -> None:
    register_external_executor('fla_parallel_deltaformer_adapter', _execute_fla_deltaformer)
    register_external_executor('fla_parallel_forgetting_attention_adapter', _execute_fla_forgetting_attention)
    register_external_executor('fla_parallel_parallax_adapter', _execute_fla_parallax_attention)
    register_external_executor('fla_parallel_wall_attention_adapter', _execute_fla_wall_attention)
    register_external_executor('fla_parallel_path_attention_adapter', _execute_fla_path_attention)
    register_external_executor('fla_parallel_moba_adapter', _execute_fla_moba_attention)
    register_external_executor('fla_fused_attnres_adapter', _execute_fla_attnres)
    register_external_executor('atma_polar_triton_adapter', _execute_atma_polar)
    register_external_executor('atma_polar_sparse_triton_adapter', _execute_atma_polar)
    register_external_executor('tda_triton_attention_adapter', _execute_tda_attention_adapter)
    register_external_executor('tucker_triton_attention_adapter', _execute_tucker_attention_adapter)
    register_external_executor('longformer_sliding_chunks_adapter', _execute_longformer_attention_adapter)
    register_external_executor('kata_parallel_triton_adapter', _execute_kata_attention_adapter)
    register_external_executor('fwpkm_selected_softmax_triton_adapter', _execute_fwpkm_selected_read)
    register_external_executor('atma_gated_delta_decode_adapter', _execute_atma_gated_delta_decode)
    register_external_executor('h3_ssm_fft_convolution_adapter', _execute_h3_ssm_fft)
    register_external_executor('hyena_fft_convolution_adapter', _execute_hyena_fftconv)
    register_external_executor('hla_second_order_triton_adapter', _execute_hla_second_order)
    register_external_executor('bdh_attention_adapter', _execute_bdh_attention)
    register_external_executor('fla_chunk_log_linear_attention_adapter', _execute_fla_log_linear_attention)
    register_external_executor('xma_rnn_triton_adapter', _execute_xma_nonlinear_rnn)
    register_external_executor('xma_gru_triton_adapter', _execute_xma_nonlinear_rnn)
    register_external_executor('xma_m2rnn_triton_adapter', _execute_xma_nonlinear_rnn)
    register_external_executor('fla_chunk_titans_linear_adapter', _execute_fla_titans_linear)
    register_external_executor('fla_chunk_ttt_linear_adapter', _execute_fla_ttt_linear)
    register_external_executor('mamba3_siso_combined_adapter', _execute_mamba3_siso_adapter)
    register_external_executor('fla_chunk_mesa_net_adapter', _execute_fla_mesa_net)
    register_external_executor('fla_fused_recurrent_rwkv4_adapter', _execute_fla_rwkv4)
    register_external_executor('fla_fused_recurrent_rwkv6_adapter', _execute_fla_rwkv6)
    register_external_executor('fla_chunk_momentum_delta_rule_adapter', _execute_fla_momentum_delta)
    register_external_executor('fla_chunk_gated_oja_adapter', _execute_fla_gated_oja)
    register_external_executor('fla_chunk_comba_adapter', _execute_fla_comba)
    register_external_executor('fla_chunk_precond_gated_delta_adapter', _execute_fla_pgdn)
    register_external_executor('fla_chunk_precond_kda_adapter', _execute_fla_pkda)
    register_external_executor('fla_chunk_abc_adapter', _execute_fla_slot_attention)
    register_external_executor('fla_chunk_gsa_adapter', _execute_fla_slot_attention)
    register_external_executor('fla_fused_recurrent_iplr_adapter', _execute_fla_generalized_delta)
    register_external_executor('fla_chunk_rwkv7_adapter', _execute_fla_generalized_delta)
    register_external_executor('fla_chunk_dplr_adapter', _execute_fla_generalized_delta)
    register_external_executor('fla_chunk_gated_delta_product_adapter', _execute_fla_gated_delta_product)
    register_external_executor('fla_chunk_kda_adapter', _execute_fla_kda)
    register_external_executor('mamba2_ssd_adapter', _execute_mamba2_ssm_library)
    register_external_executor('fla_chunk_gdn2_adapter', _execute_fla_gdn2)
    register_external_executor('fla_fused_recurrent_hgrn_adapter', _execute_fla_hgrn)
    register_external_executor('mamba_selective_scan_adapter', _execute_mamba_selective_scan)
    register_external_executor('fla_chunk_gla_adapter', _execute_fla_k2)
    register_external_executor('fla_chunk_delta_rule_adapter', _execute_fla_delta_rule)
    register_external_executor('fla_chunk_linear_attention_adapter', _execute_fla_linear_attention)
    register_external_executor('fla_fused_recurrent_gla_decode_adapter', _execute_fla_gated_additive)
    register_external_executor('fla_fused_recurrent_simple_gla_decode_adapter', _execute_fla_gated_additive)
    # Operator-keyed K2 equations shared across reference and library backends.
    register_external_executor('urm.op.k2.fft_convolution', _execute_hyena_fftconv)
    register_external_executor('urm.op.k2.two_stage_fft_convolution', _execute_h3_ssm_fft)
    register_external_executor('urm.op.k2.second_order_cumsum', _execute_hla_second_order)
    register_external_executor('fla_chunk_simple_gla_adapter', _execute_fla_gated_additive)
    register_external_executor('fla_fused_chunk_based_adapter', _execute_fla_polynomial_attention)
    register_external_executor('fla_parallel_rebased_adapter', _execute_fla_polynomial_attention)
    register_external_executor('fla_gated_delta_rule_adapter', _execute_fla_gated_delta)
    register_external_executor('facebook_sparse_delta_memory_183e7df_precomputed_route_adapter', _execute_upstream_sparse_delta)
    # Install the pinned-SDM capability probe so the compiler's SDM selectors can
    # verify the upstream checkout without importing this comparator package.
    try:
        from urm.compiler.select.anchors import set_sdm_support_probe

        from benchmarks.comparators.sdm.upstream import probe_sdm_support

        set_sdm_support_probe(probe_sdm_support)
    except Exception:  # noqa: BLE001 - the SDM checkout may be absent
        pass
