"""Transitional mixer-family compilation path (legacy, slated for deletion).

This module hosts the pre-graph family-dispatch API: ``compile_mixer`` maps a
``UnifiedMixerSpec``/``MixerRecipe`` to one of three physical kernel families
(K1 softmax, K2 recurrence, K3 sparse state), resolves the anchor by family-
and name-shaped dispatch, and executes through family-specific reference and
native paths (``CompiledMixerPlan.execute``). It also owns the external-executor
registry used by the comparator suite to bind pinned upstream anchors.

This whole path is *transitional*: every recipe is migrating to declarative
JSON graph documents (schema_version 2) compiled through
:func:`urm.compiler.pipeline.compile_graph` and executed by
:class:`urm.runtime.bind.BoundGraphPlan`. As each family completes its graph
migration, the corresponding dispatch and executor code here is deleted. New
code must not add to this module; see ``docs/planning/refactor-list.md``.

Nothing here is imported by :mod:`urm.compiler.pipeline` (the planner); the
dependency direction is strictly ``mixer`` -> ``pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any

from urm.frontend.recipes import (
    MixerRecipe,
    named_mixer_recipe as _named_mixer_recipe,
    softmax_attention_spec as _softmax_attention_spec,
)
from urm.ir.graph import (
    DecayGranularity,
    FeatureMap,
    K1Operation,
    MixerBackend,
    MixerIntent,
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
from urm.ir.program import (
    SparseStateExecutionMode,
    UnifiedMixerAccess,
)


# ======================================================================
# Mixer family compilation entry points (merged from unified_mixer).
# ======================================================================

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
class CompiledMixerPlan:
    """Serializable selection of one of the three executable mixer families."""

    spec: UnifiedMixerSpec
    intent: MixerIntent
    anchor: str
    backend: MixerBackend = MixerBackend.REFERENCE
    recipe: MixerRecipe | None = None
    compiler_result: Any | None = None
    compile_dtype: str = "float32"
    backend_selection: Any | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": 1,
            "semantic_spec": self.spec.to_dict(),
            "intent": self.intent.value,
            "physical_kernel_family": self.spec.family.value,
            "anchor": self.anchor,
            "backend": self.backend.value,
            "compile_dtype": self.compile_dtype,
            "backend_selection": (
                {
                    "requested_backend": self.backend_selection.requested_backend,
                    "selected_backend": self.backend_selection.selected_backend,
                    "request": {
                        "operation": self.backend_selection.request.operation,
                        "semantic_contract": self.backend_selection.request.semantic_contract,
                        "device": self.backend_selection.request.device,
                        "dtype": self.backend_selection.request.dtype,
                        "layout": self.backend_selection.request.layout,
                        "mode": self.backend_selection.request.mode,
                    },
                    "attempted_backends": list(self.backend_selection.attempted_backends),
                    "fallback_used": self.backend_selection.fallback_used,
                }
                if self.backend_selection is not None
                else None
            ),
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
        if self.spec.requires_attention_mask and operands.get("attention_mask") is None:
            raise ValueError(
                "this K1 operation requires a precomputed attention_mask route"
            )
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
        # Anchor-driven execution: when the compiled plan selected an external or
        # upstream anchor, resolve it through the registered executor instead of
        # redispatching on the recipe name or operation. The registry is populated
        # by the consumer (benchmarks/comparators) that provisions the external
        # capability; the core holds no comparator or upstream imports.
        external_executor = _EXTERNAL_EXECUTORS.get(self.anchor)
        if external_executor is not None:
            return external_executor(self, torch, **operands)
        # FFT-convolution and second-order-cumsum K2 equations are distinct typed
        # operations whose reference and library paths share one executor; they are
        # keyed by the recurrence operator, not a per-backend anchor. The consumer
        # registers them under the operator-keyed names below.
        if self.spec.family is MixerKernelFamily.RECURRENCE:
            operator_keyed = {
                RecurrenceOperator.FFT_CONVOLUTION: "urm.op.k2.fft_convolution",
                RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION: "urm.op.k2.two_stage_fft_convolution",
                RecurrenceOperator.SECOND_ORDER_CUMSUM: "urm.op.k2.second_order_cumsum",
            }.get(self.spec.recurrence_operator)
            if operator_keyed is not None:
                op_executor = _EXTERNAL_EXECUTORS.get(operator_keyed)
                if op_executor is not None:
                    return op_executor(self, torch, **operands)
        if self.spec.state_effect is StateEffect.IN_PLACE_SLOT_TABLE:
            if not _is_atma_gated_delta_decode_spec(self.spec):
                raise RuntimeError("unsupported in-place slot-table K2 semantics")
            if self.backend is MixerBackend.REFERENCE:
                return _execute_atma_gated_delta_reference(self, torch, **operands)
            # The LIBRARY (upstream ATMA) and any native slot-table execution are
            # resolved through the external-executor registry above.
            raise RuntimeError(
                "in-place slot-table K2 anchor was not registered by a consumer"
            )
        if self.backend is MixerBackend.LIBRARY:
            # In-core library equations that need no upstream/comparator import.
            if self.spec.family is MixerKernelFamily.SOFTMAX:
                if self.spec.k1_operation is K1Operation.DIFFERENTIAL:
                    return _execute_differential_attention(
                        self.spec, torch, library=True, **operands
                    )
                return _execute_sdpa(self.spec, torch, **operands)
            # Every other external/upstream anchor is resolved through the registry
            # at the top of this method. Reaching here means the selected anchor
            # had no registered executor: the consumer did not provision it.
            raise RuntimeError(
                f"no registered external executor for anchor {self.anchor!r}; "
                "the consumer must register it (benchmarks.comparators.executors)"
            )
        if self.backend is MixerBackend.NATIVE:
            if self.spec.family is MixerKernelFamily.SOFTMAX:
                if self.spec.k1_operation is K1Operation.DIFFERENTIAL:
                    return _execute_native_differential_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.PROJECTED:
                    return _execute_native_projected_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.LOCAL_WINDOW:
                    return _execute_native_local_window_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.GATED:
                    return _execute_native_gated_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.POSITIONAL:
                    return _execute_native_positional_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.POSITIVE_FEATURE:
                    return _execute_native_positive_feature_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.THRESHOLDED:
                    return _execute_native_thresholded_attention(self, torch, **operands)
                if self.spec.k1_operation is K1Operation.DELTA_TRANSFORM:
                    return _execute_native_delta_transform_attention(self, torch, **operands)
                from urm.backends.triton.k1.launcher import (
                    TritonOnlineSoftmaxBackend,
                )

                output = TritonOnlineSoftmaxBackend().execute(self.spec, **operands)
                return MixerResult(
                    output,
                    metadata={
                        "anchor": self.anchor,
                        "execution": "urm_native_tiled_online_softmax",
                        "backward_supported": True,
                    },
                )
            if self.spec.family is MixerKernelFamily.RECURRENCE:
                if self.spec.recurrent_layout is RecurrentLayout.MATRIX:
                    operator = self.spec.recurrence_operator
                    if operator is not RecurrenceOperator.PLAIN:
                        executor = _NATIVE_K2_OPERATOR_EXECUTORS.get(operator)
                        if executor is None:
                            raise RuntimeError(
                                "no native executor for recurrence operator "
                                f"{operator.value}"
                            )
                        return executor(self, torch, **operands)
                    return _execute_native_matrix_state_recurrence(
                        self, torch, **operands
                    )
                return _execute_native_diagonal_recurrence(self, torch, **operands)
            return _execute_native_sparse_delta(self, torch, **operands)
        if self.spec.family is MixerKernelFamily.SOFTMAX:
            if self.spec.k1_operation is K1Operation.POSITIVE_FEATURE:
                return _execute_kata_attention_reference(self.spec, torch, **operands)
            if self.spec.k1_operation is K1Operation.LOCAL_WINDOW:
                return _execute_longformer_attention_reference(
                    self.spec, torch, **operands
                )
            if self.spec.k1_operation is K1Operation.PROJECTED:
                return _execute_tucker_attention_reference(self.spec, torch, **operands)
            if self.spec.k1_operation is K1Operation.DIFFERENTIAL:
                return _execute_differential_attention(
                    self.spec, torch, library=False, **operands
                )
            if self.spec.k1_operation is K1Operation.THRESHOLDED:
                return _execute_tda_attention_reference(torch, **operands)
            if self.spec.k1_operation in {
                K1Operation.POLAR,
                K1Operation.POLAR_SPARSE,
            }:
                return _execute_polar_equation(
                    self.spec.k1_operation, torch, **operands
                )
            if self.spec.k1_operation is K1Operation.DELTA_TRANSFORM:
                return _execute_deltaformer_reference(torch, **operands)
            if self.spec.k1_operation is K1Operation.POSITIONAL:
                return _execute_parallax_reference(self.spec, torch, **operands)
            if self.spec.k1_operation is K1Operation.GATED:
                return _execute_wall_reference(self.spec, torch, **operands)
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
                return _execute_diagonal_recurrence(self.spec, torch, **operands)
            return _execute_matrix_recurrence(self.spec, torch, **operands)
        return _execute_sparse_delta(self.spec, torch, **operands)

    __call__ = execute

    def open_decode_session(self, **kwargs: Any):
        """Open a persistent-state single-token decode session for this plan.

        This is the decode/inference counterpart to :meth:`execute`. The training
        path is built for sequence throughput (autograd graph, per-token state
        history, per-call route certification); a decode session instead holds the
        persistent state, updates it in place with one fused single-token kernel
        per step under ``torch.no_grad()``, and does no per-step host work, so the
        step is CUDA-graph capturable. This is the correct way to use the kernel
        for decode; see ``urm/runtime/state.py``.

        The session type is selected from the spec family:
        - K2 matrix-state: ``MatrixStateDecodeSession`` (persistent [B,H,K,V]).
        - K2 diagonal: ``DiagonalDecodeSession`` (persistent [B,C,N]).
        - K3 sparse-state: ``SparseStateDecodeSession`` (persistent [B,S,D] memory,
          trusted native routes).
        """
        from urm.runtime.state import (
            DiagonalDecodeSession,
            MatrixStateDecodeSession,
            SparseStateDecodeSession,
        )

        if self.backend is not MixerBackend.NATIVE:
            raise RuntimeError(
                "decode sessions run the URM-native kernel; compile with "
                "backend=MixerBackend.NATIVE"
            )
        if self.spec.family is MixerKernelFamily.RECURRENCE:
            if self.spec.recurrent_layout is RecurrentLayout.MATRIX:
                return MatrixStateDecodeSession(
                    initial_state=kwargs["initial_state"],
                    scale=kwargs.get("scale"),
                    decay_granularity=kwargs.get("decay_granularity", "head"),
                    is_delta=kwargs.get("is_delta", True),
                    read_before=kwargs.get("read_before", False),
                )
            return DiagonalDecodeSession(
                initial_state=kwargs["initial_state"],
                read_before=kwargs.get("read_before", False),
            )
        if self.spec.family is MixerKernelFamily.SPARSE_DELTA:
            return SparseStateDecodeSession(
                memory=kwargs["memory"],
                read_width=kwargs["read_width"],
                write_width=kwargs["write_width"],
                read_timing_before_update=kwargs.get("read_timing_before_update", True),
            )
        raise RuntimeError(
            f"no decode session for family {self.spec.family.value}; "
            "K1 attention decode uses execute_online_softmax_decode directly "
            "(the KV cache is the persistent state)"
        )


def compile_mixer(
    spec: UnifiedMixerSpec | MixerRecipe,
    *,
    intent: MixerIntent | str = MixerIntent.INFERENCE,
    backend: MixerBackend | str = MixerBackend.REFERENCE,
    dtype: str = "float32",
    device: str | None = None,
    layout: str = "BTHD",
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
    backend_selection = None
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"unsupported mixer compile dtype {dtype!r}")
    if spec.state_effect is StateEffect.IN_PLACE_SLOT_TABLE and not (
        _is_atma_gated_delta_decode_spec(spec)
    ):
        raise ValueError(
            "in-place slot-table K2 currently supports only the exact gated-delta "
            "decode semantics"
        )
    if resolved_backend is MixerBackend.NATIVE and device not in (None, "cuda"):
        raise ValueError("URM Triton native anchors require device='cuda'")
    if (
        resolved_backend is MixerBackend.NATIVE
        and spec.family is MixerKernelFamily.SOFTMAX
        and spec.is_normalized_softmax_attention()
    ):
        from urm.compiler.select.registry import CapabilityRegistry
        from urm.backends.triton.k1.launcher import (
            TritonOnlineSoftmaxBackend,
        )

        implementation = TritonOnlineSoftmaxBackend()
        if not implementation.supports_spec(spec):
            raise ValueError(
                "Triton online-softmax backend declines the complete K1 semantics"
            )
        backend_request = implementation.request(
            device=device or "cuda",
            dtype=dtype,
            layout=layout,
            mode=resolved_intent.value,
        )
        _, backend_selection = CapabilityRegistry([implementation]).select(
            backend_request, backend=implementation.name
        )
    if resolved_backend is MixerBackend.NATIVE and not (
        spec.family is MixerKernelFamily.SPARSE_DELTA
        or (
            spec.family is MixerKernelFamily.RECURRENCE
            and spec.recurrent_layout is RecurrentLayout.DIAGONAL
        )
        or (
            spec.family is MixerKernelFamily.RECURRENCE
            and spec.recurrent_layout is RecurrentLayout.MATRIX
            and (
                _native_matrix_state_supported(spec)
                or spec.recurrence_operator in _NATIVE_K2_OPERATORS
            )
        )
        or (
            spec.family is MixerKernelFamily.SOFTMAX
            and (
                spec.is_normalized_softmax_attention()
                or spec.k1_operation in _NATIVE_K1_OPERATIONS
            )
        )
    ):
        raise ValueError(
            "URM-native anchors support K1 normalized softmax (and its covered "
            "operation variants), K3 sparse delta, K2 diagonal SSM semantics, "
            "the covered K2 matrix-state recurrences, or the distinguished K2 "
            "recurrence operators"
        )
    library_k2_anchor = None
    if (
        resolved_backend is MixerBackend.LIBRARY
        and spec.family is MixerKernelFamily.RECURRENCE
    ):
        if _is_atma_gated_delta_decode_spec(spec):
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
    from urm.compiler.select.anchors import NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME

    if resolved_backend is MixerBackend.LIBRARY:
        if spec.family is MixerKernelFamily.SOFTMAX:
            library_k1_anchors = {
                K1Operation.POLAR: "atma_polar_triton_adapter",
                K1Operation.POLAR_SPARSE: "atma_polar_sparse_triton_adapter",
                K1Operation.DELTA_TRANSFORM: "fla_parallel_deltaformer_adapter",
                K1Operation.PATH_TRANSFORM: "fla_parallel_path_attention_adapter",
                K1Operation.FORGETTING: "fla_parallel_forgetting_attention_adapter",
                K1Operation.POSITIONAL: "fla_parallel_parallax_adapter",
                K1Operation.GATED: "fla_parallel_wall_attention_adapter",
                K1Operation.BLOCK_ROUTED: "fla_parallel_moba_adapter",
                K1Operation.DEPTH: "fla_fused_attnres_adapter",
                K1Operation.THRESHOLDED: "tda_triton_attention_adapter",
                K1Operation.PROJECTED: "tucker_triton_attention_adapter",
                K1Operation.LOCAL_WINDOW: "longformer_sliding_chunks_adapter",
                K1Operation.POSITIVE_FEATURE: "kata_parallel_triton_adapter",
                K1Operation.SELECTED_READ: "fwpkm_selected_softmax_triton_adapter",
            }
            anchor = library_k1_anchors.get(
                spec.k1_operation,
                "torch.nn.functional.scaled_dot_product_attention",
            )
        elif spec.family is MixerKernelFamily.SPARSE_DELTA:
            # Legacy recipe dispatch (deleted once the graph path covers K3);
            # the pinned SDM fallback anchor is registered by the comparator
            # consumer, not core.
            anchor = "facebook_sparse_delta_memory_183e7df_precomputed_route_adapter"
        else:
            anchor = library_k2_anchor
            assert anchor is not None
    elif resolved_backend is MixerBackend.NATIVE:
        from urm.compiler.select.anchors import (
            NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME,
            NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME,
        )

        if spec.family is MixerKernelFamily.RECURRENCE:
            anchor = (
                NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME
                if spec.recurrent_layout is RecurrentLayout.MATRIX
                else NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME
            )
        else:
            anchor = {
                MixerKernelFamily.SOFTMAX: NATIVE_K1_ONLINE_SOFTMAX_ANCHOR_NAME,
                MixerKernelFamily.SPARSE_DELTA: "urm_native_sparse_state_mixer_v0",
            }[spec.family]
    from urm.compiler.pipeline import CompilationIntent, ScheduleParams, UrmCompiler

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
        backend_selection=backend_selection,
    )


def mixer_semantic_program(spec: UnifiedMixerSpec, *, dtype: str = "float32"):
    """Build the backend-independent semantic program for one mixer equation."""
    from urm.ir.program import (
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
        if spec.k1_operation in {K1Operation.POLAR, K1Operation.POLAR_SPARSE}:
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
            if spec.k1_operation is K1Operation.POLAR_SPARSE:
                integer_inputs = ("page_indices", "page_counts")
            output_names = ("output", "auxiliary_output")
        elif spec.k1_operation is K1Operation.DELTA_TRANSFORM:
            floating_inputs = ("query", "key", "value", "beta")
        elif spec.k1_operation is K1Operation.PATH_TRANSFORM:
            floating_inputs = ("query", "key", "value", "w", "beta", "g")
        elif spec.k1_operation is K1Operation.DEPTH:
            floating_inputs = ("query", "rms_weight", "residuals")
        else:
            floating_inputs = ("query", "key", "value")
        if spec.k1_operation is K1Operation.POSITIONAL:
            floating_inputs = ("query", "r", "key", "value")
        elif spec.k1_operation is K1Operation.GATED:
            floating_inputs = ("query", "key", "value", "g")
        if spec.k1_operation not in {K1Operation.POLAR, K1Operation.POLAR_SPARSE}:
            bool_inputs = ("attention_mask",)
            if spec.accepts_score_bias:
                floating_inputs += ("score_bias",)
            output_names = ("output",)
    elif spec.family is MixerKernelFamily.RECURRENCE:
        if spec.state_effect is StateEffect.IN_PLACE_SLOT_TABLE:
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
            recipe = _named_mixer_recipe("sparse_attention_core")
            recipe = replace(
                recipe,
                architecture_ids=(),
                spec=replace(
                    recipe.spec,
                    name=spec.name,
                    requires_attention_mask=True,
                ),
                component_scope=(f"{spec.name}: exact masked softmax attention core"),
            )
            return compile_mixer(recipe, intent=intent, backend=backend, dtype=dtype)
        if spec.routing.value != "dense":
            raise ValueError(
                "sparse sequence attention requires a typed SparseAttentionSpec"
            )
        recipe = MixerRecipe(
            architecture_ids=(),
            spec=_softmax_attention_spec(spec.name),
            component_scope="dense softmax attention core from frontend MixerSpec",
            required_external_stages=("Q/K/V projections and positional transforms",),
        )
        return compile_mixer(recipe, intent=intent, backend=backend, dtype=dtype)
    else:
        raise ValueError(
            f"source domain {spec.source_domain.value!r} has no K1/K2/K3 recipe"
        )

    recipe = _named_mixer_recipe(recipe_name)
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


# External-executor registry: maps a selected external/upstream anchor name to
# its executor callable. The core does NOT import comparators or upstream
# libraries; consumer applications (benchmarks/comparators) register executors
# for the external anchors they provision. ``CompiledMixerPlan.execute`` resolves
# the *selected anchor* from the compiled plan through this registry instead of
# redispatching on the recipe name or operation, so the runtime binds and invokes
# the compiled plan exactly.
_EXTERNAL_EXECUTORS: dict[str, Any] = {}


def register_external_executor(anchor_name: str, executor: Any) -> None:
    """Register ``executor`` for an external/upstream ``anchor_name``.

    The executor is called as ``executor(plan, torch, **operands)`` and must
    return a :class:`MixerResult`. Registering the same anchor twice is an
    error: an external capability must have a single, explicit owner.
    """
    if anchor_name in _EXTERNAL_EXECUTORS:
        raise ValueError(f"external executor already registered: {anchor_name}")
    _EXTERNAL_EXECUTORS[anchor_name] = executor


def clear_external_executors() -> None:
    """Remove all registered external executors (test isolation)."""
    _EXTERNAL_EXECUTORS.clear()


def _is_atma_gated_delta_decode_spec(spec: UnifiedMixerSpec) -> bool:
    """Match the complete semantic contract of ATMA's slot-table decode step."""
    expected = UnifiedMixerSpec(
        "atma_gated_delta_decode_core",
        MixerKernelFamily.RECURRENCE,
        update_rule=StateUpdateRule.DELTA,
        decay=DecayGranularity.HEAD,
        feature_map=FeatureMap.L2_NORMALIZE,
        state_effect=StateEffect.IN_PLACE_SLOT_TABLE,
    )
    return replace(spec, name=expected.name) == expected


_NATIVE_K1_OPERATIONS = frozenset(
    {
        K1Operation.DIFFERENTIAL,
        K1Operation.PROJECTED,
        K1Operation.LOCAL_WINDOW,
        K1Operation.GATED,
        K1Operation.POSITIONAL,
        K1Operation.POSITIVE_FEATURE,
        K1Operation.THRESHOLDED,
        K1Operation.DELTA_TRANSFORM,
    }
)


_NATIVE_K2_OPERATORS = frozenset(
    {
        RecurrenceOperator.TANH_RNN,
        RecurrenceOperator.GATED_RNN,
        RecurrenceOperator.MULTIPLICATIVE_RNN,
        RecurrenceOperator.FFT_CONVOLUTION,
        RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION,
        RecurrenceOperator.SECOND_ORDER_CUMSUM,
        RecurrenceOperator.REGULARIZED_SOLVE,
        RecurrenceOperator.LAYERNORM_INNER_STATE,
        RecurrenceOperator.MOMENTUM_INNER_STATE,
        RecurrenceOperator.MOMENTUM_DELTA_STATE,
        RecurrenceOperator.GATED_OJA_VALUE_CHANNEL,
        RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE,
        RecurrenceOperator.RWKV4_SCALAR_STATE,
        RecurrenceOperator.RWKV6_BONUS_CORRECTED,
        RecurrenceOperator.MAMBA2_STRUCTURED_SSM,
        RecurrenceOperator.TRAPEZOIDAL_SSM,
    }
)


def _native_matrix_state_supported(spec: UnifiedMixerSpec) -> bool:
    """Whether the native matrix-state recurrence kernel computes this spec's equation.

    The native generator lowers the matrix-state recurrence
    ``Z_t = decay*M, M_t = Z_t + k delta^T, y_t = scale * q^T M_t`` and the
    canonical-core variants the fused kernel implements: the dual-gate delta
    (gdn2), the dual-key delta (comba), the key-channel-decayed scaled read
    (kda), the factored left transitions (generalized-delta IPLR/DPLR), the
    multi-rank ordered delta updates (gated_delta_product), the query/key
    denominator normalizer (linear/based/rebased/retention forms), the
    polynomial quadratic bases (pre-expanded by the caller), the supported
    feature maps, and static head decay. A spec qualifies only when its semantic
    fields stay inside the validated envelope.

    The plain additive/no-decay configuration is declined even though it looks
    plain: the IR does not yet distinguish a plain additive recurrence from the
    exotic additive equations that share its fields (a GRU's tanh/gate
    nonlinearity, an FFT long convolution, a second-order correction, and
    similar are represented only by the ``recurrence_operator`` field, which the
    distinguished-operator dispatch handles separately). Dispatching those on the
    spec alone would silently compute the wrong equation, so the native kernel
    declines the whole plain additive/no-decay group; the distinguished
    operators route to their own kernels. Every other supported configuration is
    collision-free: no name-dependent recipe shares its semantic signature.
    """
    if spec.family is not MixerKernelFamily.RECURRENCE:
        return False
    if spec.recurrent_layout is not RecurrentLayout.MATRIX:
        return False
    if spec.recurrence_operator is not RecurrenceOperator.PLAIN:
        return False
    if spec.update_rule not in (StateUpdateRule.ADDITIVE, StateUpdateRule.DELTA):
        return False
    if spec.decay not in (
        DecayGranularity.NONE,
        DecayGranularity.HEAD,
        DecayGranularity.KEY_CHANNEL,
    ):
        return False
    if (
        spec.update_rule is StateUpdateRule.ADDITIVE
        and spec.decay is DecayGranularity.NONE
        and spec.polynomial_basis is PolynomialBasis.NONE
        and spec.normalizer is not StateNormalizer.QUERY_KEY
        and spec.transition is not StateTransition.FACTORED_MATRIX
    ):
        return False
    if spec.feature_map not in (
        FeatureMap.IDENTITY,
        FeatureMap.L2_NORMALIZE,
        FeatureMap.RELU,
        FeatureMap.ELU_PLUS_ONE,
    ):
        return False
    if spec.normalizer not in (StateNormalizer.NONE, StateNormalizer.QUERY_KEY):
        return False
    if spec.transition is StateTransition.FACTORED_MATRIX:
        if not (spec.generalized_delta_iplr or spec.generalized_delta_dplr):
            return False
    elif spec.transition is not StateTransition.POINTWISE:
        return False
    if spec.read_timing not in (ReadTiming.BEFORE_UPDATE, ReadTiming.AFTER_UPDATE):
        return False
    if spec.state_effect is not StateEffect.FUNCTIONAL:
        return False
    if any(
        (
            spec.mamba2_ssm,
            spec.log_linear_attention,
            spec.rwkv4_memory,
            spec.rwkv6_memory,
            spec.momentum_delta,
            spec.gated_oja,
            spec.preconditioned_gated_delta,
            spec.preconditioned_kda,
            spec.slot_attention,
            spec.step_size_discretization,
            spec.diagonal_hgrn,
            spec.path_attention,
            spec.deltaformer_attention,
        )
    ):
        return False
    return True


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
        return value
    if value.ndim == 3:
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




def _execute_sdpa(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    mask = operands.pop("attention_mask", None)
    score_bias = operands.pop("score_bias", None)
    if operands:
        raise TypeError(f"unexpected K1 operands: {', '.join(sorted(operands))}")
    if spec.requires_attention_mask and mask is None:
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
            combined_mask = bias.masked_fill(~combined_mask, float("-inf"))
        else:
            combined_mask = combined_mask + bias
    if (
        combined_mask is not None
        and combined_mask.is_floating_point()
        and combined_mask.dtype != query.dtype
    ):
        combined_mask = combined_mask.to(dtype=query.dtype)
    is_causal = False
    if spec.causal and q_len == k_len and combined_mask is None:
        is_causal = True
        causal_strategy = "sdpa_is_causal"
    elif spec.causal and q_len == 1 and combined_mask is None:
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


def _polar_allowed_mask(
    torch: Any, sparse: bool, query: Any, operands: dict[str, Any]
):
    batch, heads, sequence, _ = query.shape
    positions = torch.arange(sequence, device=query.device)
    allowed = positions[None, :] <= positions[:, None]
    if sparse:
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


def _execute_polar_equation(operation: K1Operation, torch: Any, **operands: Any):
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    n_keys = operands.pop("n_keys")
    v_null = operands.pop("v_null")
    null_base = operands.pop("null_base")
    null_slope_raw = operands.pop("null_slope_raw")
    len_gain_raw = operands.pop("len_gain_raw")
    mag_beta_raw = operands.pop("mag_beta_raw")
    allowed, positions = _polar_allowed_mask(
        torch, operation is K1Operation.POLAR_SPARSE, query, operands
    )
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




def _execute_softmax(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
    if spec.k1_operation is K1Operation.PATH_TRANSFORM:
        return _execute_path_attention_reference(torch, **operands)
    if spec.k1_operation is K1Operation.POSITIONAL:
        return _execute_parallax_reference(spec, torch, **operands)
    if spec.k1_operation is K1Operation.GATED:
        return _execute_wall_reference(spec, torch, **operands)
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    mask = operands.pop("attention_mask", None)
    score_bias = operands.pop("score_bias", None)
    if operands:
        raise TypeError(f"unexpected K1 operands: {', '.join(sorted(operands))}")
    if spec.requires_attention_mask and mask is None:
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
    empty_rows = torch.isneginf(scores).all(dim=-1, keepdim=True)
    safe_scores = torch.where(empty_rows, torch.zeros_like(scores), scores)
    probabilities = torch.softmax(safe_scores, dim=-1)
    probabilities = torch.where(
        empty_rows, torch.zeros_like(probabilities), probabilities
    )
    output = torch.matmul(probabilities, v).transpose(1, 2).to(value.dtype)
    return MixerResult(
        output,
        metadata={
            "anchor": "urm.unified.k1.softmax_reference.v1",
            "head_mode": "mha" if q_heads == kv_heads else "gqa_or_mqa",
            "execution": "torch_eager_reference",
        },
    )


def _execute_native_differential_attention(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
) -> MixerResult:
    """Compose differential attention natively from two reusable K1 reductions.

    Differential attention is ``softmax(q_a k_a) v - lambda * softmax(q_b k_b) v``:
    two normalized routed reductions (the reusable K1 online-softmax capability)
    combined by a learned per-head scalar. This is a held-out composition probe -
    the frozen compiler generates it from the existing K1 core with no
    architecture-specific kernel body.
    """
    spec = plan.spec
    query_a = operands.pop("query_a")
    query_b = operands.pop("query_b")
    key_a = operands.pop("key_a")
    key_b = operands.pop("key_b")
    value = operands.pop("value")
    lambda_weight = operands.pop("lambda_weight")
    if operands:
        raise TypeError(
            f"unexpected native differential K1 operands: {', '.join(sorted(operands))}"
        )
    if query_a.ndim != 4 or any(
        item.shape != query_a.shape for item in (query_b, key_a, key_b)
    ):
        raise ValueError("differential Q/K operands must share [B,T,H,K] layout")
    batch, sequence, heads, key_dim = query_a.shape
    if lambda_weight.ndim == 0:
        combine = lambda_weight
    elif lambda_weight.shape == (heads,):
        combine = lambda_weight.view(1, 1, heads, 1)
    else:
        raise ValueError("differential lambda_weight must be scalar or [H]")
    from urm.backends.triton.k1.online import execute_online_softmax

    attention_scale = spec.attention_scale or key_dim**-0.5
    output_a = execute_online_softmax(
        query_a, key_a, value,
        attention_mask=None, score_bias=None, causal=spec.causal, scale=attention_scale,
    )
    output_b = execute_online_softmax(
        query_b, key_b, value,
        attention_mask=None, score_bias=None, causal=spec.causal, scale=attention_scale,
    )
    output = output_a - combine * output_b
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_two_online_softmax_reductions_and_differential_combine",
            "backward_supported": True,
            "composition": "two K1 normalized reductions + learned per-head combine",
        },
    )


def _expand_kv_to_query_heads(key: Any, value: Any, query_heads: int) -> tuple:
    """Broadcast K/V heads over their query-head group (materialized, for the
    operation-specific K1 kernels that need a shared head count)."""
    kv_heads = key.shape[2]
    group = query_heads // kv_heads
    if group == 1:
        return key, value
    return (
        key.repeat_interleave(group, dim=2),
        value.repeat_interleave(group, dim=2),
    )


def _execute_native_projected_attention(plan, torch, **operands):
    """Tucker projected attention: expand the low-rank query by B_pre, then plain K1."""
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    b_pre = operands.pop("B_pre")
    if operands:
        raise TypeError(
            f"unexpected native projected K1 operands: {', '.join(sorted(operands))}"
        )
    expanded_query = torch.einsum(
        "btr,hrk->bthk", query.float(), b_pre.float()
    ).to(query.dtype)
    if key.ndim == 3:
        key = key[:, :, None, :]
    if value.ndim == 3:
        value = value[:, :, None, :]
    from urm.backends.triton.k1.online import execute_online_softmax

    key_dim = expanded_query.shape[-1]
    output = execute_online_softmax(
        expanded_query,
        key,
        value,
        attention_mask=None,
        score_bias=None,
        causal=spec.causal,
        scale=spec.attention_scale or key_dim**-0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_projected_query_online_softmax",
            "backward_supported": True,
            "composition": "low-rank query expansion (query @ B_pre) + K1 reduction",
        },
    )


def _execute_native_local_window_attention(plan, torch, **operands):
    """Longformer local-window attention: a sliding-window mask composed with K1."""
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    window = operands.pop("attention_window")
    if operands:
        raise TypeError(
            f"unexpected native local-window K1 operands: {', '.join(sorted(operands))}"
        )
    if not isinstance(window, int) or window <= 0:
        raise ValueError("local-window attention requires a positive attention_window")
    sequence = query.shape[1]
    positions = torch.arange(sequence, device=query.device)
    local_mask = (positions[:, None] - positions[None, :]).abs() <= window
    from urm.backends.triton.k1.online import execute_online_softmax

    key_dim = query.shape[-1]
    output = execute_online_softmax(
        query,
        key,
        value,
        attention_mask=local_mask.view(1, 1, sequence, sequence),
        score_bias=None,
        causal=spec.causal,
        scale=spec.attention_scale or key_dim**-0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_local_window_mask_online_softmax",
            "backward_supported": True,
            "composition": "sliding-window boolean mask + K1 reduction",
        },
    )


def _execute_native_gated_attention(plan, torch, **operands):
    """Wall gated attention: a per-channel log-decay q/k transform composed with K1.

    scores[t, s] = sum_k q[t, k] k[s, k] exp(prefix[t, k] - prefix[s, k]) with
    ``prefix = cumsum(g)``. This factors as a per-channel feature map on Q and K:
    ``q' = q * exp(prefix)``, ``k' = k * exp(-prefix)`` (K pre-expanded to the
    query-head count, since the decay is per query head), then a plain K1
    reduction. Note: the factored transform can overflow fp32 for long sequences
    with large decays; it is exact for the canonical coverage shapes.
    """
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("g")
    if operands:
        raise TypeError(
            f"unexpected native gated K1 operands: {', '.join(sorted(operands))}"
        )
    query_heads = query.shape[2]
    key, value = _expand_kv_to_query_heads(key, value, query_heads)
    prefix = log_decay.float().cumsum(dim=1)
    transformed_query = (query.float() * torch.exp(prefix)).to(query.dtype)
    transformed_key = (key.float() * torch.exp(-prefix)).to(key.dtype)
    from urm.backends.triton.k1.online import execute_online_softmax

    key_dim = query.shape[-1]
    output = execute_online_softmax(
        transformed_query,
        transformed_key,
        value,
        attention_mask=None,
        score_bias=None,
        causal=spec.causal,
        scale=spec.attention_scale or key_dim**-0.5,
    )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_gated_decay_transform_online_softmax",
            "backward_supported": True,
            "composition": "per-channel log-decay q/k transform + K1 reduction",
        },
    )


def _execute_native_positional_attention(plan, torch, **operands):
    """Parallax positional attention: a composition on the canonical K1 reduction.

    With softmax probabilities P and secondary scores S = r.k, the output is
    ``(P V)(1 + sum(P*S)) - (P*S) V``. P is materialized by the native
    softmax-probs kernel; the corrections are CUDA tensor arithmetic.

    Backward: the softmax-probs kernel is forward-only, so the autograd graph
    through the materialized P would miss the softmax normalization state. The
    probabilities are therefore recomputed with differentiable torch ops (the
    same masked-softmax reduction the kernel performs) whenever any input
    requires a gradient, giving the exact canonical input gradients.
    """
    spec = plan.spec
    query = operands.pop("query")
    secondary = operands.pop("r")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(
            f"unexpected native positional K1 operands: {', '.join(sorted(operands))}"
        )
    query_heads = query.shape[2]
    key_dim = query.shape[-1]
    key, value = _expand_kv_to_query_heads(key, value, query_heads)
    from urm.backends.triton.k1.online import execute_softmax_probs

    scale = spec.attention_scale or key_dim**-0.5
    needs_grad = any(
        torch.is_tensor(item) and item.requires_grad
        for item in (query, secondary, key, value)
    )
    if needs_grad:
        q = query.float().transpose(1, 2)
        k = key.float().transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        if spec.causal:
            q_len, k_len = query.shape[1], key.shape[1]
            q_pos = torch.arange(q_len, device=query.device) + (k_len - q_len)
            k_pos = torch.arange(k_len, device=query.device)
            visible = k_pos[None, :] <= q_pos[:, None]
            logits = logits.masked_fill(~visible[None, None], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
    else:
        probs = execute_softmax_probs(
            query, key, causal=spec.causal, strict=False, scale=scale
        )
    secondary_scores = torch.einsum(
        "bthk,bshk->bhts", secondary.float(), key.float()
    )
    value_h = value.float().transpose(1, 2)
    ordinary = torch.einsum("bhts,bhsv->bhtv", probs, value_h)
    correction = probs * secondary_scores
    correction_mean = correction.sum(dim=-1, keepdim=True)
    correction_out = torch.einsum("bhts,bhsv->bhtv", correction, value_h)
    output = (ordinary * (1.0 + correction_mean) - correction_out).transpose(1, 2)
    return MixerResult(
        output.to(value.dtype),
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_positional_softmax_probs_composition",
            "backward_supported": True,
            "composition": "K1 softmax probabilities + secondary-score correction",
        },
    )


def _execute_native_positive_feature_attention(plan, torch, **operands):
    """KATA positive-feature attention via the native grouped-squared-score kernel.

    Backward: the grouped-squared-score kernel is forward-only, so the scores
    are recomputed with differentiable torch ops (the same grouped squared
    reduction + L1 normalization) whenever any input requires a gradient,
    giving the exact canonical input gradients. Self-attention only (Tq == Tk).
    """
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    num_groups = operands.pop("num_groups")
    if operands:
        raise TypeError(
            f"unexpected native positive-feature K1 operands: {', '.join(sorted(operands))}"
        )
    query_heads = query.shape[2]
    key, value = _expand_kv_to_query_heads(key, value, query_heads)
    from urm.backends.triton.k1.online import execute_positive_feature

    num_groups = int(num_groups)
    needs_grad = any(
        torch.is_tensor(item) and item.requires_grad
        for item in (query, key, value)
    )
    if needs_grad:
        batch, sequence, heads, dim = query.shape
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
    else:
        output = execute_positive_feature(query, key, value, num_groups=num_groups)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_positive_feature_grouped_squared_scores",
            "backward_supported": True,
            "composition": "grouped squared scores + L1 normalization",
        },
    )


def _execute_native_thresholded_attention(plan, torch, **operands):
    """TDA thresholded differential attention via the native thresholded kernel.

    Backward: the thresholded kernel is forward-only, so each branch's scores
    are recomputed with differentiable torch ops (L2-normalized Q/K, thresholded
    squared-relu, unnormalized causal reduction) whenever any input requires a
    gradient, giving the exact canonical input gradients. Uses the paired Q/K
    branches combined by the clamped lambda coefficient.
    """
    query_a = operands.pop("query_a")
    query_b = operands.pop("query_b")
    key_a = operands.pop("key_a")
    key_b = operands.pop("key_b")
    value = operands.pop("value")
    beta = operands.pop("beta")
    lambda_weight = operands.pop("lambda_weight")
    if operands:
        raise TypeError(
            f"unexpected native thresholded K1 operands: {', '.join(sorted(operands))}"
        )
    query_heads = query_a.shape[2]
    key_a, value = _expand_kv_to_query_heads(key_a, value, query_heads)
    key_b, _ = _expand_kv_to_query_heads(key_b, value, query_heads)
    from urm.backends.triton.k1.online import execute_thresholded_attend

    beta_value = float(beta.detach() if torch.is_tensor(beta) else beta)
    coefficient = min(
        max(float(lambda_weight.detach() if torch.is_tensor(lambda_weight) else lambda_weight), 0.0),
        1.0,
    )
    needs_grad = any(
        torch.is_tensor(item) and item.requires_grad
        for item in (query_a, query_b, key_a, key_b, value)
    )
    if needs_grad:
        sequence = query_a.shape[1]
        dim = query_a.shape[-1]
        positions = torch.arange(
            1, sequence + 1, device=query_a.device, dtype=torch.float32
        )
        threshold = beta_value * torch.sqrt(2.0 * torch.log(positions) / dim)
        causal = torch.ones(
            (sequence, sequence), dtype=torch.bool, device=query_a.device
        ).tril()

        def attend(query, key):
            qn = torch.nn.functional.normalize(query.float(), p=2, dim=-1)
            kn = torch.nn.functional.normalize(key.float(), p=2, dim=-1)
            scores = torch.einsum("bthd,bshd->bhts", qn, kn)
            scores = torch.where(causal, scores, torch.zeros_like(scores))
            rectified = torch.relu(scores - threshold.view(1, 1, sequence, 1))
            weights = rectified.square()
            return torch.einsum("bhts,bshv->bthv", weights, value.float())

        attend_a = attend(query_a, key_a)
        attend_b = attend(query_b, key_b)
    else:
        attend_a = execute_thresholded_attend(query_a, key_a, value, beta=beta_value)
        attend_b = execute_thresholded_attend(query_b, key_b, value, beta=beta_value)
    output = (attend_a - coefficient * attend_b).to(value.dtype)
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_thresholded_squared_relu_differential",
            "backward_supported": True,
            "composition": "two thresholded squared-relu reductions + lambda combine",
        },
    )


def _execute_native_delta_transform_attention(plan, torch, **operands):
    """DeltaFormer delta-transform attention: triangular value solve + K1 reduction.

    Strict-causal softmax probabilities P feed a triangular value solve
    ``(I + beta*P) v' = v``; the output is the causal softmax reduction over
    ``v'``. P comes from the native softmax-probs kernel, the solve is a CUDA
    triangular solve, and the final reduction reuses the online-softmax kernel.

    Backward: the softmax-probs kernel is forward-only, so the strict-causal P
    and the final causal reduction are recomputed with differentiable torch ops
    whenever any input requires a gradient (the triangular solve is already
    differentiable), giving the exact canonical input gradients.
    """
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    beta = operands.pop("beta")
    if operands:
        raise TypeError(
            f"unexpected native delta-transform K1 operands: {', '.join(sorted(operands))}"
        )
    batch, sequence, heads, key_dim = query.shape
    key, value = _expand_kv_to_query_heads(key, value, heads)
    from urm.backends.triton.k1.online import (
        execute_online_softmax,
        execute_softmax_probs,
    )

    scale = key_dim**-0.5
    needs_grad = any(
        torch.is_tensor(item) and item.requires_grad
        for item in (query, key, value, beta)
    )
    if needs_grad:
        q = query.float().transpose(1, 2)
        k = key.float().transpose(1, 2)
        v = value.float().transpose(1, 2)
        beta_h = beta.float().transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
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
        probs = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        eye = torch.eye(sequence, device=query.device, dtype=torch.float32)
        system = eye.view(1, 1, sequence, sequence) + beta_h.unsqueeze(-1) * probs
        transformed = torch.linalg.solve_triangular(system, v, upper=False)
        causal = positions[None, :] <= positions[:, None]
        causal_scores = scores.masked_fill(~causal, -float("inf"))
        attention = torch.softmax(causal_scores, dim=-1)
        output = torch.matmul(attention, transformed)
        output = output.transpose(1, 2).to(value.dtype)
    else:
        probs = execute_softmax_probs(query, key, causal=True, strict=True, scale=scale)
        beta_h = beta.float().transpose(1, 2)
        eye = torch.eye(sequence, device=query.device, dtype=torch.float32)
        system = eye.view(1, 1, sequence, sequence) + beta_h.unsqueeze(-1) * probs
        value_h = value.float().transpose(1, 2)
        transformed = torch.linalg.solve_triangular(system, value_h, upper=False)
        transformed = transformed.transpose(1, 2).to(value.dtype)
        output = execute_online_softmax(
            query,
            key,
            transformed,
            attention_mask=None,
            score_bias=None,
            causal=True,
            scale=scale,
        )
    return MixerResult(
        output,
        metadata={
            "anchor": plan.anchor,
            "execution": "urm_native_delta_transform_triangular_solve_online_softmax",
            "backward_supported": True,
            "composition": "strict-causal P + triangular value solve + K1 reduction",
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


def _execute_diagonal_recurrence(spec: UnifiedMixerSpec, torch: Any, **operands: Any):
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






@lru_cache(maxsize=1)


@lru_cache(maxsize=1)


@lru_cache(maxsize=1)


























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






















def _execute_atma_gated_delta_reference(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    if not _is_atma_gated_delta_decode_spec(plan.spec):
        raise RuntimeError("ATMA reference requires its exact K2 semantic contract")
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


def _atma_decode_value_block(batch: int, value_dim: int) -> int:
    """Return the pinned ATMA kernel's unmasked value tile width."""
    return 64 if batch >= 256 and value_dim >= 64 else 32


def _validate_atma_decode_dimensions(batch: int, key_dim: int, value_dim: int) -> int:
    if key_dim <= 0 or key_dim & (key_dim - 1):
        raise ValueError("ATMA key width must be a positive power of two")
    block_v = _atma_decode_value_block(batch, value_dim)
    if value_dim <= 0 or value_dim % block_v:
        raise ValueError(
            "ATMA value width must be divisible by the pinned unmasked "
            f"{block_v}-wide value tile"
        )
    return block_v




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

    from urm.backends.triton.k3.state_launcher import (
        CertifiedSparseStateRoutes,
        SparseState,
        TritonSparseStateMixerBackend,
    )
    from urm.compiler.select.anchors import NATIVE_SPARSE_STATE_MIXER_ANCHOR_NAME
    from urm.ir.program import (
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




def _execute_native_diagonal_recurrence(plan: CompiledMixerPlan, torch: Any, **operands: Any):
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
    gates_one = False
    if spec.diagonal_hgrn:
        if x.ndim != 3 or log_decay.shape != x.shape:
            raise ValueError("HGRN expects x and log_decay with shape [B,T,C]")
        batch_hgrn, sequence_hgrn, channels_hgrn = x.shape
        log_decay = log_decay.unsqueeze(-1)
        input_gate = log_decay
        read_gate = log_decay
        gates_one = True
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
    from urm.backends.triton.k2.diagonal import execute_diagonal_recurrence
    from urm.compiler.select.anchors import NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME

    dtype = str(x.dtype).removeprefix("torch.")
    bound_anchor = _compile_native_diagonal_binding(
        spec, dtype=dtype, intent=plan.intent.value
    )
    if bound_anchor != NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME:
        raise RuntimeError(
            f"UrmCompiler selected {bound_anchor!r} for the native diagonal SSM plan"
        )
    output, final_state = execute_diagonal_recurrence(
        x=x,
        input_gate=input_gate,
        read_gate=read_gate,
        log_decay=log_decay,
        initial_state=initial_state,
        step_size=step_size,
        skip=skip,
        read_before=spec.read_timing is ReadTiming.BEFORE_UPDATE,
        gates_one=gates_one,
    )
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME,
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
    from urm.compiler.pipeline import CompilationIntent, ScheduleParams, UrmCompiler
    from urm.compiler.select.anchors import NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME

    compilation = UrmCompiler().compile(
        mixer_semantic_program(spec, dtype=dtype),
        intent=CompilationIntent(intent),
        schedule_params=ScheduleParams(
            anchor_overrides={"mixer": NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME}
        ),
    )
    selected = tuple(step.anchor for step in compilation.plan.steps if step.anchor)
    if selected != (NATIVE_DIAGONAL_RECURRENCE_ANCHOR_NAME,):
        raise RuntimeError(
            "UrmCompiler produced an invalid native diagonal SSM plan: "
            f"anchors={selected}"
        )
    return selected[0]


def _execute_native_matrix_state_recurrence(
    plan: CompiledMixerPlan, torch: Any, **operands: Any
):
    """Execute the native matrix-state K2 recurrence selected by semantic fields.

    Mirrors the canonical ``_execute_k2`` operand mapping: the plain form plus
    the dual-gate (gdn2), dual-key (comba), key-channel-decayed scaled read
    (kda), factored left transitions (generalized-delta IPLR/DPLR), multi-rank
    ordered updates (gated_delta_product), the query/key normalizer
    (linear/based/rebased/retention forms), the polynomial bases (pre-expanded),
    the supported feature maps, and static head decay.
    """
    spec = plan.spec
    if not _native_matrix_state_supported(spec):
        raise RuntimeError(
            "the native matrix-state anchor implements only the canonical K2 "
            "matrix-state recurrences (delta/additive update, head/key-channel "
            "decay, the covered feature maps and normalizer, pointwise and "
            "generalized-delta factored transitions, dual-gate/dual-key/"
            "multi-rank variants, and static head decay)"
        )
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    initial_state = operands.pop("initial_state", None)
    beta = operands.pop("beta", None)
    log_decay = operands.pop("log_decay", None)
    erase_gate = operands.pop("erase_gate", None)
    write_gate = operands.pop("write_gate", None)
    prediction_key = operands.pop("prediction_key", operands.pop("p", None))
    if spec.comba_rule and log_decay is None:
        log_decay = operands.pop("g", None)
    transition_alpha = operands.pop("transition_alpha", None)
    transition_beta = operands.pop("transition_beta", None)
    update_keys = operands.pop("update_keys", None)
    update_values = operands.pop("update_values", None)
    if operands:
        raise TypeError(
            "unexpected native matrix-state operands: "
            + ", ".join(sorted(operands))
        )
    if spec.gdn2_ssm and (erase_gate is None or write_gate is None):
        raise ValueError("gdn2 dual-gate delta requires erase_gate and write_gate")
    if spec.comba_rule and prediction_key is None:
        raise ValueError("comba dual-key delta requires prediction_key (p)")
    if (
        spec.update_rule is StateUpdateRule.DELTA
        and beta is None
        and not spec.gdn2_ssm
        and not spec.gated_delta_product
    ):
        raise ValueError("the delta update rule requires beta")
    if spec.decay is not DecayGranularity.NONE and log_decay is None:
        raise ValueError("this matrix-state recurrence requires log_decay")
    from urm.backends.triton.k2.matrix import (
        execute_matrix_state_recurrence,
    )
    from urm.compiler.select.anchors import NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME

    dtype = str(query.dtype).removeprefix("torch.")
    bound_anchor = _compile_native_matrix_state_binding(
        spec, dtype=dtype, intent=plan.intent.value
    )
    if bound_anchor != NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME:
        raise RuntimeError(
            f"UrmCompiler selected {bound_anchor!r} for the native matrix-state plan"
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("native matrix-state query/key/value use BTHD rank-4 layout")
    if key.shape != query.shape:
        raise ValueError("native matrix-state query and key must have identical shapes")
    batch, sequence, heads, key_dim = query.shape
    if value.shape[:3] != (batch, sequence, heads):
        raise ValueError("native matrix-state value must use [B,T,H,V] matching query")
    if spec.read_scale is not None:
        scale = spec.read_scale
    elif (
        spec.gdn2_ssm or spec.kda_delta or spec.comba_rule
        or spec.generalized_delta_iplr or spec.generalized_delta_dplr
        or spec.gated_delta_product
    ):
        scale = key_dim ** -0.5
    elif spec.attention_scale is not None:
        scale = spec.attention_scale
    else:
        scale = 1.0
    kernel_feature_map = "identity"
    if spec.polynomial_basis is not PolynomialBasis.NONE:
        poly_scale = spec.read_scale or key_dim ** -0.5
        query = _polynomial_features_torch(
            torch, query, spec.polynomial_basis, poly_scale, is_query=True
        )
        key = _polynomial_features_torch(
            torch, key, spec.polynomial_basis, poly_scale, is_query=False
        )
        scale = 1.0
    elif spec.feature_map is not FeatureMap.IDENTITY:
        kernel_feature_map = spec.feature_map.value
    if spec.static_head_decay and log_decay is not None:
        static = log_decay.reshape(-1)
        head_decay = (
            static[:heads]
            if static.numel() > 1
            else static.expand(heads)
        )
        log_decay = (
            head_decay.view(1, 1, heads)
            .expand(batch, sequence, heads)
            .contiguous()
        )
    left_transitions = None
    if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
        eye = torch.eye(key_dim, device=query.device, dtype=torch.float32)
        rank_one = torch.einsum(
            "bthk,bthl->bthkl", transition_beta.float(), transition_alpha.float()
        )
        if spec.generalized_delta_dplr:
            left_transitions = torch.diag_embed(log_decay.float().exp()) + rank_one
        else:
            left_transitions = eye.view(1, 1, 1, key_dim, key_dim) + rank_one
        left_transitions = left_transitions.contiguous()
        log_decay = None
    kernel_initial = initial_state
    if initial_state is not None and spec.state_v_first:
        kernel_initial = initial_state.transpose(-1, -2).contiguous()
    kernel_is_delta = (
        spec.update_rule is StateUpdateRule.DELTA
        and not spec.gdn2_ssm
        and not spec.gated_delta_product
    )
    result = execute_matrix_state_recurrence(
        query=query,
        key=key,
        value=value,
        log_decay=log_decay,
        beta=beta,
        initial_state=kernel_initial,
        scale=scale,
        decay_granularity=spec.decay.value,
        is_delta=kernel_is_delta,
        read_before=spec.read_timing is ReadTiming.BEFORE_UPDATE,
        retrieval_keys=prediction_key,
        erase_gate=erase_gate,
        write_gate=write_gate,
        left_transitions=left_transitions,
        update_keys=update_keys,
        update_values=update_values,
        rank_beta=beta if spec.gated_delta_product else None,
        feature_map=kernel_feature_map,
        normalizer=spec.normalizer is StateNormalizer.QUERY_KEY,
        epsilon=spec.epsilon,
    )
    if spec.normalizer is StateNormalizer.QUERY_KEY:
        output, final_state, final_normalizer = result
    else:
        output, final_state = result
        final_normalizer = None
    if spec.state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return MixerResult(
        output,
        final_state=final_state,
        final_normalizer_state=final_normalizer,
        metadata={
            "anchor": NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME,
            "execution": "urm_native_triton",
            "compiler_plan": plan.anchor,
            "urm_compiler_verified": True,
            "runtime_compiler_binding": "cached_semantic_shape",
            "runtime_binding_cache_size": _compile_native_matrix_state_binding.cache_info().currsize,
            "backward_supported": True,
        },
    )


def _polynomial_features_torch(
    torch: Any, x: Any, basis: PolynomialBasis, scale: float, *, is_query: bool
):
    """Quadratic feature expansion for the based/rebased polynomial bases.

    The query is scaled by ``scale``; the key is left unscaled (matching the
    canonical ``_polynomial_features``).
    """
    z = x.float() * scale if is_query else x.float()
    quad = z.unsqueeze(-1) * z.unsqueeze(-2)
    if basis is PolynomialBasis.BASED_TAYLOR2:
        return torch.cat(
            [
                torch.ones_like(z[..., :1]),
                z,
                quad.reshape(*z.shape[:-1], -1) / (2.0 ** 0.5),
            ],
            dim=-1,
        ).contiguous()
    if basis is PolynomialBasis.REBASED_SQUARE:
        return quad.reshape(*z.shape[:-1], -1).contiguous()
    raise ValueError(f"polynomial basis {basis} not supported natively")


def _native_k2_result(
    plan: CompiledMixerPlan,
    output,
    final_state,
    execution: str,
    *,
    backward_supported: bool = False,
):
    return MixerResult(
        output,
        final_state=final_state,
        metadata={
            "anchor": plan.anchor,
            "execution": execution,
            "backward_supported": backward_supported,
        },
    )


def _execute_native_tanh_rnn(plan, torch, **operands):
    """XMA tanh RNN: ``h_t = tanh(h_{t-1} @ W + x_t)``; output is the state."""
    query = operands.pop("query")
    weight = operands.pop("weight")
    initial_state = operands.pop("initial_state")
    if operands:
        raise TypeError(f"unexpected native tanh_rnn operands: {sorted(operands)}")
    from urm.backends.triton.k2.backward import _tanh_rnn_backwardable

    output, final = _tanh_rnn_backwardable(query, weight, initial_state)
    return _native_k2_result(
        plan, output, final, "urm_native_tanh_rnn", backward_supported=True
    )


def _execute_native_gated_rnn(plan, torch, **operands):
    """XMA GRU: reset/update-gated nonlinear state update."""
    query = operands.pop("query")
    weight = operands.pop("weight")
    forget_input = operands.pop("forget_input")
    forget_weight = operands.pop("forget_weight")
    reset_input = operands.pop("reset_input")
    reset_weight = operands.pop("reset_weight")
    initial_state = operands.pop("initial_state")
    if operands:
        raise TypeError(f"unexpected native gated_rnn operands: {sorted(operands)}")
    from urm.backends.triton.k2.backward import _gated_rnn_backwardable

    output, final = _gated_rnn_backwardable(
        query, weight, forget_input, forget_weight, reset_input, reset_weight,
        initial_state,
    )
    return _native_k2_result(
        plan, output, final, "urm_native_gated_rnn", backward_supported=True
    )


def _execute_native_multiplicative_rnn(plan, torch, **operands):
    """XMA second-order multiplicative RNN (matrix memory)."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    weight = operands.pop("weight")
    forget_input = operands.pop("forget_input")
    initial_state = operands.pop("initial_state")
    if operands:
        raise TypeError(
            f"unexpected native multiplicative_rnn operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import (
        _multiplicative_rnn_backwardable,
    )

    output, final = _multiplicative_rnn_backwardable(
        query, key, value, weight, forget_input, initial_state
    )
    return _native_k2_result(
        plan, output, final, "urm_native_multiplicative_rnn",
        backward_supported=True,
    )


def _execute_native_rwkv4_scalar_state(plan, torch, **operands):
    """RWKV-4 time-mix with a stable three-scalar-per-channel state."""
    w = operands.pop("w")
    u = operands.pop("u")
    key = operands.pop("k")
    value = operands.pop("v")
    state = operands.pop("state")
    if operands:
        raise TypeError(f"unexpected native rwkv4 operands: {sorted(operands)}")
    from urm.backends.triton.k2.backward import _rwkv4_backwardable

    output, final = _rwkv4_backwardable(w, u, key, value, state)
    return _native_k2_result(
        plan, output, final, "urm_native_rwkv4_scalar_state",
        backward_supported=True,
    )


def _execute_native_rwkv6_bonus_corrected(plan, torch, **operands):
    """RWKV-6 key-channel-decayed matrix memory with a static bonus read."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    bonus = operands.pop("bonus")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected native rwkv6 operands: {sorted(operands)}")
    from urm.backends.triton.k2.backward import _rwkv6_backwardable

    output, final = _rwkv6_backwardable(
        query, key, value, log_decay, bonus, initial_state
    )
    return _native_k2_result(
        plan, output, final, "urm_native_rwkv6_bonus_corrected",
        backward_supported=True,
    )


def _execute_native_mamba2_structured_ssm(plan, torch, **operands):
    """Mamba-2 SSD structured SSM with continuous-time head decay."""
    x = operands.pop("x")
    dt = operands.pop("dt")
    a = operands.pop("A")
    b = operands.pop("B")
    c = operands.pop("C")
    initial_states = operands.pop("initial_states", None)
    if operands:
        raise TypeError(f"unexpected native mamba2 operands: {sorted(operands)}")
    from urm.backends.triton.k2.backward import _mamba2_backwardable

    output, final = _mamba2_backwardable(x, dt, a, b, c, initial_states)
    return _native_k2_result(
        plan, output, final, "urm_native_mamba2_structured_ssm",
        backward_supported=True,
    )


def _execute_native_trapezoidal_ssm(plan, torch, **operands):
    """Mamba-3 SISO rotary angle accumulator with trapezoidal four-state SSM."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    adt = operands.pop("adt")
    dt = operands.pop("dt")
    trap = operands.pop("trap")
    query_bias = operands.pop("query_bias")
    key_bias = operands.pop("key_bias")
    angles = operands.pop("angles")
    if operands:
        raise TypeError(
            f"unexpected native trapezoidal_ssm operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import _trapezoidal_ssm_backwardable

    output, final = _trapezoidal_ssm_backwardable(
        query, key, value, adt, dt, trap, query_bias, key_bias, angles
    )
    return _native_k2_result(
        plan, output, final, "urm_native_trapezoidal_ssm", backward_supported=True
    )


def _execute_native_fft_convolution(plan, torch, **operands):
    """Hyena single implicit-filter causal FFT convolution (torch-native FFT)."""
    query = operands.pop("query")
    kernel = operands.pop("kernel")
    direct = operands.pop("direct")
    if operands:
        raise TypeError(
            f"unexpected native fft_convolution operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.inner_state import execute_fft_convolution

    output, _ = execute_fft_convolution(query=query, kernel=kernel, direct=direct)
    return _native_k2_result(plan, output, None, "urm_native_fft_convolution")


def _execute_native_two_stage_fft_convolution(plan, torch, **operands):
    """H3 two-stage causal FFT convolution (torch-native FFT)."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    ssm_kernel = operands.pop("ssm_kernel")
    ssm_k_kernel = operands.pop("ssm_k_kernel")
    ssm_k_direct = operands.pop("ssm_k_direct")
    skip = operands.pop("skip")
    if operands:
        raise TypeError(
            f"unexpected native two_stage_fft_convolution operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.inner_state import (
        execute_two_stage_fft_convolution,
    )

    output, _ = execute_two_stage_fft_convolution(
        query=query, key=key, value=value, ssm_kernel=ssm_kernel,
        ssm_k_kernel=ssm_k_kernel, ssm_k_direct=ssm_k_direct, skip=skip,
    )
    return _native_k2_result(plan, output, None, "urm_native_two_stage_fft_convolution")


def _execute_native_second_order_cumsum(plan, torch, **operands):
    """HLA masked second-order causal attention with exact streaming summaries."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    if operands:
        raise TypeError(
            f"unexpected native second_order_cumsum operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.inner_state import execute_second_order_cumsum

    output, _ = execute_second_order_cumsum(query=query, key=key, value=value)
    return _native_k2_result(plan, output, None, "urm_native_second_order_cumsum")


def _execute_native_regularized_solve(plan, torch, **operands):
    """MesaNet dual covariance-state recurrence with a regularized per-token solve."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    log_decay = operands.pop("log_decay")
    beta = operands.pop("beta")
    lamb = operands.pop("lamb")
    if operands:
        raise TypeError(
            f"unexpected native regularized_solve operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import (
        _regularized_solve_backwardable,
    )

    output, final = _regularized_solve_backwardable(
        query, key, value, log_decay, beta, lamb
    )
    return _native_k2_result(
        plan, output, final, "urm_native_regularized_solve", backward_supported=True
    )


def _execute_native_layernorm_inner_state(plan, torch, **operands):
    """TTT-Linear chunkwise inner-loss update with matrix and bias memory states."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    w = operands.pop("w")
    b = operands.pop("b")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    initial_state_bias = operands.pop("initial_state_bias", None)
    chunk_size = int(operands.pop("chunk_size", 16))
    eps = float(operands.pop("eps", 1e-6))
    if operands:
        raise TypeError(
            f"unexpected native layernorm_inner_state operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import (
        _layernorm_inner_state_backwardable,
    )

    output, final = _layernorm_inner_state_backwardable(
        query, key, value, w, b, eta, initial_state, initial_state_bias,
        chunk_size, eps,
    )
    return _native_k2_result(
        plan, output, final, "urm_native_layernorm_inner_state",
        backward_supported=True,
    )


def _execute_native_momentum_inner_state(plan, torch, **operands):
    """Titans tokenwise memory with momentum and a learned inner loss."""
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    w = operands.pop("w")
    b = operands.pop("b")
    theta = operands.pop("theta")
    alpha = operands.pop("alpha")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    chunk_size = int(operands.pop("chunk_size", 16))
    eps = float(operands.pop("eps", 1e-6))
    if operands:
        raise TypeError(
            f"unexpected native momentum_inner_state operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import (
        _momentum_inner_state_backwardable,
    )

    output, final = _momentum_inner_state_backwardable(
        query, key, value, w, b, theta, alpha, eta, initial_state, chunk_size,
        eps,
    )
    return _native_k2_result(
        plan, output, final, "urm_native_momentum_inner_state",
        backward_supported=True,
    )


def _execute_native_momentum_delta(plan, torch, **operands):
    """Momentum DeltaNet with coupled fast-weight and momentum matrix states."""
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    p = operands.pop("p")
    log_alpha = operands.pop("log_alpha")
    log_mu = operands.pop("log_mu")
    beta = operands.pop("beta")
    eta = operands.pop("eta")
    initial_state = operands.pop("initial_state", None)
    initial_momentum = operands.pop("initial_normalizer_state", None)
    if operands:
        raise TypeError(
            f"unexpected native momentum_delta operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import _momentum_delta_backwardable

    output, final = _momentum_delta_backwardable(
        query, key, value, p, log_alpha, log_mu, beta, eta, initial_state,
        initial_momentum, spec.read_scale,
    )
    return _native_k2_result(
        plan, output, final, "urm_native_momentum_delta", backward_supported=True
    )


def _execute_native_gated_oja(plan, torch, **operands):
    """Gated Oja value-channel recurrence with key residual correction."""
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    gate = operands.pop("gv")
    beta = operands.pop("beta")
    initial_state = operands.pop("initial_state", None)
    if operands:
        raise TypeError(f"unexpected native gated_oja operands: {sorted(operands)}")
    from urm.backends.triton.k2.backward import _gated_oja_backwardable

    output, final = _gated_oja_backwardable(
        query, key, value, gate, beta, initial_state, spec.read_scale
    )
    return _native_k2_result(
        plan, output, final, "urm_native_gated_oja", backward_supported=True
    )


def _execute_native_slot_attention_two_stage(plan, torch, **operands):
    """ABC/GSA two-stage slot-addressed recurrence."""
    spec = plan.spec
    query = operands.pop("query")
    key = operands.pop("key")
    value = operands.pop("value")
    if spec.name == "abc_core":
        slot_logits = operands.pop("slot_logits").float()
        cumulative = torch.logcumsumexp(slot_logits, dim=1)
        log_decay = torch.cat((cumulative[:, :1], cumulative[:, :-1]), dim=1) - cumulative
        slot_weights = torch.exp(slot_logits - cumulative)
    else:
        slot_weights = operands.pop("slot_weights")
        log_decay = operands.pop("log_decay")
    if operands:
        raise TypeError(
            f"unexpected native slot_attention operands: {sorted(operands)}"
        )
    from urm.backends.triton.k2.backward import (
        _slot_attention_backwardable,
    )

    group_size = query.shape[2] // key.shape[2]
    output, final = _slot_attention_backwardable(
        query, key, value, slot_weights, log_decay, group_size
    )
    return _native_k2_result(
        plan, output, final, "urm_native_slot_attention_two_stage",
        backward_supported=True,
    )


_NATIVE_K2_OPERATOR_EXECUTORS = {
    RecurrenceOperator.TANH_RNN: _execute_native_tanh_rnn,
    RecurrenceOperator.GATED_RNN: _execute_native_gated_rnn,
    RecurrenceOperator.MULTIPLICATIVE_RNN: _execute_native_multiplicative_rnn,
    RecurrenceOperator.RWKV4_SCALAR_STATE: _execute_native_rwkv4_scalar_state,
    RecurrenceOperator.RWKV6_BONUS_CORRECTED: _execute_native_rwkv6_bonus_corrected,
    RecurrenceOperator.MAMBA2_STRUCTURED_SSM: _execute_native_mamba2_structured_ssm,
    RecurrenceOperator.TRAPEZOIDAL_SSM: _execute_native_trapezoidal_ssm,
    RecurrenceOperator.FFT_CONVOLUTION: _execute_native_fft_convolution,
    RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION: _execute_native_two_stage_fft_convolution,
    RecurrenceOperator.SECOND_ORDER_CUMSUM: _execute_native_second_order_cumsum,
    RecurrenceOperator.REGULARIZED_SOLVE: _execute_native_regularized_solve,
    RecurrenceOperator.LAYERNORM_INNER_STATE: _execute_native_layernorm_inner_state,
    RecurrenceOperator.MOMENTUM_INNER_STATE: _execute_native_momentum_inner_state,
    RecurrenceOperator.MOMENTUM_DELTA_STATE: _execute_native_momentum_delta,
    RecurrenceOperator.GATED_OJA_VALUE_CHANNEL: _execute_native_gated_oja,
    RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE: _execute_native_slot_attention_two_stage,
}


@lru_cache(maxsize=128)
def _compile_native_matrix_state_binding(
    spec: UnifiedMixerSpec, *, dtype: str, intent: str
) -> str:
    from urm.compiler.pipeline import CompilationIntent, ScheduleParams, UrmCompiler
    from urm.compiler.select.anchors import NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME

    compilation = UrmCompiler().compile(
        mixer_semantic_program(spec, dtype=dtype),
        intent=CompilationIntent(intent),
        schedule_params=ScheduleParams(
            anchor_overrides={"mixer": NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME}
        ),
    )
    selected = tuple(step.anchor for step in compilation.plan.steps if step.anchor)
    if selected != (NATIVE_MATRIX_STATE_RECURRENCE_ANCHOR_NAME,):
        raise RuntimeError(
            "UrmCompiler produced an invalid native matrix-state plan: "
            f"anchors={selected}"
        )
    return selected[0]


@lru_cache(maxsize=128)
def _compile_native_k3_binding(
    sparse_spec: Any,
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Compile each concrete K3 semantic shape once, then reuse its verified launch."""
    from urm.compiler.pipeline import CompilationIntent, UrmCompiler
    from urm.ir.program import sparse_state_mixer_program
    from urm.ir.k3 import sparse_state_launch_schedule

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
    "MixerResult",
    "ReadTiming",
    "RecurrentLayout",
    "StateNormalizer",
    "StateEffect",
    "StateTransition",
    "StateUpdateRule",
    "UnifiedMixerSpec",
    "compile_mixer",
    "compile_frontend_mixer",
    "mixer_semantic_program",
]
