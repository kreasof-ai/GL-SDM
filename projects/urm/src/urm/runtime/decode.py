"""Persistent-state single-token decode sessions for the native K1/K2/K3 kernels.

This is the decode/inference counterpart to the training-oriented
:meth:`CompiledMixerPlan.execute` path. The training path is built for sequence
throughput: it constructs an autograd graph, allocates the per-token state
history, and (for K3) certifies routes with GPU value scans on every call. That
is the correct way to *train*, but the wrong way to *decode*.

A decode session follows the reference decode-kernel pattern (see ATMA's
``gated_delta_decode_step``): the persistent state is allocated once and updated
in place, each step runs one fused single-token kernel under ``torch.no_grad()``,
and there is no per-step host work - no autograd graph, no state-history
allocation, and (for K3) the backend is constructed once and reused. The launch
shape is fixed, so the step is CUDA-graph capturable.

The kernel math is identical to the training recurrence; only the execution path
differs. The decode-step kernels are validated against the sequential recurrence
in the test-suite (the single-token step composed over T tokens equals the
sequence scan to fp32 precision).
"""

from __future__ import annotations

from typing import Any


class MatrixStateDecodeSession:
    """Persistent-state single-token decode session for the K2 matrix-state family.

    Holds the persistent per-head ``[B, H, K, V]`` fp32 state. Each :meth:`step`
    runs the fused in-place decode-step kernel (decay, delta/additive write,
    readout) and returns the single-token output ``[B, H, V]``.
    """

    def __init__(
        self,
        *,
        initial_state: Any,
        scale: float | None,
        decay_granularity: str,
        is_delta: bool,
        read_before: bool,
    ) -> None:
        import torch

        if initial_state.dtype is not torch.float32:
            raise ValueError("the persistent matrix state is fp32")
        if initial_state.device.type != "cuda":
            raise ValueError("decode session state must be a CUDA tensor")
        self.state = initial_state.detach().clone()
        self.scale = scale
        self.decay_granularity = decay_granularity
        self.is_delta = is_delta
        self.read_before = read_before

    def step(
        self,
        *,
        query: Any,
        key: Any,
        value: Any,
        log_decay: Any = None,
        beta: Any = None,
    ) -> Any:
        """One single-token decode step, updating the persistent state in place."""
        from urm.backends.triton.recurrence.matrix_state import (
            execute_matrix_state_decode_step,
        )

        return execute_matrix_state_decode_step(
            query=query,
            key=key,
            value=value,
            log_decay=log_decay,
            beta=beta,
            state=self.state,
            scale=self.scale,
            decay_granularity=self.decay_granularity,
            is_delta=self.is_delta,
            read_before=self.read_before,
        )


class DiagonalDecodeSession:
    """Persistent-state single-token decode session for the K2 diagonal family.

    Holds the persistent ``[B, C, N]`` fp32 state. Each :meth:`step` runs one
    fused single-token diagonal recurrence update in place and returns the
    single-token output ``[B, C]``.
    """

    def __init__(self, *, initial_state: Any, read_before: bool) -> None:
        import torch

        if initial_state.dtype is not torch.float32:
            raise ValueError("the persistent diagonal state is fp32")
        if initial_state.device.type != "cuda":
            raise ValueError("decode session state must be a CUDA tensor")
        self.state = initial_state.detach().clone()
        self.read_before = read_before

    def step(
        self,
        *,
        x: Any,
        log_decay: Any,
        input_gate: Any = None,
        read_gate: Any = None,
    ) -> Any:
        """One single-token diagonal decode step, updating the state in place."""
        from urm.backends.triton.recurrence.diagonal_recurrence import (
            execute_diagonal_decode_step,
        )

        return execute_diagonal_decode_step(
            x=x,
            log_decay=log_decay,
            input_gate=input_gate,
            read_gate=read_gate,
            state=self.state,
            read_before=self.read_before,
        )


class SparseStateDecodeSession:
    """Persistent-state single-token decode session for the K3 sparse-state family.

    Holds the persistent ``[B, S, D]`` memory and constructs the route and state
    backends once. Each :meth:`step` takes the token's factorized route *scores*,
    generates trusted routes with URM's native route-selection kernel (whose
    output is self-certifying, so it bridges into the state update *without* the
    GPU value-scan certification the untrusted path pays per call), and runs the
    fused state-update kernel in place.

    This is the decode-correct path. The training-oriented path re-certifies raw
    route tensors with GPU value scans (``.item()``/``allclose`` host syncs) on
    every call; in decode the routes come from the model's route scorer, so the
    trusted native-route path applies and the per-step GPU scans are eliminated.
    """

    def __init__(
        self,
        *,
        memory: Any,
        read_width: int,
        write_width: int,
        read_timing_before_update: bool = True,
    ) -> None:
        import torch

        if memory.device.type != "cuda":
            raise ValueError("decode session memory must be a CUDA tensor")
        self.memory = memory.detach().clone()
        self.read_width = read_width
        self.write_width = write_width
        self.read_timing_before_update = read_timing_before_update
        batch, slots, value_dim = self.memory.shape
        from urm.backends.triton.sparse_state.backend import (
            TritonSparseStateMixerBackend,
        )
        from urm.backends.triton.sparse_state.route_backend import (
            TritonSparseRouteBackend,
        )
        from urm.compiler.semantic import (
            DType,
            SparseReadTiming,
            SparseRouteSelectionSpec,
            SparseStateExecutionMode,
            SparseStateMixerSpec,
            SparseStateOperation,
        )

        dtype = DType("float32" if self.memory.dtype is torch.float32 else "bfloat16")
        self._dtype = dtype
        self._batch = batch
        self._slots = slots
        self._route_backend = None
        self._write_route_backend = None
        self._route_spec = None
        self._write_route_spec = None
        self._state_spec = SparseStateMixerSpec(
            parallel=batch, sequence=1, slots_per_partition=slots, value_dim=value_dim,
            writes=write_width, reads=read_width, dtype=dtype,
            operation=SparseStateOperation.UPDATE,
            read_timing=(
                SparseReadTiming.BEFORE_UPDATE
                if read_timing_before_update
                else SparseReadTiming.AFTER_UPDATE
            ),
            mode=SparseStateExecutionMode.INFERENCE,
        )
        self._state_backend = TritonSparseStateMixerBackend(self._state_spec)

    def step(
        self,
        *,
        read_scores: Any,
        write_scores: Any,
        values: Any,
        beta: Any,
        log_decay: Any,
    ) -> Any:
        """One single-token decode step with trusted native routes, in place.

        ``read_scores``/``write_scores`` are the factorized route-score tables
        ``[B, 1, 2*sqrt(slots)]`` produced by the model's route scorer.
        """
        from urm.backends.triton.sparse_state.backend import (
            CertifiedSparseStateRoutes,
            SparseState,
        )
        from urm.backends.triton.sparse_state.route_backend import (
            CertifiedSparseRouteScores,
            TritonSparseRouteBackend,
        )
        from urm.compiler.semantic import (
            DType,
            SparseRouteSelectionSpec,
        )

        batch = self.memory.shape[0]
        if self._route_backend is None:
            self._route_spec = SparseRouteSelectionSpec(
                parallel=batch, sequence=1, source_extent=self._slots,
                route_width=self.read_width, dtype=self._dtype,
                output_index_dtype=DType("int32"),
            )
            self._route_backend = TritonSparseRouteBackend(self._route_spec)
            self._write_route_spec = SparseRouteSelectionSpec(
                parallel=batch, sequence=1, source_extent=self._slots,
                route_width=self.write_width, dtype=self._dtype,
                output_index_dtype=DType("int32"),
            )
            self._write_route_backend = TritonSparseRouteBackend(self._write_route_spec)
        read_out = self._route_backend.generate_certified(
            CertifiedSparseRouteScores.certify(self._route_spec, read_scores)
        )
        write_out = self._write_route_backend.generate_certified(
            CertifiedSparseRouteScores.certify(self._write_route_spec, write_scores)
        )
        routes = CertifiedSparseStateRoutes.from_native_generation(
            self._state_spec, read_out, write_output=write_out
        )
        prepared = self._state_backend._prepare_generated_routes(
            routes,
            values=values,
            beta=beta.reshape(batch, 1, 1),
            log_decay=log_decay.reshape(batch, 1, 1),
        )
        state = SparseState(self.memory)
        readings, updated = self._state_backend.execute(state, prepared)
        self.memory = updated.memory
        return readings

    def step_explicit(
        self,
        *,
        read_indices: Any,
        read_weights: Any,
        write_indices: Any,
        write_weights: Any,
        values: Any,
        beta: Any,
        log_decay: Any,
    ) -> Any:
        """One decode step given explicit pre-computed routes (trusted well-formed).

        This is the apples-to-apples decode comparison against an upstream that
        takes explicit route indices: the route/score generation is excluded
        (the matrix scope excludes it), so both paths receive the same routes and
        only the state update is measured. The routes are certified via the
        trusted path (no GPU value scans); the caller must guarantee they are
        well-formed (in-bounds, strictly increasing and unique within each token,
        finite nonnegative normalized weights).
        """
        from urm.backends.triton.sparse_state.backend import (
            CertifiedSparseStateRoutes,
            SparseState,
        )

        batch = self.memory.shape[0]
        routes = CertifiedSparseStateRoutes.certify_trusted(
            self._state_spec,
            read_indices,
            read_weights,
            write_indices=write_indices,
            write_weights=write_weights,
        )
        prepared = self._state_backend._prepare_generated_routes(
            routes,
            values=values,
            beta=beta.reshape(batch, 1, 1),
            log_decay=log_decay.reshape(batch, 1, 1),
        )
        state = SparseState(self.memory)
        readings, updated = self._state_backend.execute(state, prepared)
        self.memory = updated.memory
        return readings


__all__ = [
    "MatrixStateDecodeSession",
    "DiagonalDecodeSession",
    "SparseStateDecodeSession",
]
