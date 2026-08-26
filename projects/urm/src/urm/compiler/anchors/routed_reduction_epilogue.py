"""Experimental fused row-scale epilogue for routed weighted reduction.

Semantic expression (after the verified rewrite
``fold_row_scale_into_routed_reduction_epilogue``):

    output[q, d] = row_scale[q] * sum_k weights[q, k] * values[indices[q, k], d]

The per-row scale executes inside the routed-reduction mainloop's epilogue:
the fp32 accumulator is scaled while the tile is still resident, so ``base``
is never materialized as an externally visible tensor.

Contract notes:

- v1 is untouched: this module is a separate experimental anchor selected only
  when the planner requests the typed ``FINAL_SCALE_CONVERT`` visitor.
- Forward equivalence is proven against explicit references in tests.
- Backward covers ALL inputs - weights, values AND row_scale. The row-scale
  gradient needs un-scaled tiles, which are recomputed here (per the rewrite's
  ``recompute`` saved-state obligation) instead of materializing base.
- Accumulation stays fp32; outputs and gradients cast to input dtypes.

Schedule ownership: :class:`RoutedEpilogueLaunchConfig` is the SINGLE source
of truth for this anchor's execution configuration. Every field is honored by
the launchers below - the solver may only select fields that visibly reach
the Triton launch. ``benchmarks/epilogue_schedules.py`` calls these exact
implementations; there is no second set of kernels.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from urm.compiler.schedule_space import SchedulePoint
    from urm.compiler.search import CompileProbe, CompileProbeResult

ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION = 2


# -- Execution configuration ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutedEpilogueLaunchConfig:
    """The one executable configuration of the fused row-scale anchor.

    Every field maps onto a concrete launch parameter below; unsupported
    values are rejected here instead of being silently ignored. This type is
    what the compiler's verified :class:`ScheduleDecision` lowers into.
    """

    block_d: int
    num_warps: int
    num_stages: int
    grad_values_decomposition: str = "per_query"  # "per_query" | "per_route"
    grad_values_schedule: str = "segmented"  # "full_row" | "segmented"

    def __post_init__(self) -> None:
        from urm.compiler.schedule_space import (
            SUPPORTED_BLOCKS,
            SUPPORTED_STAGES,
            SUPPORTED_WARPS,
            GradValuesDecomposition,
            GradValuesSchedule,
        )

        if self.block_d not in SUPPORTED_BLOCKS:
            raise ValueError(
                f"block_d={self.block_d} not implemented (implemented: "
                f"{list(SUPPORTED_BLOCKS)})"
            )
        if self.num_warps not in SUPPORTED_WARPS:
            raise ValueError(
                f"num_warps={self.num_warps} not implemented "
                f"(implemented: {list(SUPPORTED_WARPS)})"
            )
        if self.num_stages not in SUPPORTED_STAGES:
            raise ValueError(
                f"num_stages={self.num_stages} not implemented "
                f"(implemented: {list(SUPPORTED_STAGES)})"
            )
        decompositions = {d.value for d in GradValuesDecomposition}
        if self.grad_values_decomposition not in decompositions:
            raise ValueError(
                f"grad_values_decomposition={self.grad_values_decomposition!r} "
                f"not implemented (implemented: {sorted(decompositions)})"
            )
        schedules = {s.value for s in GradValuesSchedule}
        if self.grad_values_schedule not in schedules:
            raise ValueError(
                f"grad_values_schedule={self.grad_values_schedule!r} not "
                f"implemented (implemented: {sorted(schedules)})"
            )

    @classmethod
    def from_point(cls, point: SchedulePoint) -> RoutedEpilogueLaunchConfig:
        """Build the executable config for a verified schedule point."""
        return cls(
            block_d=point.block_d,
            num_warps=point.num_warps,
            num_stages=point.num_stages,
            grad_values_decomposition=point.grad_values_decomposition,
            grad_values_schedule=point.grad_values_schedule,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RoutedEpilogueLaunchConfig:
        """Rebuild from a serialized launch configuration."""
        return cls(
            block_d=int(payload["block_d"]),
            num_warps=int(payload["num_warps"]),
            num_stages=int(payload["num_stages"]),
            grad_values_decomposition=str(payload["grad_values_decomposition"]),
            grad_values_schedule=str(payload["grad_values_schedule"]),
        )

    def to_dict(self) -> dict[str, str | int]:
        return {
            "block_d": self.block_d,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "grad_values_decomposition": self.grad_values_decomposition,
            "grad_values_schedule": self.grad_values_schedule,
        }


# -- Triton kernels ---------------------------------------------------------------


@triton.jit
def _rrs_forward_kernel(
    indices,
    weights,
    values,
    row_scale,
    output,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    scale = tl.load(row_scale + query).to(tl.float32)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
        route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(
            tl.float32
        )
        if EVEN_D:
            gathered = tl.load(values + source_row * VALUE_DIM + dimension_offsets).to(
                tl.float32
            )
        else:
            gathered = tl.load(
                values + source_row * VALUE_DIM + dimension_offsets,
                mask=dimension_offsets < VALUE_DIM,
                other=0.0,
            ).to(tl.float32)
        accumulator += route_weight * gathered
    # Typed epilogue: scale while the tile is resident; base never escapes.
    accumulator *= scale
    if EVEN_D:
        tl.store(output + query * VALUE_DIM + dimension_offsets, accumulator)
    else:
        tl.store(
            output + query * VALUE_DIM + dimension_offsets,
            accumulator,
            mask=dimension_offsets < VALUE_DIM,
        )


@triton.jit
def _rrs_forward_pipelined_kernel(
    indices,
    weights,
    values,
    row_scale,
    output,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """Explicitly pipelined forward variant (schedule ``num_stages > 1``)."""
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for route_offset in tl.range(0, ROUTE_WIDTH, num_stages=NUM_STAGES):
        source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
        route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(
            tl.float32
        )
        gathered = tl.load(
            values + source_row * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += route_weight * gathered
    accumulator *= scale
    tl.store(output + query * VALUE_DIM + dimension_offsets, accumulator, mask=mask)


@triton.jit
def _rrs_grad_weights_kernel(
    indices,
    values,
    row_scale,
    grad_output,
    grad_weights,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0)
    query = route // ROUTE_WIDTH
    route_offset = route % ROUTE_WIDTH
    source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
    scale = tl.load(row_scale + query).to(tl.float32)
    accumulator = tl.zeros((), dtype=tl.float32)
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        source_values = tl.load(
            values + source_row * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(source_values * (scale * output_gradient), axis=0)
    tl.store(grad_weights + route, accumulator)


@triton.jit
def _rrs_grad_values_per_query_segmented_kernel(
    indices,
    weights,
    row_scale,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-query decomposition; the D axis is segmented across programs."""
    query = tl.program_id(0)
    dimension_block = tl.program_id(1)
    dimension_offsets = dimension_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    output_gradient = tl.load(
        grad_output + query * VALUE_DIM + dimension_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    route_base = query * ROUTE_WIDTH
    for route_offset in tl.range(ROUTE_WIDTH):
        source_row = tl.load(indices + route_base + route_offset)
        route_weight = tl.load(weights + route_base + route_offset).to(tl.float32)
        contribution = route_weight * scale * output_gradient
        tl.atomic_add(
            grad_values + source_row * VALUE_DIM + dimension_offsets,
            contribution,
            mask=mask,
            sem="relaxed",
        )


@triton.jit
def _rrs_grad_values_per_query_fullrow_kernel(
    indices,
    weights,
    row_scale,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-query decomposition; one program walks the whole D range."""
    query = tl.program_id(0)
    scale = tl.load(row_scale + query).to(tl.float32)
    route_base = query * ROUTE_WIDTH
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        for route_offset in tl.range(ROUTE_WIDTH):
            source_row = tl.load(indices + route_base + route_offset)
            route_weight = tl.load(weights + route_base + route_offset).to(tl.float32)
            contribution = route_weight * scale * output_gradient
            tl.atomic_add(
                grad_values + source_row * VALUE_DIM + dimension_offsets,
                contribution,
                mask=mask,
                sem="relaxed",
            )


@triton.jit
def _rrs_grad_values_per_route_kernel(
    indices,
    weights,
    row_scale,
    grad_output,
    grad_values,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Per-route decomposition: grid is (Q*ROUTE_WIDTH, cdiv(D, BLOCK_D))."""
    program_id = tl.program_id(0)
    query = program_id // ROUTE_WIDTH
    route_offset = program_id % ROUTE_WIDTH
    dimension_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dimension_offsets < VALUE_DIM
    scale = tl.load(row_scale + query).to(tl.float32)
    source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
    route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(tl.float32)
    output_gradient = tl.load(
        grad_output + query * VALUE_DIM + dimension_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    tl.atomic_add(
        grad_values + source_row * VALUE_DIM + dimension_offsets,
        route_weight * scale * output_gradient,
        mask=mask,
        sem="relaxed",
    )


@triton.jit
def _rrs_grad_row_scale_kernel(
    indices,
    weights,
    values,
    grad_output,
    grad_row_scale,
    ROUTE_WIDTH: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """d(row_scale[q]) = sum_d grad_out[q, d] * base[q, d].

    base tiles are recomputed from (indices, weights, values) exactly as in
    the forward mainloop; they are never read back from memory.
    """
    query = tl.program_id(0)
    accumulator = tl.zeros((), dtype=tl.float32)
    for dimension_start in tl.range(0, VALUE_DIM, BLOCK_D):
        dimension_offsets = dimension_start + tl.arange(0, BLOCK_D)
        mask = dimension_offsets < VALUE_DIM
        inner = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for route_offset in tl.range(ROUTE_WIDTH):
            source_row = tl.load(indices + query * ROUTE_WIDTH + route_offset)
            route_weight = tl.load(weights + query * ROUTE_WIDTH + route_offset).to(
                tl.float32
            )
            gathered = tl.load(
                values + source_row * VALUE_DIM + dimension_offsets,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            inner += route_weight * gathered
        output_gradient = tl.load(
            grad_output + query * VALUE_DIM + dimension_offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(inner * output_gradient, axis=0)
    tl.store(grad_row_scale + query, accumulator)


# -- Launch metadata -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutedEpilogueLaunchInfo:
    """Metadata of one actual launch (serializable subset)."""

    kernel: str
    grid: tuple[int, ...]
    block_d: int
    num_warps: int
    num_stages: int | None = None
    # Live Triton CompiledKernel handle (register/shared-memory feedback);
    # deliberately excluded from serialization.
    handle: object = None
    extra_handles: tuple[tuple[str, object], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "grid": list(self.grid),
            "block_d": self.block_d,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


# -- Heuristics (default path; unchanged tuning) --------------------------------------


def _forward_launch(value_dim: int, queries: int) -> tuple[int, int]:
    """Mirror of the tuned v1 forward heuristic; owned by this prototype so
    future v1 retuning cannot silently change experimental numbers."""
    block_d = min(256, max(32, triton.next_power_of_2(min(value_dim, 256))))
    num_warps = 2 if queries >= 1024 else 4
    return block_d, num_warps


def _grad_launch(value_dim: int) -> tuple[int, int]:
    npd = triton.next_power_of_2(value_dim)
    if value_dim >= 128:
        return max(32, min(256, npd)), 4
    return max(32, min(256, npd)), 4


def _default_config(value_dim: int, queries: int) -> RoutedEpilogueLaunchConfig:
    block_d, num_warps = _forward_launch(value_dim, queries)
    return RoutedEpilogueLaunchConfig(
        block_d=block_d,
        num_warps=num_warps,
        num_stages=1,
        grad_values_decomposition="per_query",
        grad_values_schedule="segmented",
    )


# -- Launchers ------------------------------------------------------------------------


def launch_forward(
    config: RoutedEpilogueLaunchConfig,
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    row_scale: torch.Tensor,
) -> tuple[torch.Tensor, RoutedEpilogueLaunchInfo]:
    """Forward launch honoring every field of ``config``."""
    queries, route_width = indices.shape
    value_dim = values.shape[1]
    output = torch.empty((queries, value_dim), device=values.device, dtype=values.dtype)
    grid = (queries, triton.cdiv(value_dim, config.block_d))
    if config.num_stages == 1:
        handle = _rrs_forward_kernel[grid](
            indices,
            weights,
            values,
            row_scale,
            output,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=config.block_d,
            EVEN_D=value_dim % config.block_d == 0,
            num_warps=config.num_warps,
        )
        kernel_name = "_rrs_forward_kernel"
        stages = None
    else:
        handle = _rrs_forward_pipelined_kernel[grid](
            indices,
            weights,
            values,
            row_scale,
            output,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=config.block_d,
            NUM_STAGES=config.num_stages,
            num_warps=config.num_warps,
        )
        kernel_name = "_rrs_forward_pipelined_kernel"
        stages = config.num_stages
    info = RoutedEpilogueLaunchInfo(
        kernel=kernel_name,
        grid=grid,
        block_d=config.block_d,
        num_warps=config.num_warps,
        num_stages=stages,
        handle=handle,
    )
    return output, info


def launch_backward(
    config: RoutedEpilogueLaunchConfig,
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    row_scale: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], RoutedEpilogueLaunchInfo]:
    """Backward launch honoring the config's decomposition/schedule fields."""
    queries, route_width = indices.shape
    sources, value_dim = values.shape
    grad_output = grad_output.contiguous()
    grad_weights = torch.empty(
        weights.shape, device=weights.device, dtype=torch.float32
    )
    gw_handle = _rrs_grad_weights_kernel[(queries * route_width,)](
        indices,
        values,
        row_scale,
        grad_output,
        grad_weights,
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=config.block_d,
        num_warps=config.num_warps,
    )

    grad_values = torch.zeros(
        (sources, value_dim), device=values.device, dtype=torch.float32
    )
    if config.grad_values_decomposition == "per_query":
        if config.grad_values_schedule == "segmented":
            gv_grid = (queries, triton.cdiv(value_dim, config.block_d))
            gv_handle = _rrs_grad_values_per_query_segmented_kernel[gv_grid](
                indices,
                weights,
                row_scale,
                grad_output,
                grad_values,
                ROUTE_WIDTH=route_width,
                VALUE_DIM=value_dim,
                BLOCK_D=config.block_d,
                num_warps=config.num_warps,
            )
            gv_kernel = "_rrs_grad_values_per_query_segmented_kernel"
        else:
            gv_grid = (queries,)
            gv_handle = _rrs_grad_values_per_query_fullrow_kernel[gv_grid](
                indices,
                weights,
                row_scale,
                grad_output,
                grad_values,
                ROUTE_WIDTH=route_width,
                VALUE_DIM=value_dim,
                BLOCK_D=config.block_d,
                num_warps=config.num_warps,
            )
            gv_kernel = "_rrs_grad_values_per_query_fullrow_kernel"
    else:
        gv_grid = (queries * route_width, triton.cdiv(value_dim, config.block_d))
        gv_handle = _rrs_grad_values_per_route_kernel[gv_grid](
            indices,
            weights,
            row_scale,
            grad_output,
            grad_values,
            ROUTE_WIDTH=route_width,
            VALUE_DIM=value_dim,
            BLOCK_D=config.block_d,
            num_warps=config.num_warps,
        )
        gv_kernel = "_rrs_grad_values_per_route_kernel"

    grad_scale = torch.empty(
        row_scale.shape, device=row_scale.device, dtype=torch.float32
    )
    gs_handle = _rrs_grad_row_scale_kernel[(queries,)](
        indices,
        weights,
        values,
        grad_output,
        grad_scale,
        ROUTE_WIDTH=route_width,
        VALUE_DIM=value_dim,
        BLOCK_D=config.block_d,
        num_warps=config.num_warps,
    )
    info = RoutedEpilogueLaunchInfo(
        kernel=gv_kernel,
        grid=gv_grid,
        block_d=config.block_d,
        num_warps=config.num_warps,
        num_stages=None,
        handle=gv_handle,
        extra_handles=(
            ("_rrs_grad_weights_kernel", gw_handle),
            (gv_kernel, gv_handle),
            ("_rrs_grad_row_scale_kernel", gs_handle),
        ),
    )
    return (grad_weights, grad_values, grad_scale), info


# -- Autograd wrapper ------------------------------------------------------------------


class _RoutedReduceRowScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weights, values, row_scale, config=None):
        ctx.config = config
        ctx.save_for_backward(indices, weights, values, row_scale)
        effective = (
            config
            if config is not None
            else _default_config(values.shape[1], indices.shape[0])
        )
        output, _info = launch_forward(effective, indices, weights, values, row_scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        indices, weights, values, row_scale = ctx.saved_tensors
        effective = (
            ctx.config
            if ctx.config is not None
            else _default_config(values.shape[1], indices.shape[0])
        )
        (gw, gv, gs), _info = launch_backward(
            effective, indices, weights, values, row_scale, grad_output
        )
        return (
            None,
            gw.to(weights.dtype),
            gv.to(values.dtype),
            gs.to(row_scale.dtype),
            None,
        )


def routed_reduce_row_scale(
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    row_scale: torch.Tensor,
    config: RoutedEpilogueLaunchConfig | None = None,
) -> torch.Tensor:
    """Fused routed reduction with a typed row-scale epilogue.

    ``config=None`` keeps the historical internal heuristic (unchanged
    tuning); an explicit :class:`RoutedEpilogueLaunchConfig` - e.g. lowered
    from a verified :class:`~urm.compiler.search.ScheduleDecision` - drives
    every launch parameter visibly.
    """
    if torch.is_grad_enabled() and (
        indices.requires_grad
        or weights.requires_grad
        or values.requires_grad
        or row_scale.requires_grad
    ):
        return _RoutedReduceRowScale.apply(indices, weights, values, row_scale, config)
    effective = (
        config
        if config is not None
        else _default_config(values.shape[1], indices.shape[0])
    )
    output, _info = launch_forward(effective, indices, weights, values, row_scale)
    return output


def execute_plan_step(
    payload: Mapping[str, object],
    indices: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    row_scale: torch.Tensor,
    *,
    backward: bool = False,
    grad_output: torch.Tensor | None = None,
) -> object:
    """Execute a serialized plan step's launch configuration.

    This is the dispatch path a compiled :class:`ExecutablePlan` drives: the
    step's ``launch_config`` dictionary (serialized by the compiler) is
    rebuilt into :class:`RoutedEpilogueLaunchConfig` and launched through the
    same production launchers - compilation stays outside any timed region.
    Requires the fused plan kind.
    """
    plan_kind = str(payload.get("plan", "fused"))
    if plan_kind != "fused":
        raise ValueError(
            f"execute_plan_step lowers the fused plan only (got {plan_kind!r})"
        )
    config = RoutedEpilogueLaunchConfig.from_dict(payload)
    if backward:
        assert grad_output is not None
        grads, info = launch_backward(
            config, indices, weights, values, row_scale, grad_output
        )
        return grads, info
    return launch_forward(config, indices, weights, values, row_scale)


def routed_reduce_row_scale_metadata(
    route_width: int, value_dim: int, config: RoutedEpilogueLaunchConfig | None = None
) -> dict[str, int | str]:
    effective = config or _default_config(value_dim, queries=1)
    payload = {
        "anchor_version": ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION,
        "routes_per_program": route_width,
        **effective.to_dict(),
        "epilogue": "row_scale",
    }
    return payload


# -- Compile probing ---------------------------------------------------------------------


def _extract_resource_usage(kernel_name: str, handle: object):
    from urm.compiler.search import KernelResourceUsage

    if handle is None:
        return KernelResourceUsage(
            kernel_name=kernel_name,
            unavailable_reason="compiled_handle_unavailable",
        )
    regs = getattr(handle, "n_regs", None)
    spills = getattr(handle, "n_spills", None)
    shared = None
    if hasattr(handle, "metadata") and hasattr(handle.metadata, "shared"):
        try:
            shared = int(handle.metadata.shared)
        except (TypeError, ValueError):
            shared = None
    unavailable = None
    if regs is None and shared is None:
        unavailable = "triton_handle_exposed_no_resource_metadata"
    return KernelResourceUsage(
        kernel_name=kernel_name,
        registers_per_thread=int(regs) if regs is not None else None,
        shared_mem_bytes=shared,
        spill_bytes=int(spills) if spills is not None else None,
        unavailable_reason=unavailable,
    )


def make_triton_compile_probe(
    *,
    queries: int = 4,
    route_width: int = 2,
    sources: int = 8,
    value_dim: int = 64,
    dtype_name: str = "float32",
) -> CompileProbe:
    """Real GPU compile probe over the EXACT target specialization.

    Probes compile + launch the production kernels for the requested anchor,
    shapes, dtypes, and launch configurations (BLOCK_D, num_warps, num_stages,
    decomposition, traversal), exercising forward and backward when intent is
    training. Register/shared-memory facts flow back from the compiled handles.
    """
    if not torch.cuda.is_available():  # pragma: no cover - guarded by callers
        raise RuntimeError("make_triton_compile_probe requires CUDA")
    device = torch.device("cuda")

    def probe(context) -> CompileProbeResult:
        from urm.compiler.schedule_space import SchedulePoint
        from urm.compiler.search import CompileProbeResult

        try:
            if isinstance(context, SchedulePoint):
                point = context
                effective_anchor = "routed_reduction_row_scale_epilogue_v0"
                eff_queries = queries
                eff_sources = sources
                eff_route_width = route_width
                eff_value_dim = value_dim
                eff_dtype_name = dtype_name
                is_training = True
            else:
                point = context.schedule_point
                effective_anchor = context.anchor_name
                eff_queries = min(context.queries, 4) if context.queries > 0 else 4
                eff_sources = max(min(context.sources, 8), context.route_width)
                eff_route_width = context.route_width
                eff_value_dim = context.value_dim
                eff_dtype_name = context.dtype
                is_training = context.intent == "training"

            dtype = getattr(torch, eff_dtype_name)
            generator = torch.Generator(device=device).manual_seed(11)
            indices = torch.randint(
                0,
                eff_sources,
                (eff_queries, eff_route_width),
                device=device,
                generator=generator,
            )
            weights = torch.randn(
                (eff_queries, eff_route_width), device=device, dtype=dtype
            )
            values = torch.randn(
                (eff_sources, eff_value_dim), device=device, dtype=dtype
            )

            if effective_anchor == "routed_reduction_row_scale_epilogue_v0":
                row_scale = torch.randn((eff_queries,), device=device, dtype=dtype)
                config = RoutedEpilogueLaunchConfig.from_point(point)
                output, fwd_info = launch_forward(
                    config, indices, weights, values, row_scale
                )
                torch.cuda.synchronize()
                fwd_res = _extract_resource_usage(fwd_info.kernel, fwd_info.handle)
                resources = {"forward": fwd_res}
                max_regs = fwd_res.registers_per_thread
                max_shared = fwd_res.shared_mem_bytes or 0

                if is_training:
                    grad_output = torch.randn(
                        (eff_queries, eff_value_dim), device=device, dtype=dtype
                    )
                    (_gw, _gv, _gs), bwd_info = launch_backward(
                        config, indices, weights, values, row_scale, grad_output
                    )
                    torch.cuda.synchronize()
                    for name, handle in bwd_info.extra_handles:
                        kres = _extract_resource_usage(name, handle)
                        tag = (
                            "grad_weights"
                            if "weights" in name
                            else (
                                "grad_scale"
                                if "scale" in name or "row" in name
                                else "grad_values"
                            )
                        )
                        resources[tag] = kres
                        if kres.registers_per_thread is not None:
                            max_regs = (
                                max(max_regs, kres.registers_per_thread)
                                if max_regs is not None
                                else kres.registers_per_thread
                            )
                        if kres.shared_mem_bytes is not None:
                            max_shared = max(max_shared, kres.shared_mem_bytes)

                del output
                return CompileProbeResult(
                    ok=True,
                    registers_per_thread=max_regs,
                    shared_mem_bytes=max_shared,
                    kernel_resources=resources,
                )

            if effective_anchor == "routed_reduction_v1":
                from urm.triton_kernels.routed_reduce import routed_reduce

                output = routed_reduce(indices, weights, values)
                torch.cuda.synchronize()
                del output
                return CompileProbeResult(ok=True)

            return CompileProbeResult(
                ok=False,
                reason=f"unsupported probe anchor {effective_anchor!r}",
            )
        except Exception as error:  # noqa: BLE001 - probe failures ARE results
            return CompileProbeResult(ok=False, reason=str(error)[:200])

    return probe


__all__ = [
    "ROUTED_REDUCTION_ROW_SCALE_EPILOGUE_VERSION",
    "RoutedEpilogueLaunchConfig",
    "RoutedEpilogueLaunchInfo",
    "execute_plan_step",
    "launch_backward",
    "launch_forward",
    "make_triton_compile_probe",
    "routed_reduce_row_scale",
    "routed_reduce_row_scale_metadata",
]
