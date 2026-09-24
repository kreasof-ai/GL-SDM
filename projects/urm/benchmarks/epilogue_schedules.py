"""Schedule-parameterized launches backed by the production backend.

This module is a thin adapter over
``src/urm/backends/providers/k1/row_scale.py``. It exists so the
benchmark grid can address kernels by :class:`SchedulePoint`; the kernels
themselves live only in the production source tree, so a measured schedule
can never disagree with what the compiled backend actually executes.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from urm.backends.historical.triton_k1_row_scale import (
    RoutedEpilogueLaunchConfig,
    execute_plan_step,
)
from urm.backends.historical.triton_k1_row_scale import (
    launch_backward as _production_backward,
)
from urm.backends.historical.triton_k1_row_scale import (
    launch_forward as _production_forward,
)

__all__ = [
    "RoutedEpilogueLaunchConfig",
    "backward_launch",
    "compile_feedback_for",
    "execute_plan_step",
    "forward_launch",
    "launch_prepared_step",
    "make_inputs",
    "prepare_plan_step",
]


def _config_from(point) -> RoutedEpilogueLaunchConfig:
    """Adapt any schedule-like object/dict to the production config type."""
    if isinstance(point, RoutedEpilogueLaunchConfig):
        return point
    if isinstance(point, Mapping):
        return RoutedEpilogueLaunchConfig.from_dict(point)
    return RoutedEpilogueLaunchConfig.from_point(point)


def prepare_plan_step(payload: Mapping[str, object]) -> RoutedEpilogueLaunchConfig:
    """Decode a serialized plan step's launch configuration (compile time)."""
    return RoutedEpilogueLaunchConfig.from_dict(payload)


def launch_prepared_step(
    config: RoutedEpilogueLaunchConfig, indices, weights, values, row_scale
):
    """Steady-state dispatch of an already-decoded plan configuration.

    This is the production forward launcher; the compiled plan adds no
    per-launch work beyond calling it.
    """
    output, info = _production_forward(config, indices, weights, values, row_scale)
    return output, info


def forward_launch(point, indices, weights, values, row_scale):
    """Production forward launch for one schedule point."""
    output, info = _production_forward(
        _config_from(point), indices, weights, values, row_scale
    )
    return output, info


def backward_launch(point, indices, weights, values, row_scale, grad_output):
    """Production backward launch for one schedule point."""
    grads, info = _production_backward(
        _config_from(point), indices, weights, values, row_scale, grad_output
    )
    return grads, info


def compile_feedback_for(handle) -> dict[str, int | None]:
    """Best-effort register/shared-memory metadata from a compiled kernel."""
    from urm.backends.historical.triton_k1_row_scale import (
        _extract_resource_usage,
    )

    kres = _extract_resource_usage("kernel", handle)
    return {
        "registers_per_thread": kres.registers_per_thread,
        "shared_mem_bytes": kres.shared_mem_bytes,
    }


def make_inputs(queries, route_width, sources, value_dim, dtype_name, seed=7):
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    indices = torch.randint(
        0,
        sources,
        (queries, route_width),
        device="cuda",
        dtype=torch.int64,
        generator=generator,
    )
    weights = torch.randn(
        (queries, route_width), device="cuda", dtype=dtype, generator=generator
    )
    values = torch.randn(
        (sources, value_dim), device="cuda", dtype=dtype, generator=generator
    )
    row_scale = torch.randn((queries,), device="cuda", dtype=dtype, generator=generator)
    return indices, weights, values, row_scale
