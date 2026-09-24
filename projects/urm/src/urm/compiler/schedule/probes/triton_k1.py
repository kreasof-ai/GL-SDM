"""Exact Triton K1 compile probes.

The exact compiled specializations are probed only after legal analytical
pruning. This module lives on the *compiler* side: it imports the backend's
production launchers (``urm.backends.triton.k1``) and drives them against the
exact target specialization, collecting register/shared-memory resource facts
into the compiler's :class:`KernelResourceUsage`/:class:`CompileProbeResult`
types. The backend never imports this module or any compiler search type.
"""

from __future__ import annotations

from urm.backends.triton.k1.row_scale import (
    RoutedEpilogueLaunchConfig,
    launch_backward,
    launch_forward,
)
from urm.compiler.schedule.search import (
    CompileProbe,
    CompileProbeResult,
    KernelResourceUsage,
)


def _extract_resource_usage(kernel_name: str, handle: object) -> KernelResourceUsage:
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
    exact specialization parameters (operand dtypes, route width, value
    dimension, launch configuration: BLOCK_D, num_warps, num_stages,
    decomposition, traversal), exercising forward and backward when intent is
    training.

    Note on runtime extents: Q (queries) and S (sources) are runtime tensor
    dimensions, not compile-time Triton specialization constants. Using
    bounded representative extents for Q and S avoids excessive probe latency
    while still compiling and running the identical specialized kernels.
    Register/shared-memory facts flow back from the compiled handles.
    """
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - guarded by callers
        raise RuntimeError("make_triton_compile_probe requires CUDA")
    device = torch.device("cuda")

    def probe(context) -> CompileProbeResult:
        try:
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
                        if "weights" in name:
                            tag = "grad_weights"
                        elif "values" in name:
                            tag = "grad_values"
                        elif "scale" in name or "row" in name:
                            tag = "grad_row_scale"
                        else:
                            tag = name
                        resources[tag] = kres

                del output

                known_regs = [
                    k.registers_per_thread
                    for k in resources.values()
                    if k.registers_per_thread is not None
                ]
                max_regs = max(known_regs) if known_regs else None

                known_shared = [
                    k.shared_mem_bytes
                    for k in resources.values()
                    if k.shared_mem_bytes is not None
                ]
                max_shared = max(known_shared) if known_shared else None

                return CompileProbeResult(
                    ok=True,
                    registers_per_thread=max_regs,
                    shared_mem_bytes=max_shared,
                    kernel_resources=resources,
                )

            if effective_anchor == "routed_reduction_v1":
                from urm.backends.triton.k1.routed_reduce import routed_reduce

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


__all__ = ["make_triton_compile_probe"]
