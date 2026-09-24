"""Exact Triton K2 compile probes.

The exact compiled specializations are probed only after legal analytical
pruning. This module lives on the *compiler* side: it imports the backend's
production launchers (``urm.backends.triton.k2``) and drives them against the
exact target specialization, collecting register/shared-memory resource facts
into the compiler's :class:`KernelResourceUsage`/:class:`CompileProbeResult`
types. The backend never imports this module or any compiler search type.
"""

from __future__ import annotations

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


def _collect_compiled(
    module: object, device_index: int, *, family: str
) -> dict[str, KernelResourceUsage]:
    """Read resource facts from every compiled JITFunction the production path filled.

    Kernels are either module-level JITFunctions (K3) or returned by the
    backend's cached kernel factory ``_kernels()`` (K2); both are covered.
    """
    candidates: list[tuple[str, object]] = []
    for attr in dir(module):
        obj = getattr(module, attr)
        if type(obj).__name__ == "JITFunction":
            candidates.append((attr, obj))
    factory = getattr(module, "_kernels", None)
    if callable(factory):
        try:
            produced = factory()
        except Exception:  # noqa: BLE001 - factory failure surfaces via ok=False later
            produced = ()
        if not isinstance(produced, tuple):
            produced = (produced,)
        for obj in produced:
            if type(obj).__name__ == "JITFunction":
                candidates.append((getattr(obj, "__name__", "kernel"), obj))
    resources: dict[str, KernelResourceUsage] = {}
    for name, fn in candidates:
        caches = getattr(fn, "device_caches", None)
        if caches is None:
            continue
        cache = caches.get(device_index)
        if cache is None:
            continue
        entries = cache[0] if isinstance(cache, tuple) else cache
        for handle in entries.values():
            resources[f"{family}:{name}"] = _extract_resource_usage(name, handle)
    return resources


def make_triton_k2_compile_probe(
    *,
    batch: int = 1,
    sequence: int = 8,
    channels: int = 32,
    state_width: int = 8,
    heads: int = 2,
    key_dim: int = 16,
    value_dim: int = 16,
    dtype_name: str = "float32",
) -> CompileProbe:
    """Real GPU compile probe over the EXACT K2 target specialization.

    Probes compile + launch the production diagonal-SSM and matrix-state
    recurrence kernels for the requested specialization (channel/state widths,
    head counts, operand dtype), exercising forward and backward when intent is
    training.

    Note on runtime extents: batch/sequence are runtime tensor dimensions, not
    compile-time Triton specialization constants. Using bounded representative
    extents avoids excessive probe latency while compiling and running the
    identical specialized kernels. Register/shared-memory facts flow back from
    the compiled kernel cache.
    """
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - guarded by callers
        raise RuntimeError("make_triton_k2_compile_probe requires CUDA")
    device = torch.device("cuda")

    def probe(context) -> CompileProbeResult:
        try:
            is_training = context.intent == "training"
            dtype = getattr(torch, dtype_name)
            from urm.backends.triton.k2 import diagonal as diagonal_mod
            from urm.backends.triton.k2 import matrix as matrix_mod

            generator = torch.Generator(device=device).manual_seed(11)
            resources: dict[str, KernelResourceUsage] = {}

            def randn(*shape, requires_grad=False):
                tensor = torch.randn(
                    *shape, device=device, dtype=dtype, generator=generator
                )
                return tensor.requires_grad_() if requires_grad else tensor

            # Diagonal SSM recurrence (exact native diagonal kernel).
            x = randn(batch, sequence, channels, requires_grad=is_training)
            input_gate = randn(batch, sequence, channels, state_width, requires_grad=is_training)
            read_gate = randn(batch, sequence, channels, state_width, requires_grad=is_training)
            log_decay = -torch.rand(
                (batch, sequence, channels, state_width),
                device=device,
                generator=generator,
            ).to(dtype)
            if is_training:
                log_decay = log_decay.requires_grad_()
            output, final_state = diagonal_mod.execute_diagonal_recurrence(
                x=x,
                input_gate=input_gate,
                read_gate=read_gate,
                log_decay=log_decay,
                initial_state=None,
                step_size=None,
                skip=0.0,
                read_before=False,
            )
            if is_training:
                (output.float().square().mean() + final_state.float().square().mean()).backward()
            torch.cuda.synchronize()

            # Matrix-state recurrence (exact native matrix kernel).
            query = randn(batch, sequence, heads, key_dim, requires_grad=is_training)
            key = randn(batch, sequence, heads, key_dim, requires_grad=is_training)
            value = randn(batch, sequence, heads, value_dim, requires_grad=is_training)
            m_log_decay = -torch.rand(
                (batch, sequence, heads), device=device, generator=generator
            ).to(dtype)
            beta = torch.rand(
                (batch, sequence, heads), device=device, generator=generator
            ).to(dtype)
            if is_training:
                m_log_decay = m_log_decay.requires_grad_()
                beta = beta.requires_grad_()
            m_output, m_final = matrix_mod.execute_matrix_state_recurrence(
                query=query,
                key=key,
                value=value,
                log_decay=m_log_decay,
                beta=beta,
                initial_state=None,
                scale=key_dim**-0.5,
                decay_granularity="head",
                is_delta=True,
                read_before=False,
            )
            if is_training:
                (m_output.float().square().mean() + m_final.float().square().mean()).backward()
            torch.cuda.synchronize()

            for module, family in ((diagonal_mod, "diagonal"), (matrix_mod, "matrix")):
                resources.update(
                    _collect_compiled(module, device.index or 0, family=family)
                )

            known_regs = [
                k.registers_per_thread
                for k in resources.values()
                if k.registers_per_thread is not None
            ]
            known_shared = [
                k.shared_mem_bytes
                for k in resources.values()
                if k.shared_mem_bytes is not None
            ]
            return CompileProbeResult(
                ok=True,
                registers_per_thread=max(known_regs) if known_regs else None,
                shared_mem_bytes=max(known_shared) if known_shared else None,
                kernel_resources=resources,
            )
        except Exception as error:  # noqa: BLE001 - probe failures ARE results
            return CompileProbeResult(ok=False, reason=str(error)[:200])

    return probe


__all__ = ["make_triton_k2_compile_probe"]
