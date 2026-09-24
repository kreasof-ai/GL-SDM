"""Exact Triton K3 compile probes.

The exact compiled specializations are probed only after legal analytical
pruning. This module lives on the *compiler* side: it imports the backend's
production launchers (``urm.backends.triton.k3``) and drives them against the
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


def make_triton_k3_compile_probe(
    *,
    parallel: int = 1,
    sequence: int = 4,
    slots_per_partition: int = 64,
    value_dim: int = 32,
    writes: int = 4,
    reads: int = 4,
    dtype_name: str = "bfloat16",
) -> CompileProbe:
    """Real GPU compile probe over the EXACT K3 target specialization.

    Probes compile + launch the production route-generation and state-mixer
    kernels for the requested specialization (slots, value width, route widths,
    operand dtype, and the compiler's serialized launch configuration),
    exercising forward and backward when intent is training.

    Note on runtime extents: batch/sequence are runtime tensor dimensions, not
    compile-time Triton specialization constants. Using bounded representative
    extents avoids excessive probe latency while compiling and running the
    identical specialized kernels. Register/shared-memory facts flow back from
    the compiled kernel cache.
    """
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - guarded by callers
        raise RuntimeError("make_triton_k3_compile_probe requires CUDA")
    device = torch.device("cuda")

    def probe(context) -> CompileProbeResult:
        try:
            is_training = context.intent == "training"
            dtype = getattr(torch, dtype_name)
            slots = slots_per_partition
            if writes > slots or reads > slots:
                return CompileProbeResult(
                    ok=False, reason="route widths exceed slot count in probe shape"
                )
            from urm.backends.triton.k3 import route as route_mod
            from urm.backends.triton.k3 import state as state_mod

            generator = torch.Generator(device=device).manual_seed(11)
            rows = parallel * sequence
            score_width = 2 * round(slots**0.5)
            scores = torch.randn(
                (rows, score_width), device=device, dtype=dtype, generator=generator
            )

            resources: dict[str, KernelResourceUsage] = {}

            # Route generation (exact native route kernel).
            read_addresses, read_weights = route_mod.sparse_route_selection(
                scores, slots, reads, index_dtype=torch.int32
            )
            write_addresses, write_weights = route_mod.sparse_route_selection(
                scores, slots, writes, index_dtype=torch.int32
            )
            torch.cuda.synchronize()

            memory = torch.randn(
                (parallel, slots, value_dim), device=device, dtype=dtype
            )
            ri = read_addresses.view(parallel, sequence, reads)
            rw = read_weights.view(parallel, sequence, reads)
            wi = write_addresses.view(parallel, sequence, writes)
            ww = write_weights.view(parallel, sequence, writes)
            values = torch.randn(
                (parallel, sequence, value_dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
            beta = torch.rand(
                (parallel, sequence, 1), device=device, generator=generator
            ).to(dtype)
            log_decay = -torch.rand(
                (parallel, sequence, 1), device=device, generator=generator
            ).to(dtype)

            if is_training:
                memory = memory.requires_grad_()
                rw = rw.requires_grad_()
                ww = ww.requires_grad_()
                values = values.requires_grad_()
                beta = beta.requires_grad_()
                log_decay = log_decay.requires_grad_()
                output, updated = state_mod.sparse_state_update(
                    memory,
                    wi,
                    ww,
                    values,
                    beta,
                    log_decay,
                    ri,
                    rw,
                    read_before_update=False,
                )
                loss = output.float().square().mean() + updated.float().square().mean()
                loss.backward()
            else:
                with torch.no_grad():
                    state_mod.sparse_state_update(
                        memory,
                        wi,
                        ww,
                        values,
                        beta,
                        log_decay,
                        ri,
                        rw,
                        read_before_update=False,
                    )
            torch.cuda.synchronize()

            for module, family in ((route_mod, "route"), (state_mod, "state")):
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


__all__ = ["make_triton_k3_compile_probe"]
