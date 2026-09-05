"""Opaque torch.compile boundary for the pinned external SDM comparator.

This module is benchmark integration glue, not semantic IR and not a native
lowering. Its implementation calls the exact pinned upstream autograd kernels.
"""

from __future__ import annotations

import itertools

import torch

_HANDLES = itertools.count(1)
_ADAPTERS: dict[int, object] = {}
_FORWARD_CONTEXTS: dict[int, list[object]] = {}


def register_upstream_adapter(adapter: object) -> int:
    handle = next(_HANDLES)
    _ADAPTERS[handle] = adapter
    _FORWARD_CONTEXTS[handle] = []
    return handle


class _ManualAutogradContext:
    def save_for_backward(self, *tensors) -> None:
        self.saved_tensors = tensors


@torch.library.custom_op("urm::compiled_upstream_sdm_update", mutates_args=())
def compiled_upstream_sdm_update(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    handle: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Call the registered pinned bound method with functional output storage."""
    from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead

    adapter = _ADAPTERS[handle]
    config = adapter.config
    working = memory.clone()
    context = _ManualAutogradContext()
    readings, _unused = GatedSparseMemoryWriteRead.forward(
        context,
        working,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
        min(config.chunk_size, write_indices.shape[1]),
        True,
        config.slots_per_partition,
        write_indices.shape[0],
        False,
        "none",
        None,
    )
    _FORWARD_CONTEXTS[handle].append(context)
    # Upstream backward restores its mutable working memory; preserve the
    # forward-final state as an independent persistent-state output.
    return readings.view_as(values), working.clone()


@compiled_upstream_sdm_update.register_fake
def _compiled_upstream_sdm_update_fake(
    memory,
    write_indices,
    write_weights,
    values,
    beta,
    log_decay,
    read_indices,
    read_weights,
    handle,
):
    del (
        write_indices,
        write_weights,
        beta,
        log_decay,
        read_indices,
        read_weights,
        handle,
    )
    return values.new_empty(values.shape), memory.new_empty(memory.shape)


def _setup_context(ctx, inputs, output) -> None:
    del output
    *tensors, handle = inputs
    ctx.save_for_backward(*tensors)
    ctx.handle = handle


@torch.library.custom_op("urm::compiled_upstream_sdm_backward", mutates_args=())
def compiled_upstream_sdm_backward(
    memory: torch.Tensor,
    write_indices: torch.Tensor,
    write_weights: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    read_indices: torch.Tensor,
    read_weights: torch.Tensor,
    grad_readings: torch.Tensor,
    grad_final: torch.Tensor,
    handle: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Opaque first-order VJP through the pinned upstream backward kernel."""
    del memory, write_indices, write_weights, values, beta, log_decay
    del read_indices, read_weights
    from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead

    try:
        context = _FORWARD_CONTEXTS[handle].pop()
    except (KeyError, IndexError) as error:
        raise RuntimeError(
            "upstream compiled backward has no matching forward"
        ) from error
    context._grad_final_memory = grad_final.contiguous()
    gradients = GatedSparseMemoryWriteRead.backward(
        context,
        grad_readings.reshape(-1, grad_readings.shape[-1]).contiguous(),
        torch.empty(0, device=grad_readings.device, dtype=grad_readings.dtype),
    )
    return (
        gradients[0],
        gradients[2],
        gradients[3],
        gradients[4],
        gradients[5],
        gradients[7],
    )


@compiled_upstream_sdm_backward.register_fake
def _compiled_upstream_sdm_backward_fake(
    memory,
    write_indices,
    write_weights,
    values,
    beta,
    log_decay,
    read_indices,
    read_weights,
    grad_readings,
    grad_final,
    handle,
):
    del write_indices, read_indices, grad_readings, grad_final, handle
    return tuple(
        tensor.new_empty(tensor.shape)
        for tensor in (
            memory,
            write_weights,
            values,
            beta,
            log_decay,
            read_weights,
        )
    )


def _backward(ctx, grad_readings, grad_final):
    (
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
    ) = ctx.saved_tensors
    if grad_final is None:
        grad_final = torch.zeros_like(memory)
    (
        grad_memory,
        grad_write_weights,
        grad_values,
        grad_beta,
        grad_log_decay,
        grad_read_weights,
    ) = compiled_upstream_sdm_backward(
        memory,
        write_indices,
        write_weights,
        values,
        beta,
        log_decay,
        read_indices,
        read_weights,
        grad_readings,
        grad_final,
        ctx.handle,
    )
    return (
        grad_memory,
        None,
        grad_write_weights,
        grad_values,
        grad_beta,
        grad_log_decay,
        None,
        grad_read_weights,
        None,
    )


compiled_upstream_sdm_update.register_autograd(_backward, setup_context=_setup_context)


__all__ = [
    "compiled_upstream_sdm_backward",
    "compiled_upstream_sdm_update",
    "register_upstream_adapter",
]
