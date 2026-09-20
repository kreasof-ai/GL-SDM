"""Transparent PyTorch formulation of the URM SparseStateMixer algebra."""

from __future__ import annotations

from urm.compiler.semantic import SparseReadTiming


def torch_sparse_state_mixer(
    memory,
    read_indices,
    read_weights,
    *,
    write_indices=None,
    write_weights=None,
    values=None,
    beta=None,
    log_decay=None,
    read_timing: SparseReadTiming = SparseReadTiming.CURRENT_STATE,
    accumulation_dtype=None,
):
    """Functional ordered recurrence with fp32 arithmetic and state casts."""
    import torch

    if accumulation_dtype is None:
        accumulation_dtype = torch.float32
    state = memory.clone()
    parallel, sequence, _ = read_indices.shape
    outputs = []
    updating = write_indices is not None
    for partition in range(parallel):
        partition_outputs = []
        for token in range(sequence):
            if read_timing in {
                SparseReadTiming.CURRENT_STATE,
                SparseReadTiming.BEFORE_UPDATE,
            }:
                partition_outputs.append(
                    (
                        read_weights[partition, token]
                        .to(accumulation_dtype)
                        .unsqueeze(-1)
                        * state[partition, read_indices[partition, token]].to(
                            accumulation_dtype
                        )
                    )
                    .sum(dim=0)
                    .to(memory.dtype)
                )
            if updating:
                addresses = write_indices[partition, token]
                old = state[partition, addresses].to(accumulation_dtype)
                decayed = old * torch.exp(
                    log_decay[partition, token, 0].to(accumulation_dtype)
                )
                weights = (
                    write_weights[partition, token].to(accumulation_dtype).unsqueeze(-1)
                )
                retrieved = (weights * decayed).sum(dim=0)
                delta = beta[partition, token, 0].to(accumulation_dtype) * (
                    values[partition, token].to(accumulation_dtype) - retrieved
                )
                updated = (decayed + weights * delta).to(memory.dtype)
                # PyTorch's index_copy requires int64 even though the semantic
                # and native contracts support both int32 and int64 routes.
                partition_state = state[partition].index_copy(
                    0, addresses.to(torch.int64), updated
                )
                state = torch.cat(
                    (
                        state[:partition],
                        partition_state.unsqueeze(0),
                        state[partition + 1 :],
                    ),
                    dim=0,
                )
            if read_timing is SparseReadTiming.AFTER_UPDATE:
                partition_outputs.append(
                    (
                        read_weights[partition, token]
                        .to(accumulation_dtype)
                        .unsqueeze(-1)
                        * state[partition, read_indices[partition, token]].to(
                            accumulation_dtype
                        )
                    )
                    .sum(dim=0)
                    .to(memory.dtype)
                )
        outputs.append(torch.stack(partition_outputs))
    return torch.stack(outputs), state


__all__ = ["torch_sparse_state_mixer"]
