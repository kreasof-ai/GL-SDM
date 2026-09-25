"""Transparent PyTorch formulation of the URM SparseStateMixer algebra."""

from __future__ import annotations

from urm.ir.program import SparseReadTiming, SparseStateMixerSpec, SparseStateOperation


def sparse_delta_state(
    memory,
    read_addresses=None,
    read_weights=None,
    *,
    write_addresses=None,
    write_weights=None,
    values=None,
    beta=None,
    log_decay=None,
    spec: SparseStateMixerSpec | None = None,
    # Back-compat kwargs (legacy callsites): positional aliases and a bare
    # read_timing; when ``spec`` is omitted one is synthesized from these.
    read_indices=None,
    write_indices=None,
    read_timing: SparseReadTiming = SparseReadTiming.CURRENT_STATE,
    accumulation_dtype=None,
):
    """Canonical K3 sparse-delta state — the differentiable Torch implementation of
    the uniform address-index signature.

    Same operand names and shapes as the NumPy oracle and native launcher; the
    closed :class:`SparseStateMixerSpec` supplies read timing and accumulation
    policy. fp32 arithmetic with state casts. Returns ``(readings, updated_memory)``.
    """
    import torch

    if read_addresses is None:
        read_addresses = read_indices
    if write_addresses is None:
        write_addresses = write_indices
    if spec is None:
        from urm.ir.program import DType, SparseStateOperation

        parallel, sequence, reads = (
            int(read_addresses.shape[0]),
            int(read_addresses.shape[1]),
            int(read_addresses.shape[2]),
        )
        slots = int(memory.shape[1])
        value_dim = int(memory.shape[2])
        writes = 0 if write_addresses is None else int(write_addresses.shape[2])
        spec = SparseStateMixerSpec(
            parallel=parallel,
            sequence=sequence,
            slots_per_partition=slots,
            value_dim=value_dim,
            writes=writes,
            reads=reads,
            dtype=DType.FLOAT32,
            operation=(
                SparseStateOperation.UPDATE if write_addresses is not None else SparseStateOperation.READ_ONLY
            ),
            read_timing=read_timing,
        )
    else:
        read_timing = spec.read_timing
    if accumulation_dtype is None:
        accumulation_dtype = torch.float32
    read_indices = read_addresses
    write_indices = write_addresses
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


__all__ = ["sparse_delta_state", "torch_sparse_state_mixer"]


# Back-compat alias: the canonical name is sparse_delta_state.
torch_sparse_state_mixer = sparse_delta_state


# ---------------------------------------------------------------------------
# Provider surface (auto-discovered by urm.backends.registry)
# ---------------------------------------------------------------------------


def validate_k3_route_provenance(spec: SparseStateMixerSpec, operands) -> str | None:
    """Validate the route operands fed to a K3 state mixer (route provenance).

    The K3 contract certifies that routes are "already certified logical addresses
    and normalized weights"; on the public graph path that certification is the
    provider's job (the runtime binds raw tensors by role, and the spec's
    parallel/sequence are placeholder batch dims re-materialized from the operand
    shapes). This reuses the ``CertifiedSparseStateRoutes.certify`` value semantics
    so the reference tier enforces the route invariants: partition-local in-bounds
    addresses, strictly increasing and unique within each token, the declared route
    width, and finite nonnegative normalized weights. Returns a structured decline
    reason, or ``None`` when well-formed.
    """
    import torch

    def _check(indices, weights, width, label):
        if indices is None or weights is None:
            return f"K3 {label} routes are unbound"
        if not isinstance(indices, torch.Tensor) or indices.dtype not in (torch.int32, torch.int64):
            return f"K3 {label} addresses must be int32/int64 tensors"
        if indices.shape != weights.shape:
            return f"K3 {label} addresses and weights must share a shape"
        if indices.ndim != 3 or indices.shape[-1] != width:
            return f"K3 {label} routes must be [parallel, sequence, {width}]"
        idx = indices.to(torch.int64)
        if bool(((idx < 0) | (idx >= spec.slots_per_partition)).any().item()):
            return f"K3 {label} addresses must be partition-local and in bounds"
        if width > 1 and bool((idx[..., 1:] <= idx[..., :-1]).any().item()):
            return f"K3 {label} addresses must be strictly increasing and unique within a token"
        w = weights.to(torch.float32)
        if not bool(torch.isfinite(w).all().item()):
            return f"K3 {label} weights must be finite"
        if bool((w < 0).any().item()):
            return f"K3 {label} weights must be nonnegative"
        atol = 2e-5 if spec.dtype.value == "float32" else 4e-3
        sums = w.sum(dim=-1)
        if not bool(torch.allclose(sums, torch.ones_like(sums), atol=atol, rtol=0)):
            return f"K3 {label} weights must be normalized (sum to 1 over the route width)"
        return None

    read_err = _check(
        operands.get("read_addresses"), operands.get("read_weights"), spec.reads, "read"
    )
    if read_err is not None:
        return read_err
    if spec.operation is SparseStateOperation.UPDATE:
        write_err = _check(
            operands.get("write_addresses"), operands.get("write_weights"), spec.writes, "write"
        )
        if write_err is not None:
            return write_err
    return None


class K3TorchReferenceProvider:
    name = "urm.unified.k3.sparse_delta_reference.v1"
    family = "k3"
    tier = "reference"

    def decline(self, request) -> str | None:
        if not isinstance(request.descriptor, SparseStateMixerSpec):
            return "K3 providers require a closed SparseStateMixerSpec"
        return None

    def execute(self, request, operands):
        # Route-provenance gate: validate the route operands against the spec's
        # bounds before the equation runs (the runtime binds raw tensors by role).
        provenance = validate_k3_route_provenance(request.descriptor, operands)
        if provenance is not None:
            raise ValueError(provenance)
        outputs, state = sparse_delta_state(
            operands["memory"], operands["read_addresses"], operands["read_weights"],
            write_addresses=operands.get("write_addresses"),
            write_weights=operands.get("write_weights"),
            values=operands.get("values"),
            beta=operands.get("beta"),
            log_decay=operands.get("log_decay"),
            spec=request.descriptor,
        )
        return {"readings": outputs, "updated_memory": state}


PROVIDERS = (K3TorchReferenceProvider(),)
