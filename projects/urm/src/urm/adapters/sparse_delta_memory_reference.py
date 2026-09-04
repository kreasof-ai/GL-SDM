"""Reference levels for the frozen Facebook Sparse Delta Memory boundary.

The NumPy functions are the dependency-light semantic oracle.  The torch
functions are deliberately transparent formulations used for differential
testing; performance measurements use the original upstream implementation.
"""

from __future__ import annotations

import numpy as np


def oracle_product_key(
    scores: np.ndarray, num_keys: int, half_key: int
) -> tuple[np.ndarray, np.ndarray]:
    """Product-key top-k followed by ascending address order.

    ``scores`` has layout ``[parallel, sequence, 2 * half_key]``.  Ties are
    intentionally outside the frozen contract because upstream ``topk`` does
    not promise a cross-backend tie order.
    """
    if scores.ndim != 3 or scores.shape[-1] != 2 * half_key:
        raise ValueError("scores must have shape [P,T,2*half_key]")
    if not 0 < num_keys <= half_key * half_key:
        raise ValueError("num_keys must be in [1, half_key**2]")
    left, right = np.split(scores, 2, axis=-1)
    k_sub = min(num_keys, half_key)
    # Stable descending order is sufficient for the no-tie contract.
    left_idx = np.argsort(-left, axis=-1, kind="stable")[..., :k_sub]
    right_idx = np.argsort(-right, axis=-1, kind="stable")[..., :k_sub]
    left_val = np.take_along_axis(left, left_idx, axis=-1)
    right_val = np.take_along_axis(right, right_idx, axis=-1)
    combined = left_val[..., :, None] + right_val[..., None, :]
    flat = combined.reshape(*combined.shape[:-2], -1)
    k_final = min(num_keys, k_sub * k_sub)
    selected = np.argsort(-flat, axis=-1, kind="stable")[..., :k_final]
    i1, i2 = selected // k_sub, selected % k_sub
    chosen_left = np.take_along_axis(left_idx, i1, axis=-1)
    chosen_right = np.take_along_axis(right_idx, i2, axis=-1)
    addresses = chosen_left * half_key + chosen_right
    values = np.take_along_axis(flat, selected, axis=-1)
    order = np.argsort(addresses, axis=-1, kind="stable")
    return (
        np.take_along_axis(values, order, axis=-1),
        np.take_along_axis(addresses, order, axis=-1).astype(np.int64),
    )


def oracle_sparse_read(
    memory: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted sparse sum for ``[P,T,R]`` global addresses."""
    return np.sum(memory[indices] * weights[..., None], axis=-2)


def oracle_write_read(
    memory: np.ndarray,
    write_indices: np.ndarray,
    write_weights: np.ndarray,
    values: np.ndarray,
    beta: np.ndarray,
    log_decay: np.ndarray,
    read_indices: np.ndarray,
    read_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Token-ordered SDM recurrence, returning readings and mutated state."""
    state = memory.astype(np.float32, copy=True)
    p, sequence, _ = write_indices.shape
    readings = np.empty((p, sequence, memory.shape[-1]), dtype=np.float32)
    for parallel in range(p):
        for token in range(sequence):
            addresses = write_indices[parallel, token]
            decayed = state[addresses] * np.exp(log_decay[parallel, token, 0])
            retrieved = np.sum(
                write_weights[parallel, token, :, None] * decayed, axis=0
            )
            delta = beta[parallel, token, 0] * (values[parallel, token] - retrieved)
            state[addresses] = decayed + write_weights[parallel, token, :, None] * delta
            q_addresses = read_indices[parallel, token]
            readings[parallel, token] = np.sum(
                read_weights[parallel, token, :, None] * state[q_addresses], axis=0
            )
    return readings, state


def torch_product_key(scores, num_keys: int, half_key: int):
    """Transparent PyTorch form of upstream product-key addressing."""
    import torch

    left, right = scores.chunk(2, dim=-1)
    k_sub = min(num_keys, half_key)
    left_val, left_idx = torch.topk(left, k_sub, dim=-1)
    right_val, right_idx = torch.topk(right, k_sub, dim=-1)
    combined = left_val.unsqueeze(-1) + right_val.unsqueeze(-2)
    k_final = min(num_keys, k_sub * k_sub)
    values, selected = torch.topk(combined.flatten(-2), k_final, dim=-1)
    i1, i2 = selected // k_sub, selected % k_sub
    addresses = left_idx.gather(-1, i1) * half_key + right_idx.gather(-1, i2)
    addresses, order = addresses.sort(dim=-1)
    return values.gather(-1, order), addresses


def torch_sparse_read(memory, indices, weights):
    """Transparent PyTorch weighted sparse read."""
    return (memory[indices] * weights.unsqueeze(-1)).sum(dim=-2)


def torch_write_read(
    memory,
    write_indices,
    write_weights,
    values,
    beta,
    log_decay,
    read_indices,
    read_weights,
):
    """Transparent PyTorch token recurrence with explicit mutation order."""
    import torch

    state = memory.float().clone()
    p, sequence, _ = write_indices.shape
    outputs = []
    for parallel in range(p):
        parallel_outputs = []
        for token in range(sequence):
            addresses = write_indices[parallel, token]
            decayed = state[addresses] * torch.exp(log_decay[parallel, token, 0])
            retrieved = (
                write_weights[parallel, token].float().unsqueeze(-1) * decayed
            ).sum(dim=0)
            delta = beta[parallel, token, 0].float() * (
                values[parallel, token].float() - retrieved
            )
            updated = (
                decayed + write_weights[parallel, token].float().unsqueeze(-1) * delta
            )
            # Functional replacement preserves the explicit recurrence while
            # keeping every prior state version available to autograd.
            state = state.index_copy(0, addresses, updated)
            q_addresses = read_indices[parallel, token]
            parallel_outputs.append(
                (
                    read_weights[parallel, token].float().unsqueeze(-1)
                    * state[q_addresses]
                ).sum(dim=0)
            )
        outputs.append(torch.stack(parallel_outputs))
    return torch.stack(outputs).to(memory.dtype), state.to(memory.dtype)


def differential_backward_report(
    adapter,
    trace,
    memory,
    values,
    beta,
    log_decay,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare transparent, direct-upstream, and adapted SDM gradients.

    Every path receives independent cloned leaves. The shared scalar loss has
    nonzero weighted terms for both readings and returned final memory.
    """
    import torch

    names = (
        "initial_memory",
        "write_weights",
        "values",
        "beta",
        "log_decay",
        "read_weights",
    )
    generator = torch.Generator(device=memory.device).manual_seed(20260905)
    reading_cotangent = torch.randn(
        values.shape, device=memory.device, dtype=torch.float32, generator=generator
    )
    memory_cotangent = torch.randn(
        memory.shape, device=memory.device, dtype=torch.float32, generator=generator
    )
    grad_final_memory = (
        memory_cotangent.to(memory.dtype) / memory.numel()
    ).contiguous()

    def run(path: str):
        leaves = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                memory,
                trace.write_weights,
                values,
                beta,
                log_decay,
                trace.read_weights,
            )
        )
        path_trace = trace.with_differentiable_weights(leaves[1], leaves[5])
        if path == "torch_reference":
            readings, final_memory = torch_write_read(
                leaves[0],
                path_trace.write_indices,
                path_trace.write_weights,
                leaves[2],
                leaves[3],
                leaves[4],
                path_trace.read_indices,
                path_trace.read_weights,
            )
        elif path == "direct_upstream":
            readings, final_memory = adapter.direct_calls["update"](
                leaves[0] + 0,
                path_trace.write_indices,
                path_trace.write_weights,
                leaves[2],
                leaves[3],
                leaves[4],
                path_trace.read_indices,
                path_trace.read_weights,
                grad_final_memory=grad_final_memory,
            )
        elif path == "urm_adapter":
            from urm.adapters.sparse_delta_memory import SDMState

            readings, final_state = adapter.execute(
                SDMState(leaves[0] + 0),
                path_trace,
                leaves[2],
                leaves[3],
                leaves[4],
                grad_final_memory=grad_final_memory,
            )
            final_memory = final_state.memory
        else:  # pragma: no cover - closed internal vocabulary
            raise ValueError(path)
        reading_term = (readings.float() * reading_cotangent).mean()
        memory_term = (final_memory.float() * memory_cotangent).mean()
        # Upstream exposes final-state differentiation through the explicit
        # grad_final_memory argument because returned memory is mutated state,
        # not a differentiable result edge. The transparent path differentiates
        # the same logical loss normally.
        backward_loss = (
            reading_term + memory_term if path == "torch_reference" else reading_term
        )
        gradients = torch.autograd.grad(backward_loss, leaves)
        return gradients, float(reading_term.detach()), float(memory_term.detach())

    paths = {}
    loss_terms = {}
    for path in ("torch_reference", "direct_upstream", "urm_adapter"):
        gradients, reading_term, memory_term = run(path)
        paths[path] = gradients
        loss_terms[path] = {
            "readings_weighted_mean": reading_term,
            "final_memory_weighted_mean": memory_term,
        }

    comparisons = {}
    passed = True
    for index, name in enumerate(names):
        reference = paths["torch_reference"][index].float()
        direct = paths["direct_upstream"][index].float()
        adapted = paths["urm_adapter"][index].float()

        def compare(actual, expected):
            difference = (actual - expected).abs()
            close = torch.allclose(actual, expected, atol=atol, rtol=rtol)
            return {
                "max_abs": float(difference.max().item()),
                "max_scaled_error": float(
                    (difference / (atol + rtol * expected.abs())).max().item()
                ),
                "close": bool(close),
            }

        finite = {
            path: bool(torch.isfinite(gradients[index]).all().item())
            for path, gradients in paths.items()
        }
        direct_reference = compare(direct, reference)
        adapter_reference = compare(adapted, reference)
        adapter_direct = compare(adapted, direct)
        item_passed = (
            all(finite.values())
            and direct_reference["close"]
            and adapter_reference["close"]
            and adapter_direct["close"]
        )
        passed = passed and item_passed
        comparisons[name] = {
            "finite": finite,
            "direct_vs_torch": direct_reference,
            "adapter_vs_torch": adapter_reference,
            "adapter_vs_direct": adapter_direct,
            "passed": item_passed,
        }
    return {
        "dtype": str(memory.dtype).removeprefix("torch."),
        "atol": atol,
        "rtol": rtol,
        "loss": "weighted_mean(readings) + weighted_mean(final_memory)",
        "loss_terms": loss_terms,
        "gradients": comparisons,
        "passed": passed,
    }


__all__ = [
    "differential_backward_report",
    "oracle_product_key",
    "oracle_sparse_read",
    "oracle_write_read",
    "torch_product_key",
    "torch_sparse_read",
    "torch_write_read",
]
