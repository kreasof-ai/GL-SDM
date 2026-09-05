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


def torch_product_key_highest_address(scores, num_keys: int, half_key: int):
    """Transparent frozen URM policy: score descending, address descending ties.

    The full Cartesian formulation is intentionally dependency-light and
    transparent.  It is a semantic oracle, not the production schedule.
    Returned routes are canonicalized into increasing-address order.
    """
    if scores.shape[-1] != 2 * half_key:
        raise ValueError("scores must have width 2*half_key")
    left, right = scores.split(half_key, dim=-1)
    pair_scores = (left.unsqueeze(-1) + right.unsqueeze(-2)).flatten(-2)
    # Reversing makes stable score sorting prefer the greatest logical address.
    reversed_scores = pair_scores.flip(-1)
    selected_reversed = reversed_scores.argsort(dim=-1, descending=True, stable=True)[
        ..., :num_keys
    ]
    selected = pair_scores.shape[-1] - 1 - selected_reversed
    addresses = selected.sort(dim=-1).values
    return pair_scores.gather(-1, addresses), addresses


def product_key_selection_is_tie_free(scores, num_keys: int, half_key: int) -> bool:
    """Whether every decision used by product-key top-k has a unique value."""
    import torch

    if scores.shape[-1] != 2 * half_key:
        raise ValueError("scores must have width 2*half_key")
    left, right = scores.chunk(2, dim=-1)
    k_sub = min(num_keys, half_key)
    for half in (left, right):
        ordered = half.sort(dim=-1).values
        if bool((ordered[..., 1:] == ordered[..., :-1]).any().item()):
            return False
    left_values = torch.topk(left, k_sub, dim=-1).values
    right_values = torch.topk(right, k_sub, dim=-1).values
    products = (left_values.unsqueeze(-1) + right_values.unsqueeze(-2)).flatten(-2)
    ordered_products = products.sort(dim=-1).values
    return not bool(
        (ordered_products[..., 1:] == ordered_products[..., :-1]).any().item()
    )


def deterministic_tie_free_product_key_scores(
    *,
    parallel: int,
    sequence: int,
    half_key: int,
    num_keys: int,
    device,
    dtype,
    seed: int,
):
    """Generate deterministic random score rows, rejecting product-key ties."""
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    attempts = 0
    maximum_attempts = parallel * sequence * 10_000
    while len(rows) < parallel * sequence and attempts < maximum_attempts:
        candidate = torch.randn(
            (1, 1, 2 * half_key),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        attempts += 1
        if product_key_selection_is_tie_free(candidate, num_keys, half_key):
            rows.append(candidate)
    if len(rows) != parallel * sequence:
        raise RuntimeError(
            "could not generate deterministic tie-free product-key scores "
            f"after {attempts} attempts"
        )
    return torch.cat(rows, dim=1).reshape(parallel, sequence, 2 * half_key).contiguous()


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


def end_to_end_differential_backward_report(
    adapter,
    write_scores,
    read_scores,
    memory,
    values,
    beta,
    log_decay,
    *,
    gradient_atol: float,
    gradient_rtol: float,
    forward_atol: float,
    forward_rtol: float,
) -> dict[str, object]:
    """Certify gradients from product-key scores through final sparse state."""
    import torch

    names = (
        "write_scores",
        "read_scores",
        "initial_memory",
        "values",
        "beta",
        "log_decay",
    )
    config = adapter.config
    half_key = round(config.slots_per_partition**0.5)
    tie_free = product_key_selection_is_tie_free(
        write_scores, config.num_writes, half_key
    ) and product_key_selection_is_tie_free(read_scores, config.num_reads, half_key)
    if not tie_free:
        raise ValueError("end-to-end backward inputs contain product-key ties")

    generator = torch.Generator(device=memory.device).manual_seed(20260907)
    reading_cotangent = torch.randn(
        values.shape, device=memory.device, dtype=torch.float32, generator=generator
    )
    memory_cotangent = torch.randn(
        memory.shape, device=memory.device, dtype=torch.float32, generator=generator
    )
    grad_final_memory = (
        memory_cotangent.to(memory.dtype) / memory.numel()
    ).contiguous()

    def globalize(indices):
        offsets = (
            torch.arange(
                write_scores.shape[0], device=indices.device, dtype=torch.int64
            ).view(-1, 1, 1)
            * config.slots_per_partition
        )
        return (indices + offsets).contiguous()

    def run(path: str):
        leaves = tuple(
            tensor.detach().clone().requires_grad_(True)
            for tensor in (
                write_scores,
                read_scores,
                memory,
                values,
                beta,
                log_decay,
            )
        )
        if path == "torch_reference":
            write_values, write_indices = torch_product_key(
                leaves[0], config.num_writes, half_key
            )
            read_values, read_indices = torch_product_key(
                leaves[1], config.num_reads, half_key
            )
            write_indices = globalize(write_indices)
            read_indices = globalize(read_indices)
            write_weights = torch.softmax(write_values, dim=-1)
            read_weights = torch.softmax(read_values, dim=-1)
            readings, final_memory = torch_write_read(
                leaves[2],
                write_indices,
                write_weights,
                leaves[3],
                leaves[4],
                leaves[5],
                read_indices,
                read_weights,
            )
        elif path == "direct_upstream":
            address = adapter.direct_calls["address"]
            write_values, write_indices = address(
                leaves[0], config.num_writes, half_key
            )
            read_values, read_indices = address(leaves[1], config.num_reads, half_key)
            write_indices = globalize(write_indices)
            read_indices = globalize(read_indices)
            write_weights = adapter.layer.write_act(write_values)
            read_weights = adapter.layer.read_act(read_values)
            readings, final_memory = adapter.direct_calls["update"](
                leaves[2] + 0,
                write_indices,
                write_weights,
                leaves[3],
                leaves[4],
                leaves[5],
                read_indices,
                read_weights,
                grad_final_memory=grad_final_memory,
            )
        elif path == "urm_adapter":
            from urm.adapters.sparse_delta_memory import SDMState

            trace = adapter.generate_trace(leaves[0], leaves[1])
            write_indices = trace.write_indices
            read_indices = trace.read_indices
            write_weights = trace.write_weights
            read_weights = trace.read_weights
            readings, final_state = adapter.execute(
                SDMState(leaves[2] + 0),
                trace,
                leaves[3],
                leaves[4],
                leaves[5],
                grad_final_memory=grad_final_memory,
            )
            final_memory = final_state.memory
        else:  # pragma: no cover - closed internal vocabulary
            raise ValueError(path)

        reading_term = (readings.float() * reading_cotangent).mean()
        memory_term = (final_memory.float() * memory_cotangent).mean()
        # The pinned upstream backward reuses its mutable memory buffer as
        # scratch. Preserve forward evidence before invoking autograd.
        readings_snapshot = readings.detach().clone()
        final_memory_snapshot = final_memory.detach().clone()
        backward_loss = (
            reading_term + memory_term if path == "torch_reference" else reading_term
        )
        gradients = torch.autograd.grad(backward_loss, leaves)
        return {
            "gradients": gradients,
            "write_indices": write_indices.detach(),
            "read_indices": read_indices.detach(),
            "write_weights": write_weights.detach(),
            "read_weights": read_weights.detach(),
            "readings": readings_snapshot,
            "final_memory": final_memory_snapshot,
            "loss_terms": {
                "readings_weighted_mean": float(reading_term.detach()),
                "final_memory_weighted_mean": float(memory_term.detach()),
            },
        }

    paths = {
        path: run(path)
        for path in ("torch_reference", "direct_upstream", "urm_adapter")
    }

    def exact_report(key: str) -> dict[str, bool]:
        reference = paths["torch_reference"][key]
        direct = paths["direct_upstream"][key]
        adapted = paths["urm_adapter"][key]
        return {
            "direct_vs_torch": bool(torch.equal(direct, reference)),
            "adapter_vs_torch": bool(torch.equal(adapted, reference)),
            "adapter_vs_direct": bool(torch.equal(adapted, direct)),
        }

    def compare(
        actual, expected, *, comparison_atol: float, comparison_rtol: float
    ) -> dict[str, float | bool]:
        actual_float = actual.float()
        expected_float = expected.float()
        difference = (actual_float - expected_float).abs()
        return {
            "max_abs": float(difference.max().item()),
            "max_scaled_error": float(
                (
                    difference
                    / (comparison_atol + comparison_rtol * expected_float.abs())
                )
                .max()
                .item()
            ),
            "close": bool(
                torch.allclose(
                    actual_float,
                    expected_float,
                    atol=comparison_atol,
                    rtol=comparison_rtol,
                )
            ),
        }

    def comparison_report(key: str) -> dict[str, object]:
        reference = paths["torch_reference"][key]
        direct = paths["direct_upstream"][key]
        adapted = paths["urm_adapter"][key]
        direct_reference = compare(
            direct,
            reference,
            comparison_atol=forward_atol,
            comparison_rtol=forward_rtol,
        )
        adapter_reference = compare(
            adapted,
            reference,
            comparison_atol=forward_atol,
            comparison_rtol=forward_rtol,
        )
        adapter_direct = compare(
            adapted,
            direct,
            comparison_atol=forward_atol,
            comparison_rtol=forward_rtol,
        )
        return {
            "direct_vs_torch": direct_reference,
            "adapter_vs_torch": adapter_reference,
            "adapter_vs_direct": adapter_direct,
            "passed": bool(
                direct_reference["close"]
                and adapter_reference["close"]
                and adapter_direct["close"]
            ),
        }

    addresses = {
        "write": exact_report("write_indices"),
        "read": exact_report("read_indices"),
    }
    addresses_passed = all(
        all(comparison.values()) for comparison in addresses.values()
    )
    route_weights = {
        "write": exact_report("write_weights"),
        "read": exact_report("read_weights"),
    }
    route_weights_passed = all(
        all(comparison.values()) for comparison in route_weights.values()
    )
    forward = {
        "readings": comparison_report("readings"),
        "final_memory": comparison_report("final_memory"),
    }

    gradients = {}
    gradients_passed = True
    for index, name in enumerate(names):
        reference = paths["torch_reference"]["gradients"][index]
        direct = paths["direct_upstream"]["gradients"][index]
        adapted = paths["urm_adapter"]["gradients"][index]
        finite = {
            path: bool(torch.isfinite(result["gradients"][index]).all().item())
            for path, result in paths.items()
        }
        direct_reference = compare(
            direct,
            reference,
            comparison_atol=gradient_atol,
            comparison_rtol=gradient_rtol,
        )
        adapter_reference = compare(
            adapted,
            reference,
            comparison_atol=gradient_atol,
            comparison_rtol=gradient_rtol,
        )
        adapter_direct = compare(
            adapted,
            direct,
            comparison_atol=gradient_atol,
            comparison_rtol=gradient_rtol,
        )
        item_passed = bool(
            all(finite.values())
            and direct_reference["close"]
            and adapter_reference["close"]
            and adapter_direct["close"]
        )
        gradients_passed = gradients_passed and item_passed
        gradients[name] = {
            "finite": finite,
            "direct_vs_torch": direct_reference,
            "adapter_vs_torch": adapter_reference,
            "adapter_vs_direct": adapter_direct,
            "passed": item_passed,
        }

    return {
        "dtype": str(memory.dtype).removeprefix("torch."),
        "scope": "write/read scores through product-key top-k, Softmax, and ordered state",
        "product_key_tie_free": tie_free,
        "gradient_atol": gradient_atol,
        "gradient_rtol": gradient_rtol,
        "forward_atol": forward_atol,
        "forward_rtol": forward_rtol,
        "loss": "weighted_mean(readings) + weighted_mean(final_memory)",
        "loss_terms": {path: result["loss_terms"] for path, result in paths.items()},
        "addresses": {**addresses, "passed": addresses_passed},
        "route_weights": {**route_weights, "passed": route_weights_passed},
        "forward": forward,
        "gradients": gradients,
        "passed": bool(
            addresses_passed
            and route_weights_passed
            and all(result["passed"] for result in forward.values())
            and gradients_passed
        ),
    }


__all__ = [
    "deterministic_tie_free_product_key_scores",
    "differential_backward_report",
    "end_to_end_differential_backward_report",
    "oracle_product_key",
    "oracle_sparse_read",
    "oracle_write_read",
    "product_key_selection_is_tie_free",
    "torch_product_key",
    "torch_product_key_highest_address",
    "torch_sparse_read",
    "torch_write_read",
]
