"""Original Sparse Delta Memory direct-vs-URM comparison benchmark.

Run from ``projects/urm`` with the pinned upstream checkout on ``PYTHONPATH``.
The benchmark never vendors upstream code and fails closed on revision/runtime
incompatibility.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_OUTPUT = Path("results/sparse-delta-memory/benchmark.json")

CASES = (
    {
        "name": "smoke_read_only",
        "path": "read_only",
        "parallel": 1,
        "sequence": 1,
        "slots": 256,
        "dim": 37,
        "writes": 1,
        "reads": 1,
        "chunk": 16,
    },
    {
        "name": "prefill_batched",
        "path": "inference",
        "parallel": 2,
        "sequence": 128,
        "slots": 4096,
        "dim": 128,
        "writes": 8,
        "reads": 8,
        "chunk": 64,
    },
    {
        "name": "decode_cached",
        "path": "inference",
        "parallel": 2,
        "sequence": 1,
        "slots": 4096,
        "dim": 128,
        "writes": 8,
        "reads": 8,
        "chunk": 64,
    },
    {
        "name": "write_update",
        "path": "inference",
        "parallel": 1,
        "sequence": 32,
        "slots": 4096,
        "dim": 128,
        "writes": 16,
        "reads": 16,
        "chunk": 32,
    },
    {
        "name": "collision_heavy",
        "path": "inference",
        "parallel": 1,
        "sequence": 32,
        "slots": 256,
        "dim": 64,
        "writes": 4,
        "reads": 4,
        "chunk": 32,
        "collision_heavy": True,
    },
    {
        "name": "training_prefill_forward_only",
        "path": "training",
        "parallel": 1,
        "sequence": 64,
        "slots": 4096,
        "dim": 64,
        "writes": 8,
        "reads": 8,
        "chunk": 64,
    },
    {
        "name": "memory_capacity",
        "path": "inference",
        "parallel": 1,
        "sequence": 128,
        "slots": 262144,
        "dim": 512,
        "writes": 32,
        "reads": 32,
        "chunk": 64,
    },
)


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stats(samples: list[float]) -> dict[str, object]:
    from measurement import summarize_samples

    result = summarize_samples(samples)
    result["raw_samples_ms"] = samples
    return result


def _fraction_stats(samples: list[float]) -> dict[str, object]:
    from measurement import summarize_samples

    result = summarize_samples(samples)
    return {
        "sample_count": result["sample_count"],
        "median": result["median_ms"],
        "p95": result["p95_ms"],
        "min": result["min_ms"],
        "raw_samples": samples,
    }


def _time_cuda(call, torch):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter_ns()
    output = call()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    return wall_ms, start_event.elapsed_time(end_event), output


def _function_identity(bound_method) -> dict[str, object]:
    function = bound_method.__func__
    code = function.__code__.co_code
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "python_bytecode_sha256": hashlib.sha256(code).hexdigest(),
    }


def _same_bound_callable(direct, adapted) -> bool:
    """Factual identity check for one stored upstream bound method."""
    return (
        direct is adapted
        and direct.__self__ is adapted.__self__
        and direct.__func__ is adapted.__func__
    )


def _direct_trace(adapter, write_scores, read_scores, torch):
    key_dim = write_scores.shape[-1]
    address = adapter.direct_calls["address"]
    write_values, write_indices = address(
        write_scores, adapter.layer.args.num_writes, key_dim // 2
    )
    read_values, read_indices = address(
        read_scores, adapter.layer.args.num_reads, key_dim // 2
    )
    offsets = (
        torch.arange(write_scores.shape[0], device=write_scores.device).view(-1, 1, 1)
        * adapter.layer.slots_per_head
    )
    return (
        (write_indices + offsets).contiguous(),
        adapter.layer.write_act(write_values).contiguous(),
        (read_indices + offsets).contiguous(),
        adapter.layer.read_act(read_values).contiguous(),
    )


def _route_distribution(indices, slots: int) -> dict[str, object]:
    import torch

    local = indices % slots
    counts = torch.bincount(local.flatten(), minlength=slots)
    nonzero = counts[counts > 0]
    probs = nonzero.float() / counts.sum()
    entropy = float((-(probs * probs.log2()).sum()).item()) if len(nonzero) else 0.0
    within_duplicates = int(
        (indices[..., 1:] == indices[..., :-1]).sum().item()
        if indices.shape[-1] > 1
        else 0
    )
    return {
        "selected_addresses": indices.numel(),
        "unique_local_addresses": int((counts > 0).sum().item()),
        "collision_fraction_across_tokens": 1.0
        - float((counts > 0).sum().item()) / indices.numel(),
        "within_token_duplicate_count": within_duplicates,
        "max_hits_per_address": int(nonzero.max().item()) if len(nonzero) else 0,
        "address_entropy_bits": entropy,
    }


def _traffic(case: dict[str, object], dtype_bytes: int) -> dict[str, object]:
    p = int(case["parallel"])
    t = int(case["sequence"])
    w = int(case["writes"])
    r = int(case["reads"])
    d = int(case["dim"])
    index_bytes = 8
    read_edges = r if case["path"] == "read_only" else w + r
    read_bytes = p * t * read_edges * (d * dtype_bytes + index_bytes + dtype_bytes)
    write_bytes = p * t * w * d * dtype_bytes
    return {
        "kind": "analytical_logical_lower_bound",
        "measured": False,
        "read_bytes": read_bytes,
        "write_bytes": (
            {
                "status": "not_applicable",
                "reason": "read-only path does not mutate sparse memory",
            }
            if case["path"] == "read_only"
            else write_bytes
        ),
        "notes": "Counts logical sparse row traffic once per selected edge; cache reuse, write allocation, metadata, and kernel temporaries are not measured.",
    }


def _memory_measure(call, torch, output_bytes: int) -> dict[str, int]:
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del output
    torch.cuda.synchronize()
    return {
        "allocated_before_bytes": before,
        "peak_allocated_bytes": peak,
        "temporary_peak_bytes": max(0, peak - before - output_bytes),
        "output_bytes": output_bytes,
    }


def _paired_measure(
    direct_call,
    adapter_call,
    reset_direct,
    reset_adapter,
    *,
    samples: int,
    warmup: int,
    torch,
) -> dict[str, object]:
    for index in range(warmup):
        reset_direct()
        _time_cuda(direct_call, torch)
        reset_adapter()
        _time_cuda(adapter_call, torch)
    raw = {
        "direct_wall_ms": [],
        "adapter_wall_ms": [],
        "direct_device_ms": [],
        "adapter_device_ms": [],
        "pair_order": [],
    }
    for pair in range(samples):
        order = ("direct", "adapter") if pair % 2 == 0 else ("adapter", "direct")
        raw["pair_order"].append("AB" if pair % 2 == 0 else "BA")
        for name in order:
            if name == "direct":
                reset_direct()
                wall, device, _ = _time_cuda(direct_call, torch)
            else:
                reset_adapter()
                wall, device, _ = _time_cuda(adapter_call, torch)
            raw[f"{name}_wall_ms"].append(wall)
            raw[f"{name}_device_ms"].append(device)
    wall_fraction = [
        (adapted - direct) / direct
        for direct, adapted in zip(
            raw["direct_wall_ms"], raw["adapter_wall_ms"], strict=True
        )
    ]
    device_fraction = [
        (adapted - direct) / direct
        for direct, adapted in zip(
            raw["direct_device_ms"], raw["adapter_device_ms"], strict=True
        )
    ]
    wall_overhead_ms = [
        adapted - direct
        for direct, adapted in zip(
            raw["direct_wall_ms"], raw["adapter_wall_ms"], strict=True
        )
    ]
    device_overhead_ms = [
        adapted - direct
        for direct, adapted in zip(
            raw["direct_device_ms"], raw["adapter_device_ms"], strict=True
        )
    ]
    return {
        "sampling": "alternating paired AB/BA",
        "pairs": samples,
        "direct_wall": _stats(raw["direct_wall_ms"]),
        "adapter_wall": _stats(raw["adapter_wall_ms"]),
        "direct_device": _stats(raw["direct_device_ms"]),
        "adapter_device": _stats(raw["adapter_device_ms"]),
        "paired_wall_overhead_ms": _stats(wall_overhead_ms),
        "paired_device_overhead_ms": _stats(device_overhead_ms),
        "paired_wall_overhead_fraction": _fraction_stats(wall_fraction),
        "paired_device_overhead_fraction": _fraction_stats(device_fraction),
        "pair_order": raw["pair_order"],
    }


def _make_case(case: dict[str, object], torch):
    from urm.adapters.sparse_delta_memory import (
        MODE_INFERENCE,
        MODE_READ_ONLY,
        MODE_TRAINING,
        UrmSparseDeltaMemoryAdapter,
    )

    path = str(case["path"])
    mode = {
        "read_only": MODE_READ_ONLY,
        "inference": MODE_INFERENCE,
        "training": MODE_TRAINING,
    }[path]
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[str(case.get("dtype", "bfloat16"))]
    adapter = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=int(case["slots"]),
        value_dim=int(case["dim"]),
        num_writes=int(case["writes"]),
        num_reads=int(case["reads"]),
        chunk_size=int(case["chunk"]),
        mode=mode,
        device="cuda",
        dtype=dtype,
    )
    generator = torch.Generator(device="cuda").manual_seed(
        20260904 + sum(map(ord, str(case["name"])))
    )
    p, t, slots, dim = (
        int(case["parallel"]),
        int(case["sequence"]),
        int(case["slots"]),
        int(case["dim"]),
    )
    key_dim = 2 * round(math.sqrt(slots))
    write_scores = torch.randn(
        (p, t, key_dim), device="cuda", dtype=dtype, generator=generator
    )
    read_scores = torch.randn(
        (p, t, key_dim), device="cuda", dtype=dtype, generator=generator
    )
    if case.get("collision_heavy"):
        write_scores = write_scores[:, :1].expand(-1, t, -1).contiguous()
        read_scores = read_scores[:, :1].expand(-1, t, -1).contiguous()
    direct_trace = _direct_trace(adapter, write_scores, read_scores, torch)
    trace = adapter.generate_trace(write_scores, read_scores)
    for actual, expected in zip(
        (
            trace.write_indices,
            trace.write_weights,
            trace.read_indices,
            trace.read_weights,
        ),
        direct_trace,
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    initial = (
        torch.randn((p * slots, dim), device="cuda", dtype=dtype, generator=generator)
        * 0.05
    )
    values = (
        torch.randn((p, t, dim), device="cuda", dtype=dtype, generator=generator) * 0.05
    )
    beta = torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator)
    log_decay = (
        -torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator) * 0.15
    )
    return adapter, trace, initial, values, beta, log_decay


def _correctness(case, adapter, trace, initial, values, beta, log_decay, torch):
    from urm.adapters.sparse_delta_memory import SDMState
    from urm.adapters.sparse_delta_memory_reference import (
        oracle_sparse_read,
        oracle_write_read,
        torch_sparse_read,
        torch_write_read,
    )

    if case["path"] == "read_only":
        direct = adapter.direct_calls["read"](
            initial, trace.read_weights, trace.read_indices
        )
        adapted = adapter.read(SDMState(initial), trace)
        eager = torch_sparse_read(initial, trace.read_indices, trace.read_weights)
        oracle = oracle_sparse_read(
            initial.float().cpu().numpy(),
            trace.read_indices.cpu().numpy(),
            trace.read_weights.float().cpu().numpy(),
        )
        torch.testing.assert_close(direct, adapted, atol=0, rtol=0)
        torch.testing.assert_close(direct.float(), eager.float(), atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            direct.float().cpu(), torch.from_numpy(oracle), atol=0.02, rtol=0.02
        )
        return {
            "addresses_exact": True,
            "direct_adapter_output_max_abs": float(
                (direct.float() - adapted.float()).abs().max().item()
            ),
            "torch_output_max_abs": float(
                (direct.float() - eager.float()).abs().max().item()
            ),
            "oracle_output_max_abs": float(
                (direct.float().cpu() - torch.from_numpy(oracle)).abs().max().item()
            ),
            "state": "not_applicable_read_only",
        }
    direct_memory = initial.clone()
    direct, _ = adapter.direct_calls["update"](
        direct_memory,
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        log_decay,
        trace.read_indices,
        trace.read_weights,
    )
    adapted_state = SDMState(initial.clone())
    adapted, adapted_state = adapter.execute(
        adapted_state, trace, values, beta, log_decay
    )
    eager, eager_state = torch_write_read(
        initial,
        trace.write_indices,
        trace.write_weights,
        values,
        beta,
        log_decay,
        trace.read_indices,
        trace.read_weights,
    )
    torch.testing.assert_close(direct.float(), adapted.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        direct_memory.float(), adapted_state.memory.float(), atol=0.02, rtol=0.02
    )
    torch.testing.assert_close(direct.float(), eager.float(), atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        direct_memory.float(), eager_state.float(), atol=0.02, rtol=0.02
    )
    result = {
        "addresses_exact": True,
        "direct_adapter_output_max_abs": float(
            (direct.float() - adapted.float()).abs().max().item()
        ),
        "direct_adapter_state_max_abs": float(
            (direct_memory.float() - adapted_state.memory.float()).abs().max().item()
        ),
        "torch_output_max_abs": float(
            (direct.float() - eager.float()).abs().max().item()
        ),
        "torch_state_max_abs": float(
            (direct_memory.float() - eager_state.float()).abs().max().item()
        ),
        "cache_length_after": adapted_state.sequence_length,
    }
    if int(case["slots"]) <= 4096 and int(case["sequence"]) <= 64:
        from urm.adapters.sparse_delta_memory_reference import oracle_write_read

        oracle_out, oracle_state = oracle_write_read(
            initial.float().cpu().numpy(),
            trace.write_indices.cpu().numpy(),
            trace.write_weights.float().cpu().numpy(),
            values.float().cpu().numpy(),
            beta.float().cpu().numpy(),
            log_decay.float().cpu().numpy(),
            trace.read_indices.cpu().numpy(),
            trace.read_weights.float().cpu().numpy(),
        )
        torch.testing.assert_close(
            direct.float().cpu(), torch.from_numpy(oracle_out), atol=0.02, rtol=0.02
        )
        torch.testing.assert_close(
            direct_memory.float().cpu(),
            torch.from_numpy(oracle_state),
            atol=0.02,
            rtol=0.02,
        )
        result["oracle_output_max_abs"] = float(
            (direct.float().cpu() - torch.from_numpy(oracle_out)).abs().max().item()
        )
        result["oracle_state_max_abs"] = float(
            (direct_memory.float().cpu() - torch.from_numpy(oracle_state))
            .abs()
            .max()
            .item()
        )
    else:
        result["oracle"] = {
            "status": "not_applicable",
            "reason": "bounded oracle budget excludes this large/long case",
        }
    return result


def _benchmark_case(case, *, samples: int, warmup: int, torch):
    from urm.adapters.sparse_delta_memory import SDMState

    prep_start = time.perf_counter()
    adapter, trace, initial, values, beta, log_decay = _make_case(case, torch)
    torch.cuda.synchronize()
    prep_ms = (time.perf_counter() - prep_start) * 1000
    correctness = _correctness(
        case, adapter, trace, initial, values, beta, log_decay, torch
    )
    direct_memory = initial.clone()
    adapter_state = SDMState(initial.clone())
    adapter_storage_pointer = adapter_state.memory.data_ptr()
    persistent_decode = case["name"] == "decode_cached"
    direct_operation = adapter.direct_calls[
        "read" if case["path"] == "read_only" else "update"
    ]
    if case["path"] == "read_only":
        direct_call = lambda: direct_operation(
            direct_memory, trace.read_weights, trace.read_indices
        )
        adapter_call = lambda: adapter.read(adapter_state, trace)
        reset_direct = lambda: None
        reset_adapter = lambda: None
    else:
        direct_call = lambda: direct_operation(
            direct_memory,
            trace.write_indices,
            trace.write_weights,
            values,
            beta,
            log_decay,
            trace.read_indices,
            trace.read_weights,
        )
        adapter_call = lambda: adapter.execute(
            adapter_state, trace, values, beta, log_decay
        )

        def reset_direct():
            if not persistent_decode:
                direct_memory.copy_(initial)
                torch.cuda.synchronize()

        def reset_adapter():
            if not persistent_decode:
                adapter_state.memory.copy_(initial)
                adapter_state.sequence_length = 0
                torch.cuda.synchronize()

    paired = _paired_measure(
        direct_call,
        adapter_call,
        reset_direct,
        reset_adapter,
        samples=samples,
        warmup=warmup,
        torch=torch,
    )
    reset_direct()
    output_bytes = (
        int(case["parallel"])
        * int(case["sequence"])
        * int(case["dim"])
        * initial.element_size()
    )
    direct_memory_report = _memory_measure(direct_call, torch, output_bytes)
    reset_adapter()
    adapter_memory_report = _memory_measure(adapter_call, torch, output_bytes)
    median = paired["direct_device"]["median_ms"]
    queries = int(case["parallel"]) * int(case["sequence"])
    paired["direct_queries_per_second"] = queries / (median / 1000.0)
    paired["adapter_queries_per_second"] = queries / (
        paired["adapter_device"]["median_ms"] / 1000.0
    )
    address_direct = adapter.direct_calls["address"]
    operation_direct = direct_operation
    operation_adapter = (
        adapter._read_fn if case["path"] == "read_only" else adapter._update_fn
    )
    identical = _same_bound_callable(
        address_direct, adapter._address_fn
    ) and _same_bound_callable(operation_direct, operation_adapter)
    if not identical:
        raise RuntimeError(
            f"{case['name']}: direct and adapter paths do not share bound upstream callables"
        )
    return {
        "case": case,
        "input_preparation_ms": prep_ms,
        "correctness": correctness,
        "route_distribution": {
            "writes": {
                **_route_distribution(trace.write_indices, int(case["slots"])),
                "active": case["path"] != "read_only",
            },
            "reads": _route_distribution(trace.read_indices, int(case["slots"])),
        },
        "traffic_estimate": _traffic(case, initial.element_size()),
        "cache_persistence": (
            {
                "status": "measured",
                "upstream_invocations": warmup + samples + 1,
                "adapter_sequence_length": adapter_state.sequence_length,
                "storage_pointer_preserved": (
                    adapter_state.memory.data_ptr() == adapter_storage_pointer
                ),
            }
            if persistent_decode
            else {
                "status": "not_applicable",
                "reason": "case is read-only or resets mutable state outside each timed region",
            }
        ),
        "memory": {
            "direct": direct_memory_report,
            "adapter": adapter_memory_report,
        },
        "call_identity": {
            "address_direct": _function_identity(address_direct),
            "address_adapter_below_dispatch": _function_identity(adapter._address_fn),
            "direct": _function_identity(operation_direct),
            "adapter_below_dispatch": _function_identity(operation_adapter),
            "identity_basis": "same stored object, bound instance, and function",
            "identical": identical,
        },
        "paired_performance": paired,
    }


def _require_end_to_end_backward(report: dict[str, object], dtype_name: str) -> None:
    addresses = report.get("addresses", {})
    address_ok = (
        set(addresses) == {"write", "read", "passed"}
        and bool(addresses.get("passed", False))
        and all(
            all(bool(value) for value in addresses.get(kind, {}).values())
            for kind in ("write", "read")
        )
    )
    route_weights = report.get("route_weights", {})
    route_weight_ok = (
        set(route_weights) == {"write", "read", "passed"}
        and bool(route_weights.get("passed", False))
        and all(
            all(bool(value) for value in route_weights.get(kind, {}).values())
            for kind in ("write", "read")
        )
    )
    gradients = report.get("gradients", {})
    required_gradients = {
        "write_scores",
        "read_scores",
        "initial_memory",
        "values",
        "beta",
        "log_decay",
    }
    gradient_ok = set(gradients) == required_gradients and all(
        bool(gradient.get("passed", False)) for gradient in gradients.values()
    )
    if not (
        address_ok and route_weight_ok and gradient_ok and report.get("passed") is True
    ):
        raise RuntimeError(
            f"SDM {dtype_name} end-to-end backward certification failed: {report}"
        )


def _backward_correctness(torch) -> dict[str, object]:
    from urm.adapters.sparse_delta_memory_reference import (
        deterministic_tie_free_product_key_scores,
        end_to_end_differential_backward_report,
    )

    tolerances = {
        "float32": {
            "gradient_atol": 2.5e-5,
            "gradient_rtol": 2.0e-4,
            "forward_atol": 2.5e-3,
            "forward_rtol": 2.0e-3,
        },
        "bfloat16": {
            "gradient_atol": 5.0e-5,
            "gradient_rtol": 2.0e-2,
            "forward_atol": 5.0e-3,
            "forward_rtol": 2.0e-2,
        },
    }
    reports = {}
    for dtype_name, tolerance in tolerances.items():
        case = {
            "name": f"backward_{dtype_name}",
            "path": "training",
            "parallel": 1,
            "sequence": 16,
            "slots": 256,
            "dim": 32,
            "writes": 4,
            "reads": 4,
            "chunk": 16,
            "dtype": dtype_name,
        }
        adapter, _trace, memory, values, beta, log_decay = _make_case(case, torch)
        write_scores = deterministic_tie_free_product_key_scores(
            parallel=1,
            sequence=16,
            half_key=16,
            num_keys=4,
            device="cuda",
            dtype=memory.dtype,
            seed=20260905,
        )
        read_scores = deterministic_tie_free_product_key_scores(
            parallel=1,
            sequence=16,
            half_key=16,
            num_keys=4,
            device="cuda",
            dtype=memory.dtype,
            seed=20260906,
        )
        report = end_to_end_differential_backward_report(
            adapter,
            write_scores,
            read_scores,
            memory,
            values,
            beta,
            log_decay,
            **tolerance,
        )
        report["input_generation"] = {
            "kind": "deterministic random with rejection of product-key ties",
            "write_score_seed": 20260905,
            "read_score_seed": 20260906,
            "non_score_seed": 20260904 + sum(map(ord, str(case["name"]))),
            "loss_cotangent_seed": 20260907,
            "path_inputs": "independent clones",
        }
        _require_end_to_end_backward(report, dtype_name)
        reports[dtype_name] = report
    return {
        "measurement_scope": "untimed_correctness_only",
        "logical_loss": "weighted_mean(readings) + weighted_mean(final_memory)",
        "upstream_final_state_gradient": "grad_final_memory callable argument",
        "scope": "compiler-visible write_scores/read_scores through product-key top-k, Softmax, and ordered state",
        "passed": True,
        "dtypes": reports,
    }


def _performance_interpretation(rows: dict[str, object]) -> dict[str, object]:
    substantial_names = (
        "prefill_batched",
        "write_update",
        "collision_heavy",
        "training_prefill_forward_only",
        "memory_capacity",
    )
    tiny_names = ("smoke_read_only", "decode_cached")

    def device_overhead(name: str) -> dict[str, float]:
        paired = rows[name]["paired_performance"]
        return {
            "absolute_microseconds": paired["paired_device_overhead_ms"]["median_ms"]
            * 1000.0,
            "percent": paired["paired_device_overhead_fraction"]["median"] * 100.0,
        }

    substantial = {name: device_overhead(name) for name in substantial_names}
    tiny = {name: device_overhead(name) for name in tiny_names}
    substantial_percentages = [item["percent"] for item in substantial.values()]
    return {
        "classification": {
            "substantial_workloads": list(substantial_names),
            "tiny_host_bound_workloads": list(tiny_names),
            "basis": "fixed case classification; percentages are paired device-median overhead",
        },
        "substantial_workloads": {
            "cases": substantial,
            "median_percent_across_cases": statistics.median(substantial_percentages),
            "maximum_percent_across_cases": max(substantial_percentages),
        },
        "tiny_host_bound_workloads": tiny,
        "mature_kernel_gate": {
            "claimed": False,
            "reason": (
                "descriptive external-anchor measurements; this SDM slice has no "
                "predeclared artifact eligibility decision for the mature-kernel gate"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.samples < 3 or args.warmup < 1:
        raise ValueError("samples >= 3 and warmup >= 1 are required")

    process_start = time.perf_counter()
    torch = importlib.import_module("torch")
    runtime_import_ms = (time.perf_counter() - process_start) * 1000
    upstream_start = time.perf_counter()
    importlib.import_module("lingua.sparse_delta_memory")
    upstream_import_ms = (time.perf_counter() - upstream_start) * 1000

    from provenance import provenance, utc_now, write_artifact

    from urm.adapters.sparse_delta_memory import probe_sdm_support

    support = probe_sdm_support()
    if not support.supported:
        raise RuntimeError(f"SDM unavailable [{support.code}]: {support.reason}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # First-call timings use a bounded instance.  They are honest process-first
    # calls; cache directory pre-state is recorded rather than assumed cold.
    cold_read_case = CASES[0]
    first_start = time.perf_counter()
    read_adapter, read_trace, read_memory, *_ = _make_case(cold_read_case, torch)
    torch.cuda.synchronize()
    first_address_ms = (time.perf_counter() - first_start) * 1000
    from urm.adapters.sparse_delta_memory import SDMState

    first_read_start = time.perf_counter()
    read_adapter.read(SDMState(read_memory), read_trace)
    torch.cuda.synchronize()
    first_read_ms = (time.perf_counter() - first_read_start) * 1000
    del read_adapter, read_trace, read_memory

    cold_training_case = CASES[5]
    cold_adapter, cold_trace, cold_memory, cold_values, cold_beta, cold_decay = (
        _make_case(cold_training_case, torch)
    )
    torch.cuda.synchronize()
    first_update_start = time.perf_counter()
    cold_adapter.execute(
        SDMState(cold_memory),
        cold_trace,
        cold_values,
        cold_beta,
        cold_decay,
    )
    torch.cuda.synchronize()
    first_update_ms = (time.perf_counter() - first_update_start) * 1000
    del cold_adapter, cold_trace, cold_memory, cold_values, cold_beta, cold_decay
    torch.cuda.empty_cache()

    backward_correctness = _backward_correctness(torch)

    selected_cases = CASES[:3] if args.quick else CASES
    rows = {}
    for case in selected_cases:
        print(f"benchmarking {case['name']}", flush=True)
        rows[str(case["name"])] = _benchmark_case(
            case, samples=args.samples, warmup=args.warmup, torch=torch
        )

    configuration = {
        "cases": list(selected_cases),
        "samples": args.samples,
        "warmup": args.warmup,
    }
    command = " ".join(sys.argv)
    prov = provenance(command, configuration, include_gpu=True)
    prov["upstream"] = support.details
    prov["implementation_revision"] = _git_revision(Path(__file__).parents[3])
    cache_paths = {
        "triton": os.environ.get("TRITON_CACHE_DIR"),
        "torch_extensions": os.environ.get("TORCH_EXTENSIONS_DIR"),
    }
    artifact = {
        "schema_version": 2,
        "generated_utc": utc_now(),
        "provenance": prov,
        "cold_start": {
            "runtime_import_ms": runtime_import_ms,
            "upstream_import_ms": upstream_import_ms,
            "first_address_and_adapter_build_ms": first_address_ms,
            "first_read_call_ms": first_read_ms,
            "first_training_forward_build_and_call_ms": first_update_ms,
            "cache_paths": cache_paths,
            "classification": "process-first; cold only when supplied cache directories were empty",
        },
        "methodology": {
            "timed_region": "one original upstream read or gated_write_read call; adapter region adds typed validation/dispatch only",
            "excluded": "allocation/reset, tensor generation, cloning, address generation, cache initialization, and synchronization before start; decode_cached intentionally preserves its preallocated state across calls",
            "intrinsic_allocations": "upstream output/workspace allocation remains inside because it is intrinsic to the frozen callable API",
            "sampling": "paired alternating AB/BA with raw wall and CUDA-event samples",
            "training_timing": "forward-only; backward certification is untimed and reported separately",
            "dtype": "torch.bfloat16",
            "tolerances": {
                "output_atol": 0.02,
                "output_rtol": 0.02,
                "state_atol": 0.02,
                "state_rtol": 0.02,
            },
            "samples": args.samples,
            "warmup": args.warmup,
        },
        "semantics": {
            "contract": "docs/sparse-delta-memory.md",
            "address_layout": "global int64 [parallel,sequence,width], strictly ascending and unique within token",
            "mutation_order": "decay -> retrieve -> delta -> scatter/write -> post-update read",
            "collision_semantics": "ordered across tokens; within-token duplicate addresses rejected",
            "cache": "memory mutated in place; sequence length committed after upstream return",
            "native_urm_lowering": False,
        },
        "backward_correctness": backward_correctness,
        "performance_interpretation": _performance_interpretation(rows),
        "unsupported_cases": {
            "cpu_execution": {
                "status": "not_applicable",
                "reason": "upstream kernels require CUDA on SM80 or newer",
            },
            "float16_state": {
                "status": "not_applicable",
                "reason": "not in the frozen float32/bfloat16 adapter subset",
            },
            "duplicate_within_token": {
                "status": "not_applicable",
                "reason": "upstream product-key routing is unique within a token; adapter rejects duplicate supplied traces",
            },
            "short_training_sequence": {
                "status": "not_applicable",
                "reason": "pinned Triton training dot path requires sequence and effective chunk >= 16",
            },
            "unverified_revision": {
                "status": "not_applicable",
                "reason": "adapter only dispatches the exact clean pinned upstream revision",
            },
        },
        "cases": rows,
    }
    write_artifact(args.output, artifact)
    print(json.dumps({"cases": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
