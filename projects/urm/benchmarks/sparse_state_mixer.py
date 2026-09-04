"""Frozen kernel-only benchmark for URM-native SparseStateMixer v0.

The original repository is a comparator only. Both paths consume the same
precomputed certified routes; route production and certification are outside
all timed regions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

CASES_PATH = Path(__file__).with_name("sparse_state_mixer_cases.toml")
DEFAULT_OUTPUT = Path("results/sparse-state-mixer/discovery.json")
UPSTREAM_COMMIT = "183e7df809131b80ad4393741029d0f20fc3640b"
FORWARD_TOLERANCES = {
    "float32": {"native_reference_atol": 2e-5, "rtol": 2e-5, "upstream_atol": 2e-2},
    "bfloat16": {"native_reference_atol": 2e-2, "rtol": 2e-2, "upstream_atol": 2e-2},
}
BACKWARD_TOLERANCES = {
    "float32": {"atol": 3e-5, "rtol": 3e-4},
    "bfloat16": {"atol": 3e-2, "rtol": 3e-2},
}
SCHEMA_VERSION = 1
PROCESS_START_NS = time.perf_counter_ns()


def load_cases() -> tuple[dict[str, object], ...]:
    with CASES_PATH.open("rb") as handle:
        document = tomllib.load(handle)
    if document["freeze_status"] != "pre_tuning" or document["schema_version"] != 0:
        raise RuntimeError("SparseStateMixer benchmark grid is not the frozen v0 grid")
    cases = tuple(document["case"])
    names = [str(case["name"]) for case in cases]
    if len(names) != len(set(names)):
        raise RuntimeError("SparseStateMixer case names must be unique")
    return cases


def _dtype(torch, name: str):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[rank],
        "minimum_ms": ordered[0],
        "raw_ms": values,
    }


def _ratio_stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": ordered[rank],
        "minimum": ordered[0],
        "raw": values,
    }


def _bootstrap_median_ci(
    values: list[float], *, seed: int, resamples: int = 2000
) -> dict[str, float]:
    generator = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    medians.sort()
    return {
        "lower": medians[int(0.025 * resamples)],
        "upper": medians[min(resamples - 1, int(0.975 * resamples))],
    }


def _time_once(call, torch) -> tuple[float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start_event.record()
    result = call()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    device_ms = float(start_event.elapsed_time(end_event))
    del result
    return wall_ms, device_ms


def _paired_measure(
    direct_call,
    native_call,
    reset_direct,
    reset_native,
    *,
    samples: int,
    warmup: int,
    seed: int,
    torch,
) -> dict[str, object]:
    for _ in range(warmup):
        reset_direct()
        _time_once(direct_call, torch)
        reset_native()
        _time_once(native_call, torch)
    generator = random.Random(seed)
    raw = {
        "upstream_wall_ms": [],
        "native_wall_ms": [],
        "upstream_device_ms": [],
        "native_device_ms": [],
        "orders": [],
    }
    for _ in range(samples):
        order = ("upstream", "native")
        if generator.random() < 0.5:
            order = tuple(reversed(order))
        raw["orders"].append("AB" if order[0] == "upstream" else "BA")
        for path in order:
            if path == "upstream":
                reset_direct()
                wall, device = _time_once(direct_call, torch)
            else:
                reset_native()
                wall, device = _time_once(native_call, torch)
            raw[f"{path}_wall_ms"].append(wall)
            raw[f"{path}_device_ms"].append(device)
    device_ratios = [
        native / upstream
        for native, upstream in zip(
            raw["native_device_ms"], raw["upstream_device_ms"], strict=True
        )
    ]
    wall_ratios = [
        native / upstream
        for native, upstream in zip(
            raw["native_wall_ms"], raw["upstream_wall_ms"], strict=True
        )
    ]
    return {
        "sampling": "seeded randomized paired AB/BA",
        **{key: value for key, value in raw.items() if key == "orders"},
        "upstream_wall": _stats(raw["upstream_wall_ms"]),
        "native_wall": _stats(raw["native_wall_ms"]),
        "upstream_device": _stats(raw["upstream_device_ms"]),
        "native_device": _stats(raw["native_device_ms"]),
        "paired_device_ratio": {
            **_ratio_stats(device_ratios),
            "bootstrap_ci95_median": _bootstrap_median_ci(
                device_ratios, seed=seed + 100_000
            ),
        },
        "paired_wall_ratio": {
            **_ratio_stats(wall_ratios),
            "bootstrap_ci95_median": _bootstrap_median_ci(
                wall_ratios, seed=seed + 200_000
            ),
        },
        "native_host_dispatch_share": max(
            0.0,
            1.0
            - statistics.median(raw["native_device_ms"])
            / statistics.median(raw["native_wall_ms"]),
        ),
        "upstream_host_dispatch_share": max(
            0.0,
            1.0
            - statistics.median(raw["upstream_device_ms"])
            / statistics.median(raw["upstream_wall_ms"]),
        ),
    }


def _route_indices(case, width: int, *, offset: int, torch):
    p, t, slots = int(case["parallel"]), int(case["sequence"]), int(case["slots"])
    collision = str(case["collision"])
    rows = []
    for partition in range(p):
        tokens = []
        for token in range(t):
            if collision == "high":
                token_group = 0
            elif collision == "recurrent":
                token_group = token % 2
            else:
                token_group = token
            base = offset + partition * 131 + token_group * (width + 3)
            addresses = sorted({(base + route * 61) % slots for route in range(width)})
            if len(addresses) != width:
                raise RuntimeError(f"{case['name']}: route generator collided")
            tokens.append(addresses)
        rows.append(tokens)
    return torch.tensor(rows, device="cuda", dtype=torch.int64).contiguous()


def _globalize(indices, slots, torch):
    offsets = torch.arange(
        indices.shape[0], device=indices.device, dtype=torch.int64
    ).view(-1, 1, 1)
    return (indices + offsets * slots).contiguous()


def _make_case(case, torch):
    from urm.adapters.sparse_delta_memory import (
        MODE_INFERENCE,
        MODE_READ_ONLY,
        MODE_TRAINING,
        UrmSparseDeltaMemoryAdapter,
    )
    from urm.backends.sparse_state_mixer import (
        CertifiedSparseStateRoutes,
        SparseState,
        TritonSparseStateMixerBackend,
    )
    from urm.compiler.semantic import (
        DType,
        SparseReadTiming,
        SparseStateExecutionMode,
        SparseStateMixerSpec,
        SparseStateOperation,
    )

    p, t, slots, dim = (
        int(case["parallel"]),
        int(case["sequence"]),
        int(case["slots"]),
        int(case["dim"]),
    )
    writes, reads = int(case["writes"]), int(case["reads"])
    dtype = _dtype(torch, str(case["dtype"]))
    operation_name = str(case["operation"])
    operation = (
        SparseStateOperation.READ_ONLY
        if operation_name == "read_only"
        else SparseStateOperation.UPDATE
    )
    mode = (
        SparseStateExecutionMode.TRAINING
        if operation_name == "training"
        else SparseStateExecutionMode.INFERENCE
    )
    timing = (
        SparseReadTiming.CURRENT_STATE
        if operation is SparseStateOperation.READ_ONLY
        else SparseReadTiming.AFTER_UPDATE
    )
    spec = SparseStateMixerSpec(
        p,
        t,
        slots,
        dim,
        writes,
        reads,
        DType(str(case["dtype"])),
        operation,
        timing,
        mode=mode,
    )
    seed = 20260910 + sum(map(ord, str(case["name"])))
    generator = torch.Generator(device="cuda").manual_seed(seed)
    read_indices = _route_indices(case, reads, offset=2, torch=torch)
    read_weights = torch.softmax(
        torch.randn((p, t, reads), device="cuda", generator=generator).to(dtype),
        dim=-1,
    ).contiguous()
    write_indices = write_weights = None
    if writes:
        write_indices = _route_indices(case, writes, offset=1, torch=torch)
        write_weights = torch.softmax(
            torch.randn((p, t, writes), device="cuda", generator=generator).to(dtype),
            dim=-1,
        ).contiguous()
    routes = CertifiedSparseStateRoutes.certify(
        spec,
        read_indices,
        read_weights,
        write_indices=write_indices,
        write_weights=write_weights,
    )
    memory = (
        torch.randn((p, slots, dim), device="cuda", generator=generator).to(dtype)
        * 0.05
    ).contiguous()
    values = beta = log_decay = None
    if writes:
        values = (
            torch.randn((p, t, dim), device="cuda", generator=generator).to(dtype)
            * 0.05
        ).contiguous()
        beta = torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator)
        log_decay = (
            -torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator)
            * 0.1
        ).contiguous()
    native = TritonSparseStateMixerBackend(spec)
    prepared = native.prepare(routes, values=values, beta=beta, log_decay=log_decay)
    upstream_mode = {
        "read_only": MODE_READ_ONLY,
        "update": MODE_INFERENCE,
        "training": MODE_TRAINING,
    }[operation_name]
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=slots,
        value_dim=dim,
        num_writes=max(1, writes),
        num_reads=reads,
        mode=upstream_mode,
        dtype=dtype,
    )
    return {
        "case": case,
        "seed": seed,
        "torch": torch,
        "SparseState": SparseState,
        "spec": spec,
        "native": native,
        "prepared": prepared,
        "upstream": upstream,
        "memory": memory,
        "values": values,
        "beta": beta,
        "log_decay": log_decay,
        "global_read": _globalize(read_indices, slots, torch),
        "global_write": (
            _globalize(write_indices, slots, torch)
            if write_indices is not None
            else None
        ),
    }


def _max_abs(actual, expected) -> float:
    return float((actual.float() - expected.float()).abs().max().item())


def _correctness(bundle) -> dict[str, object]:
    import torch

    from urm.backends.sparse_state_reference import torch_sparse_state_mixer
    from urm.sparse_state_mixer import numpy_sparse_state_mixer

    case = bundle["case"]
    prepared = bundle["prepared"]
    routes = prepared.routes
    memory = bundle["memory"]
    values, beta, decay = bundle["values"], bundle["beta"], bundle["log_decay"]
    reference_output, reference_state = torch_sparse_state_mixer(
        memory,
        routes.read_indices,
        routes.read_weights,
        write_indices=routes.write_indices,
        write_weights=routes.write_weights,
        values=values,
        beta=beta,
        log_decay=decay,
        read_timing=prepared.spec.read_timing,
    )
    native_output, native_state = bundle["native"].execute(
        bundle["SparseState"](memory.clone()), prepared
    )
    if str(case["operation"]) == "read_only":
        upstream_state = memory.flatten(0, 1).clone()
        upstream_output = bundle["upstream"].direct_calls["read"](
            upstream_state, routes.read_weights, bundle["global_read"]
        )
    else:
        upstream_state = memory.flatten(0, 1).clone()
        upstream_output, _ = bundle["upstream"].direct_calls["update"](
            upstream_state,
            bundle["global_write"],
            routes.write_weights,
            values,
            beta,
            decay,
            bundle["global_read"],
            routes.read_weights,
        )
    oracle_output, oracle_state = numpy_sparse_state_mixer(
        memory.float().cpu().numpy(),
        routes.read_indices.cpu().numpy(),
        routes.read_weights.float().cpu().numpy(),
        write_indices=(
            routes.write_indices.cpu().numpy()
            if routes.write_indices is not None
            else None
        ),
        write_weights=(
            routes.write_weights.float().cpu().numpy()
            if routes.write_weights is not None
            else None
        ),
        values=values.float().cpu().numpy() if values is not None else None,
        beta=beta.float().cpu().numpy() if beta is not None else None,
        log_decay=decay.float().cpu().numpy() if decay is not None else None,
        read_timing=prepared.spec.read_timing,
    )
    errors = {
        "native_vs_torch_readings": _max_abs(native_output, reference_output),
        "native_vs_torch_state": _max_abs(native_state.memory, reference_state),
        "native_vs_numpy_readings": _max_abs(
            native_output.cpu(), torch.from_numpy(oracle_output)
        ),
        "native_vs_numpy_state": _max_abs(
            native_state.memory.cpu(), torch.from_numpy(oracle_state)
        ),
        "native_vs_upstream_readings": _max_abs(native_output, upstream_output),
        "native_vs_upstream_state": _max_abs(
            native_state.memory.flatten(0, 1), upstream_state
        ),
    }
    tolerance = FORWARD_TOLERANCES[str(case["dtype"])]
    if (
        errors["native_vs_torch_readings"] > tolerance["native_reference_atol"]
        or errors["native_vs_torch_state"] > tolerance["native_reference_atol"]
        or errors["native_vs_upstream_readings"] > tolerance["upstream_atol"]
        or errors["native_vs_upstream_state"] > tolerance["upstream_atol"]
        or not all(math.isfinite(value) for value in errors.values())
    ):
        raise RuntimeError(f"{case['name']}: forward correctness failed: {errors}")
    return {"passed": True, "raw_max_abs": errors, "tolerances": tolerance}


def _backward_correctness(bundle) -> dict[str, object]:
    import torch

    from urm.backends.sparse_state_mixer import CertifiedSparseStateRoutes, SparseState
    from urm.backends.sparse_state_reference import torch_sparse_state_mixer

    case, prepared = bundle["case"], bundle["prepared"]
    if str(case["operation"]) != "training":
        return {"status": "not_applicable", "reason": "case is not training"}
    memory = bundle["memory"]
    tensors = (
        memory,
        prepared.routes.write_weights,
        bundle["values"],
        bundle["beta"],
        bundle["log_decay"],
        prepared.routes.read_weights,
    )
    names = (
        "initial_memory",
        "write_weights",
        "values",
        "beta",
        "log_decay",
        "read_weights",
    )
    generator = torch.Generator(device="cuda").manual_seed(int(bundle["seed"]) + 7)
    reading_cotangent = torch.randn(
        bundle["values"].shape, device="cuda", generator=generator
    )
    memory_cotangent = torch.randn(memory.shape, device="cuda", generator=generator)

    def clones():
        return tuple(item.detach().clone().requires_grad_(True) for item in tensors)

    reference_leaves = clones()
    reference_output, reference_state = torch_sparse_state_mixer(
        reference_leaves[0],
        prepared.routes.read_indices,
        reference_leaves[5],
        write_indices=prepared.routes.write_indices,
        write_weights=reference_leaves[1],
        values=reference_leaves[2],
        beta=reference_leaves[3],
        log_decay=reference_leaves[4],
        read_timing=prepared.spec.read_timing,
    )
    reference_loss = (reference_output.float() * reading_cotangent).mean() + (
        reference_state.float() * memory_cotangent
    ).mean()
    reference_gradients = torch.autograd.grad(reference_loss, reference_leaves)

    native_leaves = clones()
    native_routes = CertifiedSparseStateRoutes.certify(
        prepared.spec,
        prepared.routes.read_indices,
        native_leaves[5],
        write_indices=prepared.routes.write_indices,
        write_weights=native_leaves[1],
    )
    native_output, native_state = bundle["native"].execute(
        SparseState(native_leaves[0]),
        bundle["native"].prepare(
            native_routes,
            values=native_leaves[2],
            beta=native_leaves[3],
            log_decay=native_leaves[4],
        ),
    )
    native_loss = (native_output.float() * reading_cotangent).mean() + (
        native_state.memory.float() * memory_cotangent
    ).mean()
    native_gradients = torch.autograd.grad(native_loss, native_leaves)

    upstream_leaves = clones()
    grad_final = (
        memory_cotangent.flatten(0, 1).to(memory.dtype) / memory.numel()
    ).contiguous()
    upstream_output, _ = bundle["upstream"].direct_calls["update"](
        upstream_leaves[0].flatten(0, 1) + 0,
        bundle["global_write"],
        upstream_leaves[1],
        upstream_leaves[2],
        upstream_leaves[3],
        upstream_leaves[4],
        bundle["global_read"],
        upstream_leaves[5],
        grad_final_memory=grad_final,
    )
    upstream_loss = (upstream_output.float() * reading_cotangent).mean()
    upstream_gradients = torch.autograd.grad(upstream_loss, upstream_leaves)
    tolerance = BACKWARD_TOLERANCES[str(case["dtype"])]
    report = {}
    passed = True
    for name, native, reference, upstream in zip(
        names,
        native_gradients,
        reference_gradients,
        upstream_gradients,
        strict=True,
    ):
        item = {
            "native_finite": bool(torch.isfinite(native).all().item()),
            "native_vs_torch_max_abs": _max_abs(native, reference),
            "native_vs_upstream_max_abs": _max_abs(native, upstream),
        }
        item["passed"] = bool(
            item["native_finite"]
            and torch.allclose(native.float(), reference.float(), **tolerance)
            and torch.allclose(native.float(), upstream.float(), **tolerance)
        )
        passed = passed and item["passed"]
        report[name] = item
    if not passed:
        raise RuntimeError(f"{case['name']}: backward correctness failed: {report}")
    return {"passed": True, "tolerances": tolerance, "gradients": report}


def _memory_measure(call, reset, torch) -> dict[str, int]:
    reset()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return {
        "allocated_before_bytes": before,
        "peak_allocated_bytes": peak,
        "temporary_peak_bytes": max(0, peak - before),
    }


def _forward_workloads(bundle):
    case, torch = bundle["case"], bundle["torch"]
    prepared, memory = bundle["prepared"], bundle["memory"]
    native_state = bundle["SparseState"](memory.clone())
    upstream_memory = memory.flatten(0, 1).clone()
    native_output = torch.empty(
        (int(case["parallel"]), int(case["sequence"]), int(case["dim"])),
        device="cuda",
        dtype=memory.dtype,
    )
    original_native = native_state.memory.clone()
    original_upstream = upstream_memory.clone()
    if str(case["operation"]) == "read_only":
        direct = lambda: bundle["upstream"].direct_calls["read"](
            upstream_memory, prepared.routes.read_weights, bundle["global_read"]
        )
        native = lambda: bundle["native"].execute(
            native_state, prepared, out=native_output
        )
        reset_direct = lambda: None
        reset_native = lambda: None
    else:
        direct = lambda: bundle["upstream"].direct_calls["update"](
            upstream_memory,
            bundle["global_write"],
            prepared.routes.write_weights,
            bundle["values"],
            bundle["beta"],
            bundle["log_decay"],
            bundle["global_read"],
            prepared.routes.read_weights,
        )
        native = lambda: bundle["native"].execute(
            native_state, prepared, out=native_output
        )

        def reset_direct():
            upstream_memory.copy_(original_upstream)
            torch.cuda.synchronize()

        def reset_native():
            native_state.memory.copy_(original_native)
            native_state.sequence_length = 0
            torch.cuda.synchronize()

    return direct, native, reset_direct, reset_native


def _backward_workloads(bundle):
    import torch

    from urm.backends.sparse_state_mixer import CertifiedSparseStateRoutes, SparseState

    prepared, memory = bundle["prepared"], bundle["memory"]
    tensors = (
        memory,
        prepared.routes.write_weights,
        bundle["values"],
        bundle["beta"],
        bundle["log_decay"],
        prepared.routes.read_weights,
    )
    generator = torch.Generator(device="cuda").manual_seed(int(bundle["seed"]) + 17)
    reading_cotangent = torch.randn(
        bundle["values"].shape, device="cuda", generator=generator
    ).div_(bundle["values"].numel())
    memory_cotangent = torch.randn(
        memory.shape, device="cuda", generator=generator
    ).div_(memory.numel())
    native_box: dict[str, object] = {}
    upstream_box: dict[str, object] = {}

    def clones():
        return tuple(item.detach().clone().requires_grad_(True) for item in tensors)

    def reset_native():
        leaves = clones()
        routes = CertifiedSparseStateRoutes.certify(
            prepared.spec,
            prepared.routes.read_indices,
            leaves[5],
            write_indices=prepared.routes.write_indices,
            write_weights=leaves[1],
        )
        output, state = bundle["native"].execute(
            SparseState(leaves[0]),
            bundle["native"].prepare(
                routes, values=leaves[2], beta=leaves[3], log_decay=leaves[4]
            ),
        )
        native_box["outputs"] = (output, state.memory)
        native_box["grad_outputs"] = (
            reading_cotangent.to(output.dtype),
            memory_cotangent.to(state.memory.dtype),
        )
        native_box["leaves"] = leaves
        torch.cuda.synchronize()

    def reset_upstream():
        leaves = clones()
        grad_final = memory_cotangent.flatten(0, 1).to(memory.dtype).contiguous()
        output, _ = bundle["upstream"].direct_calls["update"](
            leaves[0].flatten(0, 1) + 0,
            bundle["global_write"],
            leaves[1],
            leaves[2],
            leaves[3],
            leaves[4],
            bundle["global_read"],
            leaves[5],
            grad_final_memory=grad_final,
        )
        upstream_box["output"] = output
        upstream_box["grad_output"] = reading_cotangent.to(output.dtype)
        upstream_box["leaves"] = leaves
        torch.cuda.synchronize()

    def native_call():
        return torch.autograd.grad(
            native_box["outputs"],
            native_box["leaves"],
            grad_outputs=native_box["grad_outputs"],
        )

    def upstream_call():
        return torch.autograd.grad(
            upstream_box["output"],
            upstream_box["leaves"],
            grad_outputs=upstream_box["grad_output"],
        )

    return upstream_call, native_call, reset_upstream, reset_native


def _sentinel(call, reset, torch, samples=5) -> float:
    values = []
    for _ in range(samples):
        reset()
        _, device = _time_once(call, torch)
        values.append(device)
    return statistics.median(values)


def _paired_with_drift_retries(
    direct,
    native,
    reset_direct,
    reset_native,
    *,
    samples: int,
    warmup: int,
    seed: int,
    torch,
    case_name: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    attempts = []
    for attempt in range(3):
        before = _sentinel(direct, reset_direct, torch)
        performance = _paired_measure(
            direct,
            native,
            reset_direct,
            reset_native,
            samples=samples,
            warmup=warmup,
            seed=seed + attempt,
            torch=torch,
        )
        after = _sentinel(direct, reset_direct, torch)
        drift = abs(after - before) / before
        attempts.append(
            {
                "attempt": attempt + 1,
                "upstream_before_ms": before,
                "upstream_after_ms": after,
                "absolute_fraction": drift,
                "passed": drift <= 0.15,
            }
        )
        if drift <= 0.15:
            return performance, attempts
    raise RuntimeError(
        f"{case_name}: drift sentinel exhausted three attempts: {attempts}"
    )


def _analytical_work(case: dict[str, object]) -> dict[str, object]:
    queries = int(case["parallel"]) * int(case["sequence"])
    writes, reads, dim = (
        int(case["writes"]),
        int(case["reads"]),
        int(case["dim"]),
    )
    dtype_bytes = 4 if str(case["dtype"]) == "float32" else 2
    routes = queries * (writes + reads) * (8 + dtype_bytes)
    state_reads = queries * (writes + reads) * dim * dtype_bytes
    state_writes = queries * writes * dim * dtype_bytes
    value_gate_inputs = queries * (dim + (2 if writes else 0)) * dtype_bytes
    outputs = queries * dim * dtype_bytes
    logical_bytes = routes + state_reads + state_writes + value_gate_inputs + outputs
    useful_flops = queries * dim * (2 * reads + 6 * writes)
    return {
        "label": "analytical_logical_not_measured",
        "queries": queries,
        "route_bytes": routes,
        "state_read_bytes": state_reads,
        "state_write_bytes": state_writes,
        "value_gate_input_bytes": value_gate_inputs,
        "output_bytes": outputs,
        "total_bytes": logical_bytes,
        "useful_flops": useful_flops,
    }


def _performance_rates(
    performance: dict[str, object], work: dict[str, object]
) -> dict[str, object]:
    queries = int(work["queries"])
    logical_bytes = int(work["total_bytes"])

    def path_rates(path: str) -> dict[str, float]:
        median_ms = float(performance[f"{path}_device"]["median_ms"])
        return {
            "queries_per_second": queries * 1000.0 / median_ms,
            "median_device_us": median_ms * 1000.0,
            "analytical_logical_gb_per_second": logical_bytes / median_ms / 1e6,
        }

    return {"upstream": path_rates("upstream"), "native": path_rates("native")}


def _compiled_resource_profile() -> dict[str, object]:
    cache = Path(os.environ.get("TRITON_CACHE_DIR", Path.home() / ".triton" / "cache"))
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"status": "not_applicable", "reason": "cuobjdump unavailable"}
    variants = []
    pattern = re.compile(
        r"REG:(?P<registers>\d+).*STACK:(?P<stack>\d+).*"
        r"SHARED:(?P<shared>\d+).*LOCAL:(?P<local>\d+)"
    )
    for cubin in sorted(cache.rglob("_sparse_state_*kernel.cubin")):
        completed = subprocess.run(
            [cuobjdump, "-res-usage", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        )
        match = pattern.search(completed.stdout.replace("\n", " "))
        metadata_path = cubin.with_suffix(".json")
        if match is None or not metadata_path.exists():
            raise RuntimeError(f"cannot parse Triton resource metadata for {cubin}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        registers = int(match.group("registers"))
        num_warps = int(metadata["num_warps"])
        threads = num_warps * 32
        register_blocks = max(1, 65536 // max(1, registers * threads))
        occupancy = min(1.0, register_blocks * threads / 1536)
        variants.append(
            {
                "kernel": cubin.stem,
                "registers_per_thread": registers,
                "stack_bytes": int(match.group("stack")),
                "static_shared_bytes": int(match.group("shared")),
                "local_bytes": int(match.group("local")),
                "num_warps": num_warps,
                "register_limited_theoretical_occupancy": occupancy,
            }
        )
    if not variants:
        return {"status": "not_applicable", "reason": "no native variants compiled"}
    return {
        "status": "measured_static_binary_resources",
        "tool": cuobjdump,
        "variant_count": len(variants),
        "variants": variants,
        "spills": {
            "observed": any(item["local_bytes"] for item in variants),
            "basis": "cuobjdump LOCAL bytes; zero for all variants means no local spill storage",
        },
        "synchronization": "one CUDA event synchronization per timed sample, outside kernel; native forward kernel has no barrier",
        "atomics": "none in forward; relaxed FP32 partial-reduction atomics in training backward",
        "occupancy_basis": "register-bound theoretical ceiling on A10G; not a measured active-warp counter",
    }


def _first_call_measure(bundle) -> dict[str, object]:
    direct, native, reset_direct, reset_native = _forward_workloads(bundle)
    reset_direct()
    upstream_wall, upstream_device = _time_once(direct, bundle["torch"])
    reset_native()
    native_wall, native_device = _time_once(native, bundle["torch"])
    return {
        "scope": "first dispatch in this fresh process; includes specialization load/build",
        "upstream": {"wall_ms": upstream_wall, "device_ms": upstream_device},
        "native": {"wall_ms": native_wall, "device_ms": native_device},
    }


def benchmark_case(case, *, samples: int, warmup: int, torch) -> dict[str, object]:
    construction_start = time.perf_counter_ns()
    bundle = _make_case(case, torch)
    construction_wall_ms = (time.perf_counter_ns() - construction_start) / 1e6
    first_call = _first_call_measure(bundle)
    correctness = _correctness(bundle)
    backward_correctness = _backward_correctness(bundle)
    direct, native, reset_direct, reset_native = _forward_workloads(bundle)
    performance, drift_attempts = _paired_with_drift_retries(
        direct,
        native,
        reset_direct,
        reset_native,
        samples=samples,
        warmup=warmup,
        seed=int(bundle["seed"]),
        torch=torch,
        case_name=str(case["name"]),
    )
    memory = {
        "upstream": _memory_measure(direct, reset_direct, torch),
        "native": _memory_measure(native, reset_native, torch),
    }
    upstream_peak = memory["upstream"]["temporary_peak_bytes"]
    native_peak = memory["native"]["temporary_peak_bytes"]
    memory_limit = upstream_peak * 1.02 + 1024 * 1024
    device_ratio = performance["paired_device_ratio"]
    p95_ratio = (
        performance["native_device"]["p95_ms"]
        / performance["upstream_device"]["p95_ms"]
    )
    gate = {
        "paired_ci95_upper_within_1_10": device_ratio["bootstrap_ci95_median"]["upper"]
        <= 1.10,
        "p95_ratio": p95_ratio,
        "p95_within_1_10": p95_ratio <= 1.10,
        "memory_limit_bytes": memory_limit,
        "memory_passed": native_peak <= memory_limit,
    }
    gate["passed"] = all(
        gate[name]
        for name in (
            "paired_ci95_upper_within_1_10",
            "p95_within_1_10",
            "memory_passed",
        )
    )
    work = _analytical_work(case)
    backward = {
        "status": "not_applicable",
        "reason": "case is not a training benchmark",
    }
    backward_gate = {"status": "not_applicable"}
    if str(case["operation"]) == "training":
        backward_calls = _backward_workloads(bundle)
        backward = _paired_measure(
            *backward_calls,
            samples=samples,
            warmup=warmup,
            seed=int(bundle["seed"]) + 1,
            torch=torch,
        )
        backward_ratio = backward["paired_device_ratio"]
        backward_p95_ratio = (
            backward["native_device"]["p95_ms"] / backward["upstream_device"]["p95_ms"]
        )
        backward_memory = {
            "upstream": _memory_measure(backward_calls[0], backward_calls[2], torch),
            "native": _memory_measure(backward_calls[1], backward_calls[3], torch),
        }
        memory["backward"] = backward_memory
        backward_memory_limit = (
            backward_memory["upstream"]["temporary_peak_bytes"] * 1.02 + 1024 * 1024
        )
        backward_gate = {
            "paired_ci95_upper_within_1_10": backward_ratio["bootstrap_ci95_median"][
                "upper"
            ]
            <= 1.10,
            "p95_ratio": backward_p95_ratio,
            "p95_within_1_10": backward_p95_ratio <= 1.10,
            "memory_passed": backward_memory["native"]["temporary_peak_bytes"]
            <= backward_memory_limit,
        }
        backward_gate["passed"] = all(
            backward_gate[name]
            for name in (
                "paired_ci95_upper_within_1_10",
                "p95_within_1_10",
                "memory_passed",
            )
        )
    return {
        "case": case,
        "seed": bundle["seed"],
        "construction_wall_ms": construction_wall_ms,
        "first_call": first_call,
        "correctness": correctness,
        "backward_correctness": backward_correctness,
        "forward": performance,
        "backward": backward,
        "memory": memory,
        "drift_sentinel": {
            "threshold": 0.15,
            "maximum_attempts": 3,
            "accepted_attempt": len(drift_attempts),
            "attempts": drift_attempts,
            "passed": True,
        },
        "forward_gate": gate,
        "backward_gate": backward_gate,
        "analytical_work": work,
        "rates": _performance_rates(performance, work),
        "launch_schedule": bundle["native"].launch_schedule(),
    }


def _aggregate(rows: dict[str, object]) -> dict[str, object]:
    substantial = [
        row for row in rows.values() if row["case"]["classification"] == "substantial"
    ]
    ratios = [row["forward"]["paired_device_ratio"]["median"] for row in substantial]
    geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    return {
        "substantial_case_count": len(substantial),
        "geometric_mean_native_over_upstream": geometric_mean,
        "geometric_mean_passed": geometric_mean <= 1.0,
        "every_case_passed": all(row["forward_gate"]["passed"] for row in substantial),
        "three_process_confirmation": {
            "status": "not_applicable",
            "reason": "single-process discovery artifact",
        },
        "passed": False,
    }


def _confirmation_phase_passed(
    *,
    classification: str,
    hierarchical_upper: float,
    per_process_medians: list[float],
    per_process_p95_ratios: list[float],
    memory_passes: list[bool],
    per_process_gate_passes: list[bool],
) -> bool:
    common = hierarchical_upper <= 1.10 and all(memory_passes)
    if classification == "host_bound":
        return common and max(per_process_medians) <= 1.10
    return (
        common and max(per_process_p95_ratios) <= 1.10 and all(per_process_gate_passes)
    )


def _confirmation_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    from measurement import hierarchical_bootstrap_paired_slowdown

    case_names = tuple(runs[0]["cases"])
    cases = {}
    substantial_forward_medians = []
    all_passed = True
    for case_index, name in enumerate(case_names):
        run_rows = [run["cases"][name] for run in runs]
        phase_reports = {}
        for phase in ("forward", "backward"):
            if phase == "backward" and run_rows[0]["backward"].get("status"):
                phase_reports[phase] = {"status": "not_applicable"}
                continue
            paired_log_runs = [
                [math.log(value) for value in row[phase]["paired_device_ratio"]["raw"]]
                for row in run_rows
            ]
            median_pct, lower_pct, upper_pct = hierarchical_bootstrap_paired_slowdown(
                paired_log_runs,
                num_resamples=10_000,
                seed=20260940 + case_index + (10_000 if phase == "backward" else 0),
            )
            median, lower, upper = (
                1.0 + median_pct / 100.0,
                1.0 + lower_pct / 100.0,
                1.0 + upper_pct / 100.0,
            )
            p95_ratios = [
                row[phase]["native_device"]["p95_ms"]
                / row[phase]["upstream_device"]["p95_ms"]
                for row in run_rows
            ]
            median_ratios = [
                row[phase]["paired_device_ratio"]["median"] for row in run_rows
            ]
            p95_absolute_delta_us = [
                (
                    row[phase]["native_device"]["p95_ms"]
                    - row[phase]["upstream_device"]["p95_ms"]
                )
                * 1000.0
                for row in run_rows
            ]
            gate_name = f"{phase}_gate"
            classification = str(run_rows[0]["case"]["classification"])
            phase_passed = _confirmation_phase_passed(
                classification=classification,
                hierarchical_upper=upper,
                per_process_medians=median_ratios,
                per_process_p95_ratios=p95_ratios,
                memory_passes=[
                    bool(row[gate_name]["memory_passed"]) for row in run_rows
                ],
                per_process_gate_passes=[
                    bool(row[gate_name]["passed"]) for row in run_rows
                ],
            )
            phase_reports[phase] = {
                "hierarchical_paired_median_ratio": median,
                "hierarchical_bootstrap_ci95": {"lower": lower, "upper": upper},
                "per_process_median_ratios": median_ratios,
                "per_process_p95_ratios": p95_ratios,
                "per_process_p95_absolute_delta_us": p95_absolute_delta_us,
                "host_bound_policy_applied": classification == "host_bound",
                "passed": phase_passed,
            }
            all_passed = all_passed and phase_passed
            if (
                phase == "forward"
                and run_rows[0]["case"]["classification"] == "substantial"
            ):
                substantial_forward_medians.append(median)
        correctness_passed = all(
            row["correctness"]["passed"]
            and (
                row["backward_correctness"].get("status") == "not_applicable"
                or row["backward_correctness"]["passed"]
            )
            for row in run_rows
        )
        all_passed = all_passed and correctness_passed
        cases[name] = {
            "classification": run_rows[0]["case"]["classification"],
            "correctness_passed_all_processes": correctness_passed,
            **phase_reports,
        }
    geometric_mean = math.exp(
        sum(math.log(value) for value in substantial_forward_medians)
        / len(substantial_forward_medians)
    )
    all_passed = all_passed and geometric_mean <= 1.0
    return {
        "process_count": len(runs),
        "hierarchical_bootstrap_resamples": 10_000,
        "cases": cases,
        "substantial_forward_geometric_mean_native_over_upstream": geometric_mean,
        "geometric_mean_passed": geometric_mean <= 1.0,
        "all_case_forward_backward_correctness_latency_memory_passed": all_passed,
        "passed": all_passed,
    }


def _run_confirmation(args) -> None:
    from provenance import provenance, utc_now, write_artifact

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
    ]
    for name in args.case:
        command.extend(("--case", name))
    runs = []
    process_wall_ms = []
    with tempfile.TemporaryDirectory(prefix="urm-sparse-state-confirm-") as root:
        root_path = Path(root)
        for index in range(args.confirmation_processes):
            output = root_path / f"run-{index}.json"
            cache_root = root_path / f"cache-{index}"
            child_command = [*command, "--output", str(output)]
            environment = os.environ.copy()
            environment["TRITON_CACHE_DIR"] = str(cache_root / "triton")
            environment["TORCH_EXTENSIONS_DIR"] = str(cache_root / "torch-extensions")
            environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / "inductor")
            started = time.perf_counter_ns()
            subprocess.run(child_command, check=True, env=environment)
            process_wall_ms.append((time.perf_counter_ns() - started) / 1e6)
            runs.append(json.loads(output.read_text(encoding="utf-8")))
    revisions = {run["provenance"]["git_revision"] for run in runs}
    if len(revisions) != 1 or any(run["provenance"]["dirty_tree"] for run in runs):
        raise RuntimeError("confirmation requires one clean implementation revision")
    upstream_commits = {
        run["provenance"]["upstream"].get("installed_commit") for run in runs
    }
    upstream_dirty = {
        run["provenance"]["upstream"].get("checkout_dirty") for run in runs
    }
    if upstream_commits != {UPSTREAM_COMMIT} or upstream_dirty != {False}:
        raise RuntimeError("confirmation requires the clean pinned upstream checkout")
    grid = load_cases()
    configuration = {
        "grid": grid,
        "selected": [
            case["name"] for case in grid if not args.case or case["name"] in args.case
        ],
        "samples_per_process": args.samples,
        "warmup": args.warmup,
        "processes": args.confirmation_processes,
    }
    confirmation = _confirmation_summary(runs)
    if not confirmation["passed"]:
        raise RuntimeError(f"three-process completion gate failed: {confirmation}")
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "three_process_confirmation",
        "generated_utc": utc_now(),
        "provenance": provenance(" ".join(sys.argv), configuration, include_gpu=True),
        "methodology": {
            "fresh_processes": args.confirmation_processes,
            "fresh_per_process_caches": [
                "TRITON_CACHE_DIR",
                "TORCH_EXTENSIONS_DIR",
                "TORCHINDUCTOR_CACHE_DIR",
            ],
            "process_wall_ms": process_wall_ms,
            "sampling": "seeded randomized paired AB/BA",
            "kernel_only": True,
        },
        "runs": runs,
        "confirmation": confirmation,
    }
    write_artifact(args.output, artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--confirmation-processes", type=int, default=1)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 3 or args.warmup < 1:
        raise ValueError("samples >= 3 and warmup >= 1 are required")
    if args.confirmation_processes < 1:
        raise ValueError("confirmation-processes must be positive")
    if args.confirmation_processes > 1 and not args.child:
        _run_confirmation(args)
        return
    import torch
    from provenance import provenance, utc_now, write_artifact

    from urm.adapters.sparse_delta_memory import probe_sdm_support

    support = probe_sdm_support()
    if not support.supported:
        raise RuntimeError(
            f"pinned comparator unavailable [{support.code}]: {support.reason}"
        )
    all_cases = load_cases()
    selected = tuple(
        case for case in all_cases if not args.case or case["name"] in args.case
    )
    unknown = set(args.case) - {str(case["name"]) for case in selected}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    configuration = {
        "grid": all_cases,
        "selected": [case["name"] for case in selected],
        "samples": args.samples,
        "warmup": args.warmup,
    }
    rows = {}
    for case in selected:
        print(f"benchmarking {case['name']}", flush=True)
        rows[str(case["name"])] = benchmark_case(
            case, samples=args.samples, warmup=args.warmup, torch=torch
        )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "single_process_run",
        "generated_utc": utc_now(),
        "provenance": {
            **provenance(" ".join(sys.argv), configuration, include_gpu=True),
            "upstream": support.details,
        },
        "cold_start": {
            "process_start_to_dependency_probe_ms": (
                time.perf_counter_ns() - PROCESS_START_NS
            )
            / 1e6,
            "fresh_cache_environment": {
                name: os.environ.get(name)
                for name in (
                    "TRITON_CACHE_DIR",
                    "TORCH_EXTENSIONS_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                )
            },
        },
        "scope": {
            "kernel_only": "identical precomputed certified routes through state operation",
            "pipeline": {
                "status": "not_applicable",
                "reason": "native route-selection kernel does not exist",
            },
        },
        "fixed_tolerances": {
            "forward": FORWARD_TOLERANCES,
            "backward": BACKWARD_TOLERANCES,
        },
        "methodology": {
            "sampling": "seeded randomized paired AB/BA",
            "samples": args.samples,
            "warmup": args.warmup,
            "timed_region_excludes": "route generation/certification, tensor allocation, reset, cloning, compilation, cache initialization",
            "upstream_commit": UPSTREAM_COMMIT,
            "drift_protocol": "15% upstream sentinel threshold; at most three whole-measurement attempts, all retained",
            "mfu_mbu": {
                "status": "not_applicable",
                "reason": "random sparse state traffic and cache reuse are analytical, not hardware-counter measurements",
            },
        },
        "optimization_history": [
            {
                "id": "O1_matched_backward_cotangents",
                "kind": "measurement_correction",
                "hypothesis": "explicit preallocated output cotangents remove scalar-loss graph work excluded by the upstream API",
                "result": "retained; scalar-loss differential correctness remains separate and untimed",
                "status": "accepted",
            },
            {
                "id": "O2_bf16_state_cotangent_storage",
                "kind": "kernel_storage",
                "hypothesis": "store recurrent state cotangents in the BF16 semantic dtype while retaining FP32 arithmetic and reductions",
                "result": "retained after forward/backward differential gates and frozen memory gate",
                "status": "accepted",
            },
            {
                "id": "O3_upstream_schedule_transplant",
                "kind": "rejected_without_implementation",
                "hypothesis": "copy an upstream schedule to optimize already-passing cases",
                "result": "rejected: no measured native bottleneck and the schedule is not URM-owned evidence",
                "status": "rejected",
            },
        ],
        "native_profile": _compiled_resource_profile(),
        "cases": rows,
        "aggregate": _aggregate(rows),
    }
    write_artifact(args.output, artifact)


if __name__ == "__main__":
    main()
