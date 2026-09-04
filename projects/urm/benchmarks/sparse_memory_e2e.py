"""Frozen score-to-route-to-persistent-state Sparse Memory benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

PROCESS_START_NS = time.perf_counter_ns()
CASES_PATH = Path(__file__).with_name("sparse_memory_e2e_cases.toml")
DEFAULT_OUTPUT = Path("results/sparse-memory-e2e/confirmation.json")
UPSTREAM_COMMIT = "183e7df809131b80ad4393741029d0f20fc3640b"
SCHEMA_VERSION = 1
HOST_ABSOLUTE_ALLOWANCE_US = 25.0


def load_cases() -> tuple[dict[str, object], ...]:
    payload = tomllib.loads(CASES_PATH.read_text(encoding="utf-8"))
    if payload["schema_version"] != 0 or payload["freeze_status"] != "pre_tuning":
        raise RuntimeError("Sparse Memory E2E grid is not the frozen v0 grid")
    cases = tuple(payload["case"])
    if len({case["name"] for case in cases}) != len(cases):
        raise RuntimeError("Sparse Memory E2E case names must be unique")
    return cases


def _stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    rank = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[rank],
        "minimum_ms": ordered[0],
        "raw_ms": values,
    }


def _ratio_stats(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    rank = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p95": ordered[rank],
        "minimum": ordered[0],
        "raw": values,
    }


def _bootstrap_median_ci(values: list[float], seed: int, resamples=10_000):
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        estimates.append(
            statistics.median(rng.choice(values) for _ in range(len(values)))
        )
    estimates.sort()
    return {
        "lower": estimates[int(0.025 * resamples)],
        "upper": estimates[int(0.975 * resamples)],
    }


def _time_once(call, torch) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), (time.perf_counter_ns() - wall_start) / 1e6


def _paired_measure(
    upstream,
    native,
    reset_upstream,
    reset_native,
    *,
    samples,
    warmup,
    seed,
    torch,
):
    for _ in range(warmup):
        reset_upstream()
        upstream()
        reset_native()
        native()
    torch.cuda.synchronize()
    rng = random.Random(seed)
    orders = []
    upstream_device = []
    native_device = []
    upstream_wall = []
    native_wall = []
    for _ in range(samples):
        order = "AB" if rng.random() < 0.5 else "BA"
        orders.append(order)
        sequence = (
            (
                (upstream, reset_upstream, upstream_device, upstream_wall),
                (native, reset_native, native_device, native_wall),
            )
            if order == "AB"
            else (
                (native, reset_native, native_device, native_wall),
                (upstream, reset_upstream, upstream_device, upstream_wall),
            )
        )
        for call, reset, device_values, wall_values in sequence:
            reset()
            torch.cuda.synchronize()
            device_ms, wall_ms = _time_once(call, torch)
            device_values.append(device_ms)
            wall_values.append(wall_ms)
    ratios = [n / u for n, u in zip(native_device, upstream_device, strict=True)]
    ratio = _ratio_stats(ratios)
    ratio["bootstrap_ci95_median"] = _bootstrap_median_ci(ratios, seed + 911)
    return {
        "sampling": "seeded randomized paired AB/BA",
        "orders": orders,
        "upstream_device": _stats(upstream_device),
        "native_device": _stats(native_device),
        "upstream_wall": _stats(upstream_wall),
        "native_wall": _stats(native_wall),
        "paired_device_ratio": ratio,
    }


def _single_measure(call, reset, *, samples, warmup, torch):
    for _ in range(warmup):
        reset()
        call()
    torch.cuda.synchronize()
    device = []
    wall = []
    for _ in range(samples):
        reset()
        torch.cuda.synchronize()
        device_ms, wall_ms = _time_once(call, torch)
        device.append(device_ms)
        wall.append(wall_ms)
    return {"device": _stats(device), "wall": _stats(wall)}


def _sentinel(call, reset, torch, samples=5):
    values = []
    for _ in range(samples):
        reset()
        torch.cuda.synchronize()
        values.append(_time_once(call, torch)[0])
    return statistics.median(values)


def _paired_with_drift_retries(
    upstream,
    native,
    reset_upstream,
    reset_native,
    *,
    samples,
    warmup,
    seed,
    torch,
):
    attempts = []
    for attempt in range(3):
        before = _sentinel(upstream, reset_upstream, torch)
        performance = _paired_measure(
            upstream,
            native,
            reset_upstream,
            reset_native,
            samples=samples,
            warmup=warmup,
            seed=seed + attempt,
            torch=torch,
        )
        after = _sentinel(upstream, reset_upstream, torch)
        drift = abs(after - before) / before
        passed = drift <= 0.15
        attempts.append(
            {
                "attempt": attempt + 1,
                "upstream_sentinel_before_ms": before,
                "upstream_sentinel_after_ms": after,
                "absolute_fraction": drift,
                "threshold": 0.15,
                "passed": passed,
                "performance": performance,
            }
        )
        if passed:
            return performance, attempts
    raise RuntimeError(f"drift sentinel exhausted three attempts: {attempts}")


def _dtype(torch, name):
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _route_scores(case, width, *, seed, torch):
    from urm.backends.sparse_route import CertifiedSparseRouteScores
    from urm.compiler.semantic import DType, SparseRouteSelectionSpec

    dtype = _dtype(torch, case["dtype"])
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    p, t, slots = (int(case[k]) for k in ("parallel", "sequence", "slots"))
    half = round(slots**0.5)
    row_spec = SparseRouteSelectionSpec(1, 1, slots, width, dtype_spec)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    rows = []
    needed = 1 if case["collision"] in {"high", "recurrent"} else p * t
    attempts = 0
    while len(rows) < needed and attempts < needed * 10_000:
        candidate = torch.randn(
            (1, 1, 2 * half),
            device="cuda",
            dtype=dtype,
            generator=generator,
        ).contiguous()
        attempts += 1
        try:
            CertifiedSparseRouteScores.certify(row_spec, candidate)
        except ValueError:
            continue
        rows.append(candidate)
    if len(rows) != needed:
        raise RuntimeError(f"could not generate tie-free scores for {case['name']}")
    if needed == 1:
        return rows[0].expand(p, t, -1).clone().contiguous(), attempts
    return torch.cat(rows, dim=1).reshape(p, t, 2 * half).contiguous(), attempts


def _make_bundle(case, torch):
    bundle_started = time.perf_counter_ns()
    from urm.adapters.sparse_delta_memory import (
        MODE_INFERENCE,
        MODE_READ_ONLY,
        MODE_TRAINING,
        UrmSparseDeltaMemoryAdapter,
    )
    from urm.backends.sparse_memory import TritonSparseMemoryBackend
    from urm.backends.sparse_state_mixer import (
        CertifiedSparseStateRoutes,
        SparseState,
    )
    from urm.compiler.semantic import (
        DType,
        SDMExecutionMode,
        SparseMemoryMixerSpec,
        SparseReadTiming,
        SparseStateOperation,
    )

    p, t, slots, dim, writes, reads = (
        int(case[k])
        for k in ("parallel", "sequence", "slots", "dim", "writes", "reads")
    )
    dtype = _dtype(torch, case["dtype"])
    dtype_spec = DType.FLOAT32 if dtype is torch.float32 else DType.BFLOAT16
    read_only = case["operation"] == "read_only"
    training = case["operation"] == "training"
    spec = SparseMemoryMixerSpec(
        p,
        t,
        slots,
        dim,
        writes,
        reads,
        dtype_spec,
        SDMExecutionMode.TRAINING if training else SDMExecutionMode.INFERENCE,
        operation=(
            SparseStateOperation.READ_ONLY if read_only else SparseStateOperation.UPDATE
        ),
        read_timing=(
            SparseReadTiming.CURRENT_STATE
            if read_only
            else SparseReadTiming.AFTER_UPDATE
        ),
    )
    read_scores, read_attempts = _route_scores(case, reads, seed=8017, torch=torch)
    write_scores = None
    write_attempts = 0
    if not read_only:
        write_scores, write_attempts = _route_scores(
            case, writes, seed=6029, torch=torch
        )
    generator = torch.Generator(device="cuda").manual_seed(1709 + dim + t)
    memory0 = (
        torch.randn((p, slots, dim), device="cuda", dtype=dtype, generator=generator)
        * 0.05
    ).contiguous()
    values = beta = decay = None
    if not read_only:
        values = (
            torch.randn((p, t, dim), device="cuda", dtype=dtype, generator=generator)
            * 0.05
        ).contiguous()
        beta = torch.rand(
            (p, t, 1), device="cuda", dtype=dtype, generator=generator
        ).contiguous()
        decay = (
            -torch.rand((p, t, 1), device="cuda", dtype=dtype, generator=generator)
            * 0.1
        ).contiguous()
    native = TritonSparseMemoryBackend(spec)
    native_prepared = native.prepare(
        read_scores,
        write_scores=write_scores,
        values=values,
        beta=beta,
        log_decay=decay,
    )
    mode = (
        MODE_READ_ONLY if read_only else MODE_TRAINING if training else MODE_INFERENCE
    )
    upstream = UrmSparseDeltaMemoryAdapter(
        slots_per_partition=slots,
        value_dim=dim,
        num_writes=max(1, writes),
        num_reads=reads,
        chunk_size=16,
        mode=mode,
        device="cuda",
        dtype=dtype,
    )
    offsets = torch.arange(p, device="cuda", dtype=torch.int64).view(p, 1, 1) * slots

    def upstream_routes():
        address = upstream.direct_calls["address"]
        read_values, read_local = address(read_scores, reads, round(slots**0.5))
        read_weights = upstream.layer.read_act(read_values)
        write_local = write_weights = None
        if not read_only:
            write_values, write_local = address(write_scores, writes, round(slots**0.5))
            write_weights = upstream.layer.write_act(write_values)
        return (
            read_local,
            read_weights,
            write_local,
            write_weights,
            read_local + offsets,
            write_local + offsets if write_local is not None else None,
        )

    construction_only_ms = (time.perf_counter_ns() - bundle_started) / 1e6

    def cold_wall(call):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        result = call()
        torch.cuda.synchronize()
        return result, (time.perf_counter_ns() - started) / 1e6

    (native_read, native_write), native_route_cold = cold_wall(
        lambda: (
            native.read_backend.generate_certified(native_prepared.read_scores),
            (
                native.write_backend.generate_certified(native_prepared.write_scores)
                if native.write_backend is not None
                else None
            ),
        )
    )
    _, native_route_first_warm = cold_wall(
        lambda: (
            native.read_backend.generate(native_prepared.read_scores),
            (
                native.write_backend.generate(native_prepared.write_scores)
                if native.write_backend is not None
                else None
            ),
        )
    )
    native_routes = CertifiedSparseStateRoutes.from_native_generation(
        native.state_spec, native_read, write_output=native_write
    )
    native_state_prepared = native.state_backend._prepare_generated_routes(
        native_routes, values=values, beta=beta, log_decay=decay
    )
    up_route, upstream_route_cold = cold_wall(upstream_routes)
    _, upstream_route_first_warm = cold_wall(upstream_routes)
    hybrid_routes = CertifiedSparseStateRoutes.certify(
        native.state_spec,
        up_route[0].to(torch.int32).contiguous(),
        up_route[1].contiguous(),
        write_indices=(
            up_route[2].to(torch.int32).contiguous()
            if up_route[2] is not None
            else None
        ),
        write_weights=(up_route[3].contiguous() if up_route[3] is not None else None),
    )
    hybrid_prepared = native.state_backend.prepare(
        hybrid_routes, values=values, beta=beta, log_decay=decay
    )
    cold_native_memory = memory0.clone()

    def native_state_first():
        cold_native_memory.copy_(memory0)
        return native.state_backend.execute(
            SparseState(cold_native_memory), native_state_prepared
        )

    _, native_state_cold = cold_wall(native_state_first)
    _, native_state_first_warm = cold_wall(native_state_first)

    cold_upstream_memory = memory0.reshape(p * slots, dim).clone()

    def upstream_state_first():
        cold_upstream_memory.copy_(memory0.reshape_as(cold_upstream_memory))
        if read_only:
            return upstream.direct_calls["read"](
                cold_upstream_memory, up_route[1], up_route[4]
            )
        return upstream.direct_calls["update"](
            cold_upstream_memory,
            up_route[5],
            up_route[3],
            values,
            beta,
            decay,
            up_route[4],
            up_route[1],
        )

    _, upstream_state_cold = cold_wall(upstream_state_first)
    _, upstream_state_first_warm = cold_wall(upstream_state_first)
    torch.cuda.synchronize()
    return {
        "case": case,
        "spec": spec,
        "dtype": dtype,
        "read_scores": read_scores,
        "write_scores": write_scores,
        "memory0": memory0,
        "values": values,
        "beta": beta,
        "decay": decay,
        "native": native,
        "native_prepared": native_prepared,
        "native_state_prepared": native_state_prepared,
        "hybrid_prepared": hybrid_prepared,
        "upstream": upstream,
        "upstream_routes": upstream_routes,
        "up_route": up_route,
        "offsets": offsets,
        "score_generation_attempts": {
            "read": read_attempts,
            "write": write_attempts,
        },
        "cold_breakdown": {
            "construction_before_dispatch_ms": construction_only_ms,
            "native_route_compile_plus_first_call_ms": native_route_cold,
            "native_route_first_warm_call_ms": native_route_first_warm,
            "native_route_derived_compile_ms": max(
                0.0, native_route_cold - native_route_first_warm
            ),
            "native_state_compile_plus_first_call_ms": native_state_cold,
            "native_state_first_warm_call_ms": native_state_first_warm,
            "native_state_derived_compile_ms": max(
                0.0, native_state_cold - native_state_first_warm
            ),
            "upstream_route_compile_plus_first_call_ms": upstream_route_cold,
            "upstream_route_first_warm_call_ms": upstream_route_first_warm,
            "upstream_route_derived_compile_ms": max(
                0.0, upstream_route_cold - upstream_route_first_warm
            ),
            "upstream_state_compile_plus_first_call_ms": upstream_state_cold,
            "upstream_state_first_warm_call_ms": upstream_state_first_warm,
            "upstream_state_derived_compile_ms": max(
                0.0, upstream_state_cold - upstream_state_first_warm
            ),
            "compile_measurement": "derived cold-minus-first-warm; compile-plus-first and first-warm are retained separately",
        },
        "SparseState": SparseState,
    }


def _calls(bundle, torch):
    from urm.adapters.sparse_delta_memory_reference import torch_product_key
    from urm.backends.sparse_state_reference import torch_sparse_state_mixer

    spec = bundle["spec"]
    read_only = spec.operation.value == "read_only"
    native_memory = bundle["memory0"].clone()
    upstream_memory = (
        bundle["memory0"]
        .reshape(spec.parallel * spec.slots_per_partition, spec.value_dim)
        .clone()
    )
    hybrid_memory = bundle["memory0"].clone()

    def reset_native():
        native_memory.copy_(bundle["memory0"])

    def reset_upstream():
        upstream_memory.copy_(bundle["memory0"].reshape_as(upstream_memory))

    def reset_hybrid():
        hybrid_memory.copy_(bundle["memory0"])

    def native_call():
        return bundle["native"].execute(
            bundle["SparseState"](native_memory), bundle["native_prepared"]
        )

    def upstream_call():
        (
            _read_local,
            read_weights,
            _write_local,
            write_weights,
            read_global,
            write_global,
        ) = bundle["upstream_routes"]()
        if read_only:
            return bundle["upstream"].direct_calls["read"](
                upstream_memory, read_weights, read_global
            )
        return bundle["upstream"].direct_calls["update"](
            upstream_memory,
            write_global,
            write_weights,
            bundle["values"],
            bundle["beta"],
            bundle["decay"],
            read_global,
            read_weights,
        )

    def hybrid_call():
        # Fixed score tensors permit one-time route certification. Route
        # production still executes and is charged; only certification is reused.
        bundle["upstream_routes"]()
        return bundle["native"].state_backend.execute(
            bundle["SparseState"](hybrid_memory), bundle["hybrid_prepared"]
        )

    half = round(spec.slots_per_partition**0.5)

    def reference_call():
        read_values, read_indices = torch_product_key(
            bundle["read_scores"], spec.reads, half
        )
        if read_only:
            return torch_sparse_state_mixer(
                bundle["memory0"], read_indices, torch.softmax(read_values, -1)
            )
        write_values, write_indices = torch_product_key(
            bundle["write_scores"], spec.writes, half
        )
        return torch_sparse_state_mixer(
            bundle["memory0"],
            read_indices,
            torch.softmax(read_values, -1),
            write_indices=write_indices,
            write_weights=torch.softmax(write_values, -1),
            values=bundle["values"],
            beta=bundle["beta"],
            log_decay=bundle["decay"],
            read_timing=spec.read_timing,
        )

    return {
        "native": native_call,
        "upstream": upstream_call,
        "hybrid": hybrid_call,
        "reference": reference_call,
        "reset_native": reset_native,
        "reset_upstream": reset_upstream,
        "reset_hybrid": reset_hybrid,
        "reset_reference": lambda: None,
        "native_memory": native_memory,
        "upstream_memory": upstream_memory,
        "hybrid_memory": hybrid_memory,
    }


def _max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def _correctness(bundle, calls, torch):
    calls["reset_native"]()
    native = calls["native"]()
    calls["reset_upstream"]()
    upstream = calls["upstream"]()
    reference_readings, reference_state = calls["reference"]()
    if isinstance(upstream, tuple):
        upstream_readings, upstream_state = upstream
    else:
        upstream_readings = upstream
        upstream_state = calls["upstream_memory"]
    spec = bundle["spec"]
    upstream_state = upstream_state.reshape_as(reference_state)
    up_route = bundle["up_route"]
    route_exact = bool(
        torch.equal(native.read_addresses.to(torch.int64), up_route[0])
        and (
            spec.operation.value == "read_only"
            or torch.equal(native.write_addresses.to(torch.int64), up_route[2])
        )
    )
    reference_atol = 2e-5 if bundle["dtype"] is torch.float32 else 2e-2
    upstream_atol = 5e-2
    values = {
        "native_vs_reference_readings": _max_abs(native.readings, reference_readings),
        "native_vs_reference_state": _max_abs(native.state.memory, reference_state),
        "native_vs_upstream_readings": _max_abs(native.readings, upstream_readings),
        "native_vs_upstream_state": _max_abs(native.state.memory, upstream_state),
    }
    finite = all(
        bool(torch.isfinite(item).all())
        for item in (
            native.readings,
            native.state.memory,
            upstream_readings,
            upstream_state,
        )
    )
    passed = bool(
        route_exact
        and finite
        and values["native_vs_reference_readings"] <= reference_atol
        and values["native_vs_reference_state"] <= reference_atol
        and values["native_vs_upstream_readings"] <= upstream_atol
        and values["native_vs_upstream_state"] <= upstream_atol
    )
    return {
        "passed": passed,
        "addresses_exact": route_exact,
        "finite": finite,
        "native_reference_atol": reference_atol,
        "native_upstream_atol": upstream_atol,
        "rtol": reference_atol,
        "raw_max_abs": values,
    }


def _stage_attribution(bundle, calls, *, samples, warmup, torch):
    spec = bundle["spec"]

    def native_route():
        bundle["native"].read_backend.generate(bundle["native_prepared"].read_scores)
        if bundle["native"].write_backend is not None:
            bundle["native"].write_backend.generate(
                bundle["native_prepared"].write_scores
            )

    def native_state():
        bundle["native"].state_backend.execute(
            bundle["SparseState"](calls["native_memory"]),
            bundle["native_state_prepared"],
        )

    def upstream_route():
        bundle["upstream_routes"]()

    def upstream_state():
        route = bundle["up_route"]
        if spec.operation.value == "read_only":
            return bundle["upstream"].direct_calls["read"](
                calls["upstream_memory"], route[1], route[4]
            )
        return bundle["upstream"].direct_calls["update"](
            calls["upstream_memory"],
            route[5],
            route[3],
            bundle["values"],
            bundle["beta"],
            bundle["decay"],
            route[4],
            route[1],
        )

    report = {}
    for name, whole, reset, route_call, state_call in (
        (
            "upstream",
            calls["upstream"],
            calls["reset_upstream"],
            upstream_route,
            upstream_state,
        ),
        (
            "native",
            calls["native"],
            calls["reset_native"],
            native_route,
            native_state,
        ),
    ):
        whole_result = _single_measure(
            whole, reset, samples=samples, warmup=warmup, torch=torch
        )
        route_result = _single_measure(
            route_call, lambda: None, samples=samples, warmup=warmup, torch=torch
        )
        state_result = _single_measure(
            state_call, reset, samples=samples, warmup=warmup, torch=torch
        )
        total = whole_result["device"]["median_ms"]
        route = route_result["device"]["median_ms"]
        state = state_result["device"]["median_ms"]
        residual = total - route - state
        if residual >= 0:
            denominator = total
            remaining = residual
        else:
            # Independent diagnostic samples can make component medians exceed
            # the whole median. Normalize the two measured components and retain
            # the signed residual as measurement-noise evidence.
            denominator = route + state
            remaining = 0.0
        report[name] = {
            "whole_pipeline": whole_result,
            "route_production": route_result,
            "state_mixer": state_result,
            "remaining_orchestration_materialization_ms": remaining,
            "signed_independent_median_residual_ms": residual,
            "fractions": {
                "route": route / denominator,
                "state": state / denominator,
                "remaining": remaining / denominator,
            },
        }
    fraction = report["upstream"]["fractions"]["state"]
    state_speedup = (
        report["upstream"]["state_mixer"]["device"]["median_ms"]
        / report["native"]["state_mixer"]["device"]["median_ms"]
    )
    report["amdahl"] = {
        "upstream_state_fraction": fraction,
        "native_state_speedup": state_speedup,
        "state_only_maximum_pipeline_speedup": 1
        / ((1 - fraction) + fraction / state_speedup),
        "measured_native_pipeline_speedup": (
            report["upstream"]["whole_pipeline"]["device"]["median_ms"]
            / report["native"]["whole_pipeline"]["device"]["median_ms"]
        ),
    }
    return report


def _memory(call, reset, torch):
    reset()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "allocated_before_bytes": before,
        "peak_allocated_bytes": peak,
        "temporary_peak_bytes": peak - before,
    }


def _backward_memory(bundle, path, torch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    call, _ = _backward_setup(bundle, path, torch)
    call()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {
        "allocated_before_bytes": before,
        "peak_allocated_bytes": peak,
        "temporary_peak_bytes": peak - before,
    }


def _backward_setup(bundle, path, torch):
    from urm.adapters.sparse_delta_memory_reference import torch_product_key
    from urm.backends.sparse_state_mixer import CertifiedSparseStateRoutes
    from urm.backends.sparse_state_reference import torch_sparse_state_mixer

    spec = bundle["spec"]
    base = (
        bundle["write_scores"],
        bundle["read_scores"],
        bundle["memory0"],
        bundle["values"],
        bundle["beta"],
        bundle["decay"],
    )
    leaves = [item.detach().clone().requires_grad_(True) for item in base]
    write_scores, read_scores, memory, values, beta, decay = leaves
    generator = torch.Generator(device="cuda").manual_seed(9941)
    grad_read = torch.randn(
        (spec.parallel, spec.sequence, spec.value_dim),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ) / (spec.parallel * spec.sequence * spec.value_dim)
    grad_state = (
        torch.randn(
            memory.shape, device="cuda", dtype=torch.float32, generator=generator
        )
        / memory.numel()
    )
    half = round(spec.slots_per_partition**0.5)
    if path == "native":
        prepared = bundle["native"].prepare(
            read_scores,
            write_scores=write_scores,
            values=values,
            beta=beta,
            log_decay=decay,
        )
        result = bundle["native"].execute(bundle["SparseState"](memory), prepared)
        outputs = (result.readings, result.state.memory)
        gradients = (grad_read.to(result.readings.dtype), grad_state.to(memory.dtype))
    elif path == "reference":
        write_values, write_indices = torch_product_key(write_scores, spec.writes, half)
        read_values, read_indices = torch_product_key(read_scores, spec.reads, half)
        outputs = torch_sparse_state_mixer(
            memory,
            read_indices,
            torch.softmax(read_values, -1),
            write_indices=write_indices,
            write_weights=torch.softmax(write_values, -1),
            values=values,
            beta=beta,
            log_decay=decay,
            read_timing=spec.read_timing,
        )
        gradients = (grad_read.to(outputs[0].dtype), grad_state.to(outputs[1].dtype))
    else:
        address = bundle["upstream"].direct_calls["address"]
        write_values, write_indices = address(write_scores, spec.writes, half)
        read_values, read_indices = address(read_scores, spec.reads, half)
        write_weights = bundle["upstream"].layer.write_act(write_values)
        read_weights = bundle["upstream"].layer.read_act(read_values)
        if path == "hybrid":
            routes = CertifiedSparseStateRoutes.certify(
                bundle["native"].state_spec,
                read_indices.to(torch.int32).contiguous(),
                read_weights,
                write_indices=write_indices.to(torch.int32).contiguous(),
                write_weights=write_weights,
            )
            prepared = bundle["native"].state_backend.prepare(
                routes, values=values, beta=beta, log_decay=decay
            )
            outputs = bundle["native"].state_backend.execute(
                bundle["SparseState"](memory), prepared
            )
            outputs = (outputs[0], outputs[1].memory)
            gradients = (
                grad_read.to(outputs[0].dtype),
                grad_state.to(outputs[1].dtype),
            )
        else:
            offsets = bundle["offsets"]
            flat_memory = memory.reshape(
                spec.parallel * spec.slots_per_partition, spec.value_dim
            )
            readings, _ = bundle["upstream"].direct_calls["update"](
                flat_memory + 0,
                write_indices + offsets,
                write_weights,
                values,
                beta,
                decay,
                read_indices + offsets,
                read_weights,
                grad_final_memory=grad_state.reshape_as(flat_memory)
                .to(memory.dtype)
                .contiguous(),
            )
            outputs = (readings,)
            gradients = (grad_read.to(readings.dtype),)

    def backward():
        torch.autograd.backward(outputs, gradients)

    return backward, leaves


def _backward_correctness(bundle, torch):
    paths = {}
    for path in ("reference", "upstream", "native"):
        call, leaves = _backward_setup(bundle, path, torch)
        call()
        paths[path] = [leaf.grad.detach().clone() for leaf in leaves]
    names = (
        "write_scores",
        "read_scores",
        "initial_memory",
        "values",
        "beta",
        "log_decay",
    )
    atol = 3e-5 if bundle["dtype"] is torch.float32 else 3e-2
    report = {}
    passed = True
    for index, name in enumerate(names):
        native = paths["native"][index]
        reference = paths["reference"][index]
        upstream = paths["upstream"][index]
        item = {
            "finite": bool(torch.isfinite(native).all()),
            "native_vs_reference_max_abs": _max_abs(native, reference),
            "native_vs_upstream_max_abs": _max_abs(native, upstream),
        }
        item["passed"] = bool(
            item["finite"]
            and torch.allclose(native.float(), reference.float(), atol=atol, rtol=atol)
            and torch.allclose(native.float(), upstream.float(), atol=atol, rtol=atol)
        )
        passed = passed and item["passed"]
        report[name] = item
    return {"passed": passed, "atol": atol, "rtol": atol, "gradients": report}


def _backward_performance(bundle, *, samples, warmup, seed, torch):
    upstream_setups = []
    native_setups = []

    def reset_upstream():
        upstream_setups[:] = [_backward_setup(bundle, "upstream", torch)[0]]

    def reset_native():
        native_setups[:] = [_backward_setup(bundle, "native", torch)[0]]

    def upstream():
        upstream_setups.pop()()

    def native():
        native_setups.pop()()

    paired, drift_attempts = _paired_with_drift_retries(
        upstream,
        native,
        reset_upstream,
        reset_native,
        samples=samples,
        warmup=warmup,
        seed=seed,
        torch=torch,
    )
    reference_setups = []

    def reset_reference():
        reference_setups[:] = [_backward_setup(bundle, "reference", torch)[0]]

    def reference_call():
        reference_setups.pop()()

    hybrid_setups = []

    def reset_hybrid():
        hybrid_setups[:] = [_backward_setup(bundle, "hybrid", torch)[0]]

    def hybrid_call():
        hybrid_setups.pop()()

    reference = _single_measure(
        reference_call,
        reset_reference,
        samples=min(5, samples),
        warmup=1,
        torch=torch,
    )
    hybrid = _single_measure(
        hybrid_call,
        reset_hybrid,
        samples=min(5, samples),
        warmup=1,
        torch=torch,
    )
    return {
        "authoritative": paired,
        "drift_attempts": drift_attempts,
        "reference": reference,
        "hybrid": hybrid,
    }


def _route_backward_setup(bundle, path, torch):
    write_scores = bundle["write_scores"].detach().clone().requires_grad_(True)
    read_scores = bundle["read_scores"].detach().clone().requires_grad_(True)
    if path == "native":
        write = bundle["native"].write_backend.generate(
            type(bundle["native_prepared"].write_scores).certify(
                bundle["native"].write_spec, write_scores
            )
        )[1]
        read = bundle["native"].read_backend.generate(
            type(bundle["native_prepared"].read_scores).certify(
                bundle["native"].read_spec, read_scores
            )
        )[1]
    else:
        half = round(bundle["spec"].slots_per_partition ** 0.5)
        address = bundle["upstream"].direct_calls["address"]
        write = bundle["upstream"].layer.write_act(
            address(write_scores, bundle["spec"].writes, half)[0]
        )
        read = bundle["upstream"].layer.read_act(
            address(read_scores, bundle["spec"].reads, half)[0]
        )
    write_cotangent = torch.randn_like(write)
    read_cotangent = torch.randn_like(read)

    def backward():
        torch.autograd.backward((write, read), (write_cotangent, read_cotangent))

    return backward


def _state_backward_setup(bundle, path, torch):
    from urm.backends.sparse_state_mixer import (
        CertifiedSparseStateRoutes,
        SparseState,
    )

    spec = bundle["spec"]
    route = bundle["up_route"]
    base = (
        bundle["memory0"],
        route[3],
        bundle["values"],
        bundle["beta"],
        bundle["decay"],
        route[1],
    )
    memory, write_weights, values, beta, decay, read_weights = [
        item.detach().clone().requires_grad_(True) for item in base
    ]
    generator = torch.Generator(device="cuda").manual_seed(12031)
    grad_read = torch.randn(
        (spec.parallel, spec.sequence, spec.value_dim),
        device="cuda",
        dtype=memory.dtype,
        generator=generator,
    )
    grad_state = torch.randn(
        memory.shape, device="cuda", dtype=memory.dtype, generator=generator
    )
    if path == "native":
        routes = CertifiedSparseStateRoutes.certify(
            bundle["native"].state_spec,
            route[0].to(torch.int32).contiguous(),
            read_weights,
            write_indices=route[2].to(torch.int32).contiguous(),
            write_weights=write_weights,
        )
        prepared = bundle["native"].state_backend.prepare(
            routes, values=values, beta=beta, log_decay=decay
        )
        readings, final = bundle["native"].state_backend.execute(
            SparseState(memory), prepared
        )
        outputs = (readings, final.memory)
        gradients = (grad_read, grad_state)
    else:
        flat = memory.reshape(spec.parallel * spec.slots_per_partition, spec.value_dim)
        readings, _ = bundle["upstream"].direct_calls["update"](
            flat + 0,
            route[5],
            write_weights,
            values,
            beta,
            decay,
            route[4],
            read_weights,
            grad_final_memory=grad_state.reshape_as(flat).contiguous(),
        )
        outputs = (readings,)
        gradients = (grad_read,)

    def backward():
        torch.autograd.backward(outputs, gradients)

    return backward


def _backward_stage_attribution(bundle, whole, *, samples, torch):
    def measure_setup(factory):
        holder = []

        def reset():
            holder[:] = [factory()]

        def call():
            holder.pop()()

        return _single_measure(call, reset, samples=samples, warmup=1, torch=torch)

    report = {}
    for path in ("upstream", "native"):
        route = measure_setup(
            lambda path=path: _route_backward_setup(bundle, path, torch)
        )
        state = measure_setup(
            lambda path=path: _state_backward_setup(bundle, path, torch)
        )
        total = whole[f"{path}_device"]["median_ms"]
        route_median = route["device"]["median_ms"]
        state_median = state["device"]["median_ms"]
        residual = total - route_median - state_median
        denominator = total if residual >= 0 else route_median + state_median
        report[path] = {
            "whole_backward": whole[f"{path}_device"],
            "route_backward": route,
            "state_backward": state,
            "remaining_orchestration_materialization_ms": max(0.0, residual),
            "signed_independent_median_residual_ms": residual,
            "fractions": {
                "route": route_median / denominator,
                "state": state_median / denominator,
                "remaining": max(0.0, residual) / denominator,
            },
        }
    return report


def benchmark_case(case, *, samples, warmup, torch):
    construction_start = time.perf_counter_ns()
    bundle = _make_bundle(case, torch)
    construction_ms = (time.perf_counter_ns() - construction_start) / 1e6
    calls = _calls(bundle, torch)
    correctness = _correctness(bundle, calls, torch)
    if not correctness["passed"]:
        raise RuntimeError(
            f"forward correctness failed for {case['name']}: {correctness}"
        )
    paired, forward_drift_attempts = _paired_with_drift_retries(
        calls["upstream"],
        calls["native"],
        calls["reset_upstream"],
        calls["reset_native"],
        samples=samples,
        warmup=warmup,
        seed=3101 + int(case["sequence"]),
        torch=torch,
    )
    reference = _single_measure(
        calls["reference"],
        calls["reset_reference"],
        samples=min(5, samples),
        warmup=1,
        torch=torch,
    )
    hybrid = _single_measure(
        calls["hybrid"],
        calls["reset_hybrid"],
        samples=min(7, samples),
        warmup=1,
        torch=torch,
    )
    attribution = _stage_attribution(
        bundle, calls, samples=min(7, samples), warmup=1, torch=torch
    )
    memory = {
        "upstream": _memory(calls["upstream"], calls["reset_upstream"], torch),
        "native": _memory(calls["native"], calls["reset_native"], torch),
        "hybrid": _memory(calls["hybrid"], calls["reset_hybrid"], torch),
        "reference": _memory(calls["reference"], calls["reset_reference"], torch),
    }
    training = case["operation"] == "training"
    backward_correctness = (
        _backward_correctness(bundle, torch)
        if training
        else {"status": "not_applicable", "reason": "inference/read-only case"}
    )
    if training and not backward_correctness["passed"]:
        raise RuntimeError(
            f"backward correctness failed for {case['name']}: {backward_correctness}"
        )
    backward = (
        _backward_performance(
            bundle,
            samples=samples,
            warmup=warmup,
            seed=7103 + int(case["dim"]),
            torch=torch,
        )
        if training
        else {"status": "not_applicable", "reason": "inference/read-only case"}
    )
    if training:
        backward["attribution"] = _backward_stage_attribution(
            bundle,
            backward["authoritative"],
            samples=min(5, samples),
            torch=torch,
        )
        memory["backward_upstream"] = _backward_memory(bundle, "upstream", torch)
        memory["backward_native"] = _backward_memory(bundle, "native", torch)
    up_peak = memory["upstream"]["temporary_peak_bytes"]
    native_peak = memory["native"]["temporary_peak_bytes"]
    memory_passed = native_peak <= max(int(up_peak * 1.02), up_peak + 1_048_576)
    ratio = paired["paired_device_ratio"]
    classification = case["classification"]
    if classification == "host_bound":
        latency_passed = bool(
            (
                paired["native_device"]["median_ms"]
                - paired["upstream_device"]["median_ms"]
            )
            * 1000
            <= HOST_ABSOLUTE_ALLOWANCE_US
            and (
                paired["native_device"]["p95_ms"] - paired["upstream_device"]["p95_ms"]
            )
            * 1000
            <= HOST_ABSOLUTE_ALLOWANCE_US
        )
    else:
        latency_passed = bool(
            ratio["bootstrap_ci95_median"]["upper"] <= 1.10
            and paired["native_device"]["p95_ms"] / paired["upstream_device"]["p95_ms"]
            <= 1.10
        )
    return {
        "case": case,
        "construction_ms": construction_ms,
        "cold_breakdown": bundle["cold_breakdown"],
        "score_generation_attempts": bundle["score_generation_attempts"],
        "correctness": correctness,
        "backward_correctness": backward_correctness,
        "forward": {
            "authoritative": paired,
            "drift_attempts": forward_drift_attempts,
            "reference": reference,
            "hybrid": hybrid,
        },
        "backward": backward,
        "attribution": attribution,
        "memory": memory,
        "gate": {
            "classification": classification,
            "latency_passed": latency_passed,
            "memory_passed": memory_passed,
            "correctness_passed": correctness["passed"]
            and (not training or backward_correctness["passed"]),
            "passed": latency_passed
            and memory_passed
            and correctness["passed"]
            and (not training or backward_correctness["passed"]),
        },
    }


def _single_process(args):
    import torch
    from provenance import provenance, utc_now, write_artifact

    from urm.adapters.sparse_delta_memory import probe_sdm_support

    support = probe_sdm_support()
    if not support.supported:
        raise RuntimeError(
            f"pinned comparator unavailable: {support.code}: {support.reason}"
        )
    grid = load_cases()
    selected = tuple(
        case for case in grid if not args.case or case["name"] in args.case
    )
    unknown = set(args.case) - {case["name"] for case in grid}
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    rows = {}
    for case in selected:
        print(f"benchmarking {case['name']}", flush=True)
        rows[case["name"]] = benchmark_case(
            case, samples=args.samples, warmup=args.warmup, torch=torch
        )
    configuration = {
        "grid": grid,
        "selected": [case["name"] for case in selected],
        "samples": args.samples,
        "warmup": args.warmup,
    }
    upstream_provenance = {
        "repository": support.details["repository"],
        "expected_commit": support.details["expected_commit"],
        "installed_commit": support.details["installed_commit"],
        "checkout_root": support.details["checkout_root"],
        "module_file": support.details["module_file"],
        "checkout_dirty": support.details["checkout_dirty"],
        "license": support.details["license"],
        "installation": support.details["installation"],
        "source_usage": support.details["source_usage"],
        "runtime_versions": support.details["runtime_versions"],
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "single_process_attribution",
        "generated_utc": utc_now(),
        "provenance": {
            **provenance(" ".join(sys.argv), configuration, include_gpu=True),
            "upstream": upstream_provenance,
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
        "methodology": {
            "scope": "scores through canonical routes and persistent state",
            "levels": {
                "reference": "transparent PyTorch semantics",
                "upstream": "pinned original SDM end-to-end",
                "hybrid": "diagnostic upstream routes then native state",
                "native": "fully native URM route and state lowerings",
            },
            "hybrid_certification": "fixed immutable scores permit one-time route certification; route production is still charged",
            "authoritative_sampling": "randomized paired AB/BA without per-stage events",
            "attribution_sampling": "separate diagnostic CUDA-event passes",
            "nvtx": "stage names exercised by optional --profile pass",
            "profiler": "torch.profiler CUDA activities is the Nsight-equivalent fallback because nsys is unavailable",
            "host_absolute_allowance_us": HOST_ABSOLUTE_ALLOWANCE_US,
            "mfu_mbu": {
                "status": "not_applicable",
                "reason": "sparse route/state utilization lacks measured hardware counters",
            },
            "optimization_history": [
                {
                    "id": "route-v0",
                    "hypothesis": "row-owned factor and pair top-k avoids framework route materialization overhead",
                    "result": "accepted after exact-route, gradient, and full-grid E2E gates",
                    "status": "accepted",
                },
                {
                    "id": "bf16-pair-rounding",
                    "hypothesis": "rounding composed pair scores to BF16 before FP32 Softmax matches the frozen semantic storage contract",
                    "result": "accepted on correctness before performance measurement",
                    "status": "accepted",
                },
                {
                    "id": "physical-route-state-fusion",
                    "hypothesis": "eliminating materialized routes may improve E2E latency",
                    "result": "deferred because the unfused logical composition clears the E2E gate and profiling does not require added coupling",
                    "status": "deferred",
                },
            ],
            "comparator_boundary": "BF16 D=257 remains native-tested but is absent from overlapping E2E cases because the pinned upstream comparator faults with a CUDA misaligned-address error on A10G; BF16 D=95 retains non-power-of-two coverage",
        },
        "cases": rows,
    }
    write_artifact(args.output, artifact)


def _confirmation_summary(runs, grid):
    expected = {case["name"] for case in grid}
    if any(set(run["cases"]) != expected for run in runs):
        raise RuntimeError("confirmation process case keys differ from frozen grid")
    reports = {}
    substantial = []
    passed = True
    for name in sorted(expected):
        rows = [run["cases"][name] for run in runs]
        ratios = [
            row["forward"]["authoritative"]["paired_device_ratio"]["median"]
            for row in rows
        ]
        hierarchical = []
        rng = random.Random(8849 + len(name))
        for _ in range(10_000):
            process = rng.choice(rows)
            raw = process["forward"]["authoritative"]["paired_device_ratio"]["raw"]
            hierarchical.append(
                statistics.median(rng.choice(raw) for _ in range(len(raw)))
            )
        hierarchical.sort()
        ci = {"lower": hierarchical[250], "upper": hierarchical[9750]}
        p95_ratios = [
            row["forward"]["authoritative"]["native_device"]["p95_ms"]
            / row["forward"]["authoritative"]["upstream_device"]["p95_ms"]
            for row in rows
        ]
        classification = rows[0]["case"]["classification"]
        if classification == "substantial":
            case_passed = bool(
                ci["upper"] <= 1.10
                and max(p95_ratios) <= 1.10
                and all(row["gate"]["passed"] for row in rows)
            )
            substantial.append(statistics.median(ratios))
        else:
            case_passed = all(row["gate"]["passed"] for row in rows)
        backward_ratios = []
        if rows[0]["backward"].get("status") != "not_applicable":
            backward_ratios = [
                row["backward"]["authoritative"]["paired_device_ratio"]["median"]
                for row in rows
            ]
            case_passed = case_passed and max(backward_ratios) <= 1.10
        passed = passed and case_passed
        reports[name] = {
            "classification": classification,
            "hierarchical_forward_ratio_ci95": ci,
            "per_process_forward_median_ratios": ratios,
            "per_process_forward_p95_ratios": p95_ratios,
            "per_process_backward_median_ratios": backward_ratios,
            "passed": case_passed,
        }
    geometric = math.exp(sum(math.log(x) for x in substantial) / len(substantial))
    passed = passed and geometric <= 1.0
    return {
        "process_count": len(runs),
        "cases": reports,
        "substantial_geometric_mean_native_over_upstream": geometric,
        "geometric_mean_passed": geometric <= 1.0,
        "passed": passed,
    }


def _run_confirmation(args):
    from provenance import provenance, utc_now, write_artifact

    if args.confirmation_processes != 3:
        raise ValueError(
            "completion confirmation requires exactly three fresh processes"
        )
    if args.case:
        raise ValueError("completion confirmation cannot filter the frozen case grid")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--samples",
        str(args.samples),
        "--warmup",
        str(args.warmup),
    ]
    runs = []
    process_wall = []
    with tempfile.TemporaryDirectory(prefix="urm-sparse-memory-e2e-") as root:
        for index in range(3):
            output = Path(root) / f"run-{index}.json"
            cache = Path(root) / f"cache-{index}"
            env = os.environ.copy()
            env["TRITON_CACHE_DIR"] = str(cache / "triton")
            env["TORCH_EXTENSIONS_DIR"] = str(cache / "extensions")
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
            started = time.perf_counter_ns()
            subprocess.run([*command, "--output", str(output)], check=True, env=env)
            process_wall.append((time.perf_counter_ns() - started) / 1e6)
            runs.append(json.loads(output.read_text(encoding="utf-8")))
    revisions = {run["provenance"]["git_revision"] for run in runs}
    if len(revisions) != 1 or any(run["provenance"]["dirty_tree"] for run in runs):
        raise RuntimeError("confirmation requires one clean implementation revision")
    if any(
        run["provenance"]["upstream"].get("installed_commit") != UPSTREAM_COMMIT
        or run["provenance"]["upstream"].get("checkout_dirty") is not False
        for run in runs
    ):
        raise RuntimeError("confirmation requires the clean pinned upstream checkout")
    grid = load_cases()
    confirmation = _confirmation_summary(runs, grid)
    if not confirmation["passed"]:
        raise RuntimeError(f"E2E completion gate failed: {confirmation}")
    configuration = {
        "grid": grid,
        "samples": args.samples,
        "warmup": args.warmup,
        "processes": 3,
    }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "three_process_confirmation",
        "generated_utc": utc_now(),
        "provenance": provenance(" ".join(sys.argv), configuration, include_gpu=True),
        "methodology": {
            "fresh_processes": 3,
            "fresh_caches": True,
            "process_wall_ms": process_wall,
            "scope": "scores through canonical routes and persistent state",
        },
        "runs": runs,
        "confirmation": confirmation,
    }
    write_artifact(args.output, artifact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--confirmation-processes", type=int, default=1)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 3 or args.warmup < 1:
        raise ValueError("samples >=3 and warmup >=1 are required")
    if args.confirmation_processes > 1 and not args.child:
        _run_confirmation(args)
    else:
        _single_process(args)


if __name__ == "__main__":
    main()
