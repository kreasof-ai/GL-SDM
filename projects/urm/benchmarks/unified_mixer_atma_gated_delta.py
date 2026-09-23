"""Parity and paired decode profiles for ATMA's slot-table gated-delta kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from provenance import provenance, write_artifact
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe

EXPECTED_REVISION = "28bb3de8afbe7c0b00115e0fbff36afc9ad49c11"
RECIPE = "atma_gated_delta_decode_core"


def _kernel_module():
    from kernel import gated_delta_triton

    return gated_delta_triton


def _revision() -> str:
    root = Path(_kernel_module().__file__).resolve().parents[1]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _inputs(seed: int = 2610):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    batch, capacity, heads, key_dim, value_dim = 4, 8, 8, 128, 128

    def rand(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    query = rand(batch, 1, heads, key_dim)
    key = rand(batch, 1, heads, key_dim)
    value = rand(batch, 1, heads, value_dim)
    gamma = torch.sigmoid(rand(batch, 1, heads) * 0.15 + 2.5)
    beta = torch.sigmoid(rand(batch, 1, heads) * 0.15 - 1.0)
    state = rand(capacity, heads, key_dim, value_dim) * 0.01
    slots = torch.arange(batch, device="cuda", dtype=torch.int64)
    return {
        "query": query,
        "key": key,
        "value": value,
        "gamma": gamma,
        "beta": beta,
        "state_table": state,
        "slots": slots,
    }


def _clone(inputs):
    return {name: tensor.detach().clone() for name, tensor in inputs.items()}


def _direct(inputs):
    return _kernel_module().gated_delta_decode_step(
        inputs["query"][:, 0],
        inputs["key"][:, 0],
        inputs["value"][:, 0],
        inputs["gamma"][:, 0],
        inputs["beta"][:, 0],
        inputs["state_table"],
        inputs["slots"],
    )


def _compiled(plan, inputs):
    return plan.execute(**inputs)


def _equation(inputs):
    state = inputs["state_table"].clone()
    slots = inputs["slots"]
    q = F.normalize(inputs["query"][:, 0].float(), dim=-1)
    k = F.normalize(inputs["key"][:, 0].float(), dim=-1)
    v = inputs["value"][:, 0].float()
    gamma = inputs["gamma"][:, 0]
    beta = inputs["beta"][:, 0]
    selected = state[slots]
    decayed = gamma[..., None, None] * selected
    prediction = torch.einsum("bhkv,bhk->bhv", decayed, k)
    update = beta[..., None] * (v - prediction)
    updated = decayed + k[..., None] * update[..., None, :]
    output = torch.einsum("bhkv,bhk->bhv", updated, q)
    state[slots] = updated
    return output.unsqueeze(1), state


def _max_error(left, right) -> float:
    return (left.float() - right.float()).abs().max().item()


def _summary(samples):
    ordered = sorted(samples)
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": ordered[max(0, int(0.95 * (len(ordered) - 1)))] * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _time_one(call, inputs, initial_state):
    inputs["state_table"].copy_(initial_state)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    call(inputs)
    end.record()
    torch.cuda.synchronize()
    return time.perf_counter() - wall_start, start.elapsed_time(end) / 1000


def _profile(direct, compiled, direct_inputs, compiled_inputs, initial_state, pairs, warmup):
    for _ in range(warmup):
        _time_one(direct, direct_inputs, initial_state)
        _time_one(compiled, compiled_inputs, initial_state)
    direct_wall, compiled_wall, direct_device, compiled_device, overhead, order = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index in range(pairs):
        first, second = (
            ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
        )
        order.append(first + second)
        samples = {}
        for name in (first, second):
            call, inputs = (
                (direct, direct_inputs)
                if name == "direct"
                else (compiled, compiled_inputs)
            )
            samples[name] = _time_one(call, inputs, initial_state)
        direct_wall.append(samples["direct"][0])
        direct_device.append(samples["direct"][1])
        compiled_wall.append(samples["compiled"][0])
        compiled_device.append(samples["compiled"][1])
        overhead.append(
            (samples["compiled"][0] - samples["direct"][0])
            / samples["direct"][0]
        )
    median_overhead = statistics.median(overhead)
    return {
        "warmup_calls_per_backend": warmup,
        "paired_samples": pairs,
        "direct_wall": _summary(direct_wall),
        "compiled_wall": _summary(compiled_wall),
        "direct_device": _summary(direct_device),
        "compiled_device": _summary(compiled_device),
        "paired_compiled_overhead_fraction": {
            "median": median_overhead,
            "p95": sorted(overhead)[max(0, int(0.95 * (len(overhead) - 1)))],
            "raw_samples": overhead,
            "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
        },
        "pair_order": order,
    }


def _capture_graph(call, inputs, initial_state):
    inputs["state_table"].copy_(initial_state)
    call(inputs)
    torch.cuda.synchronize()
    inputs["state_table"].copy_(initial_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(inputs)
    torch.cuda.synchronize()
    inputs["state_table"].copy_(initial_state)
    torch.cuda.synchronize()
    return graph


def _time_graph(graph, inputs, initial_state):
    inputs["state_table"].copy_(initial_state)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return time.perf_counter() - wall_start, start.elapsed_time(end) / 1000


def _profile_graph(
    direct, compiled, direct_inputs, compiled_inputs, initial_state, pairs, warmup
):
    direct_graph = _capture_graph(direct, direct_inputs, initial_state)
    compiled_graph = _capture_graph(compiled, compiled_inputs, initial_state)
    graphs = {"direct": direct_graph, "compiled": compiled_graph}
    inputs = {"direct": direct_inputs, "compiled": compiled_inputs}
    for _ in range(warmup):
        _time_graph(direct_graph, direct_inputs, initial_state)
        _time_graph(compiled_graph, compiled_inputs, initial_state)
    direct_wall, compiled_wall, direct_device, compiled_device, overhead, order = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index in range(pairs):
        first, second = (
            ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
        )
        order.append(first + second)
        samples = {}
        for name in (first, second):
            samples[name] = _time_graph(graphs[name], inputs[name], initial_state)
        direct_wall.append(samples["direct"][0])
        direct_device.append(samples["direct"][1])
        compiled_wall.append(samples["compiled"][0])
        compiled_device.append(samples["compiled"][1])
        overhead.append(
            (samples["compiled"][0] - samples["direct"][0])
            / samples["direct"][0]
        )
    median_overhead = statistics.median(overhead)
    return {
        "execution_mode": "CUDA graph replay",
        "warmup_replays_per_backend": warmup,
        "paired_samples": pairs,
        "direct_wall": _summary(direct_wall),
        "compiled_wall": _summary(compiled_wall),
        "direct_device": _summary(direct_device),
        "compiled_device": _summary(compiled_device),
        "paired_compiled_overhead_fraction": {
            "median": median_overhead,
            "p95": sorted(overhead)[max(0, int(0.95 * (len(overhead) - 1)))],
            "raw_samples": overhead,
            "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
        },
        "pair_order": order,
    }


def run(pairs: int, warmup: int, output: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("ATMA gated-delta profiling requires CUDA")
    revision = _revision()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"ATMA source must match {EXPECTED_REVISION}, got {revision!r}"
        )
    upstream = _kernel_module()
    if not upstream.HAS_TRITON:
        raise RuntimeError("ATMA checkout reports Triton unavailable")

    inputs = _inputs()
    direct_inputs, compiled_inputs = _clone(inputs), _clone(inputs)
    plan = compile_mixer(
        named_mixer_recipe(RECIPE),
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.INFERENCE,
        dtype="float32",
    )
    reference, expected_state = _equation(inputs)
    upstream_inputs, compiled_check_inputs = _clone(inputs), _clone(inputs)
    upstream_result = _direct(upstream_inputs)
    compiled_result = _compiled(plan, compiled_check_inputs)
    torch.testing.assert_close(
        upstream_result.unsqueeze(1), reference, atol=5e-5, rtol=5e-5
    )
    torch.testing.assert_close(
        upstream_inputs["state_table"], expected_state, atol=5e-5, rtol=5e-5
    )
    torch.testing.assert_close(compiled_result.output, upstream_result.unsqueeze(1), atol=0, rtol=0)
    torch.testing.assert_close(
        compiled_check_inputs["state_table"], upstream_inputs["state_table"],
        atol=0, rtol=0,
    )

    compiled_call = lambda values: _compiled(plan, values)
    eager_profile = _profile(
        _direct,
        compiled_call,
        direct_inputs,
        compiled_inputs,
        inputs["state_table"],
        pairs,
        warmup,
    )
    graph_profile = _profile_graph(
        _direct,
        compiled_call,
        direct_inputs,
        compiled_inputs,
        inputs["state_table"],
        pairs,
        warmup,
    )
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": (
            "profile URM K2 decode dispatch against ATMA's slot-table "
            "gated-delta Triton kernel"
        ),
        "upstream": {
            "repository": "https://github.com/kreasof-ai/atma",
            "local_path": str(Path(upstream.__file__).resolve().parents[1]),
            "revision": revision,
            "loaded_module": upstream.__file__,
            "callable": "kernel.gated_delta_triton.gated_delta_decode_step",
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/atma:src python benchmarks/unified_mixer_atma_gated_delta.py",
            {"recipe": RECIPE, "pairs": pairs, "warmup": warmup},
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "case": {
            "architecture_ids": ["arch-026"],
            "semantic_scope": (
                "One in-place, slot-indexed gated-delta decode step with "
                "L2-normalized Q/K; caller supplies valid distinct slots and "
                "gamma/beta. Projections, gates, RMSNorm, output projection "
                "and full block remain external."
            ),
            "shape": {
                "batch": 4,
                "state_capacity": 8,
                "sequence": 1,
                "heads": 8,
                "key_dim": 128,
                "value_dim": 128,
                "state_layout": "capacity,heads,key_dim,value_dim",
                "dtype": "float32",
            },
            "intent_modes": ["inference_decode"],
            "compiled_anchor": plan.anchor,
            "parity": {
                "status": "pass",
                "upstream_vs_equation_output_max_abs_error": _max_error(
                    upstream_result.unsqueeze(1), reference
                ),
                "upstream_vs_equation_state_max_abs_error": _max_error(
                    upstream_inputs["state_table"], expected_state
                ),
                "compiled_vs_upstream_output_max_abs_error": _max_error(
                    compiled_result.output, upstream_result.unsqueeze(1)
                ),
                "compiled_vs_upstream_state_max_abs_error": _max_error(
                    compiled_check_inputs["state_table"],
                    upstream_inputs["state_table"],
                ),
                "tolerances": {
                    "equation_atol": 5e-5,
                    "equation_rtol": 5e-5,
                    "adapter_atol": 0.0,
                    "adapter_rtol": 0.0,
                },
                "backward": "not_supported_by_upstream_decode_kernel",
            },
            "architecture_entrypoint_parity": "not_measured",
            "performance": {
                "eager_python_dispatch_diagnostic": eager_profile,
                "cuda_graph_replay": graph_profile,
                "qualification_basis": (
                    "CUDA graph replay, matching ATMA's documented "
                    "serving/decode capture mode"
                ),
            },
        },
        "methodology": {
            "timed_work": (
                "one decode launch after restoring identical preallocated "
                "state outside the timed window"
            ),
            "sampling": (
                "paired interleaved direct/compiled calls with synchronized "
                "wall and CUDA-event timing"
            ),
            "overhead": "median of per-pair (compiled-direct)/direct wall-time fractions",
            "overhead_gate_fraction": 0.10,
            "overhead_gate_rule": "median compiled slowdown must not exceed 10%",
        },
    }
    write_artifact(output, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/unified-mixer/atma-gated-delta-decode-k2.json"),
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output)


if __name__ == "__main__":
    main()
