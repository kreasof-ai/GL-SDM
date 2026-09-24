"""Profile URM's native online K1 against pinned FlashAttention 2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import flash_attn
import torch
from flash_attn import flash_attn_func

from measurement import quantile
from provenance import provenance, write_artifact
from unified_mixer_flash import (
    DTYPE,
    KEY_DIM,
    QUERY_HEADS,
    _inputs,
    _kernel_source_hashes,
    _source_identity,
    _summary,
    _time_one,
    _forward_backward,
)
from urm.compiler.mixer import MixerBackend, MixerIntent, compile_mixer
from benchmarks.recipe_catalog import load_kernel_recipe


EXPECTED_FLASH_REVISION = "1bda8f9290cd48d030f1516f0e680cd464ef3554"
CASES = ("mha_prefill", "mqa_prefill", "gqa_prefill")


def _case_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
    recipe = case.removesuffix("_prefill")
    return _inputs(recipe, seed)


def _direct(inputs: dict[str, torch.Tensor]):
    return flash_attn_func(
        inputs["query"],
        inputs["key"],
        inputs["value"],
        dropout_p=0.0,
        softmax_scale=KEY_DIM**-0.5,
        causal=True,
    )


def _native(plan, inputs: dict[str, torch.Tensor]):
    return plan.execute(**inputs).output


def _measure_pair(direct, native, direct_inputs, native_inputs, pairs, warmup):
    cold_direct = _time_one(direct, direct_inputs, backward=False)
    cold_native = _time_one(native, native_inputs, backward=False)
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(direct, direct_inputs, backward=backward)
            _time_one(native, native_inputs, backward=backward)

    measurements = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, native_wall = [], []
        direct_device, native_device, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = (
                ("direct", "native") if index % 2 == 0 else ("native", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, device = _time_one(direct, direct_inputs, backward=backward)
                    direct_wall.append(wall)
                    direct_device.append(device)
                else:
                    wall, device = _time_one(native, native_inputs, backward=backward)
                    native_wall.append(wall)
                    native_device.append(device)
            pair_index = len(overhead)
            overhead.append(
                (native_wall[pair_index] - direct_wall[pair_index])
                / direct_wall[pair_index]
            )
        median_overhead = statistics.median(overhead)
        measurements[mode] = {
            "direct_wall": _summary(direct_wall),
            "native_wall": _summary(native_wall),
            "direct_device": _summary(direct_device),
            "native_device": _summary(native_device),
            "paired_native_overhead_fraction": {
                "median": median_overhead,
                "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
            },
            "pair_order": order,
        }
        measurements[mode]["cuda_graph_replay"] = _measure_graph_pair(
            direct,
            native,
            direct_inputs,
            native_inputs,
            backward=backward,
            pairs=pairs,
        )
    return {
        "first_direct_forward_call_ms": {
            "wall": cold_direct[0] * 1000,
            "device": cold_direct[1] * 1000,
        },
        "first_native_forward_call_ms": {
            "wall": cold_native[0] * 1000,
            "device": cold_native[1] * 1000,
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def _capture_graph(call, inputs, *, backward: bool):
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        warmup_output = call(inputs)
        if backward:
            warmup_output.float().square().mean().backward()
    torch.cuda.synchronize()
    del warmup_output
    for tensor in inputs.values():
        tensor.grad = None
    graph = torch.cuda.CUDAGraph()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        with torch.cuda.graph(graph, stream=capture_stream):
            output = call(inputs)
            if backward:
                output.float().square().mean().backward()
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)
    torch.cuda.synchronize()
    return graph


def _time_graph(graph, replays: int) -> tuple[float, float]:
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(replays):
        graph.replay()
    end_event.record()
    torch.cuda.synchronize()
    divisor = replays
    return (
        (time.perf_counter() - start_wall) / divisor,
        start_event.elapsed_time(end_event) / 1000 / divisor,
    )


def _measure_graph_pair(
    direct, native, direct_inputs, native_inputs, *, backward: bool, pairs: int
):
    direct_graph = _capture_graph(direct, direct_inputs, backward=backward)
    native_graph = _capture_graph(native, native_inputs, backward=backward)
    replays_per_sample = 20
    # Settle the captured graphs and GPU clocks before collecting paired samples.
    for _ in range(3):
        direct_graph.replay()
        native_graph.replay()
    torch.cuda.synchronize()
    direct_wall, native_wall = [], []
    direct_device, native_device, overhead, order = [], [], [], []
    for index in range(pairs):
        first, second = (
            ("direct", "native") if index % 2 == 0 else ("native", "direct")
        )
        order.append(first + second)
        for name in (first, second):
            graph = direct_graph if name == "direct" else native_graph
            wall, device = _time_graph(graph, replays_per_sample)
            if name == "direct":
                direct_wall.append(wall)
                direct_device.append(device)
            else:
                native_wall.append(wall)
                native_device.append(device)
        pair_index = len(overhead)
        overhead.append(
            (native_wall[pair_index] - direct_wall[pair_index])
            / direct_wall[pair_index]
        )
    median_overhead = statistics.median(overhead)
    return {
        "replays_per_sample": replays_per_sample,
        "direct_wall": _summary(direct_wall),
        "native_wall": _summary(native_wall),
        "direct_device": _summary(direct_device),
        "native_device": _summary(native_device),
        "paired_native_overhead_fraction": {
            "median": median_overhead,
            "p95": quantile(overhead, 0.95),
            "raw_samples": overhead,
            "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
        },
        "pair_order": order,
    }


def run(pairs: int, warmup: int, output_path: Path, only: str | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the native K1 profile requires CUDA")
    source, revision = _source_identity()
    if revision != EXPECTED_FLASH_REVISION:
        raise RuntimeError(f"expected pinned FlashAttention {EXPECTED_FLASH_REVISION}, got {revision}")
    import flash_attn_2_cuda

    native_source = (
        Path(__file__).resolve().parents[1]
        / "src/urm/backends/triton/softmax/online.py"
    )

    selected_cases = (only,) if only is not None else CASES
    if any(case not in CASES for case in selected_cases):
        raise ValueError(f"unknown profile selection: {selected_cases}")

    cases = {}
    for index, case in enumerate(selected_cases):
        recipe = "mqa" if case.startswith("mqa") else case.removesuffix("_prefill")
        operands = _case_inputs(case, seed=79301 + index)
        direct_inputs = {
            name: tensor.detach().clone().requires_grad_()
            for name, tensor in operands.items()
        }
        native_inputs = {
            name: tensor.detach().clone().requires_grad_()
            for name, tensor in operands.items()
        }
        plan_started = time.perf_counter()
        plan = compile_mixer(
            load_kernel_recipe(recipe),
            backend=MixerBackend.NATIVE,
            intent=MixerIntent.TRAINING,
            dtype="bfloat16",
        )
        plan_build_ms = (time.perf_counter() - plan_started) * 1000
        direct = _direct
        native = lambda values, selected_plan=plan: _native(selected_plan, values)

        parity_direct = _forward_backward(direct, direct_inputs)
        parity_native = _forward_backward(native, native_inputs)
        output_error = (
            (parity_direct[0].float() - parity_native[0].float()).abs().max().item()
        )
        gradient_errors = {
            name: (left.float() - right.float()).abs().max().item()
            for name, left, right in zip(
                operands, parity_direct[1], parity_native[1], strict=True
            )
        }
        torch.testing.assert_close(
            parity_native[0].float(), parity_direct[0].float(), atol=2e-2, rtol=2e-2
        )
        for direct_grad, native_grad in zip(
            parity_direct[1], parity_native[1], strict=True
        ):
            torch.testing.assert_close(
                native_grad.float(), direct_grad.float(), atol=2e-2, rtol=2e-2
            )

        performance = _measure_pair(
            direct, native, direct_inputs, native_inputs, pairs, warmup
        )
        cases[case] = {
            "architecture_ids": {
                "mha_prefill": ["arch-001", "arch-014"],
                "mqa_prefill": ["arch-002"],
                "gqa_prefill": ["arch-003"],
            }[case],
            "recipe": recipe,
            "semantic_scope": "causal normalized-softmax K1; projections, positional transforms, cache ABI, and full layer excluded",
            "shape": {
                "batch": 1,
                "query_length": operands["query"].shape[1],
                "key_length": operands["key"].shape[1],
                "query_heads": operands["query"].shape[2],
                "key_value_heads": operands["key"].shape[2],
                "key_dim": KEY_DIM,
                "value_dim": operands["value"].shape[-1],
                "dtype": str(DTYPE).removeprefix("torch."),
            },
            "upstream_callable": "flash_attn.flash_attn_interface.flash_attn_func",
            "native_anchor": plan.anchor,
            "compiler_plan_build_ms": plan_build_ms,
            "parity": {
                "status": "pass",
                "output_max_abs_error": output_error,
                "input_gradient_max_abs_errors": gradient_errors,
                "atol": 2e-2,
                "rtol": 2e-2,
            },
            "performance": performance,
        }

    config = {"cases": selected_cases, "pairs": pairs, "warmup": warmup, "dtype": "bfloat16"}
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "qualify URM native tiled online K1 against the pinned direct FlashAttention operator",
        "upstream": {
            "repository": "https://github.com/Dao-AILab/flash-attention",
            "revision": revision,
            "module_version": getattr(flash_attn, "__version__", None),
            "distribution_version": importlib.metadata.version("flash_attn"),
            "loaded_module": str(source),
            "extension_binary": str(Path(flash_attn_2_cuda.__file__).resolve()),
            "extension_sha256": hashlib.sha256(Path(flash_attn_2_cuda.__file__).read_bytes()).hexdigest(),
            "extension_build_scope": "locally narrowed dispatch; BF16 causal D=32 prefill kernel path only, so unequal-length decode is not covered",
            "kernel_source_sha256": _kernel_source_hashes(source),
        },
        "urm_native_implementation": {
            "module": "urm.backends.triton.k1.online",
            "source_path": str(native_source),
            "source_sha256": hashlib.sha256(native_source.read_bytes()).hexdigest(),
        },
        "provenance": provenance(
            "CUDA_HOME=/path/to/cuda-toolkit PYTHONPATH=/path/to/flash-attention:src python benchmarks/unified_mixer_native_k1.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one forward or one forward plus backward on preallocated leaf tensors; separate CUDA graph replay batches capture the same work",
            "sampling": "paired interleaved direct/native calls, order alternates, synchronized wall and CUDA-event timing; graph replay samples execute 20 captures per timing sample",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (native-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "per-call samples include Python dispatch and allocation; CUDA graph replay samples isolate captured GPU work and are the kernel performance gate. Both are reported independently against direct pinned FlashAttention.",
        },
        "cases": cases,
    }
    write_artifact(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("results/unified-mixer/native-k1.json"))
    parser.add_argument("--only", choices=CASES)
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(args.pairs, args.warmup, args.output, args.only)


if __name__ == "__main__":
    main()
