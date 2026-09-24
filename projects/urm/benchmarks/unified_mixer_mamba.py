"""Parity and paired K2 profiles against pinned Mamba selective scan source."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import mamba_ssm
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.pipeline import (
    diagonal_ssm_spec,
)
from urm.compiler.mixer import (
    MixerBackend,
    MixerIntent,
    compile_mixer,
)

EXPECTED_MAMBA_REVISION = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
BATCH = 1
CHANNELS = 256
SEQUENCE = 1024
STATE_WIDTH = 64


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(mamba_ssm)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not find Git root for loaded Mamba source {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_MAMBA_REVISION:
        raise RuntimeError(
            f"loaded Mamba source must match {EXPECTED_MAMBA_REVISION}, got {revision} at {source}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("the pinned Mamba source checkout must be clean")
    return source, revision


def _source_hashes(source: Path) -> dict[str, str]:
    root = next(parent for parent in source.parents if (parent / ".git").exists())
    paths = (
        "csrc/selective_scan/selective_scan.cpp",
        "csrc/selective_scan/selective_scan_fwd_fp32.cu",
        "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
        "csrc/selective_scan/selective_scan_fwd_kernel.cuh",
        "csrc/selective_scan/selective_scan_bwd_kernel.cuh",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    u = (
        torch.randn((BATCH, CHANNELS, SEQUENCE), device="cuda", generator=generator)
        * 0.2
    )
    delta = (
        torch.rand((BATCH, CHANNELS, SEQUENCE), device="cuda", generator=generator)
        * 0.1
        + 0.01
    )
    A = (
        -torch.rand((CHANNELS, STATE_WIDTH), device="cuda", generator=generator) * 2
        - 0.1
    )
    B = (
        torch.randn((BATCH, STATE_WIDTH, SEQUENCE), device="cuda", generator=generator)
        * 0.1
    )
    C = (
        torch.randn((BATCH, STATE_WIDTH, SEQUENCE), device="cuda", generator=generator)
        * 0.1
    )
    D = torch.randn((CHANNELS,), device="cuda", generator=generator) * 0.1
    return {
        name: value.requires_grad_()
        for name, value in locals().copy().items()
        if name in {"u", "delta", "A", "B", "C", "D"}
    }


def _direct(inputs: dict[str, torch.Tensor]):
    output, final_state = selective_scan_fn(
        inputs["u"],
        inputs["delta"],
        inputs["A"],
        inputs["B"],
        inputs["C"],
        inputs["D"],
        delta_softplus=False,
        return_last_state=True,
    )
    return output, final_state


def _compiled_native(plan, inputs: dict[str, torch.Tensor]):
    u, delta, A, B, C, D = (inputs[name] for name in ("u", "delta", "A", "B", "C", "D"))
    batch, _, sequence = u.shape
    result = plan.execute(
        x=u.transpose(1, 2),
        input_gate=B.permute(0, 2, 1),
        read_gate=C.permute(0, 2, 1),
        log_decay=A[None, None, :, :].expand(batch, sequence, CHANNELS, STATE_WIDTH),
        step_size=delta.transpose(1, 2),
        skip=D,
    )
    return result.output.transpose(1, 2), result.final_state


def _compiled_library(plan, inputs: dict[str, torch.Tensor]):
    result = plan.execute(
        x=inputs["u"],
        input_gate=inputs["B"],
        read_gate=inputs["C"],
        log_decay=inputs["A"],
        step_size=inputs["delta"],
        skip=inputs["D"],
    )
    return result.output.transpose(1, 2), result.final_state


def _loss(output: torch.Tensor) -> torch.Tensor:
    return output.float().square().mean()


def _forward_backward(call, inputs: dict[str, torch.Tensor]):
    for tensor in inputs.values():
        tensor.grad = None
    output, final_state = call(inputs)
    _loss(output).backward()
    return output, final_state, tuple(inputs[name].grad for name in inputs)


def _time_one(call, inputs: dict[str, torch.Tensor], *, backward: bool):
    for tensor in inputs.values():
        tensor.grad = None
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    output, _ = call(inputs)
    if backward:
        _loss(output).backward()
    end_event.record()
    torch.cuda.synchronize()
    return time.perf_counter() - start_wall, start_event.elapsed_time(end_event) / 1000


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _measure_pair(direct, compiled, direct_inputs, compiled_inputs, pairs, warmup):
    cold_direct = _time_one(direct, direct_inputs, backward=False)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False)
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(direct, direct_inputs, backward=backward)
            _time_one(compiled, compiled_inputs, backward=backward)
    measurements = {}
    for mode, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall = [], []
        direct_device, compiled_device, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, device = _time_one(direct, direct_inputs, backward=backward)
                    direct_wall.append(wall)
                    direct_device.append(device)
                else:
                    wall, device = _time_one(
                        compiled, compiled_inputs, backward=backward
                    )
                    compiled_wall.append(wall)
                    compiled_device.append(device)
            pair_index = len(overhead)
            overhead.append(
                (compiled_wall[pair_index] - direct_wall[pair_index])
                / direct_wall[pair_index]
            )
        median_overhead = statistics.median(overhead)
        measurements[mode] = {
            "direct_wall": _summary(direct_wall),
            "compiled_wall": _summary(compiled_wall),
            "direct_device": _summary(direct_device),
            "compiled_device": _summary(compiled_device),
            "paired_compiled_overhead_fraction": {
                "median": median_overhead,
                "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {"limit_fraction": 0.10, "pass": median_overhead <= 0.10},
            },
            "pair_order": order,
        }
    return {
        "first_direct_forward_call_ms": {
            "wall": cold_direct[0] * 1000,
            "device": cold_direct[1] * 1000,
        },
        "first_compiled_forward_call_after_direct_autotune_ms": {
            "wall": cold_compiled[0] * 1000,
            "device": cold_compiled[1] * 1000,
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def run(
    pairs: int, warmup: int, output_path: Path, *, include_native_candidate: bool
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the Mamba unified mixer profile requires CUDA")
    source, revision = _source_identity()
    import selective_scan_cuda

    operands = _inputs(seed=55403)
    direct_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    library_inputs = {
        name: tensor.detach().clone().requires_grad_()
        for name, tensor in operands.items()
    }
    spec = diagonal_ssm_spec("mamba1_selective_scan", step_size_discretization=True)
    plan_started = time.perf_counter()
    library_plan = compile_mixer(
        spec,
        backend=MixerBackend.LIBRARY,
        intent=MixerIntent.TRAINING,
        dtype="float32",
    )
    library_plan_build_ms = (time.perf_counter() - plan_started) * 1000
    direct = _direct
    compiled_library = lambda values: _compiled_library(library_plan, values)

    direct_result = _forward_backward(direct, direct_inputs)
    library_result = _forward_backward(compiled_library, library_inputs)
    value_tolerance = 2e-6
    gradient_tolerance = 2e-6

    def parity(candidate):
        output_error = (
            (direct_result[0].float() - candidate[0].float()).abs().max().item()
        )
        state_error = (
            (direct_result[1].float() - candidate[1].float()).abs().max().item()
        )
        gradient_errors = {
            name: (left.float() - right.float()).abs().max().item()
            for name, left, right in zip(
                operands, direct_result[2], candidate[2], strict=True
            )
        }
        torch.testing.assert_close(
            candidate[0].float(),
            direct_result[0].float(),
            atol=value_tolerance,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            candidate[1].float(),
            direct_result[1].float(),
            atol=value_tolerance,
            rtol=2e-5,
        )
        for direct_grad, candidate_grad in zip(
            direct_result[2], candidate[2], strict=True
        ):
            torch.testing.assert_close(
                candidate_grad.float(),
                direct_grad.float(),
                atol=gradient_tolerance,
                rtol=2e-5,
            )
        return {
            "status": "pass",
            "output_max_abs_error": output_error,
            "final_state_max_abs_error": state_error,
            "input_gradient_max_abs_errors": gradient_errors,
            "tolerances": {
                "output_atol": value_tolerance,
                "state_atol": value_tolerance,
                "gradient_atol": gradient_tolerance,
                "relative_tolerance": 2e-5,
            },
            "final_state_gradient": "not included; upstream selective_scan backward does not propagate last-state cotangent",
        }

    library_parity = parity(library_result)
    performance = _measure_pair(
        direct, compiled_library, direct_inputs, library_inputs, pairs, warmup
    )
    native_candidate = None
    if include_native_candidate:
        native_inputs = {
            name: tensor.detach().clone().requires_grad_()
            for name, tensor in operands.items()
        }
        plan_started = time.perf_counter()
        native_plan = compile_mixer(
            spec,
            backend=MixerBackend.NATIVE,
            intent=MixerIntent.TRAINING,
            dtype="float32",
        )
        native_plan_build_ms = (time.perf_counter() - plan_started) * 1000
        compiled_native = lambda values: _compiled_native(native_plan, values)
        native_result = _forward_backward(compiled_native, native_inputs)
        native_parity = parity(native_result)
        native_performance = _measure_pair(
            direct, compiled_native, direct_inputs, native_inputs, pairs, warmup
        )
        native_candidate = {
            "anchor": native_plan.anchor,
            "compiler_plan_build_ms": native_plan_build_ms,
            "parity": native_parity,
            "performance": native_performance,
        }
    cases = {
        "mamba1_selective_scan": {
            "architecture_ids": ["arch-043"],
            "semantic_scope": "selective diagonal state scan; convolution, projections, activation gate and cache ABI excluded",
            "shape": {
                "batch": BATCH,
                "sequence": SEQUENCE,
                "channels": CHANNELS,
                "state_width": STATE_WIDTH,
                "dtype": "float32",
            },
            "upstream_callable": "mamba_ssm.ops.selective_scan_interface.selective_scan_fn",
            "compiled_anchor": library_plan.anchor,
            "compiler_plan_build_ms": library_plan_build_ms,
            "parity": library_parity,
            "performance": performance,
            **({"native_candidate": native_candidate} if native_candidate else {}),
        }
    }
    config = {
        "recipe": "mamba1_selective_scan",
        "pairs": pairs,
        "warmup": warmup,
        "shape": [BATCH, CHANNELS, SEQUENCE, STATE_WIDTH],
        "dtype": "float32",
        "include_native_candidate": include_native_candidate,
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified K2 library and native diagonal plans with pinned Mamba selective scan",
        "upstream": {
            "repository": "https://github.com/state-spaces/mamba",
            "revision": revision,
            "module_version": getattr(mamba_ssm, "__version__", None),
            "loaded_module": str(source),
            "extension_binary": str(Path(selective_scan_cuda.__file__).resolve()),
            "extension_sha256": hashlib.sha256(
                Path(selective_scan_cuda.__file__).read_bytes()
            ).hexdigest(),
            "extension_build_scope": "FP32 real selective-scan C++/CUDA translation units from the pinned revision, built with the installed Torch C++20 ABI",
            "kernel_source_sha256": _source_hashes(source),
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/mamba-checkout:src python benchmarks/unified_mixer_mamba.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one source call or one unified library/native plan call, optionally followed by output-only backward",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "the library profile compares plan dispatch overhead against the same source kernel; the separate native candidate maps the same source operands into URM's diagonal K2 equation and includes only view construction",
        },
        "cases": cases,
    }
    write_artifact(output_path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--include-native-candidate",
        action="store_true",
        help="also profile the experimental URM Triton diagonal scan",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/unified-mixer/mamba-k2.json")
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(
        args.pairs,
        args.warmup,
        args.output,
        include_native_candidate=args.include_native_candidate,
    )


if __name__ == "__main__":
    main()
