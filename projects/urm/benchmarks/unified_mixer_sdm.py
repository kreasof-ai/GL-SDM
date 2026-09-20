"""Parity and paired profiles for unified K3 against pinned SDM operators."""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from lingua.sparse_delta_memory.layer import SparseDeltaMemory, SparseDeltaMemoryArgs
from measurement import quantile
from provenance import provenance, write_artifact

from urm.compiler.unified_mixer import (
    MixerBackend,
    MixerIntent,
    compile_mixer,
    sparse_delta_spec,
)

EXPECTED_SDM_REVISION = "183e7df809131b80ad4393741029d0f20fc3640b"
ROUTE_WIDTH = 4
SLOTS = 256
SEQUENCE = 16
VALUE_DIM = 32
DTYPES = (torch.float32, torch.bfloat16)
GRADIENT_NAMES = (
    "memory",
    "write_weights",
    "values",
    "beta",
    "log_decay",
    "read_weights",
)


def _source_identity() -> tuple[Path, str]:
    source = Path(inspect.getfile(SparseDeltaMemory)).resolve()
    repository = next(
        (parent for parent in source.parents if (parent / ".git").exists()), None
    )
    if repository is None:
        raise RuntimeError(f"could not find Git root for loaded SDM source {source}")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != EXPECTED_SDM_REVISION:
        raise RuntimeError(
            f"loaded SDM source must match {EXPECTED_SDM_REVISION}, got {revision} at {source}"
        )
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if dirty:
        raise RuntimeError("the pinned SDM source checkout must be clean")
    return source, revision


def _routes(
    dtype: torch.dtype, *, offset: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = (
        torch.tensor(
            [
                [
                    [(offset + route * 11) % SLOTS for route in range(ROUTE_WIDTH)]
                    for _ in range(SEQUENCE)
                ]
            ],
            dtype=torch.int64,
            device="cuda",
        )
        .sort(dim=-1)
        .values.contiguous()
    )
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weights = torch.softmax(
        torch.randn((1, SEQUENCE, ROUTE_WIDTH), device="cuda", generator=generator).to(
            dtype
        ),
        dim=-1,
    ).contiguous()
    return indices, weights


def _inputs(dtype: torch.dtype, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    write_indices, write_weights = _routes(dtype, offset=1, seed=seed + 1)
    # Distinct product-key lanes keep reads separate while writes collide across tokens.
    read_indices, read_weights = _routes(dtype, offset=2, seed=seed + 2)
    return {
        "memory": (
            torch.randn(
                (1, SLOTS, VALUE_DIM), device="cuda", dtype=dtype, generator=generator
            )
            * 0.05
        ).contiguous(),
        "read_indices": read_indices,
        "read_weights": read_weights,
        "write_indices": write_indices,
        "write_weights": write_weights,
        "values": (
            torch.randn(
                (1, SEQUENCE, VALUE_DIM),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.05
        ).contiguous(),
        "beta": torch.rand(
            (1, SEQUENCE, 1), device="cuda", dtype=dtype, generator=generator
        ).contiguous(),
        "log_decay": (
            -torch.rand(
                (1, SEQUENCE, 1), device="cuda", dtype=dtype, generator=generator
            )
            * 0.1
        ).contiguous(),
    }


def _fresh_inputs(
    template: dict[str, torch.Tensor], *, backward: bool
) -> dict[str, torch.Tensor]:
    differentiable = {
        "memory",
        "read_weights",
        "write_weights",
        "values",
        "beta",
        "log_decay",
    }
    return {
        name: value.detach().clone().requires_grad_(backward and name in differentiable)
        for name, value in template.items()
    }


def _direct_call(
    layer: SparseDeltaMemory, inputs: dict[str, torch.Tensor], final_cotangent=None
):
    batch, slots, value_dim = inputs["memory"].shape
    memory = inputs["memory"].flatten(0, 1) + 0
    grad_final_memory = None
    if final_cotangent is not None:
        grad_final_memory = (
            final_cotangent.flatten(0, 1).to(inputs["memory"].dtype)
            / inputs["memory"].numel()
        ).contiguous()
    output, _ = layer.gated_write_read(
        memory,
        inputs["write_indices"],
        inputs["write_weights"],
        inputs["values"],
        inputs["beta"],
        inputs["log_decay"],
        inputs["read_indices"],
        inputs["read_weights"],
        grad_final_memory=grad_final_memory,
    )
    return output, memory.view(batch, slots, value_dim)


def _compiled_call(plan, inputs: dict[str, torch.Tensor]):
    result = plan.execute(**inputs)
    return result.output, result.final_state


def _loss(output, final_state, output_cotangent, state_cotangent):
    return (output.float() * output_cotangent).mean() + (
        final_state.float() * state_cotangent
    ).mean()


def _time_one(call, template, output_cotangent, state_cotangent, *, backward, direct):
    inputs = _fresh_inputs(template, backward=backward)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    output, final_state = (
        call(
            inputs,
            state_cotangent if backward and direct else None,
        )
        if direct
        else call(inputs)
    )
    if backward:
        if direct:
            (output.float() * output_cotangent).mean().backward()
        else:
            _loss(output, final_state, output_cotangent, state_cotangent).backward()
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


def _measure_pair(
    direct, compiled, template, output_cotangent, state_cotangent, pairs, warmup
):
    cold_direct = _time_one(
        direct, template, output_cotangent, state_cotangent, backward=False, direct=True
    )
    cold_compiled = _time_one(
        compiled,
        template,
        output_cotangent,
        state_cotangent,
        backward=False,
        direct=False,
    )
    for _ in range(warmup):
        for backward in (False, True):
            _time_one(
                direct,
                template,
                output_cotangent,
                state_cotangent,
                backward=backward,
                direct=True,
            )
            _time_one(
                compiled,
                template,
                output_cotangent,
                state_cotangent,
                backward=backward,
                direct=False,
            )

    measurements = {}
    for label, backward in (("forward", False), ("forward_backward", True)):
        direct_wall, compiled_wall = [], []
        direct_device, compiled_device, overhead, order = [], [], [], []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, device = _time_one(
                        direct,
                        template,
                        output_cotangent,
                        state_cotangent,
                        backward=backward,
                        direct=True,
                    )
                    direct_wall.append(wall)
                    direct_device.append(device)
                else:
                    wall, device = _time_one(
                        compiled,
                        template,
                        output_cotangent,
                        state_cotangent,
                        backward=backward,
                        direct=False,
                    )
                    compiled_wall.append(wall)
                    compiled_device.append(device)
            pair_index = len(overhead)
            overhead.append(
                (compiled_wall[pair_index] - direct_wall[pair_index])
                / direct_wall[pair_index]
            )
        median_overhead = statistics.median(overhead)
        measurements[label] = {
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


def _check_parity(layer, plan, template, dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260947)
    output_cotangent = torch.randn(
        (1, SEQUENCE, VALUE_DIM), device="cuda", generator=generator
    )
    state_cotangent = torch.randn(
        (1, SLOTS, VALUE_DIM), device="cuda", generator=generator
    )
    direct_inputs = _fresh_inputs(template, backward=True)
    compiled_inputs = _fresh_inputs(template, backward=True)
    direct_output, direct_state = _direct_call(layer, direct_inputs, state_cotangent)
    direct_state = direct_state.detach().clone()
    compiled_output, compiled_state = _compiled_call(plan, compiled_inputs)
    (direct_output.float() * output_cotangent).mean().backward()
    _loss(compiled_output, compiled_state, output_cotangent, state_cotangent).backward()

    value_atol, value_rtol = (2e-2, 2e-2) if dtype is torch.bfloat16 else (2e-2, 3e-3)
    gradient_atol, gradient_rtol = (
        (3e-2, 3e-2) if dtype is torch.bfloat16 else (3e-5, 3e-4)
    )
    torch.testing.assert_close(
        compiled_output.float(), direct_output.float(), atol=value_atol, rtol=value_rtol
    )
    torch.testing.assert_close(
        compiled_state.float(), direct_state.float(), atol=value_atol, rtol=value_rtol
    )
    gradients = {}
    for name in GRADIENT_NAMES:
        direct_grad = direct_inputs[name].grad.float()
        compiled_grad = compiled_inputs[name].grad.float()
        torch.testing.assert_close(
            compiled_grad, direct_grad, atol=gradient_atol, rtol=gradient_rtol, msg=name
        )
        gradients[name] = (compiled_grad - direct_grad).abs().max().item()
    return (
        {
            "status": "pass",
            "output_max_abs_error": (compiled_output.float() - direct_output.float())
            .abs()
            .max()
            .item(),
            "final_state_max_abs_error": (compiled_state.float() - direct_state.float())
            .abs()
            .max()
            .item(),
            "input_gradient_max_abs_errors": gradients,
            "tolerances": {
                "output_atol": value_atol,
                "output_rtol": value_rtol,
                "state_atol": value_atol,
                "state_rtol": value_rtol,
                "gradient_atol": gradient_atol,
                "gradient_rtol": gradient_rtol,
            },
        },
        output_cotangent,
        state_cotangent,
    )


def _nvcc_version() -> str | None:
    cuda_home = Path(__import__("os").environ.get("CUDA_HOME", ""))
    nvcc = cuda_home / "bin/nvcc" if cuda_home else Path("nvcc")
    try:
        return (
            subprocess.check_output([str(nvcc), "--version"], text=True)
            .splitlines()[-1]
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _distribution_version() -> str | None:
    distributions = importlib.metadata.packages_distributions().get("lingua", ())
    for name in distributions:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def diagnose_overlapping_routes(output: Path) -> None:
    """Record equation parity and the pinned SDM VJP on overlapping routes."""
    if not torch.cuda.is_available():
        raise RuntimeError("the SDM overlap diagnostic requires CUDA")
    source, revision = _source_identity()
    cases = {}
    for index, dtype in enumerate(DTYPES):
        dtype_name = "bfloat16" if dtype is torch.bfloat16 else "float32"
        template = _inputs(dtype, seed=7731 + index)
        template["read_indices"] = template["write_indices"].clone()
        template["read_weights"] = template["write_weights"].clone()
        args = SparseDeltaMemoryArgs(
            dim=VALUE_DIM,
            num_writes=ROUTE_WIDTH,
            num_reads=ROUTE_WIDTH,
            slots_per_head=SLOTS,
            memory_block_size=64,
            backprop_on_memory=False,
            snapshot_quant="none",
        )
        layer = SparseDeltaMemory(args, layer_id=0).cuda().train()
        plan_args = {
            "intent": MixerIntent.TRAINING,
            "dtype": dtype_name,
        }
        native_plan = compile_mixer(
            sparse_delta_spec(), backend=MixerBackend.NATIVE, **plan_args
        )
        reference_plan = compile_mixer(
            sparse_delta_spec(), backend=MixerBackend.REFERENCE, **plan_args
        )
        generator = torch.Generator(device="cuda").manual_seed(20260947)
        output_cotangent = torch.randn(
            (1, SEQUENCE, VALUE_DIM), device="cuda", generator=generator
        )
        state_cotangent = torch.randn(
            (1, SLOTS, VALUE_DIM), device="cuda", generator=generator
        )

        direct_inputs = _fresh_inputs(template, backward=True)
        native_inputs = _fresh_inputs(template, backward=True)
        reference_inputs = _fresh_inputs(template, backward=True)
        direct_output, direct_state = _direct_call(
            layer, direct_inputs, state_cotangent
        )
        direct_state = direct_state.detach().clone()
        native_output, native_state = _compiled_call(native_plan, native_inputs)
        reference_output, reference_state = _compiled_call(
            reference_plan, reference_inputs
        )
        direct_loss = (direct_output.float() * output_cotangent).mean()
        native_loss = _loss(
            native_output, native_state, output_cotangent, state_cotangent
        )
        reference_loss = _loss(
            reference_output, reference_state, output_cotangent, state_cotangent
        )
        direct_loss.backward()
        native_loss.backward()
        reference_loss.backward()

        value_atol, value_rtol = (
            (2e-2, 2e-2) if dtype is torch.bfloat16 else (2e-2, 3e-3)
        )
        gradient_atol, gradient_rtol = (
            (3e-2, 3e-2) if dtype is torch.bfloat16 else (3e-5, 3e-4)
        )

        def compare(left, right, atol, rtol):
            return {
                "max_abs_error": (left.float() - right.float()).abs().max().item(),
                "pass": torch.allclose(
                    left.float(), right.float(), atol=atol, rtol=rtol
                ),
            }

        def comparison(
            left_output,
            left_state,
            left_inputs,
            right_output,
            right_state,
            right_inputs,
            value_atol,
            value_rtol,
            gradient_atol,
            gradient_rtol,
        ):
            gradients = {
                name: compare(
                    left_inputs[name].grad,
                    right_inputs[name].grad,
                    gradient_atol,
                    gradient_rtol,
                )
                for name in GRADIENT_NAMES
            }
            components = {
                "output": compare(left_output, right_output, value_atol, value_rtol),
                "final_state": compare(left_state, right_state, value_atol, value_rtol),
                "input_gradients": gradients,
            }
            passed = all(
                item["pass"]
                for component in components.values()
                for item in (
                    component.values() if component is gradients else (component,)
                )
            )
            return {"status": "pass" if passed else "measured_fail", **components}

        cases[dtype_name] = {
            "architecture_ids": ["arch-047"],
            "semantic_scope": "K3 state update with read and write routes overlapping at every token",
            "shape": {
                "batch": 1,
                "sequence": SEQUENCE,
                "slots": SLOTS,
                "value_dim": VALUE_DIM,
                "read_width": ROUTE_WIDTH,
                "write_width": ROUTE_WIDTH,
                "route_pattern": "read addresses equal write addresses; writes repeat across tokens; addresses are unique within each token",
                "dtype": dtype_name,
            },
            "parity_tolerances": {
                "output_atol": value_atol,
                "output_rtol": value_rtol,
                "state_atol": value_atol,
                "state_rtol": value_rtol,
                "gradient_atol": gradient_atol,
                "gradient_rtol": gradient_rtol,
            },
            "native_vs_equation_reference": comparison(
                native_output,
                native_state,
                native_inputs,
                reference_output,
                reference_state,
                reference_inputs,
                value_atol,
                value_rtol,
                gradient_atol,
                gradient_rtol,
            ),
            "native_vs_pinned_upstream": comparison(
                native_output,
                native_state,
                native_inputs,
                direct_output,
                direct_state,
                direct_inputs,
                value_atol,
                value_rtol,
                gradient_atol,
                gradient_rtol,
            ),
        }

    config = {
        "dtypes": ["float32", "bfloat16"],
        "sequence": SEQUENCE,
        "slots": SLOTS,
        "value_dim": VALUE_DIM,
        "route_width": ROUTE_WIDTH,
        "route_pattern": "read addresses equal write addresses",
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "diagnose native K3 equation parity and pinned SDM backward behavior for overlapping routes",
        "upstream": {
            "repository": "https://github.com/facebookresearch/sparse-delta-memory",
            "revision": revision,
            "loaded_module": str(source),
            "distribution_version": _distribution_version(),
        },
        "provenance": provenance(
            "CUDA_HOME=/path/to/cuda-toolkit PYTHONPATH=/path/to/sparse-delta-memory:src python benchmarks/unified_mixer_sdm.py --overlap-diagnostic",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvcc": _nvcc_version(),
        },
        "methodology": {
            "comparison": "native K3 plan and equation reference versus direct pinned SDM operator",
            "timing": "not measured; this artifact records a parity diagnostic, not a performance profile",
            "upstream_callable": "lingua.sparse_delta_memory.layer.SparseDeltaMemory.gated_write_read",
            "upstream_extension": "pinned source CUDA extension built with the CUDA 13 nvcc/CCCL toolchain",
        },
        "cases": cases,
    }
    write_artifact(output, payload)


def run(pairs: int, warmup: int, output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the SDM unified mixer profile requires CUDA")
    source, revision = _source_identity()
    cases = {}
    for index, dtype in enumerate(DTYPES):
        template = _inputs(dtype, seed=44047 + index)
        args = SparseDeltaMemoryArgs(
            dim=VALUE_DIM,
            num_writes=ROUTE_WIDTH,
            num_reads=ROUTE_WIDTH,
            slots_per_head=SLOTS,
            memory_block_size=64,
            backprop_on_memory=False,
            snapshot_quant="none",
        )
        layer = SparseDeltaMemory(args, layer_id=0).cuda().train()
        plan = compile_mixer(
            sparse_delta_spec(),
            backend=MixerBackend.NATIVE,
            intent=MixerIntent.TRAINING,
            dtype="bfloat16" if dtype is torch.bfloat16 else "float32",
        )
        direct = lambda inputs, cotangent=None, layer=layer: _direct_call(
            layer, inputs, cotangent
        )
        compiled = lambda inputs, plan=plan: _compiled_call(plan, inputs)
        parity, output_cotangent, state_cotangent = _check_parity(
            layer, plan, template, dtype
        )
        performance = _measure_pair(
            direct,
            compiled,
            template,
            output_cotangent,
            state_cotangent,
            pairs,
            warmup,
        )
        cases["float32" if dtype is torch.float32 else "bfloat16"] = {
            "architecture_ids": ["arch-047"],
            "semantic_scope": "K3 routed sparse delta state update; product-key score generation and model projections excluded",
            "shape": {
                "batch": 1,
                "sequence": SEQUENCE,
                "slots": SLOTS,
                "value_dim": VALUE_DIM,
                "read_width": ROUTE_WIDTH,
                "write_width": ROUTE_WIDTH,
                "route_pattern": "write routes repeat across tokens; read routes are disjoint; each route is unique within its token",
                "dtype": "bfloat16" if dtype is torch.bfloat16 else "float32",
            },
            "upstream_callable": "lingua.sparse_delta_memory.layer.SparseDeltaMemory.gated_write_read",
            "compiled_anchor": plan.anchor,
            "parity": parity,
            "performance": performance,
        }

    config = {
        "dtypes": ["float32", "bfloat16"],
        "pairs": pairs,
        "warmup": warmup,
        "sequence": SEQUENCE,
        "slots": SLOTS,
        "value_dim": VALUE_DIM,
        "route_width": ROUTE_WIDTH,
    }
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified native K3 compiler plan with direct pinned SDM state operator",
        "upstream": {
            "repository": "https://github.com/facebookresearch/sparse-delta-memory",
            "revision": revision,
            "loaded_module": str(source),
            "distribution_version": _distribution_version(),
        },
        "provenance": provenance(
            "CUDA_HOME=/path/to/cuda-toolkit PYTHONPATH=/path/to/sparse-delta-memory:src python benchmarks/unified_mixer_sdm.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nvcc": _nvcc_version(),
        },
        "methodology": {
            "timed_work": "direct SDM gated_write_read or compiled K3 plan, with one forward or forward plus backward",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing; fresh state buffers cloned before each timer",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "interpretation": "compares direct source operation with the full unified native K3 plan; route score production and model projections are excluded",
            "upstream_extension": "pinned source CUDA extension built with the CUDA 13 nvcc/CCCL toolchain before paired timing",
        },
        "cases": cases,
    }
    write_artifact(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--overlap-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path(
        "results/unified-mixer/sdm-k3-overlap-diagnostic.json"
        if args.overlap_diagnostic
        else "results/unified-mixer/sdm-k3.json"
    )
    if not args.overlap_diagnostic and (args.pairs < 1 or args.warmup < 0):
        parser.error("--pairs must be positive and --warmup nonnegative")
    if args.overlap_diagnostic:
        diagnose_overlapping_routes(output)
    else:
        run(args.pairs, args.warmup, output)


if __name__ == "__main__":
    main()
