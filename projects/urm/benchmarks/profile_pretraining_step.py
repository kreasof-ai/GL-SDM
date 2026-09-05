"""One-step model-level Sparse Memory attribution with NVTX-compatible ranges."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pretraining_step import _one_step, load_frozen_config, prefetched_batches
from provenance import provenance, utc_now, write_artifact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["urm_native"], required=True)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--emit-nvtx", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pretraining-step/native-profile-authority-v2.json"),
    )
    args = parser.parse_args()

    import torch

    from urm.pretraining import FP32AdamW, URMDecoderLM
    from urm.sparse_state_profile import EVENTS

    payload, config = load_frozen_config(diagnostic=args.diagnostic)
    torch.manual_seed(811)
    torch.cuda.manual_seed_all(811)
    model = URMDecoderLM(config, args.backend).cuda().to(torch.bfloat16)
    optimizer = FP32AdamW(model.named_parameters())
    batches = prefetched_batches(config, 811, 2, torch)
    kwargs = {
        "gradient_clip": float(payload["optimizer"]["gradient_clip_norm"]),
        "record_correctness": False,
        "torch": torch,
    }
    _one_step(
        model, optimizer, batches[: config.gradient_accumulation], config, **kwargs
    )
    torch.cuda.synchronize()
    model.enable_profiling(True)
    EVENTS.clear()

    measured = batches[config.gradient_accumulation : 2 * config.gradient_accumulation]
    trace_path = args.output.with_suffix(".trace.json")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if args.emit_nvtx:
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            _one_step(model, optimizer, measured, config, **kwargs)
        torch.cuda.synchronize()
        print("NVTX step complete; capture this command under nsys --trace=cuda,nvtx")
        return

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        started = time.perf_counter_ns()
        _one_step(model, optimizer, measured, config, **kwargs)
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - started) / 1e6
    profile.export_chrome_trace(str(trace_path))
    ranges = []
    for event in profile.key_averages():
        # CPU range attribution already includes descendant CUDA work. CUDA
        # entries bearing the same range name would count it a second time.
        if (
            event.key.startswith("pretraining::")
            and event.device_type == torch.autograd.DeviceType.CPU
        ):
            device_total = getattr(
                event, "device_time_total", getattr(event, "cuda_time_total", 0.0)
            )
            ranges.append(
                {
                    "name": event.key,
                    "count": int(event.count),
                    "cpu_time_total_us": float(event.cpu_time_total),
                    "device_time_total_us": float(device_total),
                    "cpu_memory_usage_bytes": int(event.cpu_memory_usage),
                    "device_memory_usage_bytes": int(event.device_memory_usage),
                }
            )
    plans = [mixer._executor.serialized_plan() for mixer in model.sparse_mixers()]
    artifact_provenance = provenance(" ".join(sys.argv), payload, include_gpu=True)
    artifact = {
        "schema_version": 1,
        "artifact_kind": "pretraining_step_profile",
        "generated_utc": utc_now(),
        "provenance": artifact_provenance,
        "backend": args.backend,
        "diagnostic": args.diagnostic,
        "configuration": config.to_dict(),
        "scope": "one_complete_eager_optimizer_step_after_one_warmup_step",
        "trace_file": str(trace_path),
        "nvtx_command": (
            "nsys profile --trace=cuda,nvtx python benchmarks/profile_pretraining_step.py "
            f"--backend {args.backend} --emit-nvtx"
        ),
        "ranges": ranges,
        "state_stage_spans": {
            "scope": "profiled_cuda_spans_include_state_forward_backward_allocations_and_orchestration; not_acceptance_timing",
            "profiled_step_wall_ms": wall_ms,
            "forward_ms": sum(
                start.elapsed_time(end)
                for phase, start, end in EVENTS
                if phase == "forward"
            ),
            "backward_ms": sum(
                start.elapsed_time(end)
                for phase, start, end in EVENTS
                if phase == "backward"
            ),
            "replaceable_fraction_diagnostic": sum(
                start.elapsed_time(end) for _, start, end in EVENTS
            )
            / wall_ms,
            "span_count": len(EVENTS),
        },
        "compiler_execution": {
            "layer_count": config.layers,
            "plan": plans[0] if plans else None,
            "all_layers_identical": bool(plans)
            and all(plan == plans[0] for plan in plans),
            "benchmark_direct_backend_instantiation": False,
        },
    }
    import json

    from jsonschema import validate

    schema_path = Path(__file__).with_name(
        "pretraining-step-profile-authority-schema.json"
    )
    validate(artifact, json.loads(schema_path.read_text(encoding="utf-8")))
    write_artifact(args.output, artifact)


if __name__ == "__main__":
    main()
