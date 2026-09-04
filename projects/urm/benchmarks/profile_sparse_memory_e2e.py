"""NVTX and CUDA-profiler validation of Sparse Memory E2E stage attribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sparse_memory_e2e as benchmark
import torch
from provenance import provenance, utc_now, write_artifact


def main() -> None:
    from urm.adapters.sparse_delta_memory import probe_sdm_support

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="long_nonpower_bf16")
    parser.add_argument(
        "--output", type=Path, default=Path("results/sparse-memory-e2e/profile.json")
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("results/sparse-memory-e2e/profile-trace.json"),
    )
    args = parser.parse_args()
    case = next(
        (item for item in benchmark.load_cases() if item["name"] == args.case), None
    )
    if case is None:
        raise ValueError(f"unknown frozen case {args.case!r}")
    support = probe_sdm_support()
    if not support.supported:
        raise RuntimeError(
            f"pinned comparator unavailable: {support.code}: {support.reason}"
        )
    bundle = benchmark._make_bundle(case, torch)
    calls = benchmark._calls(bundle, torch)

    def upstream_state_stage():
        spec = bundle["spec"]
        route = bundle["up_route"]
        memory = calls["upstream_memory"]
        if spec.operation.value == "read_only":
            return bundle["upstream"].direct_calls["read"](memory, route[1], route[4])
        return bundle["upstream"].direct_calls["update"](
            memory,
            route[5],
            route[3],
            bundle["values"],
            bundle["beta"],
            bundle["decay"],
            route[4],
            route[1],
        )

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
        for path in ("upstream", "native", "hybrid"):
            reset = calls[f"reset_{path}"]
            reset()
            torch.cuda.synchronize()
            label = f"sparse_memory_e2e::{path}::whole_pipeline"
            with torch.profiler.record_function(label), torch.cuda.nvtx.range(label):
                calls[path]()
            torch.cuda.synchronize()
        diagnostic_stages = {
            "upstream::route_production": bundle["upstream_routes"],
            "native::route_production": lambda: (
                bundle["native"].read_backend.generate(
                    bundle["native_prepared"].read_scores
                ),
                (
                    bundle["native"].write_backend.generate(
                        bundle["native_prepared"].write_scores
                    )
                    if bundle["native"].write_backend is not None
                    else None
                ),
            ),
            "upstream::state_mixer": upstream_state_stage,
            "native::state_mixer": lambda: bundle["native"].state_backend.execute(
                bundle["SparseState"](calls["native_memory"]),
                bundle["native_state_prepared"],
            ),
        }
        for stage, call in diagnostic_stages.items():
            if stage == "native::state_mixer":
                calls["reset_native"]()
            elif stage == "upstream::state_mixer":
                calls["reset_upstream"]()
            torch.cuda.synchronize()
            label = f"sparse_memory_e2e::{stage}"
            with torch.profiler.record_function(label), torch.cuda.nvtx.range(label):
                call()
            torch.cuda.synchronize()
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(args.trace))
    events = []
    for event in profile.key_averages():
        if "sparse_memory" in event.key or event.self_device_time_total > 0:
            events.append(
                {
                    "name": event.key,
                    "count": event.count,
                    "cpu_time_total_us": event.cpu_time_total,
                    "self_device_time_total_us": event.self_device_time_total,
                }
            )
    profile_provenance = provenance(
        " ".join(sys.argv), {"case": case}, include_gpu=True
    )
    profile_provenance["upstream"] = {
        key: support.details[key]
        for key in (
            "repository",
            "expected_commit",
            "installed_commit",
            "checkout_root",
            "module_file",
            "checkout_dirty",
            "license",
            "installation",
            "source_usage",
            "runtime_versions",
        )
    }
    artifact = {
        "schema_version": 1,
        "artifact_kind": "stage_profile",
        "generated_utc": utc_now(),
        "provenance": profile_provenance,
        "case": case,
        "profiler": {
            "tool": "torch.profiler CPU+CUDA activities (Nsight Systems unavailable)",
            "nvtx_ranges": [
                "sparse_memory_e2e::upstream::whole_pipeline",
                "sparse_memory_e2e::native::whole_pipeline",
                "sparse_memory_e2e::hybrid::whole_pipeline",
                "sparse_memory_e2e::upstream::route_production",
                "sparse_memory_e2e::native::route_production",
                "sparse_memory_e2e::upstream::state_mixer",
                "sparse_memory_e2e::native::state_mixer",
            ],
            "trace": str(args.trace),
            "events": events,
        },
    }
    write_artifact(args.output, artifact)


if __name__ == "__main__":
    main()
