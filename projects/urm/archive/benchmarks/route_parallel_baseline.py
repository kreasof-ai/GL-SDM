"""Fresh unchanged-model baseline and complete persistent-state diagnostic audit.

This never installs a candidate in a model. It reuses authority-v2's actual
execution, full frozen timings and separate eager replay. Profiles are separate
passes; compiled attribution uses observed CUDA kernel events, not eager ranges.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pretraining_step as baseline
import torch
from provenance import provenance, utc_now, write_artifact
from route_parallel_experiment import clean_revision
from state_tensor_audit import persistent_audit


def state_kernel_events(profile):
    return [
        {"name": x.name, "duration_us": x.time_range.elapsed_us()}
        for x in profile.events()
        if x.device_type == torch.autograd.DeviceType.CUDA
        and "sparse_state_update" in x.name
    ]


def child(args):
    source = clean_revision()
    original = baseline._one_step
    retained = {}
    state_hashes = []

    def audited_step(execution, optimizer, batches, config, **kwargs):
        model = getattr(execution, "_orig_mod", execution)
        if not retained:
            retained.update(
                execution=execution,
                model=model,
                optimizer=optimizer,
                batches=batches,
                config=config,
            )
        if kwargs["record_correctness"]:
            # Diagnostic-only reset and full hash; original performs the same reset.
            model.reset_state()
            state_hashes.append(
                baseline._tensor_digest(
                    (
                        (i, x.persistent_memory)
                        for i, x in enumerate(model.sparse_mixers())
                    ),
                    torch,
                )
            )
            retained.setdefault("initial_state_sha256", state_hashes[-1])
        return original(execution, optimizer, batches, config, **kwargs)

    baseline._one_step = audited_step
    try:
        baseline._child(args)
    finally:
        baseline._one_step = original
    result = json.loads(args.output.read_text())
    result["clean_source_commit"] = source
    result["matched_initial"]["persistent_state_sha256"] = retained[
        "initial_state_sha256"
    ]
    result["state_reset_audit"] = {
        "hashes_before_actual_correctness_and_separate_eager_replay": state_hashes,
        "all_match_initial": all(
            x == retained["initial_state_sha256"] for x in state_hashes
        ),
    }
    if not result["state_reset_audit"]["all_match_initial"]:
        raise RuntimeError("persistent state not matched before correctness")
    if args.backend == "urm_native":
        # Do not enable Python record-function ranges in compiled graph: profile
        # the actual production path and identify actual device kernel events.
        execution, optimizer, config, batches = (
            retained[k] for k in ("execution", "optimizer", "config", "batches")
        )
        kwargs = {"gradient_clip": 1.0, "record_correctness": False, "torch": torch}
        original(execution, optimizer, batches, config, **kwargs)
        torch.cuda.synchronize()
        profiles = []
        for repeat in range(3):
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
            with torch.profiler.profile(activities=activities) as profile:
                started = time.perf_counter_ns()
                original(execution, optimizer, batches, config, **kwargs)
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter_ns() - started) / 1e6
            events = state_kernel_events(profile)
            fwd = (
                sum(x["duration_us"] for x in events if "backward" not in x["name"])
                / 1000
            )
            bwd = (
                sum(x["duration_us"] for x in events if "backward" in x["name"]) / 1000
            )
            counts = {
                "forward": sum("backward" not in x["name"] for x in events),
                "backward": sum("backward" in x["name"] for x in events),
            }
            if counts != {"forward": 48, "backward": 48}:
                raise RuntimeError(
                    f"expected 48 actual state forward and backward launches, got {counts}"
                )
            profiles.append(
                {
                    "repeat": repeat,
                    "step_wall_ms": wall_ms,
                    "forward_kernel_ms": fwd,
                    "backward_kernel_ms": bwd,
                    "state_kernel_fraction": (fwd + bwd) / wall_ms,
                    "launch_counts": counts,
                    "launches": events,
                }
            )
        result["state_profiles"] = {
            "mode": args.mode,
            "scope": "separate_pass_actual_device_state_kernels_only; no clones/zeroing/casts credited; use candidate COMPLETE-stage cost for conservative screen",
            "passes": profiles,
        }
    # Artifacts have new names and are not passed off as authority-v2 schema.
    write_artifact(args.output, result)


def compare(left, right, tolerances):
    result = baseline._compare_correctness(left, right, torch, tolerances)
    for index, step in enumerate(result["steps"]):
        states = [
            torch.load(row["state_files"][index], weights_only=True, mmap=True)
            for row in (left, right)
        ]
        step["persistent_state_full_audit"] = persistent_audit(
            *states,
            left["correctness"][index]["microbatch_persistent_state_checksums"],
            right["correctness"][index]["microbatch_persistent_state_checksums"],
        )
    result["new_tensor_audit_changes_acceptance"] = False
    return result


def parent(args):
    source = clean_revision()
    frozen, _ = baseline.load_frozen_config()
    tolerances = frozen["correctness"]["bfloat16"]
    artifact = {
        "schema_version": 1,
        "artifact_kind": "ordered_route_parallel_corrected_model_baseline",
        "generated_utc": utc_now(),
        "provenance": provenance(" ".join(sys.argv), frozen),
        "clean_source_commit": source,
        "frozen_contract": frozen,
        "candidate_in_model": False,
        "modes": {},
        "eager_vs_compiled": {},
    }
    with tempfile.TemporaryDirectory(prefix="urm-route-baseline-") as root:
        root = Path(root)
        # Finish one seed's cross-mode comparisons before releasing its snapshots.
        # This bounds disk without dropping any accumulation microbatch/layer.
        all_rows = {mode: [] for mode in frozen["execution"]["modes"]}
        all_comparisons = {mode: [] for mode in all_rows}
        all_internal = {mode: [] for mode in all_rows}
        for seed in frozen["measurement"]["paired_seeds"]:
            seed_rows = {}
            for mode, pairs in all_rows.items():
                order = ["upstream_sdm", "urm_native"]
                random.Random(seed + 77).shuffle(order)
                rows = {}
                for backend in order:
                    output = root / f"{seed}-{mode}-{backend}.json"
                    cache = root / f"{seed}-{mode}-{backend}-cache"
                    env = os.environ.copy()
                    for name, subdir in (
                        ("TRITON_CACHE_DIR", "triton"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                        ("TORCH_EXTENSIONS_DIR", "extensions"),
                    ):
                        env[name] = str(cache / subdir)
                    command = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--child",
                        "--backend",
                        backend,
                        "--mode",
                        mode,
                        "--seed",
                        str(seed),
                        "--output",
                        str(output),
                    ]
                    print("baseline child:", " ".join(command), flush=True)
                    subprocess.run(command, env=env, check=True)
                    rows[backend] = json.loads(output.read_text())
                left, right = rows["upstream_sdm"], rows["urm_native"]
                all_comparisons[mode].append(compare(left, right, tolerances))
                all_internal[mode].append(
                    baseline._compare_correctness(
                        left, right, torch, tolerances, internal=True
                    )
                )
                for row in (left, right):
                    for path in row.pop("gradient_files"):
                        Path(path).unlink()
                pairs.append((left, right))
                seed_rows[mode] = rows
            for backend in ("urm_native", "upstream_sdm"):
                left, right = (
                    seed_rows["eager"][backend],
                    seed_rows["compile_fullgraph"][backend],
                )
                artifact["eager_vs_compiled"].setdefault(backend, []).append(
                    compare(left, right, tolerances)
                )
            for rows in seed_rows.values():
                for row in rows.values():
                    for path in row.pop("state_files"):
                        Path(path).unlink()
            for mode, pairs in all_rows.items():
                artifact["modes"][mode] = {
                    "pairs": [{"upstream": a, "native": b} for a, b in pairs],
                    "correctness": all_comparisons[mode],
                    "eager_internal_gradient_diagnostics": all_internal[mode],
                    "performance": baseline._ratio_summary(pairs),
                }
            # Durable checkpoint after each full seed, not just at the end.
            artifact["completed_seeds"] = [
                pair[0]["seed"] for pair in all_rows["eager"]
            ]
            artifact["complete"] = (
                artifact["completed_seeds"] == frozen["measurement"]["paired_seeds"]
            )
            write_artifact(args.output, artifact)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--child", action="store_true")
    p.add_argument("--backend", choices=("urm_native", "upstream_sdm"))
    p.add_argument("--mode", choices=("eager", "compile_fullgraph"))
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    args.diagnostic = False
    (child if args.child else parent)(args)


if __name__ == "__main__":
    main()
