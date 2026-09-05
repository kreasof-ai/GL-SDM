"""Frozen-shape isolated route/residency experiment, never model integration.

Run correctness first, then timing in fresh processes from the same clean commit.
Upstream comparisons are reported separately, never substituted for the oracle.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from provenance import provenance, utc_now, write_artifact
from sparse_state_triangular import (
    BACKWARD,
    NAMES,
    ORACLE_FORWARD,
    UPSTREAM_FORWARD,
    make_case,
    run_reference,
)
from state_tensor_audit import tensor_audit

from urm.compiler.semantic import SparseReadTiming
from urm.experiments import route_parallel as rp
from urm.triton_kernels import sparse_state_mixer as native

IMPLEMENTATIONS = ("native", "route_global", "route_resident")
MODES = ("eager", "compile_fullgraph")


def clean_revision():
    # Unlike old provenance, reject untracked code as well as tracked edits.
    rows = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], text=True
    ).splitlines()
    bad = [row for row in rows if "/results/" not in row]
    if bad:
        raise RuntimeError(
            f"authoritative experiment requires committed clean source: {bad}"
        )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def identity(sample):
    wi, ri, leaves, cy, cf = sample
    digest = hashlib.sha256()
    for x in (wi, ri, *leaves, cy, cf):
        digest.update(str((x.shape, x.dtype)).encode())
        digest.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def certify_sample(sample):
    wi, ri, data, _cy, _cf = sample
    for indices in (wi, ri):
        assert bool(((indices >= 0) & (indices < data[0].shape[1])).all())
        assert bool((indices.sort(-1).values.diff(dim=-1) > 0).all())


def production_case(dtype=torch.bfloat16, pattern="mixed", seed=1701):
    """Exact state shape; random routes plus nonzero state, not a model surrogate."""
    torch.manual_seed(seed)
    p, t, s, d, width = 12, 1024, 4096, 64, 64
    device = "cuda"
    if pattern == "repeated":
        wi = torch.arange(width, device=device).expand(p, t, -1).contiguous()
        ri = (wi + 32).contiguous()
    else:
        # Product-key structure and sorted routes like the production route stage.
        from urm.triton_kernels.sparse_route import sparse_route_selection

        scores = [torch.randn(p, t, 128, device=device, dtype=dtype) for _ in range(2)]
        wi, ww = sparse_route_selection(scores[0], s, width)
        ri, rw = sparse_route_selection(scores[1], s, width)
    if pattern == "repeated":
        ww = torch.softmax(torch.randn(p, t, width, device=device, dtype=dtype), -1)
        rw = torch.softmax(torch.randn(p, t, width, device=device, dtype=dtype), -1)
    leaves = [
        torch.randn(p, s, d, device=device, dtype=dtype) * 0.05,
        ww,
        torch.randn(p, t, d, device=device, dtype=dtype) * 0.05,
        torch.rand(p, t, 1, device=device, dtype=dtype),
        -torch.rand(p, t, 1, device=device, dtype=dtype) * 0.1,
        rw,
    ]
    cy = torch.randn(p, t, d, device=device) / (p * t * d)
    cf = torch.randn(p, s, d, device=device) / (p * s * d)
    return wi, ri, leaves, cy, cf


def function(kind, before):
    if kind == "native":

        def fn(m, wi, w, v, b, g, ri, q):
            return native.sparse_state_update(
                m, wi, w, v, b, g, ri, q, read_before_update=before
            )
    else:

        def fn(m, wi, w, v, b, g, ri, q):
            return rp.update(
                m,
                wi,
                w,
                v,
                b,
                g,
                ri,
                q,
                resident=kind == "route_resident",
                before=before,
            )

    return fn


def evaluate(fn, sample):
    wi, ri, data, cy, cf = sample
    leaves = [x.detach().clone().requires_grad_() for x in data]
    m, w, v, b, g, q = leaves
    y, f = fn(m, wi, w, v, b, g, ri, q)
    loss = (y.float() * cy).sum() + (f.float() * cf).sum()
    grads = torch.autograd.grad(loss, leaves)
    return {
        "readings": y.detach().float().cpu(),
        "final_memory": f.detach().float().cpu(),
        "gradients": {
            n: x.detach().float().cpu() for n, x in zip(NAMES, grads, strict=True)
        },
        "joint_loss": float(loss.detach()),
    }


def compare(actual, expected, dtype, upstream=False):
    tol = (UPSTREAM_FORWARD if upstream else ORACLE_FORWARD)[dtype]
    reports = {
        n: tensor_audit(actual[n], expected[n], tol)
        for n in ("readings", "final_memory")
    }
    reports["gradients"] = {
        n: tensor_audit(
            actual["gradients"][n], expected["gradients"][n], BACKWARD[dtype]
        )
        for n in NAMES
    }
    reports["passed"] = all(
        reports[n]["passed"] for n in ("readings", "final_memory")
    ) and all(x["passed"] for x in reports["gradients"].values())
    return reports


def base_artifact(kind, config, development):
    revision = None if development else clean_revision()
    return {
        "schema_version": 1,
        "artifact_kind": kind,
        "generated_utc": utc_now(),
        "provenance": provenance(" ".join(sys.argv), config),
        "clean_source_commit": revision,
        "development_only": development,
        "configuration": config,
        "comparison_revisions": {
            "native": subprocess.check_output(
                ["git", "rev-parse", "bdae3d7"], text=True
            ).strip(),
            "oracle": subprocess.check_output(
                ["git", "rev-parse", "bdae3d7"], text=True
            ).strip(),
            "upstream": "183e7df809131b80ad4393741029d0f20fc3640b",
            "m2rnn": "384ed0a7bd82ced1f40609603dd541cac5416844",
            "atma": "fcefd2d75f9db73c16850343e20b57b3b729b3ea",
        },
        "runtime_environment": {
            "torch_cpu_threads": torch.get_num_threads(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "gpu_properties": str(torch.cuda.get_device_properties(0)),
        },
        "production_integration": False,
    }


def correctness(args):
    config = {
        "modes": list(MODES),
        "implementations": list(IMPLEMENTATIONS),
        "oracle_tolerances": ORACLE_FORWARD,
        "upstream_tolerances": UPSTREAM_FORWARD,
        "gradient_tolerances": BACKWARD,
        "loss": "readings_and_final_memory",
        "upstream_comparison": "separate_not_candidate_acceptance",
        "intermediate_state_casts": "every_token_forward_and_reverse; read_gradient_before_write_vjp",
        "native_compiled_scope": "production_shapes_and_rounding_stress_129; both_candidate_modes_cover_every_case",
    }
    artifact = base_artifact(
        "ordered_route_parallel_correctness", config, args.development
    )
    rp.library()
    specs = []
    lengths = {
        "repeated": [1, 16, 17, 65, 129],
        "mixed": [15, 32, 33, 63, 64],
        "disjoint": [1, 16, 17, 63, 64],
        "rounding_stress": [64, 129],
    }
    if args.smoke:
        lengths = {"mixed": [17], "rounding_stress": [129]}
    for dtype, (pattern, ts), before in itertools.product(
        ("float32", "bfloat16"), lengths.items(), (False, True)
    ):
        for t in ts:
            specs.append((dtype, pattern, t, before, False))
    if not args.smoke:
        for dtype, pattern, before in itertools.product(
            ("float32", "bfloat16"), ("mixed", "repeated"), (False, True)
        ):
            specs.append((dtype, pattern, 1024, before, True))
    rows = []
    # Use one compiled callable per specialization, dynamic=False/fullgraph=True.
    # Compilation/correctness here are never timed as performance.
    for index, (dtype, pattern, t, before, production) in enumerate(specs):
        sample = (
            production_case(getattr(torch, dtype), pattern)
            if production
            else make_case(getattr(torch, dtype), t, pattern, 64)
        )
        digest = identity(sample)
        certify_sample(sample)
        timing = (
            SparseReadTiming.BEFORE_UPDATE if before else SparseReadTiming.AFTER_UPDATE
        )
        oracle = run_reference("per_token_cast_oracle", sample, 16, timing)
        upstream = run_reference("pinned_upstream", sample, 16, timing)
        native_result = evaluate(function("native", before), sample)
        row = {
            "dtype": dtype,
            "pattern": pattern,
            "sequence": t,
            "before": before,
            "production_shape": production,
            "matched_inputs_sha256": digest,
            "upstream_vs_oracle": compare(upstream, oracle, dtype),
            "native_vs_oracle": compare(native_result, oracle, dtype),
            "comparisons": [],
        }
        for kind, mode in itertools.product(IMPLEMENTATIONS, MODES):
            if (
                kind == "native"
                and mode == "compile_fullgraph"
                and not (production or (pattern == "rounding_stress" and t == 129))
            ):
                continue
            torch._dynamo.reset()
            from torch._dynamo.utils import counters

            counters.clear()
            fn = function(kind, before)
            if mode == "compile_fullgraph":
                fn = torch.compile(fn, fullgraph=True, dynamic=False)
            result = (
                native_result
                if kind == "native" and mode == "eager"
                else evaluate(fn, sample)
            )
            row["comparisons"].append(
                {
                    "implementation": kind,
                    "mode": mode,
                    "against_native": compare(result, native_result, dtype),
                    "against_oracle": compare(result, oracle, dtype),
                    "against_upstream": compare(result, upstream, dtype, True),
                    "graph_breaks": sum(counters["graph_break"].values()),
                    "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
                }
            )
        if identity(sample) != digest:
            raise RuntimeError(
                "a functional comparison mutated matched initial inputs/state"
            )
        rows.append(row)
        print(
            f"correctness {index + 1}/{len(specs)} {dtype} {pattern} T={t} before={before}",
            flush=True,
        )
        del oracle, upstream, native_result, sample, result
    artifact["cases"] = rows
    artifact["accepted_against_native_and_oracle"] = all(
        c["against_native"]["passed"]
        and c["against_oracle"]["passed"]
        and c["graph_breaks"] == 0
        and (c["mode"] == "eager" or c["unique_graphs"] == 1)
        for r in rows
        for c in r["comparisons"]
    )
    artifact["upstream_all_passed_separate"] = all(
        c["against_upstream"]["passed"] for r in rows for c in r["comparisons"]
    )
    write_artifact(args.output, artifact)
    return artifact["accepted_against_native_and_oracle"]


def stage_functions(kind, sample):
    wi, ri, data, cy, cf = sample
    m, w, v, b, g, q = data
    dy, df = cy.to(m.dtype), cf.to(m.dtype)
    if kind == "native":

        def forward():
            return native._sparse_state_update_forward(
                m.clone(),
                wi,
                w,
                v,
                b,
                g,
                ri,
                q,
                read_before_update=False,
                save_selected=True,
            )

        def backward(wh, rh):
            return native._sparse_state_update_backward(
                dy, df, wh, rh, wi, w, v, b, g, ri, q, read_before_update=False
            )
    else:
        resident = kind == "route_resident"

        def forward():
            return rp.forward(m, wi, w, v, b, g, ri, q, resident, False)

        def backward(wh, rh):
            return rp.backward(dy, df, wh, rh, wi, w, v, b, g, ri, q, resident, False)

    return forward, backward


def timed(call, iterations=20, warmups=10):
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall, device = [], []
    for _ in range(iterations):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize()
        now = time.perf_counter_ns()
        start.record()
        result = call()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter_ns() - now) / 1e6)
        device.append(start.elapsed_time(end))
        del result
    return {
        "wall_ms": wall,
        "cuda_span_ms": device,
        "median_wall_ms": statistics.median(wall),
        "median_cuda_ms": statistics.median(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }


def performance(args):
    accepted = json.loads(args.correctness.read_text())
    if not args.development and (
        not accepted["accepted_against_native_and_oracle"]
        or accepted["development_only"]
        or accepted["clean_source_commit"] != clean_revision()
    ):
        raise RuntimeError(
            "authoritative timing requires accepted correctness from this clean source commit"
        )
    config = {
        "shape": [12, 1024, 4096, 64, 64, 64],
        "dtype": "bfloat16",
        "D_tile": 4,
        "warmups": 10,
        "iterations": 20,
        "seeds": [1701, 2903, 4409],
        "scope": "forward_with_selected_histories+backward; all required copies, allocations, zeroing, casts, sync and host orchestration included; no diagnostic scans",
    }
    artifact = base_artifact(
        "ordered_route_parallel_performance", config, args.development
    )
    artifact["build"] = rp.library()[1]
    rows = []
    for seed in config["seeds"]:
        sample = production_case(seed=seed)
        certify_sample(sample)
        order = list(IMPLEMENTATIONS)
        random.Random(seed).shuffle(order)
        for kind in order:
            for mode in MODES:
                torch._dynamo.reset()
                fwd, bwd = stage_functions(kind, sample)
                if mode == "compile_fullgraph":
                    fwd, bwd = (
                        torch.compile(fwd, fullgraph=True, dynamic=False),
                        torch.compile(bwd, fullgraph=True, dynamic=False),
                    )
                hist = fwd()
                wh, rh = hist[2:]

                def both(fwd=fwd, bwd=bwd):
                    outputs = fwd()
                    return outputs, bwd(*outputs[2:])

                row = {
                    "seed": seed,
                    "implementation": kind,
                    "mode": mode,
                    "input_sha256": identity(sample),
                    "forward": timed(fwd),
                    "backward": timed(lambda bwd=bwd, wh=wh, rh=rh: bwd(wh, rh)),
                    "combined": timed(both),
                }
                rows.append(row)
                print(
                    f"timing {seed} {kind} {mode}: {row['combined']['median_wall_ms']:.4f} ms",
                    flush=True,
                )
                del hist, wh, rh
    artifact["measurements"] = rows
    write_artifact(args.output, artifact)
    return True


def resources(args):
    artifact = base_artifact(
        "ordered_route_parallel_launch_resources", {}, args.development
    )
    sample = production_case()
    m = sample[2][0]
    artifact["cuda_build"] = rp.library()[1]
    artifact["resources"] = {
        kind: {
            phase: rp.resource_report(
                m, sample[0], kind == "route_resident", False, phase == "backward"
            )
            for phase in ("forward", "backward")
        }
        for kind in IMPLEMENTATIONS[1:]
    }
    audit_rows = []
    for mode in MODES:
        rp.enable_audit(True)
        for kind in IMPLEMENTATIONS[1:]:
            fn = function(kind, False)
            if mode == "compile_fullgraph":
                fn = torch.compile(fn, fullgraph=True, dynamic=False)
            evaluate(fn, sample)
        audit_rows.append({"mode": mode, "launches": rp.AUDIT.copy()})
        rp.enable_audit(False)
    artifact["cuda_launch_audits"] = audit_rows
    # Real native launch metadata and assembly; hooks isolated from timing processes.
    from triton import knobs
    from triton.compiler.compiler import CompiledKernel

    original, previous = CompiledKernel.launch_metadata, knobs.runtime.launch_enter_hook
    observed = []
    asm_dir = args.output.with_suffix(".assembly")
    asm_dir.mkdir(parents=True, exist_ok=True)

    def hook(kernel, grid, stream, *arguments):
        if "sparse_state_update" in kernel.name:
            stem = "native_backward" if "backward" in kernel.name else "native_forward"
            for fmt in ("ptx", "cubin"):
                value = kernel.asm[fmt]
                path = asm_dir / f"{stem}.{fmt}"
                path.write_bytes(value if isinstance(value, bytes) else value.encode())
            observed.append(
                {
                    "kernel": kernel.name,
                    "grid": list(grid),
                    "registers": kernel.n_regs,
                    "spills": kernel.n_spills,
                    "shared": kernel.metadata.shared,
                    "warps": kernel.metadata.num_warps,
                    "stages": kernel.metadata.num_stages,
                    "constants": {
                        kernel.src.fn.arg_names[k[0]]
                        if isinstance(k, tuple)
                        else str(k): v
                        for k, v in kernel.src.constants.items()
                    },
                }
            )
            driver = ctypes.CDLL("libcuda.so.1")
            occupancy = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
            occupancy.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_size_t,
            ]
            blocks = ctypes.c_int()
            status = occupancy(
                ctypes.byref(blocks),
                kernel.function,
                kernel.metadata.num_warps * 32,
                kernel.metadata.shared,
            )
            if status:
                raise RuntimeError(f"native occupancy query failed: {status}")
            observed[-1]["max_active_blocks_per_sm"] = blocks.value
            observed[-1]["theoretical_occupancy_fraction_sm86"] = (
                blocks.value * kernel.metadata.num_warps / 48
            )
        return original(kernel, grid, stream, *arguments)

    knobs.runtime.launch_enter_hook = lambda _: None
    CompiledKernel.launch_metadata = hook
    try:
        for mode in MODES:
            fn = function("native", False)
            if mode == "compile_fullgraph":
                fn = torch.compile(fn, fullgraph=True, dynamic=False)
            evaluate(fn, sample)
    finally:
        CompiledKernel.launch_metadata, knobs.runtime.launch_enter_hook = (
            original,
            previous,
        )
    for row in observed:
        assert row["grid"][:2] == [12, 16] and row["warps"] == 2 and row["stages"] == 3
        for k, v in {
            "BLOCK_D": 4,
            "SEQUENCE": 1024,
            "SLOTS": 4096,
            "VALUE_DIM": 64,
            "WRITE_WIDTH": 64,
            "READ_WIDTH": 64,
            "READ_BEFORE_UPDATE": False,
        }.items():
            assert row["constants"][k] == v
    artifact["native_launches"] = observed
    artifact["native_launches_verified"] = len(observed) >= 4
    # Keep text disassembly for independent inspection, no inference from tensor size.
    binaries = [Path(artifact["cuda_build"]["binary"]), *asm_dir.glob("*.cubin")]
    for binary in binaries:
        result = subprocess.run(
            ["cuobjdump", "--dump-sass", str(binary)],
            capture_output=True,
            text=True,
            check=True,
        )
        (asm_dir / f"{binary.stem}.sass").write_text(result.stdout)
    artifact["assembly_directory"] = str(asm_dir)
    artifact["hardware_counters"] = (
        "not_measured_by_this_command; collect separately with Nsight Compute if permitted"
    )
    write_artifact(args.output, artifact)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("correctness", "performance", "resources"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and not args.development:
        raise ValueError("smoke checks cannot authorize measurement")
    with contextlib.nullcontext():
        passed = {
            "correctness": correctness,
            "performance": performance,
            "resources": resources,
        }[args.phase](args)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
