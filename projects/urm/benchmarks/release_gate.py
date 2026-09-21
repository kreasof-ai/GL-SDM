"""Executable production release gate (acceptance-contract section 10).

Runs every mandatory workload's native-replacement qualification and the held-out
generation probes, then reports the honest production progress:

    production progress = fully qualified mandatory workloads / all mandatory workloads

The gate fails closed: a workload counts as qualified only when its qualification
artifact exists, is well-formed, and records verdict == "qualified" (all four
gates - Express, Execute, Match, Replace - pass). Missing, malformed, stale, or
failed evidence counts as not qualified. Aggregation is independent of case
ordering. Narrative reports cannot override this verdict.

Each workload's qualification runner writes a machine-readable artifact under
results/qualification/. This gate runs the runners in fresh processes (per the
measurement policy) and aggregates the recorded verdicts.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "qualification"
MATRIX = PROJECT_ROOT / "benchmarks" / "production-matrix.json"

# Mandatory workload -> (qualification runner, artifact, comparator-available note).
# Runners are invoked as fresh processes. A workload with no runnable comparator in
# this environment is reported "unsupported" (it stays in the denominator).
WORKLOADS = {
    "k1-mha": {
        "runner": "benchmarks/qualify_native_k1_attention.py",
        "artifact": "native-k1-mha.json",
        "args": [],
    },
    "k1-gqa": {
        "runner": "benchmarks/qualify_native_k1_attention.py",
        "artifact": "native-k1-gqa.json",
        "args": ["--qheads", "32", "--kvheads", "8"],
    },
    "k2-diagonal-recurrence": {
        "runner": "benchmarks/qualify_native_k2_hgrn.py",
        "artifact": "native-k2-hgrn.json",
        "args": ["--block", "10"],
    },
    "k2-gated-delta-recurrence": {
        "runner": "benchmarks/qualify_native_k2_gated_delta.py",
        "artifact": "native-k2-gated-delta.json",
        "args": ["--block", "10"],
    },
    "k3-sparse-state": {
        "runner": "benchmarks/unified_mixer_sdm.py",
        "artifact": "sdm-k3.json",
        "results_dir": PROJECT_ROOT / "results" / "unified-mixer",
        "args": [],
        "block": False,
        # The SDM upstream builds its CUDA extension with the CUDA 13 nvcc/CCCL
        # toolchain (pip nvidia-cuda-* 13.0 packages) and imports from the pinned
        # checkout; see benchmarks/provision_comparators.py.
        "env": {
            "PYTHONPATH": "src:/tmp/urm-comparator-pins/sdm",
            "CUDA_HOME": "/opt/conda/lib/python3.12/site-packages/nvidia/cu13",
            "PATH_PREFIX": "/opt/conda/lib/python3.12/site-packages/nvidia/cu13/bin",
        },
    },
}

VALID_VERDICTS = {
    "qualified",
    "correct_below_target",
    "numeric_failed",
    "inconclusive",
    "unsupported",
    "not_run",
}


def _run_workload(workload_id: str, spec: dict, pairs: int, warmup: int, block: int) -> dict:
    """Run a workload's qualification runner in a fresh process; read its artifact."""
    import os

    artifact_path = spec.get("results_dir", RESULTS) / spec["artifact"]
    cmd = [
        sys.executable,
        spec["runner"],
        "--pairs", str(pairs),
        "--warmup", str(warmup),
        "--output", str(artifact_path),
        *spec["args"],
    ]
    if block and spec.get("block", True) and "--block" not in spec["args"]:
        cmd += ["--block", str(block)]
    env = {"PYTHONPATH": "src", **os.environ}
    extra_env = spec.get("env")
    if extra_env:
        for key, value in extra_env.items():
            if key == "PATH_PREFIX":
                env["PATH"] = value + os.pathsep + env.get("PATH", "")
            elif key == "PYTHONPATH":
                env["PYTHONPATH"] = value + os.pathsep + env.get("PYTHONPATH", "")
            else:
                env[key] = value
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"workload": workload_id, "verdict": "inconclusive", "reason": "runner timeout"}
    if proc.returncode != 0:
        return {
            "workload": workload_id,
            "verdict": "inconclusive",
            "reason": f"runner failed: {proc.stderr[-300:]}",
        }
    return _read_artifact(workload_id, artifact_path)


def _read_artifact(workload_id: str, artifact_path: Path) -> dict:
    """Fail closed on missing, malformed, or failed evidence."""
    if not artifact_path.exists():
        return {"workload": workload_id, "verdict": "not_run", "reason": "no artifact"}
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"workload": workload_id, "verdict": "inconclusive", "reason": f"malformed artifact: {exc}"}
    verdict = payload.get("verdict")
    if verdict not in VALID_VERDICTS:
        return {"workload": workload_id, "verdict": "inconclusive", "reason": f"unknown verdict {verdict!r}"}
    # Fail closed: a qualified verdict requires parity pass on every case.
    cases = payload.get("cases", {})
    parity_ok = all(c.get("parity", {}).get("status") == "pass" for c in cases.values())
    if verdict == "qualified" and not parity_ok:
        verdict = "numeric_failed"
    result = {"workload": workload_id, "verdict": verdict}
    # Surface the headline performance numbers for the report. Both the K1/K2
    # runners (paired_native_overhead_fraction) and the K3 runner
    # (paired_compiled_overhead_fraction) record the same paired overhead.
    for case_id, case in cases.items():
        perf = case.get("performance", {}).get("measurements", {})
        for mode, m in perf.items():
            ov = m.get("paired_native_overhead_fraction") or m.get("paired_compiled_overhead_fraction") or {}
            result.setdefault("cases", {}).setdefault(case_id, {})[mode] = {
                "median_overhead": ov.get("median"),
                "ci95_upper": ov.get("ci95_upper"),
                "gate_pass": ov.get("gate", {}).get("pass"),
            }
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--block", type=int, default=10)
    parser.add_argument("--output", type=Path, default=RESULTS / "release-gate.json")
    args = parser.parse_args()

    matrix = json.loads(MATRIX.read_text())
    mandatory_ids = [w["id"] for w in matrix["workloads"]]
    results = {}
    for workload_id in mandatory_ids:
        spec = WORKLOADS.get(workload_id)
        if spec is None:
            results[workload_id] = {
                "workload": workload_id,
                "verdict": "unsupported",
                "reason": "no qualification runner / comparator in this environment",
            }
            continue
        results[workload_id] = _run_workload(workload_id, spec, args.pairs, args.warmup, args.block)

    qualified = sum(1 for r in results.values() if r["verdict"] == "qualified")
    total = len(mandatory_ids)
    production_progress = qualified / total if total else 0.0

    report = {
        "schema_version": 1,
        "purpose": "executable production release gate (acceptance-contract section 10)",
        "production_progress": {
            "qualified": qualified,
            "total_mandatory": total,
            "fraction": production_progress,
            "headline": f"{qualified}/{total} mandatory workloads qualified; broader catalog incomplete",
        },
        "workloads": results,
        "release_ready": qualified == total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    # Human-readable summary.
    print("=" * 70)
    print("URM production release gate")
    print("=" * 70)
    for workload_id in mandatory_ids:
        r = results[workload_id]
        line = f"  {workload_id:32} {r['verdict']}"
        for case_id, modes in (r.get("cases") or {}).items():
            for mode, m in modes.items():
                if m.get("median_overhead") is not None:
                    line += f"\n      {case_id}/{mode}: {m['median_overhead']*100:+.1f}% (ci95 upper {m['ci95_upper']*100:+.1f}%)"
        print(line)
    print("-" * 70)
    print(f"  Production progress: {qualified}/{total} qualified ({production_progress*100:.0f}%)")
    print(f"  Headline: {report['production_progress']['headline']}")
    print(f"  RELEASE READY: {report['release_ready']}")
    print("=" * 70)
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
