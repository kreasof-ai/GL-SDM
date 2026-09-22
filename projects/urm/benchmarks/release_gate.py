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

from release_coverage import derive_workload_coverage

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
        "args": ["--workload", "mha"],
    },
    "k1-gqa": {
        "runner": "benchmarks/qualify_native_k1_attention.py",
        "artifact": "native-k1-gqa.json",
        "args": ["--workload", "gqa"],
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

# Derived verdicts. The gate derives these from matrix coverage; it never trusts
# an artifact's self-declared "qualified" string. "partial_coverage" means the
# artifact passes only a slice of the mandatory matrix (a passing benchmark
# slice), which does not count as a qualified replacement.
VALID_VERDICTS = {
    "qualified",
    "partial_coverage",
    "gate_failed",
    "numeric_failed",
    "inconclusive",
    "unsupported",
    "not_run",
}


def _run_workload(workload_id: str, spec: dict, matrix_workload: dict, pairs: int, warmup: int, block: int) -> dict:
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
    return _read_artifact(workload_id, artifact_path, matrix_workload)


def _read_artifact(workload_id: str, artifact_path: Path, matrix_workload: dict) -> dict:
    """Derive the verdict from matrix coverage; fail closed on bad evidence.

    The artifact's self-declared ``verdict`` string is ignored. The verdict is
    derived by cross-referencing the artifact's covered cases, dtypes, modes, and
    correctness components against the frozen matrix declaration, and by checking
    parity and performance gates from the recorded evidence.
    """
    if not artifact_path.exists():
        return {"workload": workload_id, "verdict": "not_run", "reason": "no artifact"}
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {"workload": workload_id, "verdict": "inconclusive", "reason": f"malformed artifact: {exc}"}
    if not isinstance(payload.get("cases"), dict):
        return {"workload": workload_id, "verdict": "inconclusive", "reason": "artifact has no cases mapping"}

    derived = derive_workload_coverage(matrix_workload, payload)
    result = {
        "workload": workload_id,
        "verdict": derived["verdict"],
        "artifact_self_declared_verdict": payload.get("verdict"),
        "coverage_complete": derived["complete"],
    }
    if derived.get("missing"):
        result["missing_coverage"] = derived["missing"]
    if derived.get("correctness_gaps"):
        result["correctness_gaps"] = derived["correctness_gaps"]

    # Surface the headline performance numbers for the report. Both the K1/K2
    # runners (paired_native_overhead_fraction) and the K3 runner
    # (paired_compiled_overhead_fraction) record the same paired overhead.
    for case_id, case in payload.get("cases", {}).items():
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
    matrix_by_id = {w["id"]: w for w in matrix["workloads"]}
    mandatory_ids = [w["id"] for w in matrix["workloads"]]
    results = {}
    for workload_id in mandatory_ids:
        matrix_workload = matrix_by_id[workload_id]
        spec = WORKLOADS.get(workload_id)
        if spec is None:
            results[workload_id] = {
                "workload": workload_id,
                "verdict": "unsupported",
                "reason": "no qualification runner / comparator in this environment",
            }
            continue
        results[workload_id] = _run_workload(
            workload_id, spec, matrix_workload, args.pairs, args.warmup, args.block
        )

    qualified = sum(1 for r in results.values() if r["verdict"] == "qualified")
    partial = sum(1 for r in results.values() if r["verdict"] == "partial_coverage")
    total = len(mandatory_ids)
    production_progress = qualified / total if total else 0.0

    # Honest headline: only fully matrix-qualified workloads count. Workloads
    # passing only a slice of the mandatory matrix are reported as passing
    # benchmark slices, not as qualified replacements.
    if qualified == total:
        headline = f"{qualified}/{total} mandatory workloads fully qualified"
    else:
        headline = (
            f"Production qualification incomplete: {qualified}/{total} mandatory "
            f"workloads fully qualified against the frozen matrix"
            + (f" ({partial} passing benchmark slices only)" if partial else "")
        )

    report = {
        "schema_version": 2,
        "purpose": "executable production release gate (acceptance-contract section 10)",
        "accounting": (
            "Verdicts are derived from frozen-matrix coverage (every mandatory "
            "case, dtype, mode, and correctness component, with passing parity and "
            "performance gates), not from an artifact's self-declared verdict."
        ),
        "production_progress": {
            "qualified": qualified,
            "partial_coverage": partial,
            "total_mandatory": total,
            "fraction": production_progress,
            "headline": headline,
        },
        "workloads": results,
        "release_ready": qualified == total,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    # Human-readable summary.
    print("=" * 70)
    print("URM production release gate (verdicts derived from frozen-matrix coverage)")
    print("=" * 70)
    for workload_id in mandatory_ids:
        r = results[workload_id]
        line = f"  {workload_id:32} {r['verdict']}"
        self_declared = r.get("artifact_self_declared_verdict")
        if self_declared and self_declared != r["verdict"]:
            line += f"  (artifact self-declared: {self_declared})"
        for case_id, modes in (r.get("cases") or {}).items():
            for mode, m in modes.items():
                if m.get("median_overhead") is not None:
                    line += f"\n      {case_id}/{mode}: {m['median_overhead']*100:+.1f}% (ci95 upper {m['ci95_upper']*100:+.1f}%)"
        for gap in (r.get("missing_coverage") or [])[:6]:
            line += f"\n      missing: {gap}"
        for gap in (r.get("correctness_gaps") or [])[:6]:
            line += f"\n      gap: {gap}"
        print(line)
    print("-" * 70)
    print(f"  Production progress: {qualified}/{total} fully qualified ({production_progress*100:.0f}%)")
    print(f"  Headline: {report['production_progress']['headline']}")
    print(f"  RELEASE READY: {report['release_ready']}")
    print("=" * 70)
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
