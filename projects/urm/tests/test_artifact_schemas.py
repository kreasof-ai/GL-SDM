"""CPU-only validation of committed benchmark artifacts against their schemas.

Schema validation must run even when CUDA, Triton, FlashAttention, or FLA is
unavailable, so this module deliberately imports only ``json``/``jsonschema``.
It also recomputes the headline adapter-overhead numbers from the committed
attention artifact so documentation cannot drift from the artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import validate

PROJECT_ROOT = Path(__file__).parents[1]
RESULTS = PROJECT_ROOT / "results"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"artifact not committed yet: {path.relative_to(PROJECT_ROOT)}")
    return _load(path)


def test_committed_attention_artifact_validates_against_schema() -> None:
    schema = _load(PROJECT_ROOT / "benchmarks" / "attention-result-schema.json")
    validate(_artifact("attention/dense-causal.json"), schema)


def test_committed_gated_delta_rule_artifact_validates_against_schema() -> None:
    schema = _load(PROJECT_ROOT / "benchmarks" / "gated-delta-rule-result-schema.json")
    validate(_artifact("fla-gated-delta-rule/benchmark.json"), schema)


def _steady_state_overhead_rows(artifact: dict) -> list[dict]:
    """Flatten paired adapter-overhead statistics from the attention artifact."""
    min_seq = artifact["methodology"]["steady_state_min_seq"]
    rows: list[dict] = []
    for case in artifact["cases"].values():
        sequence = case["case"]["sequence"]
        for direct_impl, modes in case["adapter_overhead"].items():
            for mode, stats in modes.items():
                fraction = stats["wall_fraction"]
                rows.append(
                    {
                        "sequence": sequence,
                        "direct_impl": direct_impl,
                        "mode": mode,
                        "median": fraction["median"],
                        "ci_upper": fraction["bootstrap_ci95_median"]["upper"],
                        "gate_pass": stats["gate"]["pass"],
                    }
                )
    return rows, min_seq


def test_committed_epilogue_artifact_validates_against_schema() -> None:
    schema = _load(PROJECT_ROOT / "benchmarks" / "compiler-epilogue-schema.json")
    validate(_artifact("compiler/routed-scale-epilogue/benchmark.json"), schema)
    artifact = _artifact("compiler/routed-scale-epilogue/benchmark.json")
    # Correctness must be inside the declared envelope for every dtype,
    # with assert_close semantics: gap <= atol + rtol*|reference|.
    for dtype, stats in artifact["correctness_by_dtype"].items():
        allowed = stats["atol"] + stats["rtol"] * stats["reference_max_abs"]
        assert stats["max_abs_error_vs_eager_reference"] <= allowed, dtype
    # Both plans and both regimes must be present, host-bound separated.
    names = {case["case"]["name"] for case in artifact["forward_cases"]}
    assert {"decode_hostbound", "prefill_gpu_bound"} <= names


def test_committed_compilation_matrix_validates_against_schema() -> None:
    schema = _load(PROJECT_ROOT / "benchmarks" / "compilation-matrix-schema.json")
    artifact = _artifact("compiler/compilation-matrix.json")
    validate(artifact, schema)
    summary = artifact["summary"]
    assert summary["escape_hatch_count"] == 0
    assert summary["presets_evaluated"] >= 15
    # Architecture and schedule namespaces stay distinguishable per row.
    for row in artifact["rows"]:
        assert set(row["architecture_params"]) >= {"preset", "routing"}
        assert isinstance(row["schedule_params"], dict)
    # Honest coverage metrics must be present and internally consistent:
    # the routing skeleton rate never implies full-architecture coverage.
    assert (
        summary["full_architecture_compile_rate"]
        <= summary["routing_skeleton_compile_rate"]
    )


def test_committed_solver_artifacts_validate_against_schemas() -> None:
    for schema_name, artifact_name in (
        (
            "routed-epilogue-selection-schema.json",
            "compiler/solver/routed-epilogue-selection.json",
        ),
        ("placement-selection-schema.json", "compiler/solver/placement-selection.json"),
        ("unsat-diagnostics-schema.json", "compiler/solver/unsat-diagnostics.json"),
    ):
        schema = _load(PROJECT_ROOT / "benchmarks" / schema_name)
        artifact = _artifact(artifact_name)
        validate(artifact, schema)
        # Provenance is mandatory and complete in every solver artifact.
        provenance = artifact["provenance"]
        for field in (
            "git_revision",
            "dirty_tree",
            "benchmark_command",
            "config_hash",
            "solver_version",
            "constraint_model_hash",
        ):
            assert field in provenance, (artifact_name, field)


def test_committed_epilogue_selection_semantic_facts() -> None:
    """Pin committed GPU schedule selection evidence semantically."""
    artifact = _artifact("compiler/solver/routed-epilogue-selection.json")
    prov = artifact["provenance"]
    assert prov["dirty_tree"] is False
    for env_field in (
        "python",
        "pytorch",
        "triton",
        "cuda",
        "driver",
        "gpu",
        "solver_version",
    ):
        assert prov.get(env_field) is not None, (
            f"provenance {env_field} must be non-null"
        )

    decision = artifact["schedule_decision"]
    assert decision["compile_status"] == "succeeded"
    assert decision["schedule"] == decision["launch_config"]

    attempts = decision["attempts"]
    assert attempts, "must have at least one attempt"
    selected_attempt = attempts[-1]
    assert selected_attempt["compile_status"] == "succeeded"

    expected_kernels = {"forward", "grad_weights", "grad_values", "grad_row_scale"}
    for kres_container in (
        decision.get("kernel_resources"),
        selected_attempt.get("kernel_resources"),
    ):
        assert kres_container is not None
        assert set(kres_container.keys()) == expected_kernels
        for tag, record in kres_container.items():
            assert record.get("kernel_name"), (
                f"resource entry for {tag} missing kernel_name"
            )
            regs = record.get("registers_per_thread")
            smem = record.get("shared_mem_bytes")
            unavail = record.get("unavailable_reason")
            assert (regs is not None and smem is not None) or (unavail is not None), (
                f"resource {tag} must have values or an unavailable reason"
            )

    kres = decision["kernel_resources"]
    known_regs = [
        k["registers_per_thread"]
        for k in kres.values()
        if k.get("registers_per_thread") is not None
    ]
    if known_regs:
        assert decision["registers_per_thread"] == max(known_regs)
    known_smem = [
        k["shared_mem_bytes"]
        for k in kres.values()
        if k.get("shared_mem_bytes") is not None
    ]
    if known_smem:
        assert decision["shared_mem_bytes"] == max(known_smem)

    legality = artifact["legality"]
    assert legality["exact_set_agreement"] is True
    assert legality["reference_matches_model_sweep"] is True
    assert legality["agreement"] is True
    assert legality["legality_accuracy"] == pytest.approx(1.0)
    assert legality["solver_assignments_checked"] >= 1

    empirical = artifact["empirical"]
    assert empirical is not None
    assert empirical["measured_points"] > 0
    for sample in empirical["samples"]:
        assert sample["sample_count"] > 0
        assert sample["min_ms"] <= sample["median_ms"] <= sample["p95_ms"]

    overhead = artifact["dispatch_overhead"]
    assert overhead is not None
    assert overhead["direct_median_ms"] > 0
    assert overhead["compiler_driven_median_ms"] > 0
    assert isinstance(overhead["overhead_pct"], (int, float))
    assert isinstance(overhead["gate_pass"], bool)
    assert isinstance(overhead["delta_mean_ms"], (int, float))
    assert isinstance(overhead["delta_median_ms"], (int, float))
    assert isinstance(overhead["delta_std_ms"], (int, float))
    assert overhead["paired_samples_count"] > 0
    assert overhead["batch_launches_per_sample"] > 0


def test_committed_compilation_matrix_semantic_facts() -> None:
    """Pin committed GPU compilation matrix evidence semantically."""
    artifact = _artifact("compiler/compilation-matrix.json")
    prov = artifact["provenance"]
    assert prov["probe_mode"] == "required"
    assert prov["probing_active"] is True
    assert prov["dirty_tree"] is False
    for env_field in (
        "python",
        "pytorch",
        "triton",
        "cuda",
        "driver",
        "gpu",
        "solver_version",
    ):
        assert prov.get(env_field) is not None, (
            f"provenance {env_field} must be non-null"
        )

    summary = artifact["summary"]
    assert isinstance(summary["compile_probe_failures"], int)
    assert isinstance(summary["nogoods_added"], int)

    expected_kernels = {"forward", "grad_weights", "grad_values", "grad_row_scale"}
    for row in artifact["rows"]:
        decisions = row.get("schedule_decisions", [])
        for decision in decisions:
            if decision is None:
                continue
            assert decision["compile_status"] == "succeeded"
            for attempt in decision.get("attempts", []):
                assert attempt["compile_status"] == "succeeded"
            kres = decision.get("kernel_resources")
            assert kres is not None
            assert set(kres.keys()) == expected_kernels
            for record in kres.values():
                assert record.get("kernel_name")


def test_cpu_compilation_matrix_with_probe_off(tmp_path: Path) -> None:
    """Validate that --probe off generates a valid CPU-safe matrix without claiming GPU facts."""
    import subprocess
    import sys

    output_file = tmp_path / "matrix-off.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "benchmarks" / "compilation_matrix.py"),
            "--probe",
            "off",
            "--output",
            str(output_file),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0

    matrix = json.loads(output_file.read_text(encoding="utf-8"))
    schema = _load(PROJECT_ROOT / "benchmarks" / "compilation-matrix-schema.json")
    validate(matrix, schema)

    prov = matrix["provenance"]
    assert prov["probe_mode"] == "off"
    assert prov["probing_active"] is False
    assert prov["pytorch"] is None
    assert prov["triton"] is None
    assert prov["cuda"] is None
    assert prov["driver"] is None
    assert prov["gpu"] is None

    summary = matrix["summary"]
    assert summary["compile_probe_failures"] is None
    assert summary["nogoods_added"] is None

    for row in matrix["rows"]:
        for decision in row.get("schedule_decisions", []):
            if decision is not None:
                assert decision["compile_status"] == "not_probed"
                assert decision["registers_per_thread"] is None
                assert decision["shared_mem_bytes"] is None
                assert decision["kernel_resources"] is None


def test_committed_unsat_diagnostics_all_map() -> None:
    artifact = _artifact("compiler/solver/unsat-diagnostics.json")
    assert artifact["summary"]["all_unsat"]
    assert artifact["summary"]["all_cores_mapped"]
    assert artifact["summary"]["cases_run"] >= 9


def test_committed_epilogue_selection_agrees_with_exhaustive() -> None:
    artifact = _artifact("compiler/solver/routed-epilogue-selection.json")
    legality = artifact["legality"]
    assert legality["agreement"] is True
    assert legality["legality_accuracy"] == pytest.approx(1.0)
    z3 = artifact["z3_selection"]
    assert z3["status"] == "sat"
    assert z3["verified"] is True
    assert z3["verification_failures"] == []


def test_attention_headline_overhead_matches_documented_values() -> None:
    """Docs quote artifact-derived numbers; this test pins them together."""
    artifact = _artifact("attention/dense-causal.json")
    rows, min_seq = _steady_state_overhead_rows(artifact)
    if not rows:
        pytest.skip("no attention overhead rows committed")
    steady = [row for row in rows if row["sequence"] >= min_seq]

    # Every steady-state case must pass the <=5% median overhead gate.
    assert all(row["gate_pass"] for row in steady)

    fa_steady = [row for row in steady if row["direct_impl"] == "flash_attn"]
    worst_median = max(row["median"] for row in fa_steady)
    worst_ci_upper = max(row["ci_upper"] for row in fa_steady)

    # Documented headline values (README.md, triton-optimization-report.md):
    # "approximately +2.32% median" and "approximately +4.53% CI upper bound".
    assert worst_median == pytest.approx(0.0232, abs=5e-4), (
        f"worst steady-state FA median is {worst_median:.4f}; update the "
        "documented headline value to match the artifact"
    )
    assert worst_ci_upper == pytest.approx(0.0453, abs=5e-4), (
        f"worst steady-state FA CI upper bound is {worst_ci_upper:.4f}; update "
        "the documented headline value to match the artifact"
    )
