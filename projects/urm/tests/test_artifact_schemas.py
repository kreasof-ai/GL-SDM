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


def test_committed_sparse_delta_memory_artifact_validates_against_schema() -> None:
    schema = _load(
        PROJECT_ROOT / "benchmarks" / "sparse-delta-memory-result-schema.json"
    )
    artifact = _artifact("sparse-delta-memory/benchmark.json")
    validate(artifact, schema)
    assert artifact["schema_version"] == 2
    assert set(artifact["cases"]) >= {
        "smoke_read_only",
        "prefill_batched",
        "decode_cached",
        "write_update",
        "collision_heavy",
        "training_prefill_forward_only",
        "memory_capacity",
    }
    output_atol = artifact["methodology"]["tolerances"]["output_atol"]
    state_atol = artifact["methodology"]["tolerances"]["state_atol"]
    assert artifact["methodology"]["training_timing"].startswith("forward-only")
    backward = artifact["backward_correctness"]
    assert backward["passed"] is True
    assert backward["scope"].startswith("compiler-visible write_scores/read_scores")
    assert backward["measurement_scope"] == "untimed_correctness_only"
    assert set(backward["dtypes"]) == {"float32", "bfloat16"}
    for dtype, report in backward["dtypes"].items():
        assert report["dtype"] == dtype
        assert report["passed"] is True
        assert report["product_key_tie_free"] is True
        assert report["input_generation"]["path_inputs"] == "independent clones"
        assert report["addresses"]["passed"] is True
        assert report["route_weights"]["passed"] is True
        assert set(report["gradients"]) == {
            "write_scores",
            "read_scores",
            "initial_memory",
            "values",
            "beta",
            "log_decay",
        }
    decode_cache = artifact["cases"]["decode_cached"]["cache_persistence"]
    assert decode_cache["status"] == "measured"
    assert decode_cache["storage_pointer_preserved"] is True
    assert (
        decode_cache["adapter_sequence_length"] == decode_cache["upstream_invocations"]
    )
    for case in artifact["cases"].values():
        assert case["correctness"]["addresses_exact"] is True
        assert case["correctness"]["direct_adapter_output_max_abs"] <= output_atol
        if "direct_adapter_state_max_abs" in case["correctness"]:
            assert case["correctness"]["direct_adapter_state_max_abs"] <= state_atol
        assert case["call_identity"]["identical"] is True
        assert (
            case["call_identity"]["address_direct"]
            == case["call_identity"]["address_adapter_below_dispatch"]
        )
        assert (
            case["call_identity"]["direct"]
            == case["call_identity"]["adapter_below_dispatch"]
        )
        paired = case["paired_performance"]
        assert len(paired["pair_order"]) == paired["pairs"]
        for sample_name in (
            "direct_wall",
            "adapter_wall",
            "direct_device",
            "adapter_device",
            "paired_wall_overhead_ms",
            "paired_device_overhead_ms",
        ):
            assert (
                len(paired[sample_name]["raw_samples_ms"])
                == paired[sample_name]["sample_count"]
            )
        for sample_name in (
            "paired_wall_overhead_fraction",
            "paired_device_overhead_fraction",
        ):
            assert (
                len(paired[sample_name]["raw_samples"])
                == paired[sample_name]["sample_count"]
            )
    interpretation = artifact["performance_interpretation"]
    assert interpretation["mature_kernel_gate"]["claimed"] is False
    assert set(interpretation["substantial_workloads"]["cases"]) == {
        "prefill_batched",
        "write_update",
        "collision_heavy",
        "training_prefill_forward_only",
        "memory_capacity",
    }
    for name in ("smoke_read_only", "decode_cached"):
        observation = interpretation["tiny_host_bound_workloads"][name]
        paired = artifact["cases"][name]["paired_performance"]
        assert observation["absolute_microseconds"] == pytest.approx(
            paired["paired_device_overhead_ms"]["median_ms"] * 1000
        )
        assert observation["percent"] == pytest.approx(
            paired["paired_device_overhead_fraction"]["median"] * 100
        )


def test_committed_sparse_state_mixer_confirmation_validates() -> None:
    schema = _load(
        PROJECT_ROOT / "benchmarks" / "sparse-state-mixer-result-schema.json"
    )
    artifact = _artifact("sparse-state-mixer/confirmation.json")
    validate(artifact, schema)
    assert artifact["schema_version"] == 1
    assert artifact["confirmation"]["passed"] is True
    assert len(artifact["runs"]) == 3
    assert {run["provenance"]["git_revision"] for run in artifact["runs"]} == {
        artifact["provenance"]["git_revision"]
    }
    assert all(run["provenance"]["dirty_tree"] is False for run in artifact["runs"])
    for run in artifact["runs"]:
        assert run["provenance"]["upstream"]["installed_commit"] == (
            "183e7df809131b80ad4393741029d0f20fc3640b"
        )
        for row in run["cases"].values():
            for phase in ("forward", "backward"):
                measured = row[phase]
                if measured.get("status") == "not_applicable":
                    continue
                count = measured["upstream_device"]["count"]
                assert len(measured["orders"]) == count
                for path in (
                    "upstream_wall",
                    "native_wall",
                    "upstream_device",
                    "native_device",
                ):
                    assert len(measured[path]["raw_ms"]) == count
                assert len(measured["paired_device_ratio"]["raw"]) == count
            assert row["correctness"]["passed"] is True
            if row["case"]["operation"] == "training":
                assert row["backward_correctness"]["passed"] is True
                assert set(row["backward_correctness"]["gradients"]) == {
                    "initial_memory",
                    "write_weights",
                    "values",
                    "beta",
                    "log_decay",
                    "read_weights",
                }


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
    sparse = next(
        row
        for row in artifact["rows"]
        if row["architecture_params"]["preset"] == "sparse_delta_memory"
    )
    assert sparse["valid_semantic_programs"] == 2
    assert {
        "facebook_sparse_delta_memory_183e7df_external_adapter",
        "urm_native_sparse_state_mixer_v0",
    } <= set(sparse["anchors_selected"])
    assert sparse["anchor_dispatch_counts"] == {
        "facebook_sparse_delta_memory_183e7df_external_adapter": 1,
        "urm_native_sparse_state_mixer_v0": 1,
    }
    assert summary["native_lowering_rate"] + summary["upstream_adapter_rate"] <= 1


def test_committed_solver_artifacts_validate_against_schemas() -> None:
    for schema_name, artifact_name in (
        (
            "routed-epilogue-selection-schema.json",
            "compiler/solver/routed-epilogue-selection.json",
        ),
        (
            "routed-epilogue-stability-schema.json",
            "compiler/solver/routed-epilogue-stability.json",
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
    from measurement import quantile

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
        raw = sample.get("raw_samples_ms")
        assert raw is not None and len(raw) == sample["sample_count"]
        # Raw samples recompute the serialized summary statistics exactly
        recomputed_median = round(quantile(raw, 0.5), 4)
        recomputed_p95 = round(quantile(raw, 0.95), 4)
        recomputed_min = round(min(raw), 4)
        assert sample["median_ms"] == pytest.approx(recomputed_median, abs=1e-4)
        assert sample["p95_ms"] == pytest.approx(recomputed_p95, abs=1e-4)
        assert sample["min_ms"] == pytest.approx(recomputed_min, abs=1e-4)
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


def test_committed_stability_artifact_semantic_facts() -> None:
    """Pin committed GPU multi-run stability facts semantically."""
    from measurement import quantile

    artifact = _artifact("compiler/solver/routed-epilogue-stability.json")
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

    runs = artifact["runs"]
    assert len(runs) >= 5, f"expected at least 5 independent runs, got {len(runs)}"

    for run in runs:
        assert run["measured_points"] >= 45
        assert run["best_point"] is not None
        assert run["best_median_ms"] > 0
        assert run["solver_selected"] is not None
        assert run["solver_median_ms"] > 0
        assert isinstance(run["solver_regret_pct"], (int, float))
        assert run["heuristic_selected"] is not None
        assert run["heuristic_median_ms"] > 0
        assert isinstance(run["heuristic_regret_pct"], (int, float))
        # Environment conditions snapshots
        for tag in ("operating_conditions_before", "operating_conditions_after"):
            cond = run.get(tag)
            assert cond is not None, f"missing {tag} snapshot"
            assert cond.get("gpu_name") or cond.get("unavailable_reason")

    stats = artifact["cross_run_statistics"]
    assert stats["schedules_measured_in_all_runs"] >= 45
    assert len(stats["pairwise_spearman_correlations"]) >= 10
    assert (
        stats["min_spearman_correlation"]
        <= stats["median_spearman_correlation"]
        <= stats["max_spearman_correlation"]
    )
    assert len(stats["pairwise_top5_jaccard"]) >= 10
    assert stats["median_abs_drift_pct"] >= 0.0

    winning_set = artifact["winning_set"]
    assert winning_set["representative_best"] is not None
    assert winning_set["representative_best_median_ms"] > 0
    assert winning_set["exploratory_candidate_set"]
    assert isinstance(winning_set["is_winner_stable_across_runs"], bool)

    robust_regret = artifact["robust_regret"]
    for agent in ("solver", "heuristic"):
        agent_regret = robust_regret[agent]
        assert len(agent_regret["per_run_regret_pct"]) == len(runs)
        assert agent_regret["ci95_lower_pct"] <= agent_regret["ci95_upper_pct"]
        classification = agent_regret["classification"]
        assert classification in {"pass", "fail", "inconclusive"}
        if agent_regret["ci95_upper_pct"] <= agent_regret["target_pct"]:
            assert classification == "pass"
        elif agent_regret["ci95_lower_pct"] > agent_regret["target_pct"]:
            assert classification == "fail"
        else:
            assert classification == "inconclusive"

    per_sched = artifact["per_schedule_stability"]
    assert len(per_sched) >= 45
    for sched_entry in per_sched:
        assert sched_entry["schedule"] is not None
        assert len(sched_entry["rank_per_run"]) == len(runs)
        assert len(sched_entry["raw_samples_by_run"]) == len(runs)
        assert len(sched_entry["per_run_medians_ms"]) == len(runs)
        for run_idx, raw_run in enumerate(sched_entry["raw_samples_by_run"]):
            recomputed = round(quantile(raw_run, 0.5), 4)
            assert sched_entry["per_run_medians_ms"][run_idx] == pytest.approx(
                recomputed, abs=1e-4
            )
        recomputed_med_of_meds = round(
            quantile(sched_entry["per_run_medians_ms"], 0.5), 4
        )
        assert sched_entry["median_of_medians_ms"] == pytest.approx(
            recomputed_med_of_meds, abs=1e-4
        )


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
