"""CPU-only validation of committed benchmark artifacts against their schemas.

Schema validation must run even when CUDA, Triton, FlashAttention, or FLA is
unavailable, so this module deliberately imports only ``json``/``jsonschema``.
It also recomputes the headline adapter-overhead numbers from the committed
attention artifact so documentation cannot drift from the artifacts.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest
from jsonschema import validate

PROJECT_ROOT = Path(__file__).parents[1]
RESULTS = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))


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
    import tomllib

    grid = tomllib.loads(
        (PROJECT_ROOT / "benchmarks" / "sparse_state_mixer_cases.toml").read_text(
            encoding="utf-8"
        )
    )["case"]
    expected_case_names = {case["name"] for case in grid}
    assert all(set(run["cases"]) == expected_case_names for run in artifact["runs"])
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


def test_committed_sparse_memory_e2e_confirmation_validates() -> None:
    schema = _load(PROJECT_ROOT / "benchmarks" / "sparse-memory-e2e-result-schema.json")
    artifact = _artifact("sparse-memory-e2e/confirmation.json")
    validate(artifact, schema)
    assert artifact["schema_version"] == 2
    assert artifact["confirmation"]["passed"] is True
    assert len(artifact["runs"]) == 3

    import tomllib

    grid = tomllib.loads(
        (PROJECT_ROOT / "benchmarks" / "sparse_memory_e2e_cases.toml").read_text(
            encoding="utf-8"
        )
    )["case"]
    expected = {case["name"] for case in grid}
    assert set(artifact["confirmation"]["cases"]) == expected
    assert all(set(run["cases"]) == expected for run in artifact["runs"])
    assert all(run["provenance"]["dirty_tree"] is False for run in artifact["runs"])
    assert all(
        run["provenance"]["upstream"]["installed_commit"]
        == "183e7df809131b80ad4393741029d0f20fc3640b"
        and run["provenance"]["upstream"]["checkout_dirty"] is False
        for run in artifact["runs"]
    )
    for run in artifact["runs"]:
        assert set(run["methodology"]["levels"]) == {
            "reference",
            "upstream",
            "hybrid",
            "native",
        }
        for row in run["cases"].values():
            assert row["correctness"]["addresses_exact"] is True
            assert row["correctness"]["passed"] is True
            if row["case"]["operation"] == "training":
                assert row["backward_correctness"]["passed"] is True
                assert set(row["backward_correctness"]["gradients"]) == {
                    "write_scores",
                    "read_scores",
                    "initial_memory",
                    "values",
                    "beta",
                    "log_decay",
                }


def test_committed_sparse_memory_e2e_initial_attribution_validates() -> None:
    schema = _load(PROJECT_ROOT / "benchmarks" / "sparse-memory-e2e-result-schema.json")
    artifact = _artifact("sparse-memory-e2e/attribution-initial.json")
    validate(artifact, schema)
    assert artifact["artifact_kind"] == "single_process_attribution"
    assert artifact["provenance"]["dirty_tree"] is False
    assert all(row["correctness"]["passed"] for row in artifact["cases"].values())


def test_committed_sparse_memory_e2e_profile_validates() -> None:
    schema = _load(
        PROJECT_ROOT / "benchmarks" / "sparse-memory-e2e-profile-schema.json"
    )
    artifact = _artifact("sparse-memory-e2e/profile.json")
    validate(artifact, schema)
    assert artifact["provenance"]["dirty_tree"] is False
    assert artifact["provenance"]["upstream"]["installed_commit"] == (
        "183e7df809131b80ad4393741029d0f20fc3640b"
    )
    ranges = set(artifact["profiler"]["nvtx_ranges"])
    assert {
        "sparse_memory_e2e::upstream::route_production",
        "sparse_memory_e2e::native::route_production",
        "sparse_memory_e2e::upstream::state_mixer",
        "sparse_memory_e2e::native::state_mixer",
    } <= ranges


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
def test_committed_solver_artifacts_validate_against_schemas() -> None:
    for schema_name, artifact_name in (
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
def test_committed_unsat_diagnostics_all_map() -> None:
    artifact = _artifact("compiler/solver/unsat-diagnostics.json")
    assert artifact["summary"]["all_unsat"]
    assert artifact["summary"]["all_cores_mapped"]
    assert artifact["summary"]["cases_run"] >= 9
def test_inference_throughput_table_matches_committed_artifacts() -> None:
    """The inference throughput + MFU doc must regenerate exactly.

    The table is a rollup over the committed release-gate artifacts (native and
    upstream wall times per case/dtype/mode); this keeps the serving comparison
    from drifting from the validated measurements.
    """
    import inference_report

    documented = (
        PROJECT_ROOT / "docs" / "validation" / "inference-throughput.md"
    ).read_text(encoding="utf-8")
    regenerated = inference_report.render_markdown(inference_report.build_rows())
    assert documented == regenerated, (
        "docs/validation/inference-throughput.md is out of sync with the "
        "committed artifacts; regenerate it with "
        "`PYTHONPATH=src:benchmarks python benchmarks/inference_report.py`"
    )


def test_alignment_doc_matches_committed_artifacts() -> None:
    """The gradient-alignment + decoding-KL doc must regenerate exactly.

    The gradient-alignment rows come from the committed qualification artifacts;
    the decoding KL divergence is a fixed-seed live measurement. This keeps the
    alignment evidence from drifting from the validated numbers.
    """
    import alignment_report

    documented = (
        PROJECT_ROOT / "docs" / "validation" / "alignment.md"
    ).read_text(encoding="utf-8")
    regenerated = alignment_report.render_markdown(
        alignment_report._gradient_rows(), alignment_report._kl_divergence_rows()
    )
    assert documented == regenerated, (
        "docs/validation/alignment.md is out of sync; regenerate it with "
        "`PYTHONPATH=src:benchmarks python benchmarks/alignment_report.py`"
    )


def _select_cases(data: dict | None, comparison: dict) -> tuple[list[dict], str | None]:
    """Resolve the exact artifact cases a comparison is qualified on.

    The register names cases explicitly (``recipe`` or ``recipes``); there is no
    "all cases" fallback. Missing or malformed evidence returns an error so the
    caller cannot emit a pass claim.
    """
    if not data or not isinstance(data.get("cases"), dict):
        return [], "artifact_missing"
    cases = data["cases"]
    named = comparison.get("recipes")
    if named is None:
        recipe = comparison.get("recipe")
        named = [recipe] if recipe is not None else []
    if not named:
        return [], "no_cases_named"
    selected: list[dict] = []
    for name in named:
        case = cases.get(name)
        if not isinstance(case, dict):
            return [], f"case_missing:{name}"
        selected.append(case)
    return selected, None


def _aggregate_status(statuses: list[str | None]) -> str:
    """Combine per-case parity verdicts into one order-independent status."""
    if any(status == "fail" for status in statuses):
        return "fail"
    if statuses and all(status == "pass" for status in statuses):
        return "pass"
    return "incomplete"


def _aggregate_case(data: dict | None, comparison: dict) -> tuple[str, float | None, float | None]:
    """Return (parity_status, worst forward overhead, worst fwd+bwd overhead)."""
    selected, error = _select_cases(data, comparison)
    if error is not None:
        return "incomplete", None, None
    statuses: list[str | None] = []
    forwards: list[float] = []
    forward_backwards: list[float] = []
    for case in selected:
        statuses.append(case.get("parity", {}).get("status"))
        measurements = case.get("performance", {}).get("measurements", {})
        forward = (
            measurements.get("forward", {}).get("paired_compiled_overhead_fraction", {})
            or {}
        ).get("median")
        forward_backward = (
            measurements.get("forward_backward", {}).get("paired_compiled_overhead_fraction", {})
            or {}
        ).get("median")
        if forward is not None:
            forwards.append(forward)
        if forward_backward is not None:
            forward_backwards.append(forward_backward)
    return (
        _aggregate_status(statuses),
        max(forwards) if forwards else None,
        max(forward_backwards) if forward_backwards else None,
    )


def _comparison_artifact(cases: dict[str, str]) -> dict:
    """Build a minimal artifact whose cases carry only a parity status."""
    return {
        "cases": {
            name: {"parity": {"status": status}, "performance": {"measurements": {}}}
            for name, status in cases.items()
        }
    }


def test_aggregate_case_never_hides_a_failure() -> None:
    """A failed case must fail the rollup regardless of case ordering."""
    passing = _comparison_artifact({"a": "pass", "b": "pass"})
    comparison = {"recipes": ["a", "b"]}
    assert _aggregate_case(passing, comparison)[0] == "pass"

    # Same two cases, opposite insertion order: the verdict must not change.
    fail_last = _comparison_artifact({"a": "pass", "b": "fail"})
    fail_first = _comparison_artifact({"b": "fail", "a": "pass"})
    assert _aggregate_case(fail_last, comparison)[0] == "fail"
    assert _aggregate_case(fail_first, comparison)[0] == "fail"


def test_aggregate_case_marks_incomplete_evidence() -> None:
    """Missing, malformed, or non-pass evidence must not surface as a pass."""
    comparison = {"recipes": ["a", "b"]}
    # A case with no recorded status is incomplete, not a pass.
    partial = _comparison_artifact({"a": "pass", "b": None})
    assert _aggregate_case(partial, comparison)[0] == "incomplete"
    # A named case absent from the artifact is incomplete.
    missing = _comparison_artifact({"a": "pass"})
    assert _aggregate_case(missing, comparison)[0] == "incomplete"
    # No artifact at all is incomplete.
    assert _aggregate_case(None, comparison)[0] == "incomplete"


def test_aggregate_case_uses_only_named_cases() -> None:
    """Unrelated cases in the artifact must not be substituted in."""
    # The artifact carries an extra failing case the comparison did not name;
    # the rollup is over the named cases only and stays a pass.
    artifact = _comparison_artifact({"a": "pass", "unrelated": "fail"})
    assert _aggregate_case(artifact, {"recipe": "a"})[0] == "pass"
    # A comparison that names no case cannot claim a pass.
    assert (
        _aggregate_case(artifact, {"recipe": None, "recipes": None})[0]
        == "incomplete"
    )
def test_production_matrix_validates_against_schema() -> None:
    """The frozen production replacement matrix must stay well-formed."""
    schema = _load(PROJECT_ROOT / "benchmarks" / "production-matrix-schema.json")
    validate(_load(PROJECT_ROOT / "benchmarks" / "production-matrix.json"), schema)


def test_production_matrix_spans_all_three_families() -> None:
    """The mandatory envelope must cover K1, K2, and K3 with a comparator each."""
    matrix = _load(PROJECT_ROOT / "benchmarks" / "production-matrix.json")
    families = {workload["family"] for workload in matrix["workloads"]}
    assert families == {"K1", "K2", "K3"}
    for workload in matrix["workloads"]:
        comparator = workload["comparator"]
        # A competitive comparator must be frozen with an exact revision.
        assert comparator["revision"], workload["id"]
        assert comparator["callable"], workload["id"]
        # Every workload must declare at least one training and one serving mode
        # where the family supports it, and freeze a performance budget.
        assert workload["performance_budget"], workload["id"]
        assert workload["cases"], workload["id"]


def test_every_register_comparison_names_real_cases() -> None:
    """The register must name explicit cases that exist in each artifact.

    This pins the explicit artifact-case mapping: no comparison may rely on an
    implicit "all cases" fallback or point at a case that does not exist.
    """
    register = _load(PROJECT_ROOT / "benchmarks" / "architecture-coverage.json")
    for architecture in register["architectures"]:
        comparison = architecture.get("kernel_upstream_comparison")
        if not comparison:
            continue
        artifact = comparison.get("artifact")
        data = _load(PROJECT_ROOT / artifact) if artifact else None
        selected, error = _select_cases(data, comparison)
        assert error is None, (
            f"{architecture['architecture']}: comparison evidence error {error}; "
            "name explicit recipe/recipes that exist in the artifact"
        )
        assert selected, f"{architecture['architecture']}: no cases selected"


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
