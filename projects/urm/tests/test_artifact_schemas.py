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
