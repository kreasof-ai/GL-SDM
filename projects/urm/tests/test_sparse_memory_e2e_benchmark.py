"""Dependency-light protocol tests for Sparse Memory E2E confirmation."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import sparse_memory_e2e as benchmark


def test_frozen_e2e_grid_has_complete_unique_coverage() -> None:
    cases = benchmark.load_cases()
    assert len(cases) == 19
    assert len({case["name"] for case in cases}) == 19
    assert {case["dtype"] for case in cases} == {"float32", "bfloat16"}
    assert {case["operation"] for case in cases} == {
        "read_only",
        "update",
        "training",
    }
    assert {case["classification"] for case in cases} == {
        "host_bound",
        "substantial",
    }
    assert any(
        case["writes"] == 64 and case["operation"] == "training" for case in cases
    )
    assert any(case["dim"] == 95 for case in cases)
    assert any(case["slots"] == 65536 for case in cases)


@pytest.mark.parametrize("processes", [1, 2, 4])
def test_e2e_completion_requires_exactly_three_processes(processes) -> None:
    args = Namespace(
        confirmation_processes=processes,
        case=[],
        samples=3,
        warmup=1,
        output=Path("unused.json"),
    )
    with pytest.raises(ValueError, match="exactly three"):
        benchmark._run_confirmation(args)


def test_e2e_completion_rejects_case_filtering() -> None:
    args = Namespace(
        confirmation_processes=3,
        case=["decode_fp32"],
        samples=3,
        warmup=1,
        output=Path("unused.json"),
    )
    with pytest.raises(ValueError, match="cannot filter"):
        benchmark._run_confirmation(args)


def test_e2e_schema_is_strict_version_one() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "benchmarks"
            / "sparse-memory-e2e-result-schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["$defs"]["singleRun"]["additionalProperties"] is False
    assert schema["$defs"]["caseResult"]["additionalProperties"] is False
    assert schema["$defs"]["provenance"]["additionalProperties"] is False
    assert schema["$defs"]["upstream"]["additionalProperties"] is False
