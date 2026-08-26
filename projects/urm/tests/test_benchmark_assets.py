from __future__ import annotations

import json
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_routed_reduction_cases_are_named_and_within_v1_contract() -> None:
    with (PROJECT_ROOT / "benchmarks" / "cases.toml").open("rb") as handle:
        cases = tomllib.load(handle)["routed_reduction"]["case"]

    names = [case["name"] for case in cases]
    assert names
    assert len(names) == len(set(names))
    for case in cases:
        assert 0 < case["route_width"] <= 64
        assert case["route_width"] <= case["sources"]
        assert case["dtype"] in {"float16", "bfloat16", "float32"}
        assert case["distribution"] in {"uniform", "skewed", "recurrent_reuse"}


def test_result_schema_is_valid_json_and_versioned() -> None:
    schema = json.loads(
        (PROJECT_ROOT / "benchmarks" / "result-schema.json").read_text(encoding="utf-8")
    )
    assert schema["properties"]["schema_version"]["const"] == 1
    assert "measurements" in schema["required"]
