from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]


def _load_schema(name: str) -> dict:
    return json.loads((PROJECT_ROOT / "benchmarks" / name).read_text(encoding="utf-8"))


def test_profiling_schema_exists_and_versioned() -> None:
    schema = _load_schema("profiling-schema.json")
    assert schema["properties"]["schema_version"]["const"] == 1
    for field in ("timing", "flops_model", "mfu", "memory_traffic"):
        assert field in schema["properties"], field
        assert field in schema["required"], field
    # Routed reduction accumulates through FP32 CUDA-core ops; the MFU
    # denominator must not silently switch to a tensor-core peak.
    description = json.dumps(schema)
    assert "tensor-core peaks are prohibited" in description


def test_result_schema_accepts_optional_profiling_extension() -> None:
    from jsonschema import validate

    schema = _load_schema("result-schema.json")
    base = {
        "schema_version": 1,
        "case": {"name": "x"},
        "environment": {
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "python": "3",
            "torch": "2",
            "triton": "3",
            "gpu": "g",
        },
        "measurements": [
            {"backend": "b", "median_ms": 1.0, "p95_ms": 2.0, "samples": 1}
        ],
    }
    validate(base, schema)
    extended = {**base, "profiling": {"mfu_pct": 12.5}}
    validate(extended, schema)


def test_device_limits_record_measured_denominators() -> None:
    path = PROJECT_ROOT / "results" / "device-limits.json"
    if not path.exists():
        pytest.skip("device limits not yet measured")
    data = json.loads(path.read_text())
    assert data["bandwidth"]["sustainable_gbps"] > 100
    assert data["fp32_cuda_core"]["tf32_disabled"] is True
    assert data["fp32_cuda_core"]["fp32_cuda_core_tfps_measured"] > 1
    vendor = data["vendor_reference"]
    assert data["bandwidth"]["sustainable_gbps"] != vendor["a10g_spec_mem_gbps"], (
        "bandwidth denominator must be measured, not the marketing number"
    )
