from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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


def test_compilation_matrix_probe_off_does_not_import_torch_or_triton(
    tmp_path: Path,
) -> None:
    """The --probe off path must never import torch or triton."""
    code = (
        "import sys\n"
        "import benchmarks.compilation_matrix as cm\n"
        "assert 'torch' not in sys.modules, f'torch in sys.modules: {sys.modules.get(\"torch\")}'\n"
        "assert 'triton' not in sys.modules, f'triton in sys.modules: {sys.modules.get(\"triton\")}'\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    assert res.returncode == 0, res.stderr + "\n" + res.stdout


def test_compilation_matrix_probe_required_fails_when_cuda_unavailable(
    tmp_path: Path,
) -> None:
    """Probe mode 'required' fails with a concise diagnostic when CUDA is unavailable."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    out_file = tmp_path / "matrix-fail.json"
    res = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "benchmarks" / "compilation_matrix.py"),
            "--probe",
            "required",
            "--output",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    assert res.returncode != 0
    assert "Probe mode 'required' failed" in res.stderr
    assert "CUDA is unavailable" in res.stderr


def test_compilation_matrix_probe_auto_falls_back_when_cuda_unavailable(
    tmp_path: Path,
) -> None:
    """Probe mode 'auto' falls back to no probe when CUDA is unavailable."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    out_file = tmp_path / "matrix-auto-fallback.json"
    res = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "benchmarks" / "compilation_matrix.py"),
            "--probe",
            "auto",
            "--output",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["provenance"]["probing_active"] is False
    assert data["summary"]["compile_probe_failures"] is None
    assert data["summary"]["nogoods_added"] is None


def test_compilation_matrix_probe_required_on_gpu(tmp_path: Path) -> None:
    """Probe mode 'required' on GPU produces succeeded schedule decisions with resources."""
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
    except ImportError:
        pytest.skip("Torch unavailable")

    out_file = tmp_path / "matrix-required-gpu.json"
    res = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "benchmarks" / "compilation_matrix.py"),
            "--probe",
            "required",
            "--output",
            str(out_file),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    assert res.returncode == 0, res.stderr
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["provenance"]["probe_mode"] == "required"
    assert data["provenance"]["probing_active"] is True
    assert isinstance(data["summary"]["compile_probe_failures"], int)
    assert isinstance(data["summary"]["nogoods_added"], int)

    expected_kernels = {"forward", "grad_weights", "grad_values", "grad_row_scale"}
    for row in data["rows"]:
        for dec in row.get("schedule_decisions", []):
            if dec is not None:
                assert dec["compile_status"] == "succeeded"
                assert dec["kernel_resources"] is not None
                assert set(dec["kernel_resources"].keys()) == expected_kernels
