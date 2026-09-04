from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]


def test_sparse_delta_callable_identity_is_measured_not_assumed() -> None:
    module_path = PROJECT_ROOT / "benchmarks" / "sparse_delta_memory.py"
    spec = importlib.util.spec_from_file_location("urm_sdm_benchmark", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    same_bound_callable = module._same_bound_callable

    class Example:
        def call(self):
            return None

    instance = Example()
    stored = instance.call
    assert same_bound_callable(stored, stored) is True
    # A newly materialized bound-method object shares function/instance but is
    # not the exact callable stored below direct and adapter dispatch.
    assert same_bound_callable(stored, instance.call) is False


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


def test_compilation_matrix_probe_required_fails_when_dependencies_missing(
    tmp_path: Path,
) -> None:
    """Probe mode 'required' fails with a concise diagnostic when torch/triton is missing."""
    out_file = tmp_path / "matrix-fail-dep.json"
    benchmarks_dir = str(PROJECT_ROOT / "benchmarks")
    code = (
        "import sys\n"
        f"sys.path.insert(0, r'{benchmarks_dir}')\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['triton'] = None\n"
        "sys.argv = ['compilation_matrix.py', '--probe', 'required', '--output', "
        f"r'{out_file}']\n"
        "import benchmarks.compilation_matrix as cm\n"
        "try:\n"
        "    cm.main()\n"
        "    sys.exit(0)\n"
        "except RuntimeError as e:\n"
        "    sys.stderr.write(str(e))\n"
        "    sys.exit(1)\n"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    assert res.returncode != 0
    assert "Probe mode 'required' failed" in res.stderr
    assert "dependency missing" in res.stderr


def test_compilation_matrix_probe_required_fails_when_cuda_unavailable(
    tmp_path: Path,
) -> None:
    """Probe mode 'required' fails with a concise diagnostic when CUDA is unavailable."""
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("Torch and Triton required to test CUDA unavailability diagnostic")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    out_file = tmp_path / "matrix-fail-cuda.json"
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
