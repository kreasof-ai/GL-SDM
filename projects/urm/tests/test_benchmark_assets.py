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

    valid = {
        "addresses": {
            "write": {
                "direct_vs_torch": True,
                "adapter_vs_torch": True,
                "adapter_vs_direct": True,
            },
            "read": {
                "direct_vs_torch": True,
                "adapter_vs_torch": True,
                "adapter_vs_direct": True,
            },
            "passed": True,
        },
        "route_weights": {
            "write": {
                "direct_vs_torch": True,
                "adapter_vs_torch": True,
                "adapter_vs_direct": True,
            },
            "read": {
                "direct_vs_torch": True,
                "adapter_vs_torch": True,
                "adapter_vs_direct": True,
            },
            "passed": True,
        },
        "gradients": {
            name: {"passed": True}
            for name in (
                "write_scores",
                "read_scores",
                "initial_memory",
                "values",
                "beta",
                "log_decay",
            )
        },
        "passed": True,
    }
    module._require_end_to_end_backward(valid, "float32")
    for invalid in (
        {**valid, "addresses": {"passed": False}},
        {
            **valid,
            "gradients": {
                **valid["gradients"],
                "write_scores": {"passed": False},
            },
        },
    ):
        with pytest.raises(RuntimeError, match="end-to-end backward"):
            module._require_end_to_end_backward(invalid, "float32")
