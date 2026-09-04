"""Dependency-light checks for the native confirmation protocol."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import sparse_state_mixer as benchmark


def test_drift_retry_retains_rejected_attempt(monkeypatch) -> None:
    sentinels = iter((1.0, 1.4, 1.0, 1.1))
    seeds = []
    monkeypatch.setattr(benchmark, "_sentinel", lambda *_args: next(sentinels))

    def paired(*_args, **kwargs):
        seeds.append(kwargs["seed"])
        return {"seed": kwargs["seed"]}

    monkeypatch.setattr(benchmark, "_paired_measure", paired)
    performance, attempts = benchmark._paired_with_drift_retries(
        None,
        None,
        None,
        None,
        samples=3,
        warmup=1,
        seed=17,
        torch=None,
        case_name="synthetic",
    )
    assert seeds == [17, 18]
    assert performance == {"seed": 18}
    assert [attempt["passed"] for attempt in attempts] == [False, True]
    assert attempts[0]["absolute_fraction"] == pytest.approx(0.4)


def test_drift_retry_exhaustion_fails_closed(monkeypatch) -> None:
    sentinels = iter((1.0, 1.4, 1.0, 1.4, 1.0, 1.4))
    monkeypatch.setattr(benchmark, "_sentinel", lambda *_args: next(sentinels))
    monkeypatch.setattr(benchmark, "_paired_measure", lambda *_args, **_kwargs: {})
    with pytest.raises(RuntimeError, match="exhausted three attempts"):
        benchmark._paired_with_drift_retries(
            None,
            None,
            None,
            None,
            samples=3,
            warmup=1,
            seed=17,
            torch=None,
            case_name="synthetic",
        )


def test_host_bound_confirmation_uses_absolute_latency_policy() -> None:
    assert benchmark._confirmation_phase_passed(
        classification="host_bound",
        hierarchical_upper=0.7,
        per_process_medians=[0.6, 0.7, 0.8],
        per_process_p95_ratios=[0.8, 0.9, 1.4],
        memory_passes=[True, True, True],
        per_process_gate_passes=[True, True, False],
    )
    assert not benchmark._confirmation_phase_passed(
        classification="substantial",
        hierarchical_upper=0.7,
        per_process_medians=[0.6, 0.7, 0.8],
        per_process_p95_ratios=[0.8, 0.9, 1.4],
        memory_passes=[True, True, True],
        per_process_gate_passes=[True, True, False],
    )


@pytest.mark.parametrize("processes", [1, 2, 4])
def test_completion_confirmation_requires_exactly_three_processes(processes) -> None:
    args = Namespace(
        confirmation_processes=processes,
        case=[],
        samples=3,
        warmup=1,
        output=Path("unused.json"),
    )
    with pytest.raises(ValueError, match="exactly three"):
        benchmark._run_confirmation(args)


def test_completion_confirmation_rejects_case_filtering() -> None:
    args = Namespace(
        confirmation_processes=3,
        case=["decode_fp32"],
        samples=3,
        warmup=1,
        output=Path("unused.json"),
    )
    with pytest.raises(ValueError, match="cannot filter"):
        benchmark._run_confirmation(args)
