"""Representative impossible cases produce mapped, human-readable cores."""

from __future__ import annotations

import pytest

from urm.compiler.solver import FeasibilityPass, FeasibilityStatus, z3_available
from urm.compiler.unsat_catalog import (
    REPRESENTATIVE_UNSAT_CASES,
    describe_unsat,
)

pytestmark = pytest.mark.skipif(
    not z3_available(), reason="z3-solver optional extra not installed"
)


@pytest.mark.parametrize("case", REPRESENTATIVE_UNSAT_CASES, ids=lambda case: case.name)
def test_case_is_unsat_with_mapped_core(case) -> None:
    result = FeasibilityPass().run(case.build())
    assert result.status is FeasibilityStatus.UNSAT
    record = describe_unsat(case, result.unsat_core_names)
    assert record["core_mapped"], (case.name, result.unsat_core_names)
    # The concise message never leaks raw solver formulas.
    assert "z3" not in record["concise_message"].lower()
    assert record["diagnostic_code"] == "unsat_constraints"


def training_forward_only_message() -> str:
    case = REPRESENTATIVE_UNSAT_CASES[0]
    result = FeasibilityPass().run(case.build())
    return case.concise_message if result.unsat_core_names else ""


def test_training_case_message_matches_charter_example() -> None:
    message = training_forward_only_message()
    assert "training" in message and "forward-only" in message
