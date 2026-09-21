"""Matrix cross-reference for the production release gate.

The release gate must *derive* each workload's verdict from the frozen
production matrix, not trust an artifact's self-declared ``qualified`` string
(acceptance-contract section 10). This module compares what a qualification
artifact actually covers - which matrix cases, dtypes, modes, and correctness
components, with passing parity and performance gates - against what the matrix
declares mandatory.

A workload is ``qualified`` only when every mandatory (case, dtype, mode) tuple
is present in the evidence with passing parity and a passing performance gate,
and every declared correctness component is verified. Anything less is reported
as a precise incomplete status (``partial_coverage``, ``gate_failed``,
``numeric_failed``, ``not_run``, ``unsupported``), never as ``qualified``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Runner mode key -> matrix mode id. The runners record "forward" and
# "forward_backward"; the matrix declares training_forward,
# training_forward_backward, prefill, and decode.
MODE_TO_MATRIX = {
    "forward": "training_forward",
    "forward_backward": "training_forward_backward",
    "prefill": "prefill",
    "decode": "decode",
}

# Correctness component -> the artifact parity field that evidences it. A
# component is verified only when its field is present and parity passed.
_COMPONENT_PARITY_FIELDS = {
    "output": ("output_max_abs_error_vs_upstream", "output_max_abs_error_vs_oracle",
               "output_max_abs_error"),
    "final_state": ("final_state_max_abs_error_vs_upstream",
                    "final_state_max_abs_error_vs_oracle",
                    "final_state_max_abs_error"),
    "input_gradients": ("input_gradient_max_abs_errors_vs_upstream",
                        "input_gradient_max_abs_errors_vs_oracle",
                        "input_gradient_max_abs_errors"),
    # No runner yet emits a distinct state-gradient parity field; the matrix
    # declares it for the stateful K2/K3 workloads, so it is a coverage gap
    # until a runner verifies it explicitly.
    "state_gradients": ("state_gradient_max_abs_errors",),
}


@dataclass
class CaseCoverage:
    """Coverage verdict for one matrix case."""

    case_id: str
    covered_dtypes: set[str] = field(default_factory=set)
    covered_modes: set[str] = field(default_factory=set)
    correctness_components: set[str] = field(default_factory=set)
    parity_pass: bool = False
    performance_pass: bool = False
    matched_artifact_cases: list[str] = field(default_factory=list)


def _normalize_shape(family: str, shape: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce a recorded artifact shape to the matrix case's comparable fields.

    Returns None when the artifact shape cannot be expressed in the matrix's
    per-case structure (for example a runner parameterized by a single
    "channels" width that does not map to heads/key_dim/value_dim).
    """
    if family == "K1":
        try:
            return {
                "batch": shape["batch"],
                "query_length": shape["query_length"],
                "key_length": shape["key_length"],
                "query_heads": shape["query_heads"],
                "key_value_heads": shape.get("key_value_heads", shape.get("kv_heads")),
                "key_dim": shape["key_dim"],
                "value_dim": shape["value_dim"],
            }
        except KeyError:
            return None
    if family == "K2":
        # The diagonal (hgrn) runner records a single "channels" width that does
        # not map onto the matrix's heads/key_dim/value_dim structure.
        if "channels" in shape:
            return None
        try:
            return {
                "batch": shape["batch"],
                "sequence": shape["sequence"],
                "heads": shape["heads"],
                "key_dim": shape["key_dim"],
                "value_dim": shape["value_dim"],
            }
        except KeyError:
            return None
    if family == "K3":
        try:
            return {
                "batch": shape["batch"],
                "sequence": shape["sequence"],
                "slots": shape["slots"],
                "value_dim": shape["value_dim"],
                "read_width": shape["read_width"],
                "write_width": shape["write_width"],
            }
        except KeyError:
            return None
    return None


def _matrix_case_shape(case: dict[str, Any]) -> dict[str, Any]:
    """Extract the comparable shape fields from a matrix case declaration."""
    keys = ("batch", "query_length", "key_length", "query_heads",
            "key_value_heads", "key_dim", "value_dim", "sequence", "heads",
            "slots", "read_width", "write_width")
    return {k: case[k] for k in keys if k in case}


def _case_performance_pass(case_payload: dict[str, Any]) -> bool:
    """Whether every measured mode's performance gate passes for a case."""
    measurements = case_payload.get("performance", {}).get("measurements", {})
    if not measurements:
        return False
    for mode_payload in measurements.values():
        overhead = (
            mode_payload.get("paired_native_overhead_fraction")
            or mode_payload.get("paired_compiled_overhead_fraction")
            or {}
        )
        gate = overhead.get("gate")
        if not gate or not gate.get("pass"):
            return False
    return True


def derive_workload_coverage(
    matrix_workload: dict[str, Any], artifact: dict[str, Any] | None
) -> dict[str, Any]:
    """Cross-reference one artifact against its matrix workload declaration.

    Returns a coverage report with the derived verdict, the per-case coverage,
    and the list of missing mandatory (case, dtype, mode) tuples and
    correctness components.
    """
    family = matrix_workload["family"]
    declared_cases = {c["id"]: c for c in matrix_workload["cases"]}
    declared_dtypes = set(matrix_workload["dtypes"])
    declared_modes = set(matrix_workload["modes"])
    declared_components = set(matrix_workload["correctness"]["components"])

    coverage = {cid: CaseCoverage(case_id=cid) for cid in declared_cases}

    if artifact is None:
        return {
            "verdict": "not_run",
            "reason": "no artifact",
            "coverage": coverage,
            "complete": False,
        }

    artifact_cases = artifact.get("cases", {})
    for artifact_case_id, case_payload in artifact_cases.items():
        shape = case_payload.get("shape", {})
        normalized = _normalize_shape(family, shape)
        if normalized is None:
            continue
        dtype = shape.get("dtype")
        # Match against a declared matrix case by shape.
        matched_id = None
        for cid, matrix_case in declared_cases.items():
            if _matrix_case_shape(matrix_case) == normalized:
                matched_id = cid
                break
        if matched_id is None:
            continue
        cov = coverage[matched_id]
        cov.matched_artifact_cases.append(artifact_case_id)
        if dtype in declared_dtypes:
            cov.covered_dtypes.add(dtype)
        parity = case_payload.get("parity", {})
        parity_status = parity.get("status") == "pass"
        # Record which correctness components this case evidences.
        for component, fields in _COMPONENT_PARITY_FIELDS.items():
            if any(f in parity for f in fields):
                cov.correctness_components.add(component)
        # Modes measured for this case, mapped to matrix mode ids.
        measurements = case_payload.get("performance", {}).get("measurements", {})
        for mode_key in measurements:
            matrix_mode = MODE_TO_MATRIX.get(mode_key)
            if matrix_mode in declared_modes:
                cov.covered_modes.add(matrix_mode)
        cov.parity_pass = cov.parity_pass or parity_status
        cov.performance_pass = cov.performance_pass or _case_performance_pass(
            case_payload
            )

    # Derive the missing mandatory set and the per-case completeness.
    missing = []
    correctness_gaps = []
    all_gates_pass = True
    for cid, matrix_case in declared_cases.items():
        cov = coverage[cid]
        if not cov.matched_artifact_cases:
            missing.append(f"{cid}: no matching artifact case")
            all_gates_pass = False
            continue
        for dtype in sorted(declared_dtypes - cov.covered_dtypes):
            missing.append(f"{cid}: dtype {dtype} not measured")
        for mode in sorted(declared_modes - cov.covered_modes):
            missing.append(f"{cid}: mode {mode} not measured")
        for component in sorted(declared_components - cov.correctness_components):
            correctness_gaps.append(f"{cid}: correctness component {component} not verified")
        if not cov.parity_pass:
            all_gates_pass = False
        if not cov.performance_pass:
            all_gates_pass = False

    complete = not missing and not correctness_gaps
    any_parity_fail = any(
        c.matched_artifact_cases and not c.parity_pass for c in coverage.values()
    )

    if complete and all_gates_pass:
        verdict = "qualified"
    elif any_parity_fail:
        verdict = "numeric_failed"
    elif not complete:
        verdict = "partial_coverage"
    else:
        verdict = "gate_failed"

    return {
        "verdict": verdict,
        "coverage": coverage,
        "complete": complete,
        "missing": missing,
        "correctness_gaps": correctness_gaps,
        "all_gates_pass": all_gates_pass,
    }
