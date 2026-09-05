"""Additional full-state diagnostics; does not change historical acceptance."""

from __future__ import annotations

import torch

# Reuse the existing state-forward differential envelope for diagnosis ONLY.
STATE_DIAGNOSTIC_TOLERANCE = {"atol": 0.02, "rtol": 0.02}


def tensor_audit(candidate, reference, tolerance):
    candidate, reference = candidate.detach().float(), reference.detach().float()
    if candidate.shape != reference.shape:
        raise ValueError("state comparison shape mismatch")
    delta = (candidate - reference).abs()
    finite_c, finite_r = torch.isfinite(candidate), torch.isfinite(reference)
    valid = finite_c & finite_r
    allowance = tolerance["atol"] + tolerance["rtol"] * reference.abs()
    violations = ~valid | (delta > allowance)
    ranked = torch.where(valid, delta, torch.full_like(delta, float("inf")))
    flat = int(ranked.flatten().argmax())
    coord = []
    remainder = flat
    for extent in reversed(candidate.shape):
        coord.append(remainder % extent)
        remainder //= extent
    coordinate = tuple(reversed(coord))
    finite = bool(valid.all())

    def number(x):
        return float(x) if bool(torch.isfinite(x)) else str(float(x))

    return {
        "shape": list(candidate.shape),
        "elements": candidate.numel(),
        "candidate_nonfinite": int((~finite_c).sum()),
        "reference_nonfinite": int((~finite_r).sum()),
        "finite": finite,
        "max_abs": float(delta.max()) if finite else None,
        "rms": float(delta.square().mean().sqrt()) if finite else None,
        "tolerance": tolerance,
        "violating_elements": int(violations.sum()),
        "worst_coordinate": list(coordinate),
        "worst_candidate": number(candidate[coordinate]),
        "worst_reference": number(reference[coordinate]),
        "worst_allowance": number(allowance[coordinate]),
        "passed": finite and not bool(violations.any()),
    }


def persistent_audit(left_states, right_states, left_checksums, right_checksums):
    rows = []
    for microbatch, (left, right) in enumerate(
        zip(left_states, right_states, strict=True)
    ):
        for block, (reference, candidate) in enumerate(zip(left, right, strict=True)):
            report = tensor_audit(candidate, reference, STATE_DIAGNOSTIC_TOLERANCE)
            # Preserve original GPU-produced means; a new CPU reduction is not
            # a valid replacement for the historical checksum measurement.
            checksum_error = abs(
                right_checksums[microbatch][block]["mean"]
                - left_checksums[microbatch][block]["mean"]
            )
            checksum_passed = checksum_error <= 2e-6
            rows.append(
                {
                    "microbatch": microbatch,
                    "block": block,
                    **report,
                    "diagnostic_only": True,
                    "historical_checksum_normalized_error": checksum_error,
                    "historical_checksum_tolerance": 2e-6,
                    "historical_checksum_passed": checksum_passed,
                    "checksum_and_tensor_diagnostic_disagree": checksum_passed
                    != report["passed"],
                }
            )
    return rows
