"""Model-relative Amdahl screening; never substitutes for full acceptance."""

import math


def project_state_replacement(baseline_ratio, state_fraction, complete_stage_speedup):
    if not (math.isfinite(baseline_ratio) and baseline_ratio > 0):
        raise ValueError("baseline ratio must be finite and positive")
    if not (math.isfinite(state_fraction) and 0 <= state_fraction <= 1):
        raise ValueError("replaceable state fraction must be in [0, 1]")
    if not (math.isfinite(complete_stage_speedup) and complete_stage_speedup > 0):
        raise ValueError("complete-stage speedup must be finite and positive")
    floor = baseline_ratio * (1 - state_fraction)
    predicted = baseline_ratio * (
        (1 - state_fraction) + state_fraction / complete_stage_speedup
    )
    return {
        "baseline_native_upstream_ratio": baseline_ratio,
        "replaceable_state_fraction": state_fraction,
        "complete_stage_speedup": complete_stage_speedup,
        "predicted_native_upstream_ratio": predicted,
        "infinite_state_speedup_ratio_floor": floor,
        "required_speedup_for_1_05": (
            0.0
            if state_fraction == 0 and floor <= 1.05
            else baseline_ratio * state_fraction / (1.05 - floor)
            if floor < 1.05
            else None
        ),
        "threefold_screen_passed": complete_stage_speedup >= 3,
        "projection_gate_passed": predicted <= 1.05,
        "model_acceptance": "requires_unchanged_full_grid_after_numerical_acceptance_and_credible_complete_stage_measurement",
    }


def main():
    import argparse
    import json
    from pathlib import Path

    from provenance import provenance, utc_now, write_artifact

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/pretraining-step/confirmation-authority-v2.json"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("results/pretraining-step/native-profile-authority-v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pretraining-step/state-screening-authority-v2.json"),
    )
    parser.add_argument("--assumed-speedup", type=float, default=3.0)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    profile = json.loads(args.profile.read_text())
    if (
        profile["diagnostic"]
        or profile["configuration"]
        != baseline["modes"]["eager"]["pairs"][0]["native"]["configuration"]
    ):
        raise ValueError("screening requires a profile of the frozen primary model")
    fraction = profile["state_stage_spans"]["replaceable_fraction_diagnostic"]
    ratio = baseline["modes"]["eager"]["performance"][
        "geometric_mean_optimizer_step_ratio"
    ]
    config = {
        "baseline": str(args.baseline),
        "profile": str(args.profile),
        "mode": "eager",
        "assumed_complete_stage_speedup": args.assumed_speedup,
    }
    write_artifact(
        args.output,
        {
            "schema_version": 1,
            "artifact_kind": "pretraining_state_replacement_screening",
            "generated_utc": utc_now(),
            "provenance": provenance("pretraining_projection.py", config),
            "configuration": config,
            "formula": "R_baseline * ((1 - f) + f / s)",
            "projection": project_state_replacement(
                ratio, fraction, args.assumed_speedup
            ),
            "fraction_scope": profile["state_stage_spans"]["scope"],
            "speedup_status": "assumed_not_measured",
            "integration_eligible": False,
            "limitation": "Screening only: profile fraction is diagnostic and no candidate complete-stage speedup has been measured. Numerical acceptance, a credible measured projection <=1.05, and the unchanged full model grid remain required.",
        },
    )


if __name__ == "__main__":
    main()
