"""Optional Nsight counters in a separate process; never timed acceptance."""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from provenance import write_artifact
from route_parallel_experiment import base_artifact, production_case, stage_functions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", choices=("native", "route_global", "route_resident"))
    p.add_argument("--ncu", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--development", action="store_true")
    args = p.parse_args()
    if args.probe:
        sample = production_case()
        fwd, bwd = stage_functions(args.probe, sample)
        _y, _f, wh, rh = fwd()
        bwd(wh, rh)
        torch.cuda.synchronize()
        return
    artifact = base_artifact(
        "ordered_route_parallel_hardware_counters",
        {"ncu": str(args.ncu)},
        args.development,
    )
    artifact["ncu_version"] = subprocess.check_output(
        [str(args.ncu), "--version"], text=True
    )
    artifact["runs"] = []
    for kind in ("native", "route_global", "route_resident"):
        command = [
            str(args.ncu),
            "--target-processes",
            "all",
            "--kernel-name",
            "regex:ordered_|sparse_state_update",
            "--launch-count",
            "2",
            "--set",
            "full",
            "--csv",
            sys.executable,
            str(Path(__file__).resolve()),
            "--probe",
            kind,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        artifact["runs"].append(
            {
                "implementation": kind,
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "counters_available": "ERR_NVGPUCTRPERM"
                not in result.stdout + result.stderr
                and result.returncode == 0,
            }
        )
    artifact["all_counters_available"] = all(
        row["counters_available"] for row in artifact["runs"]
    )
    write_artifact(args.output, artifact)


if __name__ == "__main__":
    main()
