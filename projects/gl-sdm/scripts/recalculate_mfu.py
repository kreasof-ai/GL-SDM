"""Recalculate fixed-depth MFU from saved timings without rerunning the model.

Preserve measurement provenance and add a separate fingerprint for accounting.
Adaptive results require per-token positional telemetry and cannot be reconstructed
from a global mean, so this script refuses them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from gl_sdm.experiments import metrics
from gl_sdm.model import create_model


def recalculate(result):
    cfg = result["config"]
    if cfg["arch_type"] != "gl_sdm" or cfg["gl_reasoning"] != "fixed":
        raise ValueError("saved-timing recalculation requires fixed-depth GL-SDM")
    with torch.device("meta"):
        model = create_model(cfg)
    rows = [result["training"]] if "training" in result else result["curve"]
    for row in rows:
        if "tokens" not in row:
            continue
        tokens, passes = row["tokens"], cfg["gl_max_steps"]
        if row["mean_reasoning_steps"] != passes or row["max_reasoning_steps"] != passes or row["executed_reasoner_calls"] != tokens * passes:
            raise ValueError("saved depth telemetry disagrees with the fixed config")
        updated = metrics.utilization(model, tokens, row["seconds"], row["peak_tflops"] * 1e12)
        for name in ("num_params", "active_params", "memory_params"):
            if row[name] != updated[name]:
                raise ValueError(f"saved {name} does not match the model")
        row.update(updated)
        if "mfu" in row:
            row["mfu"] = row["mfu_6nd_pct"]
    result["mfu_accounting"] = {
        "method": "execution-weighted 6ND; recalculated from unchanged original timings",
        "implementation_sha256": hashlib.sha256(Path(metrics.__file__).read_bytes()).hexdigest(),
        "measurement_source_sha256": result["runtime"]["source_sha256"],
        "depth_basis": "fixed config independently agrees with saved measured depth telemetry",
        "timings_rerun": False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--write", action="store_true", help="update artifacts; otherwise only print recalculated values")
    args = parser.parse_args()
    for path in args.artifacts:
        result = recalculate(json.loads(path.read_text()))
        if args.write:
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        rows = [result["training"]] if "training" in result else result["curve"]
        print(path.name, [(row.get("step"), row["mfu_6nd_pct"], row["unique_parameter_6nd_pct"])
                          for row in rows if "mfu_6nd_pct" in row])


if __name__ == "__main__":
    main()
