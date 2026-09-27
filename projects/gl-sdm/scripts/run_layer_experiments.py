"""Run the four untied 16-layer models, one GPU process at a time.

Workloads never shrink or switch kernels on failure. The manifest records every
command and exit status; failed checks are retained as failed checks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=("transformer", "gdn2", "sdm", "gl_sdm"),
                        default=["transformer", "gdn2", "sdm", "gl_sdm"])
    parser.add_argument("--phases", nargs="+", choices=("verify", "verify-fp32", "benchmark", "train"),
                        default=["verify", "verify-fp32", "benchmark", "train"])
    parser.add_argument("--output-dir", type=Path, default=Path("projects/gl-sdm/results/sequence_2048"))
    parser.add_argument("--train-steps", type=int, default=20, help="explicit short pilot; use 1000 for the configured training budget")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--allocator-config", default="max_split_size_mb:512",
                        help="same explicit CUDA allocator configuration for every model")
    args = parser.parse_args()
    if args.train_steps < 1:
        parser.error("--train-steps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    entries = json.loads(manifest.read_text()) if manifest.exists() else []
    failed = False
    environment = {**os.environ, "PYTORCH_ALLOC_CONF": args.allocator_config}
    for phase in args.phases:
        for arch in args.models:
            cfg_path = Path(f"projects/gl-sdm/configs/{arch}.json")
            cfg = json.loads(cfg_path.read_text())
            attempt = 1 + sum(e["model"] == arch and e["phase"] == phase for e in entries)
            name = f"{arch}_{phase.replace('-', '_')}" + (f"_attempt{attempt}" if attempt > 1 else "")
            if phase == "verify-fp32":
                cfg.update(dtype="float32")
                cfg_path = args.output_dir / f"{arch}_verify_fp32_config.json"
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
            command = [sys.executable, "-m", "gl_sdm.cli", "verify" if phase == "verify-fp32" else phase, "--config", str(cfg_path)]
            if phase.startswith("verify"):
                length = max(129, cfg.get("gl_chunk_size", 1) + 1)
                command += ["--verify-batch-size", "1", "--length", str(length)]
            if phase == "benchmark":
                command += ["--iterations", str(args.iterations), "--warmup", str(args.warmup)]
            if phase == "train":
                # Preserve model/workload/kernel settings; report the smaller
                # pilot training budget explicitly instead of overwriting them.
                cfg.update(max_steps=args.train_steps, val_freq=max(1, args.train_steps // 2))
                cfg_path = args.output_dir / f"{arch}_pilot_config.json"
                cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
                command[command.index("--config") + 1] = str(cfg_path)
                artifact = Path("projects/gl-sdm/checkpoints") / args.output_dir.name / arch
            else:
                artifact = args.output_dir / f"{name}.json"
            command += ["--output", str(artifact)]
            log = args.output_dir / f"{name}.log"
            print(f"Starting {name}: {' '.join(command)}", flush=True)
            with log.open("w") as output:
                status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=environment).returncode
            entry = {"model": arch, "phase": phase, "command": command,
                     "attempt": attempt, "exit_code": status, "allocator_config": args.allocator_config,
                     "log": str(log), "artifact": str(artifact)}
            entries.append(entry)
            manifest.write_text(json.dumps(entries, indent=2) + "\n")
            failed = failed or status != 0
            print(f"Finished {name}: exit {status}; {log}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
