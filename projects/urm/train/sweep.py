"""Benchmark sweep: 100M-class training runs across the native registry rows.

Runs each row as a subprocess (CUDA memory isolation between rows), records the
TrainResult JSON per row under the output dir. OOM fallback is an opt-in diagnostic;
the default campaign retains an 8192-token effective batch and microbatch. Memory
heavy rows use activation checkpointing. Failures remain error records.

Usage:
    PYTHONPATH=src:. python -m train.sweep --out-dir results/sweep
    PYTHONPATH=src:. python -m train.sweep --out-dir results/sweep --rows gla,deltanet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# The benchmark configuration: ~100M-param class (dense = 102.7M; exact per-row
# counts are in each result JSON — the mixers' projection structures differ).
TIMEOUT_S = 1500              # per-row wallclock cap (compile + train + gates)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URM native-row benchmark sweep")
    p.add_argument("--out-dir", default="results/sweep")
    p.add_argument("--rows", default=None, help="comma-separated subset (default: all native)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=9)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--timeout", type=int, default=TIMEOUT_S)
    p.add_argument("--batch-tokens", type=int, default=8192)
    p.add_argument("--microbatch-tokens", type=int, default=8192)
    p.add_argument("--allow-oom-fallback", action="store_true",
                   help="diagnostic only; retain the effective batch and record every attempt")
    p.add_argument("--eager", action="store_true")
    p.add_argument("--include-reference", action="store_true",
                   help="also run reference-tier rows (mamba1)")
    return p.parse_args()


def _run_row(row: str, args: argparse.Namespace, microbatch_tokens: int,
             out_file: Path, log_file: Path) -> dict:
    cmd = [
        sys.executable, "-m", "train.run",
        "--mixer", row,
        "--steps", str(args.steps),
        "--layers", str(args.layers),
        "--width", str(args.width),
        "--num-heads", str(args.num_heads),
        "--head-dim", str(args.head_dim),
        "--sequence-length", str(args.sequence_length),
        "--batch-tokens", str(args.batch_tokens),
        "--microbatch-tokens", str(microbatch_tokens),
        "--out", str(out_file),
    ]
    from train.campaign import use_checkpointing
    if use_checkpointing(row):
        cmd.append("--activation-checkpointing")
    if args.eager:
        cmd.append("--eager")
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    with log_file.open("a") as log:
        log.write(f"\nCOMMAND: {cmd!r}\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env=env, timeout=args.timeout)
    if proc.returncode != 0:
        # The OOM text lives in the subprocess log, not the exit path — sniff the log
        # so the microbatch fallback actually triggers on CUDA OOM.
        log_text = log_file.read_text() if log_file.exists() else ""
        if "out of memory" in log_text.lower():
            raise _CudaOOM(f"CUDA OOM at this microbatch (see {log_file})")
        raise RuntimeError(f"exit {proc.returncode} (see {log_file})")
    return json.loads(out_file.read_text())


class _CudaOOM(Exception):
    """CUDA out-of-memory in the row's subprocess (triggers the microbatch fallback)."""


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, ".")
    from train.registry import MIXER_REGISTRY
    from train.campaign import use_checkpointing, valid_cached_result
    from train.harness import TrainConfig
    from dataclasses import asdict

    if args.rows:
        rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    else:
        rows = sorted(
            n for n, m in MIXER_REGISTRY.items()
            if m.tier == "native" or (args.include_reference and m.tier != "native")
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] {len(rows)} rows -> {out_dir}", flush=True)

    summary = []
    for i, row in enumerate(rows, 1):
        out_file = out_dir / f"{row}.json"
        log_file = out_dir / f"{row}.log"
        expected = asdict(TrainConfig(
            mixer=row, layers=args.layers, width=args.width, num_heads=args.num_heads,
            head_dim=args.head_dim, sequence_length=args.sequence_length,
            steps=args.steps, batch_tokens=args.batch_tokens,
            microbatch_tokens=args.microbatch_tokens, compile_model=not args.eager,
            activation_checkpointing=use_checkpointing(row),
        ))
        if out_file.exists():
            rec = json.loads(out_file.read_text())
            if valid_cached_result(rec, expected):
                print(f"[sweep] {i}/{len(rows)} {row}: cached "
                      f"(mfu={rec.get('mfu', 0):.3f})", flush=True)
                summary.append(rec)
                continue
        log_file.write_text("")
        t0 = time.time()
        rec: dict = {"mixer": row}
        ladder = (args.microbatch_tokens,)
        if args.allow_oom_fallback:
            ladder += tuple(mb for mb in (2048, 1024, 512)
                            if mb < args.microbatch_tokens and args.batch_tokens % mb == 0)
        attempts = []
        for attempt, mb in enumerate(ladder):
            attempts.append(mb)
            try:
                rec = _run_row(row, args, mb, out_file, log_file)
                if attempt:
                    rec["microbatch_fallback"] = mb
                    out_file.write_text(json.dumps(rec, indent=2))
                break
            except subprocess.TimeoutExpired:
                rec = {"mixer": row, "error": f"timeout after {args.timeout}s"}
                break
            except _CudaOOM:
                if attempt < len(ladder) - 1:
                    print(f"[sweep] {row}: OOM at {mb}, retrying at "
                          f"{ladder[attempt + 1]}", flush=True)
                    continue
                rec = {"mixer": row, "error": f"CUDA OOM at all of {ladder}"}
                break
            except Exception as e:  # noqa: BLE001 — record and continue the sweep
                rec = {"mixer": row, "error": str(e)}
                break
        rec["attempted_microbatches"] = attempts
        if "error" in rec:
            rec.update(measurement_version=2, config=expected)
        out_file.write_text(json.dumps(rec, indent=2, allow_nan=False))
        dt = time.time() - t0
        status = (f"mfu={rec['mfu']:.3f} tok/s={rec['throughput_tokens_s']:.0f} "
                  f"mem={rec['peak_memory_gib']:.1f}GiB" if "mfu" in rec
                  else f"ERROR: {rec['error'][:80]}")
        print(f"[sweep] {i}/{len(rows)} {row}: {status} ({dt:.0f}s)", flush=True)
        summary.append(rec)

    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for r in summary if "mfu" in r)
    print(f"[sweep] done: {n_ok}/{len(summary)} rows ok -> {out_dir}/_summary.json",
          flush=True)


if __name__ == "__main__":
    main()
