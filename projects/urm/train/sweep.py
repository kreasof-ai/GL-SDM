"""Benchmark sweep: 100M-class training runs across the native registry rows.

Runs each row as a subprocess (CUDA memory isolation between rows), records the
TrainResult JSON per row under the output dir, and automatically falls back from the
primary 8192-token microbatch to 2048 for memory-capped rows (the chunked-K Based
state history is the binding constraint). Rows that fail record an error JSON so the
sweep completes and reports honestly.

Usage:
    PYTHONPATH=src:. python -m train.sweep --out-dir results_sweep
    PYTHONPATH=src:. python -m train.sweep --out-dir results_sweep --rows gla,deltanet
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
PRIMARY_MICROBATCH = 8192     # B=16 at T=512 — puts efficient rows in the 40-50% MFU band
FALLBACK_MICROBATCH = 2048    # B=4 — the largest the chunked-K Based state history fits
TIMEOUT_S = 1500              # per-row wallclock cap (compile + train + gates)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URM native-row benchmark sweep")
    p.add_argument("--out-dir", default="results_sweep")
    p.add_argument("--rows", default=None, help="comma-separated subset (default: all native)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=9)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--timeout", type=int, default=TIMEOUT_S)
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
        "--batch-tokens", str(microbatch_tokens),
        "--microbatch-tokens", str(microbatch_tokens),
        "--out", str(out_file),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "src:."
    with log_file.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              env=env, timeout=args.timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"exit {proc.returncode} (see {log_file})")
    return json.loads(out_file.read_text())


def main() -> None:
    args = _parse_args()
    sys.path.insert(0, ".")
    from train.registry import MIXER_REGISTRY

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
        if out_file.exists():
            rec = json.loads(out_file.read_text())
            print(f"[sweep] {i}/{len(rows)} {row}: cached "
                  f"(mfu={rec.get('mfu', 0):.3f})", flush=True)
            summary.append(rec)
            continue
        t0 = time.time()
        rec: dict = {"mixer": row}
        for attempt, mb in enumerate((PRIMARY_MICROBATCH, FALLBACK_MICROBATCH)):
            try:
                rec = _run_row(row, args, mb, out_file, log_file)
                if attempt:
                    rec["microbatch_fallback"] = mb
                    out_file.write_text(json.dumps(rec, indent=2))
                break
            except subprocess.TimeoutExpired:
                rec = {"mixer": row, "error": f"timeout after {args.timeout}s"}
                break
            except Exception as e:  # noqa: BLE001 — record and continue the sweep
                msg = str(e)
                if "out of memory" in msg.lower() and attempt == 0:
                    print(f"[sweep] {row}: OOM at {PRIMARY_MICROBATCH}, "
                          f"retrying at {FALLBACK_MICROBATCH}", flush=True)
                    continue
                rec = {"mixer": row, "error": msg}
                break
        if "error" in rec:
            out_file.write_text(json.dumps(rec, indent=2))
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
