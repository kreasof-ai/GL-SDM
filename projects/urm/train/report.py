"""Aggregate the benchmark sweep + upstream baseline results into a comparison table.

Reads the per-row TrainResult JSONs from the sweep dir (URM native rows) and the
upstream dir (fast-kernel baselines), joins the rows that have an upstream arm, and
emits a markdown report: MFU, throughput, checkpoint parity, KL divergence, peak
memory, exact params — ours vs upstream where an upstream kernel exists.

Usage:
    PYTHONPATH=src:. python -m train.report --sweep-dir results_sweep \
        --upstream-dir results_upstream --out results_report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(dirpath: str) -> dict[str, dict]:
    out = {}
    for f in sorted(Path(dirpath).glob("*.json")):
        if f.name.startswith("_"):
            continue
        rec = json.loads(f.read_text())
        out[rec.get("mixer", f.stem)] = rec
    return out


def _fmt(v, spec="{:.3f}"):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return spec.format(v)
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results_sweep")
    ap.add_argument("--upstream-dir", default="results_upstream")
    ap.add_argument("--out", default="results_report.md")
    args = ap.parse_args()

    ours = _load(args.sweep_dir)
    upstream = _load(args.upstream_dir) if Path(args.upstream_dir).exists() else {}

    lines = []
    lines.append("# URM native-row benchmark: 100M-class training coverage\n")
    lines.append("Config: width=768, layers=9, heads=12, head_dim=64, seq=512, "
                 "finewebedu, 10 steps, compiled+bf16 (URM rows), eager (upstream). "
                 "MFU denominator: A10G adopted achievable bf16 peak = 70 TFLOPS.\n")

    # ---- URM native rows ----
    lines.append("\n## URM native rows\n")
    lines.append("| mixer | params | MFU | tok/s | ckpt | KL | peak GiB | loss |")
    lines.append("|---|---|---|---|---|---|---|---|")
    n_ok = 0
    for name in sorted(ours):
        r = ours[name]
        if "error" in r:
            lines.append(f"| {name} | — | ERROR | — | — | — | — | {r['error'][:40]} |")
            continue
        n_ok += 1
        mb = f" (mb{r['microbatch_fallback']})" if "microbatch_fallback" in r else ""
        lines.append(
            f"| {name}{mb} | {r['params']:,} | {_fmt(r['mfu'])} | "
            f"{_fmt(r['throughput_tokens_s'], '{:.0f}')} | {_fmt(r['checkpoint_aligned'])} | "
            f"{_fmt(r['kl_divergence'], '{:.2e}')} | {_fmt(r['peak_memory_gib'], '{:.2f}')} | "
            f"{_fmt(r['final_loss'], '{:.3f}')} |"
        )
    lines.append(f"\n_{n_ok}/{len(ours)} native rows completed._\n")

    # ---- Upstream comparison ----
    if upstream:
        lines.append("\n## Upstream fast-kernel comparison\n")
        lines.append("Same 100M-class config; upstream rows run eager (the fla chunk "
                     "kernels fail torch.compile here; the URM rows compile through the "
                     "opaque-op boundary).\n")
        lines.append("| row | MFU (urm/up) | tok/s (urm/up) | peak GiB (urm/up) | "
                     "params (urm/up) | KL |")
        lines.append("|---|---|---|---|---|---|")
        for name in sorted(upstream):
            u = upstream[name]
            o = ours.get(name)
            if "error" in u:
                lines.append(f"| {name} | — / ERROR | — | — | — | — |")
                continue
            o_mfu = _fmt(o['mfu']) if o and 'mfu' in o else "—"
            o_tps = _fmt(o['throughput_tokens_s'], '{:.0f}') if o and 'mfu' in o else "—"
            o_mem = _fmt(o['peak_memory_gib'], '{:.2f}') if o and 'mfu' in o else "—"
            o_par = f"{o['params']:,}" if o and 'params' in o else "—"
            kl = _fmt(o['kl_divergence'], '{:.2e}') if o and 'kl_divergence' in o else "—"
            lines.append(
                f"| {name} | {o_mfu} / {_fmt(u['mfu'])} | {o_tps} / "
                f"{_fmt(u['throughput_tokens_s'], '{:.0f}')} | {o_mem} / "
                f"{_fmt(u['peak_memory_gib'], '{:.2f}')} | {o_par} / {u['params']:,} | {kl} |"
            )

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {args.out} ({n_ok} urm rows, {len(upstream)} upstream rows)")


if __name__ == "__main__":
    main()
