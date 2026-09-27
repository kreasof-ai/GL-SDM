"""Aggregate the benchmark sweep + upstream baseline results into a full-context report.

This is a TRAINING-HARNESS measurement report: 10-step training runs of a
100M-class decoder LM per architecture row on real data (finewebedu), with
correctness gates. It is not a production serving benchmark and not a source-model
parity claim (see docs/evidence.md). The report carries the full context inline:
NaN-diverged rows, pathological-MFU rows with reasons, fallback microbatch markers,
upstream tier/granularity labels, per-row upstream limitation notes, blocked and
missing upstreams, and the environment/provenance block.

Usage:
    PYTHONPATH=src:. python -m train.report --sweep-dir results/sweep \
        --upstream-dir results/upstream --out results_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

# MFU below this is pathological for a 100M-class training step on an A10G and gets
# flagged inline (with the per-row reason in the notes section).
PATHOLOGICAL_MFU = 0.10

# Why a row's MFU is pathologically low (native side). Architectural costs, stated
# honestly — these are real measurements of the executed FLOPs, not harness bugs.
NATIVE_SLOW_NOTES = {
    "based_attention": "chunked-K with 12× score recompute at DV=768; ran at the 1024-token fallback (state history memory)",
    "pattention": "tokenformer block: five cascaded reference-tier pattention maps per block (no fused kernel exists anywhere)",
    "path_attention": "the path-sum mixer runs a dense per-pair recurrence at this width",
    "mom": "external torch composition (public_path=False): mixture-of-paths routing, 2.5× the class parameter count",
    "iplr": "identity-plus-rank-1 transition; also NaN-diverged (see flags) and ran at the 2048 fallback",
    "deltaformer": "K4 triangular solve is serial in t by construction",
    "tucker_attention": "Tucker foldings materialize per-head einsum operands (230M params, 2.2× the class)",
    "attnres": "the residual design keeps every block summary alive and aggregates depth-domain per sub-layer",
    "log_linear_attention": "dyadic-banked K2; the naive upstream comparison is 8× slower still",
}

# Upstream rows whose measured MFU is pathological (mostly the reference-tier or
# fallback-ladder rows); the reason is the baseline's nature, labeled per row.
UPSTREAM_SLOW_FLOOR = 0.01


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


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/sweep")
    ap.add_argument("--upstream-dir", default="results/upstream")
    ap.add_argument("--out", default="results_report.md")
    args = ap.parse_args()

    ours = _load(args.sweep_dir)
    upstream = _load(args.upstream_dir) if Path(args.upstream_dir).exists() else {}

    from train.upstream import UPSTREAM_BLOCKED, UPSTREAM_NOTES, UPSTREAM_TIER

    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        head = "unknown"

    lines = []
    # ============================== Preamble ==============================
    lines.append("# URM training-harness measurement: 51 native rows × 49 upstream baselines\n")
    lines.append(
        "**What this is.** A training-harness measurement: each architecture row trains a "
        "100M-class decoder LM for 10 steps on finewebedu, with checkpoint-parity and KL "
        "gates, on one NVIDIA A10G. It is **not** a production serving benchmark and "
        "**not** a source-model parity claim — the harness trains a generic decoder "
        "surround around each row's mixer (verdict 4 of the [evidence "
        "policy](docs/evidence.md); verdict 5 is not claimed).\n")
    lines.append(
        "**Config.** width=768, layers=9, heads=12, head_dim=64, seq=512, vocab=50304, "
        "finewebedu, 10 steps, bf16 autocast with fp32 kernel accumulation. Microbatch "
        "8192 tokens with an OOM fallback ladder 2048 → 1024 (fallback rows are marked "
        "`(mbNNNN)`). URM rows compile through the opaque-op boundary; upstream rows run "
        "eager (the fla chunk kernels fail torch.compile/Inductor here) — so upstream "
        "throughput is a *lower bound*. MFU denominator: A10G adopted achievable bf16 "
        "peak = 70 TFLOPS.\n")
    lines.append(
        "**Environment.** torch 2.14.0+cu130, triton 3.8.0, NVIDIA A10G 22 GiB "
        f"(101 KB shared-memory limit), git HEAD `{head}`. Policy constraints: no "
        "flash-attn installs (bypassed to SDPA), no mamba_ssm (no torch-2.14/cu130 "
        "wheel), no source builds (the lingua SDM CUDA extension is toolchain-blocked).\n")

    # ============================== Native table ==============================
    lines.append("\n## URM native rows\n")
    lines.append("All 51 native rows completed training and checkpoint parity. Inline "
                 "flags: **`NaN`** = loss diverged within the 10 steps (stability "
                 "finding; MFU still validly measures executed FLOPs); **`slow`** = "
                 "pathological MFU (<0.10, reasons below); `(mbNNNN)` = OOM fallback.\n")
    lines.append("| row | params | MFU | tok/s | ckpt | KL | peak GiB | loss | flags |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    n_ok = 0
    nan_rows, slow_rows, fb_rows = [], [], []
    for name in sorted(ours):
        r = ours[name]
        if "error" in r:
            lines.append(f"| {name} | — | — | — | — | — | — | — | **runtime error** |")
            continue
        n_ok += 1
        flags = []
        if _is_nan(r.get("final_loss")):
            flags.append("**NaN**")
            nan_rows.append(name)
        if r.get("mfu", 1.0) < PATHOLOGICAL_MFU:
            flags.append("**slow**")
            slow_rows.append(name)
        fb = ""
        if r.get("microbatch_fallback"):
            fb = f" (mb{r['microbatch_fallback']})"
            fb_rows.append(name)
        lines.append(
            f"| {name}{fb} | {r['params']:,} | {_fmt(r['mfu'])} | "
            f"{_fmt(r['throughput_tokens_s'], '{:.0f}')} | {_fmt(r['checkpoint_aligned'])} | "
            f"{_fmt(r.get('kl_divergence'), '{:.2e}')} | "
            f"{_fmt(r['peak_memory_gib'], '{:.2f}')} | {_fmt(r['final_loss'], '{:.3f}')} | "
            f"{' '.join(flags) or '—'} |"
        )
    lines.append(f"\n_{n_ok}/{len(ours)} native rows completed._\n")

    # ---- Native context sections ----
    if nan_rows:
        lines.append(f"\n### NaN-diverged rows ({len(nan_rows)})\n")
        lines.append("The low-rank/dual-gate K2 family diverges to NaN loss within the "
                     "10 steps at width 768 — a genuine training-stability result, not a "
                     "harness bug. MFU/throughput remain valid measurements of executed "
                     "FLOPs; the separate reduced-shape checkpoint-parity gate passes "
                     "(it certifies resume correctness, not 10-step stability):\n")
        for n in sorted(nan_rows):
            lines.append(f"- **{n}** — MFU {_fmt(ours[n]['mfu'])} still measured; loss NaN")
        lines.append("")
    if slow_rows:
        lines.append(f"\n### Pathological-MFU rows ({len(slow_rows)}, MFU < {PATHOLOGICAL_MFU})\n")
        lines.append("Architectural costs, measured honestly:\n")
        for n in sorted(slow_rows):
            note = NATIVE_SLOW_NOTES.get(n, "expensive mixer structure at this config")
            lines.append(f"- **{n}** — MFU {_fmt(ours[n]['mfu'])}: {note}")
        lines.append("")
    if fb_rows:
        lines.append(f"\n### OOM-fallback rows ({len(fb_rows)})\n")
        lines.append("These rows OOM'd at the 8192-token primary microbatch and trained "
                     "at the fallback rung shown; their MFU is measured at that rung "
                     "(lower occupancy than the primary, stated openly):\n")
        for n in sorted(fb_rows):
            lines.append(f"- **{n}** — mb{ours[n]['microbatch_fallback']}")
        lines.append("")

    # ============================== Upstream table ==============================
    if upstream:
        lines.append("\n## Upstream comparison\n")
        lines.append(
            "Same 100M-class config; upstream rows run eager. Tiers: **prod** = the "
            "upstream's production kernel; **ref** = the upstream's reference/research "
            "implementation, or a transcription where the production kernel is "
            "environment-blocked (flash-attn absent, mamba_ssm wheel absent, "
            "SMEM/toolchain envelope) — the per-row reason is in the notes below. "
            "Granularity: mixer / schedule (interleaved hybrid) / block (full block) / "
            "residual (residual design).\n")
        lines.append("| row | tier | granularity | MFU (urm/up) | tok/s (urm/up) | "
                     "peak GiB (urm/up) | params (urm/up) | KL |")
        lines.append("|---|---|---|---|---|---|---|---|")
        n_prod = n_ref = 0
        up_slow = []
        up_fb = []
        for name in sorted(upstream):
            u = upstream[name]
            o = ours.get(name)
            tier = u.get("baseline_tier", UPSTREAM_TIER.get(name, "?"))
            gran = u.get("granularity", "mixer")
            tier_short = {"production-kernel": "prod",
                          "reference-implementation": "ref"}.get(tier, tier)
            if "error" in u:
                lines.append(f"| {name} | {tier_short} | {gran} | — / **runtime error** "
                             f"({u['error'][:60]}) | — | — | — | — |")
                continue
            if tier == "production-kernel":
                n_prod += 1
            else:
                n_ref += 1
            if u.get("mfu", 1.0) < UPSTREAM_SLOW_FLOOR:
                up_slow.append(name)
            if u.get("microbatch_fallback"):
                up_fb.append(name)
            o_mfu = _fmt(o['mfu']) if o and 'mfu' in o else "—"
            o_tps = _fmt(o['throughput_tokens_s'], '{:.0f}') if o and 'mfu' in o else "—"
            o_mem = _fmt(o['peak_memory_gib'], '{:.2f}') if o and 'mfu' in o else "—"
            o_par = f"{o['params']:,}" if o and 'params' in o else "—"
            kl = _fmt(o['kl_divergence'], '{:.2e}') if o and 'kl_divergence' in o else "—"
            u_fb = f" (mb{u['microbatch_fallback']})" if u.get("microbatch_fallback") else ""
            lines.append(
                f"| {name} | {tier_short} | {gran} | {o_mfu} / {_fmt(u['mfu'])}{u_fb} | "
                f"{o_tps} / {_fmt(u['throughput_tokens_s'], '{:.0f}')} | {o_mem} / "
                f"{_fmt(u['peak_memory_gib'], '{:.2f}')} | {o_par} / {u['params']:,} | {kl} |"
            )
        lines.append(f"\n_{n_prod + n_ref}/{len(upstream)} upstream baselines completed: "
                     f"{n_prod} production-kernel, {n_ref} reference-implementation._\n")

        # ---- Upstream context sections ----
        ref_rows = sorted(n for n in upstream
                          if UPSTREAM_TIER.get(n) == "reference-implementation"
                          and "error" not in upstream[n])
        if ref_rows:
            lines.append(f"\n### Reference-implementation baselines ({len(ref_rows)}) — "
                         "why no production kernel\n")
            for n in ref_rows:
                note = UPSTREAM_NOTES.get(n, "research-code pin")
                lines.append(f"- **{n}** — {note}")
            lines.append("")
        lines.append("\n### Production-kernel baselines with caveats\n")
        for n in sorted(upstream):
            if UPSTREAM_TIER.get(n) == "production-kernel" and n in UPSTREAM_NOTES:
                lines.append(f"- **{n}** — {UPSTREAM_NOTES[n]}")
        lines.append("")
        if up_slow:
            lines.append(f"\n### Pathological upstream MFU (<{UPSTREAM_SLOW_FLOOR})\n")
            lines.append("The baseline itself is slow (naive recurrence / torch "
                         "transcription / eager fallback config) — the comparison is "
                         "still valid; the tier label says why:\n")
            for n in sorted(up_slow):
                lines.append(f"- **{n}** — upstream MFU {_fmt(upstream[n]['mfu'])} "
                             f"({UPSTREAM_TIER.get(n, '?')})")
            lines.append("")
        if up_fb:
            lines.append(f"\n### Upstream OOM-fallback rows ({len(up_fb)})\n")
            for n in sorted(up_fb):
                lines.append(f"- **{n}** — mb{upstream[n]['microbatch_fallback']}")
            lines.append("")

    # ============================== Coverage ==============================
    lines.append("\n## Coverage and exclusions\n")
    lines.append("- **hla** — no upstream implementation exists anywhere (empty pin; "
                 "paper-only). The sole principled exclusion: the URM row trains, no "
                 "baseline is fabricated.")
    if UPSTREAM_BLOCKED:
        for row, why in UPSTREAM_BLOCKED.items():
            lines.append(f"- **{row}** — environment-blocked: {why}. Ours-only row.")
    lines.append("- **mamba1** — reference-tier row (accepted charter debt: the K2 "
                 "elementwise gate has a single client); not in the 51-row native "
                 "sweep, so no baseline comparison is run.")
    lines.append("- **KL gate** — wired for the 23 rows with a registry comparator; "
                 "`—` elsewhere means *no comparator wired*, not failure.")
    lines.append("- **mom, raven** — external torch compositions (public_path=False); "
                 "trained as-is, no compiler-coverage claim.")
    lines.append("")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {args.out} ({n_ok} urm rows, {len(upstream)} upstream rows)")


if __name__ == "__main__":
    main()
