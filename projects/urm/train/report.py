"""Report verified training measurements and eligible production comparisons.

Reference implementations and failed production runs never supply the comparison
arm. Results with different shapes, batches, execution policies, or source hashes
are kept visible as diagnostics and are excluded from the paired table.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PATHOLOGICAL_MFU = 0.10
PROTOCOL_KEYS = (
    "vocab_size", "sequence_length", "layers", "width", "num_heads", "head_dim",
    "mlp_ratio", "batch_tokens", "microbatch_tokens", "steps", "seed",
    "compile_model", "bf16", "activation_checkpointing",
)
KERNEL_REPLACEMENT_ROWS = {
    "comba", "gdn2", "gated_delta_product", "dplr", "rwkv7", "mamba2",
    "log_linear_attention", "log_linear_mamba2", "dense_attention", "attnres", "samba_attention", "raven", "tda",
    "sdm",
}
SLOW_NOTES = {
    "attnres": "depth aggregation over full-width residual sources",
    "based_attention": "normalized recurrent Taylor-feature state",
    "deltaformer": "strict-causal correction and block triangular solve",
    "iplr": "factored identity-plus-rank-one state transition",
    "mom": "eight variable-length expert streams with routing and packing",
    "path_attention": "Householder score correction and block triangular solve",
    "pattention": "five count-scaled parameter-token softmax contractions per decoder block",
    "tucker_attention": "full-rank Tucker foldings and 768-wide per-head attention",
    "log_linear_attention": "four saved state banks and activation recomputation",
    "log_linear_mamba2": "four saved state banks and activation recomputation",
    "tda": "threshold-ReLU-square forward/backward with restored Q/K/V gradients",
}


def _load(directory):
    return {r.get("mixer", p.stem): r
            for p in sorted(Path(directory).glob("*.json")) if not p.name.startswith("_")
            for r in [json.loads(p.read_text())]}


def verified(record):
    if record.get("measurement_version") != 2 or "error" in record:
        return False
    trace = record.get("loss_trace", [])
    return (record.get("checkpoint_aligned") is True
            and len(trace) == record.get("steps", 0) > 0
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in trace)
            and math.isfinite(record.get("final_loss", math.nan))
            and record.get("source_fingerprint") is not None
            and all(math.isfinite(record.get(k, math.nan)) and record.get(k, 0) > 0
                    for k in ("mfu", "wallclock_s", "throughput_tokens_s")))


def comparison_reason(ours, upstream):
    if not verified(ours) or not verified(upstream):
        return "training/gates not verified under measurement version 2"
    if upstream.get("baseline_tier") != "production-kernel":
        return "production training kernel unavailable"
    if ours.get("source_fingerprint") != upstream.get("source_fingerprint"):
        return "different source fingerprints"
    if ours.get("environment") != upstream.get("environment"):
        return "different device/precision environment"
    oc, uc = ours.get("config", {}), upstream.get("config", {})
    mismatches = [key for key in PROTOCOL_KEYS if key not in oc or key not in uc or oc[key] != uc[key]]
    if mismatches:
        return "different " + ", ".join(mismatches)
    if ours.get("microbatch_fallback") or upstream.get("microbatch_fallback"):
        return "diagnostic OOM fallback"
    return None


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    if not math.isfinite(value):
        return "invalid"
    return f"{value:.{digits}f}"


def render(ours, upstream):
    ok = {n: r for n, r in ours.items() if verified(r)}
    pairs = {n: (r, upstream[n]) for n, r in ok.items() if n in upstream
             and comparison_reason(r, upstream[n]) is None}
    lines = ["# URM training benchmark", "",
             f"**{len(ok)}/{len(ours)} URM configurations trained with finite losses and passed the checkpoint resume check.** "
             f"{len(pairs)} also have a usable upstream production-kernel comparison.", "",
             "Each run trains a nine-layer decoder on FineWebEdu for ten measured updates after two warmup updates. "
             "Batch and microbatch are both 8192 tokens. Results measure decoder training on an A10G GPU.", "",
             "MFU is shown as a percentage: estimated model work (`6 × parameters × tokens`, plus mixer/state work) "
             "divided by elapsed time and 70 TFLOPS. It is an estimate, not a GPU hardware counter.", "",
             "## URM results", "",
             "Peak memory includes temporary activations. Memory change is the last minus first allocation "
             "after a training step. The checkpoint column records whether saving, reloading and resuming passed.", "",
             "| Model | Parameters | MFU (%) | Tokens/s | Checkpoint | Peak (GiB) | Final loss | Memory change (GiB) | Notes |",
             "|---|---:|---:|---:|:---:|---:|---:|---:|---|"]
    for name, r in sorted(ours.items()):
        if not verified(r):
            reason = r.get("error", "unverified/legacy measurement").replace("|", "/").replace("\n", " ")
            lines.append(f"| {name} | — | — | — | — | — | — | — | {reason} |")
            continue
        notes = []
        if r['mfu'] < PATHOLOGICAL_MFU:
            notes.append("MFU below 10%")
        if r['config'].get('activation_checkpointing'):
            notes.append("activation checkpointing")
        if r.get('microbatch_fallback'):
            notes.append(f"reduced microbatch: {r['microbatch_fallback']} tokens (diagnostic)")
        if r.get('sdm_execution', {}).get('schedule') == 'torch-chunked':
            notes.append("external state implementation")
        if r.get('sdm_execution', {}).get('schedule') == 'native-k3':
            notes.append("native chunks")
        memory = r.get('memory_trace_gib', [])
        drift = memory[-1] - memory[0] if memory else None
        lines.append(f"| {name} | {r['params']:,} | {_fmt(100 * r['mfu'], 1)} | {_fmt(r['throughput_tokens_s'], 0)} | "
                     f"✓ | {_fmt(r['peak_memory_gib'], 2)} | {_fmt(r['final_loss'])} | {_fmt(drift)} | {', '.join(notes) or '—'} |")
    slow = [n for n, r in ok.items() if r['mfu'] < PATHOLOGICAL_MFU]
    if slow:
        lines += ["", f"### Rows below 10% MFU ({len(slow)})", "",
                  "These rows need further profiling. The operations below describe their implementations; "
                  "they have not been confirmed as the bottlenecks.", ""]
        lines += [f"- **{n}**: {SLOW_NOTES.get(n, 'requires further profiling')} (MFU {_fmt(100 * ok[n]['mfu'], 1)}%)."
                  for n in sorted(slow)]
    lines += ["", "## Production-kernel measurements", "",
              "Every pair uses the same decoder dimensions, data, batch sizes, precision, optimizer, "
              "compilation and activation-checkpointing settings. Both runs must pass the training checks "
              "and record the same hardware and benchmark source version. "
              "Failed upstream kernels are excluded; a reference implementation cannot replace them.", "",
              "Each cell shows **URM / upstream**. The tables separate comparisons that replace only the "
              "mixer kernel from those that use a different mixer module."]
    groups = (
        (True, "Same projections and routing; only the kernel changes",
         "Both runs use the same projections, gates and routing code. The upstream run replaces "
         "the URM mixer kernel with the upstream production kernel."),
        (False, "Different mixer modules",
         "The upstream run uses a separate mixer implementation. Its projections, gates or other "
         "layers can differ, as can its parameter count. These numbers compare decoder implementations; "
         "they do not isolate kernel speed."),
    )
    for kernel_only, title, explanation in groups:
        group = {n: pair for n, pair in pairs.items()
                 if (n in KERNEL_REPLACEMENT_ROWS) == kernel_only}
        if not group:
            continue
        lines += ["", f"### {title} ({len(group)})", "", explanation, "",
                  "| Model | MFU (%) | Tokens/s | Peak (GiB) | Parameters |",
                  "|---|---:|---:|---:|---:|"]
        for n, (o, u) in sorted(group.items()):
            lines.append(f"| {n} | {_fmt(100 * o['mfu'], 1)} / {_fmt(100 * u['mfu'], 1)} | "
                         f"{_fmt(o['throughput_tokens_s'], 0)} / {_fmt(u['throughput_tokens_s'], 0)} | "
                         f"{_fmt(o['peak_memory_gib'], 2)} / {_fmt(u['peak_memory_gib'], 2)} | {o['params']:,} / {u['params']:,} |")
    lines += ["", "### Production adapters", "",
              "The upstream runs use pinned production kernels. Loading requirements and numerical "
              "adjustments are documented in [benchmark implementation details](../docs/benchmark.md#corrections).", ""]
    if "sdm" in ok and ok["sdm"].get("sdm_execution", {}).get("schedule") == "native-k3":
        lines += ["SDM runs through native URM with chunks of up to 128 tokens. Its upstream comparison uses "
                  "the original Meta CUDA/Triton kernels with 64-token chunks and the same projections and routes. "
                  "See [SDM results and historical MFU accounting](../docs/sdm-optimization.md).", ""]
    lines += ["## Upstreams excluded from production comparison", "",
              "These rows have no usable production-kernel comparison. The reason is listed for each row. "
              "Reference or research implementations can be run separately with "
              "`train.upstream --include-reference`.", ""]
    for n, u in sorted(upstream.items()):
        if n in pairs:
            continue
        reason = u.get('reason') or u.get('error') or comparison_reason(ours.get(n, {}), u)
        lines.append(f"- **{n}**: {reason}.")
    for n in sorted(set(ours) - set(upstream)):
        lines.append(f"- **{n}**: no upstream measurement available.")
    lines += ["", "## Measurement details", "",
              "- Decoder: width 768, 9 layers, 12 heads, head dimension 64, sequence length 512, vocabulary 50,304.",
              "- BF16 training with FP32 kernel accumulation. Both runs compile the model around Python plan "
              "dispatch and upstream kernels that execute outside the compiled graph.",
              "- GPU timing is synchronized. Optimizer settings and gradient clipping match in both runs.",
              "- Activation checkpointing is enabled for the rows marked in the table. "
              "It recomputes activations to reduce memory use and is separate from the checkpoint resume check.",
              "- OOM retries are disabled. Failed runs are recorded at the requested batch size.", ""]
    if ok:
        env = next(iter(ok.values())).get("environment", {})
        hashes = sorted({r['source_fingerprint'][:12] for r in ok.values()})
        lines += [f"Torch {env.get('torch', '?')}; GPU: {env.get('device', '?')}; "
                  f"measurement version 2. Benchmark source hashes: `{', '.join(hashes)}`. "
                  "SDM was remeasured after native integration; the other rows retain their accepted measurements. "
                  "Each URM/upstream pair has matching source hashes.", ""]
        memory_ranges = [max(trace) - min(trace) for r in ok.values()
                         for trace in [r.get('memory_trace_gib', [])] if trace]
        if len(memory_ranges) == len(ok):
            lines += [f"The largest change in step-end allocation within any URM run is "
                      f"{_fmt(max(memory_ranges) * 2**20, 1)} KiB. "
                      "This includes intermediate steps, not just the first and last.", ""]
        upstream_ranges = [max(trace) - min(trace) for _, r in pairs.values()
                           for trace in [r.get('memory_trace_gib', [])] if trace]
        if upstream_ranges:
            lines += [f"The largest step-end allocation change among compared upstream runs is "
                      f"{_fmt(max(upstream_ranges) * 2**20, 1)} KiB.", ""]
    lines += ["Full configs, losses, memory traces and source hashes are recorded in the per-model "
              "JSONs under [sweep/](sweep/) and [upstream/](upstream/). "
              "See [how to reproduce the benchmark](../docs/benchmark.md#running).", "",
              "## What the checks establish", "",
              "Checkpoint resume is tested on smaller models without compilation. Full-size runs separately "
              "check losses and gradient norms at every update. Output-distribution comparisons (KL) are "
              "recorded in JSON where available. Ten training updates do not establish long-run convergence "
              "or equivalence to complete upstream models and serving workloads. "
              "See the [evidence policy](../docs/evidence.md).", ""]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-dir', default='results/sweep')
    parser.add_argument('--upstream-dir', default='results/upstream')
    parser.add_argument('--out', default='results/report.md')
    args = parser.parse_args()
    ours, upstream = _load(args.sweep_dir), _load(args.upstream_dir)
    Path(args.out).write_text(render(ours, upstream))
    print(f'[report] wrote {args.out}')


if __name__ == '__main__':
    main()
