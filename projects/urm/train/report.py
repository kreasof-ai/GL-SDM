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
SHARED_FRONTEND_ROWS = {
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
    lines = ["# URM training measurements — corrected campaign", "",
             f"**Coverage.** {len(ok)}/{len(ours)} URM rows have finite training trajectories and passing checkpoint gates; "
             f"{len(pairs)} have eligible production-kernel measurements.", "",
             "Each row trains a decoder surround on finewebedu for 10 measured steps after two full optimizer warmup steps. "
             "This measures the training harness; it does not claim source-model or serving parity. "
             "See the [evidence policy](../docs/evidence.md).", "",
             "**Protocol.** width=768, layers=9, heads=12, head_dim=64, sequence=512, vocab=50304. "
             "Effective batch and microbatch are both 8192 tokens. Both arms compile the surround; "
             "Python plan dispatch and unsupported upstream kernels remain eager boundaries. "
             "Optimizer roles and clipping are identical in both arms. bf16 autocast with fp32 kernel accumulation; "
             "timing is synchronized before and after measurement. "
             "MFU is an approximate parameter/state FLOP estimate divided by the adopted 70 TFLOPS A10G peak, "
             "not a hardware utilization counter.", "",
             "Memory-heavy rows use the same explicit activation-checkpointing policy in both arms. "
             "Based uses its upstream default of 16 query/key features and 64 value channels. "
             "OOM retries are disabled in this campaign; failures are recorded without substituting a smaller batch. "
             "Non-finite losses or gradient norms fail the run.", ""]
    lines += ["TDA and Differential Attention training use existing public native calls with external "
              "differentiable merges to retain projection and mixing-weight gradients. "
              "Two core autograd-wrapper fixes retain only flags/shapes rather than bias/mask or gate tensors, "
              "preventing graph retention after backward; kernel math is unchanged.", ""]
    if "sdm" in ok and ok["sdm"].get("sdm_execution", {}).get("schedule") == "native-k3":
        lines += ["SDM executes its complete public route/update/read graph through the native K3 provider. "
                  "The compiler selects a generic guarded chunk schedule (maximum chunk size 128) by typed "
                  "state properties, including decay and all operand/state gradients; the ordered scan remains available. "
                  "The external SDM state-schedule file has been removed. Its shared-frontend baseline calls the "
                  "unmodified pinned Meta CUDA/Triton kernels (chunk size 64), built with an isolated matching "
                  "CUDA toolkit. The independent Torch reference remains unchanged. "
                  "See [SDM measurements and historical MFU accounting](../docs/sdm-optimization.md). "
                  "Accepted non-SDM measurements retain their original source fingerprints; each paired row "
                  "still requires identical fingerprints in both arms.", ""]
    if ok:
        env = next(iter(ok.values())).get("environment", {})
        hashes = sorted({r['source_fingerprint'][:12] for r in ok.values()})
        lines += [f"**Provenance.** torch {env.get('torch', '?')}, {env.get('device', '?')}; "
                  f"measurement version 2; source fingerprint(s) `{', '.join(hashes)}`. "
                  "Each JSON contains its actual config, full source hash, loss trajectory, memory trajectory, "
                  "and attempted microbatches. FLA and Mamba production adapters verify their pinned sources.", ""]
        memory_ranges = [max(trace) - min(trace) for r in ok.values()
                         for trace in [r.get('memory_trace_gib', [])] if trace]
        if len(memory_ranges) == len(ok):
            lines += [f"**Memory audit.** The largest within-run step-end allocation range across verified URM rows "
                      f"is {_fmt(max(memory_ranges))} GiB over the measured steps. "
                      "This checks intermediate steps as well as the first-to-last drift.", ""]
        upstream_ranges = [max(trace) - min(trace) for _, r in pairs.values()
                           for trace in [r.get('memory_trace_gib', [])] if trace]
        if upstream_ranges:
            lines += [f"The largest step-end allocation range among eligible upstream runs is "
                      f"{_fmt(max(upstream_ranges) * 2**20, 1)} KiB.", ""]
    lines += ["## URM rows", "",
              "`slow` marks approximate MFU below 0.10. `ckpt` means activation checkpointing is enabled; "
              "checkpoint correctness is reported separately. Memory drift is the last minus first step-end allocation.", "",
              "| row | params | MFU | tok/s | checkpoint gate | peak GiB | loss | memory drift GiB | flags |",
              "|---|---:|---:|---:|:---:|---:|---:|---:|---|"]
    for name, r in sorted(ours.items()):
        if not verified(r):
            reason = r.get("error", "unverified/legacy measurement").replace("|", "/").replace("\n", " ")
            lines.append(f"| {name} | — | — | — | — | — | — | — | {reason} |")
            continue
        flags = []
        if r['mfu'] < PATHOLOGICAL_MFU:
            flags.append("slow")
        if r['config'].get('activation_checkpointing'):
            flags.append("ckpt")
        if r.get('microbatch_fallback'):
            flags.append(f"diagnostic mb{r['microbatch_fallback']}")
        if r.get('sdm_execution', {}).get('schedule') == 'torch-chunked':
            flags.append("external state schedule")
        if r.get('sdm_execution', {}).get('schedule') == 'native-k3':
            flags.append("native K3 chunk schedule")
        memory = r.get('memory_trace_gib', [])
        drift = memory[-1] - memory[0] if memory else None
        lines.append(f"| {name} | {r['params']:,} | {_fmt(r['mfu'])} | {_fmt(r['throughput_tokens_s'], 0)} | "
                     f"✓ | {_fmt(r['peak_memory_gib'], 2)} | {_fmt(r['final_loss'])} | {_fmt(drift)} | {', '.join(flags) or '—'} |")
    slow = [n for n, r in ok.items() if r['mfu'] < PATHOLOGICAL_MFU]
    if slow:
        lines += ["", f"### Remaining low-MFU rows ({len(slow)})", "",
                  "These measurements remain visible. The listed work explains the execution path, "
                  "and does not establish that the implementation is optimal.", ""]
        lines += [f"- **{n}**: {SLOW_NOTES.get(n, 'requires further profiling')} (MFU {_fmt(ok[n]['mfu'])})."
                  for n in sorted(slow)]
    lines += ["", "## Production-kernel measurements", "",
              "Only verified runs with matching shapes, effective batch, microbatch, precision, compilation policy, "
              "checkpointing policy, environment, and source fingerprint enter this table. "
              "There is no reference-kernel replacement after an upstream failure. "
              "`shared` uses a common external frontend; `family` is an architecture-family baseline whose "
              "projections or other mixer-side layers can differ. Parameter counts are shown explicitly; "
              "family measurements are not isolated kernel speedup claims.", "",
              "| row | scope | MFU (URM / upstream) | tok/s (URM / upstream) | peak GiB (URM / upstream) | params (URM / upstream) |",
              "|---|---|---:|---:|---:|---:|"]
    for n, (o, u) in sorted(pairs.items()):
        scope = 'shared' if n in SHARED_FRONTEND_ROWS else 'family'
        lines.append(f"| {n} | {scope} | {_fmt(o['mfu'])} / {_fmt(u['mfu'])} | "
                     f"{_fmt(o['throughput_tokens_s'], 0)} / {_fmt(u['throughput_tokens_s'], 0)} | "
                     f"{_fmt(o['peak_memory_gib'], 2)} / {_fmt(u['peak_memory_gib'], 2)} | {o['params']:,} / {u['params']:,} |")
    lines += ["", "### Production adapters", "",
              "- Mamba-2 uses the unmodified pinned SSD Triton package without importing its optional CUDA extension; "
              "RWKV-7 uses the pinned chunk kernel with `chunk_size=16`.",
              "- Comba, GDN2, DeltaProduct, DPLR, RWKV-7, Mamba-2 and both log-linear rows share the URM external frontend. "
              "The upstream call replaces only the mixer kernel.",
              "- Samba uses the same Mamba-2/RoPE schedule with pinned SSD and production SDPA. "
              "Raven shares the eight-slot/top-k-two deterministic frontend and uses pinned chunk GSA. "
              "Equal duplication of all slots meets its 16-slot backward minimum while preserving outputs and gradients.",
              "- TDA supplies contiguous head batches and gradients, applies the native query scaling, "
              "and matches the registered differential merge (identical paths with lambda=0.5). "
              "Supported Triton launch options select IEEE fp32 dots and one pipeline stage.",
              "- Log-linear attention uses independent heads as single-group batches and an A10G one-stage pipeline. "
              "Operands and level scales are bf16, and the last scale repeats for the capped bank. "
              "The upstream checkout remains unmodified.", "",
              "- SDM uses the verified original Meta sparse-IP/gather CUDA extensions and Triton WY kernels. "
              "Partition-local identity padding, autocast isolation, and a saved terminal snapshot adapt the "
              "production API without changing its source or substituting a reference kernel.", "",
              "## Upstreams excluded from production comparison", "",
              "Unavailable kernels, research implementations, failed training, and mismatched measurements are listed "
              "without paired throughput. Reference implementations can be requested as separate diagnostics using "
              "`train.upstream --include-reference`; they remain excluded from the table above.", ""]
    for n, u in sorted(upstream.items()):
        if n in pairs:
            continue
        reason = u.get('reason') or u.get('error') or comparison_reason(ours.get(n, {}), u)
        lines.append(f"- **{n}**: {reason}.")
    for n in sorted(set(ours) - set(upstream)):
        lines.append(f"- **{n}**: no upstream measurement available.")
    lines += ["", "## Validation limits", "",
              "Checkpoint gates use reduced eager models and certify resume behavior, not full-scale accuracy. "
              "Finite full-scale loss and gradient norms are checked separately at every update. "
              "KL is retained in the per-row JSON wherever a comparator is wired; absent KL is not a parity claim. "
              "These ten-step runs do not establish long-run convergence.", ""]
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
