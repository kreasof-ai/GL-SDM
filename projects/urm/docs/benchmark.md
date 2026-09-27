# The training-harness benchmark

Mirrors `train/` and the committed result JSONs. This is a **training-harness
measurement**: 10-step training runs of a 100M-class decoder LM per architecture
row, on real data, with correctness gates — not a production serving benchmark and
not a source-model parity claim (see [evidence.md](evidence.md)).

## Configuration

| Field | Value |
|---|---|
| Model | width=768, layers=9, heads=12, head_dim=64, vocab=50304 |
| Data | finewebedu (`data/finewebedu10B/finewebedu_train_*.bin` via `train/data.py`) |
| Steps | 10 |
| Microbatch | 8192 tokens primary; OOM fallback ladder 2048 → 1024 |
| MFU denominator | A10G adopted achievable bf16 peak = **70 TFLOPS** |
| Precision | bf16 autocast, fp32 accumulation in kernels |
| URM rows | compiled through the opaque-op boundary |
| Upstream rows | eager (fla chunk kernels fail torch.compile/Inductor here) |
| Hardware/software | NVIDIA A10G 22 GiB, torch 2.14.0+cu130, triton 3.8.0 |

## Drivers

Run from `projects/urm` with `PYTHONPATH=src:.`:

```sh
python -m train.sweep --out-dir results/sweep                      # 51 native rows
python -m train.upstream --out-dir results/upstream --subprocess   # 49 upstream baselines
python -m train.report --sweep-dir results/sweep \
    --upstream-dir results/upstream --out results_report.md        # the joined report
```

- `train/sweep.py` — one subprocess per row (CUDA memory isolation), 8192→2048→1024
  microbatch fallback on CUDA OOM (the OOM text is sniffed from the subprocess log),
  per-row JSON + log; failures record error JSONs so the sweep completes honestly.
- `train/upstream.py` — same isolation and fallback ladder for the baseline arm;
  each baseline is granularity-matched to the URM row and labeled
  `production-kernel` vs `reference-implementation`.
- `train/run.py` — single-row entry (`train.run --mixer <row>`).
- `train/harness.py` — the training loop, MFU accounting, and the gates.

## Gates

Every row reports, per run:

1. **Training completes** with finite or honestly-recorded-NaN loss (a NaN
   trajectory is a stability finding, still reported — the MFU numerator remains a
   valid measurement of the executed FLOPs).
2. **Checkpoint parity** — save at step N, reload, verify the resumed run. Rows on
   deterministic kernels must match bitwise; `atomic_backward`/stateful rows certify
   by bitwise state-dict round-trip plus loss-trajectory match (≤1e-2/step) —
   bitwise equality is unachievable by design under relaxed-atomic backward.
3. **KL(URM ‖ pinned upstream oracle)** on identical operands where the registry has
   a wired comparator (23 of 51 native rows); `—` where no comparator is wired.
4. **MFU / throughput / peak memory** — measured over the timed training steps.

## Protocol constraints

- **Backend freeze.** `src/urm/` is frozen for the campaign; the sole exception is a
  demonstrated correctness bug whose fix re-verifies that family's tests and re-runs
  that row (exercised once: the based int64 addressing fix).
- **No flash-attn installs** (too heavy for this box) — bypassed to SDPA.
  **No mamba_ssm** (no prebuilt torch-2.14/cu130 wheel) — mamba rows use fla's own
  Triton mamba layers. **No source builds** that could destabilize the machine
  (the lingua SDM CUDA extension is toolchain-blocked; sdm runs its pinned law in
  torch).
- The upstream arm's eager execution and any reference-implementation tiering are
  labeled per row in the report, never silently blended with production kernels.

## Current results

[results_report.md](../results_report.md) is generated from the committed JSONs
(`results/sweep/`, `results/upstream/`). Headline: 51/51 native rows complete;
49 runnable upstream baselines + 1 environment-blocked (`log_linear_mamba2`) +
`hla` with no upstream anywhere (empty pin). Native median MFU 0.28; 8 rows ≥ 0.40.
The report's tables carry the full context: NaN rows, pathological-MFU rows with
reasons, fallback microbatch markers, upstream tier/granularity labels, blocked and
missing upstreams.
