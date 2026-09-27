# Evidence and claim policy

What a measurement in this repo is allowed to claim. The machine catalog is
`extra/architecture-coverage.json`; the benchmark records are
`results/sweep/` + `results/upstream/` + `results/report.md`.

## Five independent verdicts

Every architecture/mode/shape/dtype/device reports these **separately**:

1. **Represented** — a closed semantic descriptor preserves the exact equation,
   effects and state/cache contract; unsupported fields decline.
2. **Reference executable** — the public graph executes through independent
   numpy/torch references with output, parameter, route and state-gradient checks.
3. **Native qualified** — a URM-owned provider selected by the public compiler runs
   forward/backward as claimed and matches the reference. An upstream adapter is a
   separate library tier.
4. **Performance qualified** — the complete timed boundary meets a predeclared
   paired throughput/memory gate, with raw samples. Training, prefill and decode are
   separate.
5. **Source-model qualified** — an external complete model module matches the pinned
   source unit's layer arrangement, parameters, logits, gradients and cache
   continuation. Fragment parity cannot set this verdict.

Do not collapse these into one "coverage" number. Any result must state the timed
boundary (kernel fragment / harness decoder / complete source model), provider tier
(native / library / reference), revision, hardware/software environment, precision
policy, intent mode, shape and state initialization. Missing upstream modes are
`upstream_unavailable`, not passes.

## Current status at HEAD

| Scope | Evidence | Claim allowed |
|---|---|---|
| Registry | 52 rows: 51 native-tier + 1 reference-tier (`mamba1`, charter debt) | Native execution of the 51 rows' mixers on the A10G |
| Parity gates | `tests/` — 676 passed / 41 skipped; per-architecture gates verify each row against its pinned oracle | The rows' equations match the pinned sources at tested shapes |
| Training-harness benchmark | 51/51 native rows complete the 10-step 100M-class run with checkpoint parity; 49 upstream baselines joined | Harness-level MFU/throughput/memory comparison at the stated config — **not** source-model performance qualification (verdict 5 is not claimed) |
| Upstream coverage | 50/51 native rows have an upstream baseline (31 production-kernel, 18 reference-implementation, 1 environment-blocked); `hla` has no upstream anywhere | Row-level comparison at matched granularity with the labeled tier |

### What is *not* claimed

- **No source-model qualification** (verdict 5): the harness trains a generic
  decoder surround with the row's mixer — not the upstream's complete model
  arrangement.
- **No decode/prefill-serving numbers**: the benchmark measures training steps only.
- **KL gate** is wired for the 23 rows with a registry comparator; `—` elsewhere is
  "no comparator wired", not failure and not success.
- **Upstream asymmetry**: upstream rows run eager (their kernels fail torch.compile
  here) while URM rows compile — the upstream numbers are a lower bound on upstream
  throughput, stated openly in the report.
- Six native rows (comba, dplr, gated_delta_product, gdn2, iplr, rwkv7) diverge to
  NaN loss within the 10 steps at width 768 — a training-stability finding; their
  MFU is still a valid measurement of executed FLOPs and is reported as such.

## Historical records

The prior qualification harness was retired in full (git history retains it): the
release-gate aggregation (`release_gate.py`, `release_coverage.py`,
`production-matrix.json`), the per-workload drivers and profiling tools
(`dense_attention.py`, `gated_delta_rule.py`, `sparse_*.py`, `pretraining_step.py`,
`measurement.py`, …), their committed artifacts (`results/attention/`,
`results/compiler/`, `results/qualification/`, `results/unified-mixer/`,
`results/sparse-*/`, `results/device-limits.json`) and the schema gates
(`test_artifact_schemas.py` and the 14 `*-schema.json` files), plus the earlier
stale-doc deletions (`results/validation/`, `docs/validation/` and their
generators). The architecture register's per-row parity/profile evidence fields,
which pointed at those artifacts, were stripped (schema v3 → v4); the register
keeps its live role — source identity, pinned revisions and construction backlog.
The current evidence is the training-harness campaign: `results/sweep/`,
`results/upstream/`, `results/report.md`, gated by `tests/` (per-architecture
parity against `extra/comparators/`, plus the harness gates).
