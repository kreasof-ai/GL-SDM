# URM — agent notes

Semantic-to-execution compiler for routed sequence models. K1 (streamed reduction),
K2 (compact fixed-address state), K3 (indexed mutable state) mixer families.
Architectures compose public kernel calls with external projections/MLP outside
`src/urm`.

## Layout

- `src/urm/` — frontend, ir, compiler, runtime, backends (the BACKEND: see freeze below).
- `architectures/` — external model modules, one per catalog row, HF-modeling
  granularity (layer / block / model / residual design).
- `train/` — the benchmark harness: `registry.py` (the catalog), `model.py`
  (`URMDecoderLM` + `MixerSpec`), `harness.py` (train/MFU/gates), `sweep.py`
  (native-row sweep driver), `upstream.py` (upstream baselines), `report.py` (join).
- `tests/` — per-architecture parity gates (`test_architectures_<row>.py`) + the
  training-harness gate (`test_training_harness.py`).
- `extra/comparators/` — pinned upstream adapters (revision/SHA-verified,
  AST-extracted). Forward-parity adapters, not trainable modules. Also in
  `extra/`: `provision_comparators.py` (pins provisioning),
  `architecture-coverage.json` + `coverage_register.py` (the source-identity
  register and its generated doc), `recipe_catalog.py` (recipes access).
- `results/sweep/`, `results/upstream/` — committed per-row campaign JSONs +
  logs feeding `results_report.md`. `data/finewebedu10B/` — the training shards.
- `/tmp/urm-comparator-pins/` — the pinned upstream checkouts (fla, mamba, sdm/lingua,
  tucker, tpa, samba, kata, differential, hopfield, conformer, longformer,
  sparse_transformer, hla_higher_order (empty — paper only), …).

## The registry and granularity (train/registry.py, train/model.py)

Every row is a `MixerSpec` with a faithful HF-modeling `granularity`:

- `mixer` (default) — builder(dim, heads, head_dim, intent, target) → [B,T,C] mixer
  module in the shared DecoderBlock.
- `schedule` — builder(layer_idx, …) → per-layer mixer CHOICE (interleaved hybrids;
  samba = mamba2-K2 even / attention odd).
- `block` — builder(…) → full decoder block owning norms/residual/MLP (patention =
  tokenformer: Q/K/V/O and the MLP are all Pattention).
- `residual` — builder(dim, layers, …) → a residual design threaded across standard
  dense-attention blocks (attnres: depth-domain softmax aggregation; the row under
  test is the residual law, not a mixer).

`tier` is `"native"` (production URM kernel) or `"reference"` (mamba1 only — accepted
charter debt). `atomic_backward=True` marks rows whose native backward uses relaxed
`tl.atomic_add` (indexed-K1 clients: dsa, longformer, sparse_transformer, nsa; K3 via
`stateful`) — the checkpoint gate certifies those by loss-trajectory + bitwise
round-trip structure, not bitwise param equality.

## Backend freeze protocol

The backend (`src/urm/`) is FROZEN for the benchmark campaign. Sole exception: a
demonstrated correctness bug, whose fix must re-verify that kernel family's tests and
re-run that row's sweep before the report ships. Frontend (`architectures/`) and
harness (`train/`) work continue freely.

## Running the benchmark (from projects/urm, PYTHONPATH=src:.)

```sh
python -m pytest tests/ -q                                   # full suite (676 passed / 41 skipped)
python -m train.sweep --out-dir results/sweep                # native rows (51)
python -m train.upstream --out-dir results/upstream --subprocess   # upstream baselines (49 runnable + 1 blocked)
python -m train.report --sweep-dir results/sweep --upstream-dir results/upstream --out results_report.md
```

Sweep driver: per-row subprocess isolation, 8192→2048→1024 microbatch fallback on
CUDA OOM (the OOM text is sniffed from the subprocess log, not the exit path); the
upstream driver has the same ladder. Config: width=768, layers=9, heads=12,
head_dim=64, seq=512, finewebedu, 10 steps. MFU denominator: A10G adopted achievable
bf16 peak = 70 TFLOPS. Docs live in `docs/` (mirrors the code — see its README).

## Conventions

- Native rows compile through the opaque-op boundary; upstream rows run EAGER
  (fla chunk kernels fail torch.compile/Inductor here).
- No flash-attn installs (too heavy for this box); bypass to SDPA. mamba_ssm has no
  prebuilt torch-2.14/cu130 wheel — mamba rows use fla's own Triton mamba layers.
- No source builds that could destabilize the machine (the lingua SDM CUDA extension
  is environment-blocked: nvcc 12.9 vs cu13 headers; sdm upstream runs the pinned
  law in torch, labeled reference-implementation).
- Upstream baselines are labeled `production-kernel` vs `reference-implementation`
  in the report; granularity-matched to the URM row.
- hla has NO upstream (empty pin) — the sole principled exclusion.
  log_linear_mamba2's upstream chunk kernel exceeds the A10G SMEM envelope and ships
  no reference variant — recorded environment-blocked (`UPSTREAM_BLOCKED`), not
  fabricated.
- fla Triton JIT compiles are slow on first use (>120s/layer); the shared
  `~/.triton` cache warms subsequent runs. Pass generous shell timeouts.
