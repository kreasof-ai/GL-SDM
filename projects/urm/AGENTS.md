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
  logs feeding `results/report.md`. `data/finewebedu10B/` — the training shards.
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

The user's explicit follow-up authorizes internalizing the chunked decayed-delta
optimization into the existing native K3 provider/compiler path. This exception
requires name-independent selection, independent-client admission, full-state/VJP
gates, and new matched SDM measurements. It does not unfreeze other backend families
or permit SDM-specific branches in the Torch reference backend.

## Running the benchmark (from projects/urm, PYTHONPATH=src:.)

```sh
python -m pytest tests/ -q                                   # full suite
python -m train.sweep --out-dir results/sweep                # native rows (51)
python -m train.upstream --out-dir results/upstream --subprocess   # production baselines + explicit unavailable records
python -m train.report --sweep-dir results/sweep --upstream-dir results/upstream --out results/report.md
```

Both drivers isolate every row in a subprocess. Effective batch and microbatch
remain 8192 tokens; memory-heavy rows use matched activation checkpointing.
Smaller microbatches are opt-in diagnostics and excluded from production comparisons.
Config: width=768, layers=9, heads=12,
head_dim=64, seq=512, finewebedu, 10 steps. MFU denominator: A10G adopted achievable
bf16 peak = 70 TFLOPS. Docs live in `docs/` (mirrors the code — see its README).

## Conventions

- Both arms compile the surround; Python plans and unsupported upstream kernels
  execute behind eager boundaries. Dynamic routing is also an eager boundary.
- No flash-attn installs (too heavy for this box). Mamba-2's pinned pure Triton SSD
  kernel is loaded without its optional CUDA-extension package initializer.
- No source builds that could destabilize the machine. The user-authorized SDM
  production baseline uses an isolated, pinned CUDA 13 toolkit under
  `~/.cache/urm/sdm-cuda13` and extension cache under `~/.cache/urm/sdm-extensions`;
  it does not replace the base CUDA/Python environment. Provision it with
  `python extra/provision_sdm_cuda.py`; see `docs/sdm-optimization.md`.
- The SDM benchmark executes its complete public K3 graph (`public_path=True`).
  The native provider owns the generic guarded chunk schedule. Its actual pinned
  CUDA/Triton comparator stays external and shares the native routes/projections.
- Production comparisons require finite training, checkpoint gates, matching
  configs/environment/source fingerprints, and production kernels. Research and
  failed implementations have no paired throughput in the report.
- hla has NO upstream (empty pin). Research-only and environment-blocked
  baselines are also explicit production exclusions.
  Both log-linear rows use the pinned chunk kernel with bf16 operands, one
  pipeline stage, and independent-head batches to fit the A10G SMEM envelope.
- fla Triton JIT compiles are slow on first use (>120s/layer); the shared
  `~/.triton` cache warms subsequent runs. Pass generous shell timeouts.
