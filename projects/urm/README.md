# Project III: Unified Routed Mixer

[Read the complete research proposal](../../docs/research-program.md#proposal-iii).

URM is the first active project in the Global Liquid SDM program. It asks
whether attention, expert routing, parameter-token mixing, linear recurrence,
and sparse memory access can share a constrained execution contract while still
lowering to competitive specialized kernels.

## Current milestone

The routed-reduction/compiler-validation tranche remains complete and frozen.
Phase 3 is now active with one bounded family-expansion result: an external
baseline integration of the original authors' Sparse Delta Memory repository.
The prior milestone plus this adapter have been validated on an NVIDIA A10G
CUDA host:

- a typed `MixerSpec` contract;
- a dependency-light NumPy correctness oracle;
- deterministic dense, masked, top-k, and write-merge semantics;
- canonical specs for attention, MoE, parameter-token, recurrence, and GL-SDM;
- compositional detail specs for advanced MoE, recurrent, and learned
  sparse-attention flavors;
- an explicit baseline matrix and benchmark shape grid;
- a validated Triton routed-reduction backend (forward 2.6-3.0x and backward
  1.2-1.6x faster than the transparent PyTorch baseline across committed
  cases), with roofline profiling, measured device-limit denominators;
- a four-level dense-causal-attention comparator (methodology v2: no cloning
  in timed regions, paired interleaved direct-vs-adapter sampling with
  bootstrap-CI overhead distributions) whose URM dispatch overhead stays within
  a +2.32% worst steady-state median (worst bootstrap-CI upper bound +4.53%;
  values derived from results/attention/dense-causal.json); and
- a four-level FLA gated-delta-rule comparator (fp32 recurrent oracle, eager
  recurrence, pinned flash-linear-attention 0.5.2 direct, and the same calls
  behind a typed URM adapter) with separate prefill and token-by-token
  decode regimes and the frozen contract in docs/fla-gated-delta-rule.md; and
- a four-level original Sparse Delta Memory comparison using pinned commit
  `183e7df809131b80ad4393741029d0f20fc3640b`: exact product-key traces, sparse
  reads, ordered gated updates, persistent decode state, direct upstream calls,
  and the identical methods behind a typed URM adapter. This is an **SDM
  baseline integration**, not a native/page-local URM kernel; see
  [the frozen SDM contract](docs/sparse-delta-memory.md); and
- a layered compiler (docs/compiler-charter.md): typed semantic IR over logical
  domains with explicit locality/effect models, two verified reparameterization
  rules with deterministic traces, a simulated routing-to-communication
  planner, an analytical cost model, and a NAS-facing compilation API - proven
  by a CODA-inspired routed-reduction row-scale epilogue prototype
  (results/compiler/) whose fused plan avoids materializing `base` entirely; and
- compiler-to-kernel schedule integration closure: `UrmCompiler.compile()` now
  runs a candidate-bound schedule search (`compiler/search.py`: Z3 solve or
  deterministic heuristic fallback -> independent verification -> optional GPU
  compile probe -> bounded nogood/retry), and the selected
  `RoutedEpilogueLaunchConfig` is serialized into the executable plan and
  drives the PRODUCTION Triton anchor launchers (benchmarks call the same
  implementations); the committed selection artifact reports genuine raw-sample
  medians and p95s with seeded interleaved rounds; and
- schedule-measurement closure: discovery stability is explicitly exploratory,
  while `routed_epilogue_confirmation.py` freezes the shortlist and reference,
  performs randomized paired AB/BA measurements in fresh processes, fails
  closed on persistent sentinel drift, retains full-precision raw blocks and
  per-run provenance, and applies a hierarchical-bootstrap upper-bound rule.
  The committed confirmation artifact classifies 3 of 8 schedules inside the
  2.5% practical margin, selects the deterministic
  `block_d=128/num_warps=4/num_stages=1/per_query/segmented` representative,
  and correctly excludes the solver-selected schedule (6.42% confidence upper
  bound).

The routed-reduction/compiler-validation tranche is still closed and was not
reopened. The new SDM slice does **not** claim that deferred MoE, Mamba,
FlexAttention, distributed-runtime, transactional GL-SDM, or native page-local
SDM lowerings exist.

See [the compiler charter](docs/compiler-charter.md), the
[CODA retrospective](docs/coda-retrospective.md),
[Triton backend preparation](docs/triton-backend.md),
[the optimization and profiling report](docs/triton-optimization-report.md),
and `benchmarks/validated-environment.json` for exact versions.

Linear recurrence is represented but deliberately not executed by the generic
oracle: ordered scan semantics require a dedicated lowering.

## Quick start

From this directory:

```powershell
python -m pytest
$env:PYTHONPATH = "src"
python benchmarks/reference_smoke.py --family attention
```

The smoke timer checks that the harness runs. It is not a publishable performance
measurement. GPU benchmarking requires a Linux CUDA/ROCm environment with the
appropriate optional dependencies.

## Project layout

```text
projects/urm/
|-- benchmarks/
|   |-- cases.toml               # canonical shape and trace grid
|   |-- routed_reduce.py         # CUDA-event PyTorch/Triton comparison
|   |-- routed_scale_epilogue.py # materialized vs fused epilogue prototype
|   |-- epilogue_schedules.py    # SchedulePoint adapter over PRODUCTION kernels
|   |-- measurement.py           # raw samples, tested percentiles, interleaving
|   |-- routed_epilogue_selection.py # solver-guided schedule selection (GPU)
|   |-- routed_epilogue_stability.py # exploratory fresh-process rank stability
|   |-- routed_epilogue_confirmation.py # paired confirmatory schedule decision
|   |-- placement_selection.py   # solver-guided simulated mesh placement
|   |-- unsat_diagnostics.py     # representative impossible problems + cores
|   |-- compilation_matrix.py    # NAS-facing preset compilation matrix
|   |-- provenance.py            # shared artifact provenance capture
|   |-- compare_results.py       # constraint checker (median/p95/memory gates)
|   |-- measure_device_limits.py # measured HBM bandwidth and compute peaks
|   |-- profile_roofline.py      # MFU/MBU and per-kernel roofline profiling
|   |-- dense_attention.py       # four-level dense causal attention comparator
|   |-- sparse_delta_memory.py   # original-SDM direct/adapter comparison
|   |-- result-schema.json       # benchmark result schema (+ optional profiling)
|   |-- profiling-schema.json    # roofline artifact schema
|   |-- attention-result-schema.json  # attention comparator schema
|   |-- sparse-delta-memory-result-schema.json # SDM comparator schema
|   |-- compiler-epilogue-schema.json # fused-epilogue comparison schema
|   |-- compilation-matrix-schema.json # NAS compilation matrix schema
|   |-- routed-epilogue-selection-schema.json # schedule selection schema
|   |-- routed-epilogue-stability-schema.json # exploratory stability schema
|   |-- routed-epilogue-confirmation-schema.json # paired confirmation schema
|   |-- placement-selection-schema.json       # placement selection schema
|   |-- unsat-diagnostics-schema.json         # unsat diagnostics schema
|   |-- validated-environment.json    # exact validated dependency versions
|   |-- triton_preflight.py      # environment and GPU readiness report
|   `-- reference_smoke.py       # dependency-light harness smoke test
|-- docs/
|   |-- architecture.md          # target contract and non-goals
|   |-- baselines.md             # comparator catalog and scope rules
|   |-- benchmarking.md          # correctness, measurement, and acceptance gates
|   |-- compiler-charter.md      # normative compiler invariants (v1)
|   |-- kernel-generation.md     # normative kernel-generation pipeline
|   |-- coda-retrospective.md    # CODA strategy mapping and adoption decisions
|   |-- triton-backend.md        # routed-reduction v1 contract and workflows
|   |-- triton-optimization-report.md # optimization, profiling, comparator report
|   |-- fla-gated-delta-rule.md  # frozen FLA gated delta-rule contract (v1)
|   `-- sparse-delta-memory.md   # frozen original-SDM adapter contract (v1)
|-- results/
|   |-- baseline/ ... final/     # before/after benchmark matrices per change
|   |-- profiling/               # committed roofline summaries
|   |-- device-limits.json       # measured bandwidth / FP32 / BF16 peaks
|   |-- attention/               # dense causal attention comparison artifacts
|   |-- fla-gated-delta-rule/    # gated delta-rule comparison artifacts
|   |-- sparse-delta-memory/     # original-SDM comparison artifact
|   `-- compiler/                # epilogue prototype, compilation matrix,
|                                #   solver selection/stability/confirmation
|-- src/urm/
|   |-- backend.py               # explicit backend protocol and registry
|   |-- adapters/                # pinned upstream adapters (dense attention,
|   |                            #   gated delta rule, original SDM) + references
|   |-- backends/                # reference and optimized routed-reduction backends
|   |-- compiler/                # semantic IR, locality/effects, verified
|   |                            #   rewrites, anchors, planner, cost model,
|   |                            #   bounded schedule search (compiler/search.py)
|   |-- ir.py                    # restricted typed operator contract
|   |-- presets.py               # canonical semantic-family specifications
|   |-- routed_reduction.py      # frozen v1 tensor/capability contract
|   |-- triton_kernels/          # lazy GPU kernels and autograd wrappers
|   `-- reference.py             # slow NumPy oracle and transactional merge
`-- tests/                       # contract, oracle, adapter, schema, compiler tests
```

## Baseline strategy

The [baseline catalog](docs/baselines.md) separates four comparison levels:

1. a semantic oracle;
2. a transparent framework implementation;
3. the best applicable pinned upstream kernel; and
4. the URM lowering.

Initial optimized comparators are PyTorch SDPA and FlexAttention,
FlashAttention, Flash Linear Attention, Mamba selective scan, MegaBlocks,
SonicMoE, and the authors' original
[Sparse Delta Memory](https://github.com/facebookresearch/sparse-delta-memory)
implementation where the hardware supports them. FLA covers linear recurrent
mixers; it is not used as the SDM kernel source. The project adds one maintained
adapter per distinct semantic/kernel family, not one adapter for every published
model.

## Phased execution

### Phase 0 - contract and oracle (complete)

- Freeze v0 routing, normalization, mutation, residency, and collision enums.
- Validate dense/masked reduction, stable top-k routing, and transactional merge.
- Make result metadata and benchmark cases reproducible.
- Validate the routed-reduction Triton backend on the target Linux CUDA host.
- Optimize it under the frozen contract and profile the result
  ([report](docs/triton-optimization-report.md)).

### Phase 1 - framework baselines (closed for the validated tranche)

- PyTorch SDPA math and flash-dispatch adapters are complete for dense causal
  attention via `src/urm/adapters` and `benchmarks/dense_attention.py`; the URM
  dispatch overhead gate of 5% median is met with a +2.32% worst steady-state
  median measured on every
  steady-state shape under paired interleaved sampling.
- The FLA gated-delta-rule comparator is complete as a typed four-level,
  prefill-plus-decode comparison; see `docs/fla-gated-delta-rule.md`.
- Correctness, route, timing, memory, and provenance metadata are captured in
  validated result schemas for the covered families.
- Transparent top-k MoE and sparse-memory family adapters remain in the
  expansion backlog; they are not prerequisites for this tranche's closure.

### Phase 2 - specialized kernels (closed for the routed-reduction tranche)

- The routed-reduction v1 forward/backward Triton family is implemented,
  optimized, roofline-profiled, and preserved behind a frozen tensor contract.
- Direct-vs-URM dispatch comparisons are complete for dense causal attention
  and FLA gated delta rule; routed reduction is not misrepresented as a
  replacement for those kernels.
- FlexAttention/block-sparse, Mamba-family, optimized MoE, original SDM, and
  page-local memory lowerings are deferred family-expansion slices. Each must
  enter through its own frozen contract and upstream comparator.

### Phase 2.5 - compiler architecture and schedule validation (complete)

- Layered semantic/execution IR with locality and effect models
  (docs/compiler-charter.md).
- Verified reparameterization: registered rules, deterministic traces,
  positive and negative differential tests.
- CODA-inspired routed-reduction row-scale epilogue prototype with measured
  avoided materialization (results/compiler/routed-scale-epilogue/).
- Simulated routing-to-communication planner and analytical cost model.
- NAS-facing compilation API plus a preset compilation matrix artifact.
- Compiler-to-kernel schedule integration closure: verified schedules are
  selected inside `compile()`, probed on GPU when available, serialized in
  the executable plan, and executed by the production Triton anchor
  (results/compiler/solver/routed-epilogue-selection.json).
- Cross-run measurement closure: the stability artifact is explicitly an
  exploratory discovery diagnostic, while the confirmation artifact is the
  canonical paired equivalence decision with raw blocks, drift gates,
  cross-child provenance validation, and a conservative fallback.
- Known limitation carried forward outside this closed tranche:
  `decode_placement()` keeps only the first owner for replicated
  items and `placement_metrics()` divides item bytes by the replication
  factor; every committed placement case uses replication factor 1, so no
  committed artifact is affected. Multi-owner replication decoding and byte
  accounting require a separate systems-path change.

### Phase 3 - systems and family-expansion path (active)

- Complete: external original-SDM baseline adapter, frozen typed contract,
  compiler capability selection, differential tests, and dedicated benchmark.
- Fuse overlay composition and reduction.
- Add page grouping, prefetch, recurrent address reuse, and distributed sharding.
- Evaluate GL-SDM traces without making GL-SDM's architectural success a URM
  dependency.

## Acceptance gate

URM succeeds when the restricted contract covers the target families without
arbitrary tensor escape hatches and each family can select or generate a
competitive specialized backend. The working targets and stop conditions are in
the [benchmark protocol](docs/benchmarking.md). Closure of the current tranche
establishes that standard for routed reduction and the covered adapters; it is
not evidence that every deferred family already meets it.
