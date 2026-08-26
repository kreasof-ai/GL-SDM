# Project III: Unified Routed Mixer

[Read the complete research proposal](../../docs/research-program.md#proposal-iii).

URM is the first active project in the Global Liquid SDM program. It asks
whether attention, expert routing, parameter-token mixing, linear recurrence,
and sparse memory access can share a constrained execution contract while still
lowering to competitive specialized kernels.

## Current milestone

Milestone zero (semantics, oracle, benchmark discipline) is complete and
validated on an NVIDIA A10G CUDA host. The routed-reduction Triton backend is
optimized, profiled, and compared against a pinned production kernel:

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
- a layered compiler (docs/compiler-charter.md): typed semantic IR over logical
  domains with explicit locality/effect models, two verified reparameterization
  rules with deterministic traces, a simulated routing-to-communication
  planner, an analytical cost model, and a NAS-facing compilation API - proven
  by a CODA-inspired routed-reduction row-scale epilogue prototype
  (results/compiler/) whose fused plan avoids materializing `base` entirely.

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
|   |-- compilation_matrix.py    # NAS-facing preset compilation matrix
|   |-- compare_results.py       # constraint checker (median/p95/memory gates)
|   |-- measure_device_limits.py # measured HBM bandwidth and compute peaks
|   |-- profile_roofline.py      # MFU/MBU and per-kernel roofline profiling
|   |-- dense_attention.py       # four-level dense causal attention comparator
|   |-- result-schema.json       # benchmark result schema (+ optional profiling)
|   |-- profiling-schema.json    # roofline artifact schema
|   |-- attention-result-schema.json  # attention comparator schema
|   |-- compiler-epilogue-schema.json # fused-epilogue comparison schema
|   |-- compilation-matrix-schema.json # NAS compilation matrix schema
|   |-- validated-environment.json    # exact validated dependency versions
|   |-- triton_preflight.py      # environment and GPU readiness report
|   `-- reference_smoke.py       # dependency-light harness smoke test
|-- docs/
|   |-- architecture.md          # target contract and non-goals
|   |-- baselines.md             # comparator catalog and scope rules
|   |-- benchmarking.md          # correctness, measurement, and acceptance gates
|   |-- compiler-charter.md      # normative compiler invariants (v1)
|   |-- coda-retrospective.md    # CODA strategy mapping and adoption decisions
|   |-- triton-backend.md        # routed-reduction v1 contract and workflows
|   |-- triton-optimization-report.md # optimization, profiling, comparator report
|   `-- fla-gated-delta-rule.md  # frozen FLA gated delta-rule contract (v1)
|-- results/
|   |-- baseline/ ... final/     # before/after benchmark matrices per change
|   |-- profiling/               # committed roofline summaries
|   |-- device-limits.json       # measured bandwidth / FP32 / BF16 peaks
|   |-- attention/               # dense causal attention comparison artifacts
|   |-- fla-gated-delta-rule/    # gated delta-rule comparison artifacts
|   `-- compiler/                # epilogue prototype + compilation matrix
|-- src/urm/
|   |-- backend.py               # explicit backend protocol and registry
|   |-- adapters/                # pinned upstream adapters (dense attention,
|   |                            #   gated delta rule) and shared references
|   |-- backends/                # reference and optimized routed-reduction backends
|   |-- compiler/                # semantic IR, locality/effects, verified
|   |                            #   rewrites, anchors, planner, cost model
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

### Phase 1 - framework baselines (in progress)

- Add PyTorch SDPA math and flash-dispatch adapters (dense causal attention is
  done via `src/urm/adapters` and `benchmarks/dense_attention.py`; the URM
  dispatch overhead gate of 5% median is met with a +2.32% worst steady-state
  median measured on every
  steady-state shape under paired interleaved sampling).
- Add the FLA gated-delta-rule comparator (typed adapter, four levels,
  prefill + decode; see docs/fla-gated-delta-rule.md) - first recurrence
  family comparator landed.
- Add a transparent top-k MoE and sparse-memory gather/scatter baseline.
- Capture correctness, route, and memory-traffic metadata in one result schema.

### Phase 2 - specialized kernels (in progress)

- Add FlexAttention/block-sparse, FLA/Mamba recurrence, and optimized MoE
  adapters.
- Add an external adapter to the original SDM Triton/CUDA implementation.
- Prototype URM page-local gather-reduce and write-merge kernels only after
  matching the original SDM semantics and traces.
- Compare URM dispatch overhead with direct upstream calls.

### Phase 2.5 - compiler architecture (this iteration)

- Layered semantic/execution IR with locality and effect models
  (docs/compiler-charter.md).
- Verified reparameterization: registered rules, deterministic traces,
  positive and negative differential tests.
- CODA-inspired routed-reduction row-scale epilogue prototype with measured
  avoided materialization (results/compiler/routed-scale-epilogue/).
- Simulated routing-to-communication planner and analytical cost model.
- NAS-facing compilation API plus a preset compilation matrix artifact.

### Phase 3 - systems path

- Fuse overlay composition and reduction.
- Add page grouping, prefetch, recurrent address reuse, and distributed sharding.
- Evaluate GL-SDM traces without making GL-SDM's architectural success a URM
  dependency.

## Acceptance gate

URM succeeds when the restricted contract covers the target families without
arbitrary tensor escape hatches and each family can select or generate a
competitive specialized backend. The working targets and stop conditions are in
the [benchmark protocol](docs/benchmarking.md).
