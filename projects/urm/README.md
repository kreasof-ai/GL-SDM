# Project III: Unified Routed Mixer

[Read the complete research proposal](../../docs/research-program.md#proposal-iii).

URM is the first active project in the Global Liquid SDM program. It asks
whether attention, expert routing, parameter-token mixing, linear recurrence,
and sparse memory access can share a constrained execution contract while still
lowering to competitive specialized kernels.

## Current milestone

Milestone zero establishes semantics and benchmark discipline before GPU kernel
work:

- a typed `MixerSpec` contract;
- a dependency-light NumPy correctness oracle;
- deterministic dense, masked, top-k, and write-merge semantics;
- canonical specs for attention, MoE, parameter-token, recurrence, and GL-SDM;
- compositional detail specs for advanced MoE, recurrent, and learned
  sparse-attention flavors;
- an explicit baseline matrix and benchmark shape grid.

The first Triton vertical slice is now scaffolded: a frozen routed-reduction
tensor contract, transparent PyTorch baseline, lazy Triton forward/backward
backend, capability checks, GPU tests, environment preflight, and reproducible
benchmark output. GPU execution still requires validation on a supported Linux
CUDA host; see [Triton backend preparation](docs/triton-backend.md).

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
|   |-- cases.toml             # canonical shape and trace grid
|   |-- routed_reduce.py       # CUDA-event PyTorch/Triton comparison
|   |-- triton_preflight.py    # environment and GPU readiness report
|   `-- reference_smoke.py     # dependency-light harness smoke test
|-- docs/
|   |-- architecture.md        # target contract and non-goals
|   |-- baselines.md           # comparator catalog and scope rules
|   `-- benchmarking.md        # correctness, measurement, and acceptance gates
|-- src/urm/
|   |-- backend.py             # explicit backend protocol and registry
|   |-- backends/              # reference and future optimized adapters
|   |-- ir.py                  # restricted typed operator contract
|   |-- presets.py             # canonical semantic-family specifications
|   |-- routed_reduction.py    # frozen v1 tensor/capability contract
|   |-- triton_kernels/        # lazy GPU kernels and autograd wrappers
|   `-- reference.py           # slow NumPy oracle and transactional merge
`-- tests/                     # contract and oracle tests
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

### Phase 0 - contract and oracle (current)

- Freeze v0 routing, normalization, mutation, residency, and collision enums.
- Validate dense/masked reduction, stable top-k routing, and transactional merge.
- Make result metadata and benchmark cases reproducible.
- Validate the routed-reduction Triton backend on the target Linux CUDA host.

### Phase 1 - framework baselines

- Add PyTorch SDPA math and flash-dispatch adapters.
- Add a transparent top-k MoE and sparse-memory gather/scatter baseline.
- Capture correctness, route, and memory-traffic metadata in one result schema.

### Phase 2 - specialized kernels

- Add FlexAttention/block-sparse, FLA/Mamba recurrence, and optimized MoE
  adapters.
- Add an external adapter to the original SDM Triton/CUDA implementation.
- Prototype URM page-local gather-reduce and write-merge kernels only after
  matching the original SDM semantics and traces.
- Compare URM dispatch overhead with direct upstream calls.

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
