# Project I: Global Liquid SDM

[Read the complete proposal](../../docs/research-program.md#proposal-i).

## Research question

Can a weight-tied model allocate variable computation per token while repeatedly
accessing one globally shared, model-scale sparse delta memory, improving the
quality-compute-capacity frontier?

## This project owns

- The global sparse address space and read operator.
- Fixed and adaptive weight-tied recurrent reasoning.
- Frozen-snapshot reads and buffered, transactional commits.
- Memory sharing, capacity, routing-contention, and write-timing experiments.
- Architecture-level throughput and HBM-residency measurements.

It does not own multi-timescale consolidation policy or a general mixer compiler.
Those belong to CSDM and URM respectively.

## First milestone

Build the smallest dense or small-memory reference model that can reproduce:

1. the write-every-step versus snapshot-and-commit comparison; and
2. the capacity comparison between layer-local and globally shared memory.

Before scaling, fix one transaction scope, one halting objective, one fixed-depth
control, and the key/versioning policy.

## Acceptance gate

Proceed when the implementation has deterministic snapshot/commit behavior and
shows a reproducible improvement on a resource-matched quality/compute/capacity
frontier. Treat collapsed halting, excessive global contention, or gains that
vanish after bandwidth matching as negative results.

## Expected output

A paper-quality architecture study plus a reusable transactional memory operator
that CSDM can extend and URM can use as a systems workload.
