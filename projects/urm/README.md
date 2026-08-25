# Project III: Unified Routed Mixer

[Read the complete proposal](../../docs/research-program.md#proposal-iii).

## Research question

Can attention, expert routing, parameter-token mixing, linear recurrence, and
sparse memory access share a constrained execution representation without
becoming an unoptimizable general tensor language?

## Independence

URM is independently falsifiable. It may be evaluated with several mixer
families even if GL-SDM does not validate architecturally; GL-SDM is its flagship
transactional read/write workload.

## This project owns

- A typed gather-score-weighted-reduce-scatter intermediate representation.
- Declared routing, normalization, mutation, residency, determinism, and
  collision semantics.
- Reference lowering and numerical-equivalence tests.
- Page grouping, HBM/SRAM staging, prefetch, fusion, and distributed sharding.
- Microkernel, locality, fusion, distributed, and end-to-end benchmarks.

It should not absorb model-quality hypotheses belonging to GL-SDM or continual
learning policy belonging to CSDM.

## First milestone

Specify a minimal reference IR only after collecting representative address
traces. Cover at least one read-only mixer and the GL-SDM transactional path,
then compare each lowering with a direct reference implementation.

## Acceptance gate

Proceed when the restricted IR covers the target operators with numerical
equivalence and competitive specialized backends. Treat material abstraction
overhead, routing-locality collapse, or unrelated per-mixer special cases as
negative results.

## Expected output

A systems paper, constrained operator specification, profiler, and specialized
read/write kernel implementations.
