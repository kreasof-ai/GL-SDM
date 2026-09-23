# Compiler acceptance requirements

Status: active gates for supported operations and lowerings. These are the
criteria a lowering must meet before it is advertised as supported; measured
evidence against them lives in the [master coverage table](master-table.md).

## Semantics and numerical correctness

Specify shapes, logical domains, selection/collision rules, read/update ordering,
state lifetime, and storage/accumulation/commit precision. Compare each lowering
against an independent reference under its declared dtype envelope. Cover
repeated routes, degenerate inputs, partial chunks, long state histories and
unsupported combinations. Unsupported behavior must decline explicitly.

Rewrites require derivations and numerical validation. Changed rounding points
require an explicit floating-point equivalence policy. Validate and benchmark
the exact settings used in execution.

## Training and inference

Training requires every declared operand gradient, including decay and persistent
state. Validate VJPs against an independent adjoint or finite differences using
both output and final-state losses. Opaque forward kernels need a validated
backward or explicit differentiable recomputation.

Prefill and decode require state-continuation, reset/commit and mode-compatibility
tests. Certify each mode separately; forward-only tests do not establish training.

## Compiler and runtime integration

Verify frontend-to-executable slices, plan serialization round trips, anchor
selection, schedule-to-launch agreement, capability declines and diagnostics.
Check cache invalidation for properties affecting semantics or generated code.
Core imports and references must work without optional GPU dependencies. Validate
provider integration on every advertised target.

## Performance evidence

Define workload, execution boundary, numerical contract, versions and source
revision before measurement. Separate compilation/warmup from steady-state work.
Keep synchronized validation outside timing. Measure complete training steps,
prefill and decode separately, including required backward work.

Report latency distributions, throughput, memory, raw samples and provenance.
Use matched baselines and explicit acceptance margins. Distinguish useful model
FLOPs from implementation FLOPs and measured hardware denominators. An isolated
kernel speedup does not establish model speedup. Retain negative results.

## Release evidence

For each advertised capability, record operation/version, rewrite, numerical
policy, shapes, modes, provider, backward coverage and test/benchmark artifacts.
Mark experimental and unsupported paths explicitly. Distinguish proposed,
derived, implemented and validated capabilities in documentation.

The [parity plan](parity.md) supplies family-specific baselines, measurement
levels, numerical qualification and the initial performance acceptance policy.
