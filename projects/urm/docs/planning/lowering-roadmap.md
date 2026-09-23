# Compiler lowering coverage and implementation roadmap

Status: bounded engineering roadmap. Earlier coverage claims were exploratory
hypotheses, not the supported architecture matrix.

## Construction specifications

Implement all three families: [softmax](../kernels/softmax-attention.md),
[linear/delta](../kernels/linear-delta.md), and
[sparse delta](../kernels/sparse-delta.md). The [coverage matrix](coverage.md)
lists each capability, its evidence and completion work. The
[parity plan](../validation/parity.md) defines baseline selection, numerical and
training gates, and the initial 1.05 latency-ratio acceptance target.

The three contracts are construction documents; their existence does not mean
all three native implementations are complete.

## Evidence levels

Every mapping is proposed, derived, implemented, or validated. A derivation proves
a mathematical correspondence under stated assumptions; a validated implementation
also establishes its numerical, gradient, state and execution-mode contract.
Passing one backend or shape does not certify another backend or configuration.

| Scope | Present evidence | Next gate |
|---|---|---|
| Existing routed-reduction compiler | Implemented with contract/plan regressions | Preserve during restructuring |
| Dense attention adapter | Implemented; retained acceptance artifacts | Frontend integration under its exact supported contract |
| Gated-delta adapter | Implemented; retained acceptance artifacts | Explicit training and inference capability integration |
| Native sparse route/state composite | Implemented; retained acceptance and negative performance evidence | Preserve semantic and schedule regressions |
| Sparse-slot chunk algebra | Derived; NumPy differential and VJP tests | Repair and validate concrete GPU lowering |
| Historical PyTorch/Triton dual-form prototypes | Experimental; known correctness defects | Decay, cross-block state, complete backward and numerical stability |
| Other named architectures, parameter-axis mixing and inner optimizers | Proposed mappings only | Individual recurrence, state and gradient derivations |

This table records the repository's evidence; historical GPU measurements were
not rerun during this restructuring.

## Engineering milestones

1. Establish frontend, compiler, runtime, backend and oracle ownership. Retain
   compatibility imports.
2. Integrate one explicit contract per initial family: softmax, linear/delta and
   sparse slot. Reuse `UrmCompiler`, registered rewrites and execution anchors.
3. Implement the corrected chunked sparse-slot equations. Verify repeated routes,
   strong decay, zero weights on selected slots, cross-chunk state, partial chunks,
   output/final-state losses and all differentiable operands.
4. Validate each exact runtime path, then optimize and measure full workloads.
   Keep precision semantics separate from schedules and label useful versus
   executed FLOP accounting.
5. Add provider integration and broader coverage only behind these contracts.

## Boundaries requiring derivation

A value residual `v - M k` does not by itself prevent a triangular rewrite: it is
the dependency eliminated by the delta-rule derivation. Input-dependent coefficients
can also be precomputed when they depend on available input projections rather
than unknown evolving state. These observations neither prove nor disprove support
for any complete named architecture; audit its actual equations first.

Likewise, mask expressibility does not establish an efficient sparse implementation;
parameter-axis attention does not automatically implement arbitrary expert MLPs;
and adding an inner-optimization callback does not prove kernel-level equivalence.

There is no established percentage of literature coverage, count of unexplored
architectures, universal three-source implementation, or MFU target guaranteed by
this roadmap. The named comparison campaign and extension axes are explicit construction work;
qualification is per architecture and mode rather than a blanket percentage.

## Production work packages

| Package | Concrete deliverable | Exit gate |
|---|---|---|
| P0: comparator registry | Resolve the named register into pinned callable/capability records and mandatory fixtures | Every attempted run has exact identity, semantics and mode support; blockers stay visible |
| P1: core closure | One complete native frontend-to-runtime slice for each of K1/K2/K3 | Independent VJP/state checks and confirmed model parity on Wave 1 fixtures |
| P2: composite and diagonal state | A1/A5 plus typed state bundles, normalizers and route producers | Wave 2 variants and synthetic cross-architecture reuse checks |
| P3: generalized transitions and axes | A3/A8/A11, cache composition and source-derived transforms | Wave 3 native or explicit blocked status; no silent scalar-delta substitution |
| P4: broader updates | A4/A6/A7/A9/A10/A12/A13 where required by named fixtures | Exact semantics, gradient policy, cost bounds and per-provider qualification |
| P5: release matrix | Automated frozen campaign, support report and reproducible artifacts | Every advertised architecture/mode/device passes its mandatory gates |

These packages are a plan, not newly implemented functionality. Use existing
compiler rewrites, execution anchors and benchmark utilities. The source of truth
for targets is the [named register](coverage.md), for design changes the
[axes contract](../compiler/generality-axes.md), and for run admission the
[parity campaign](../validation/parity.md). Review implementation PRs against those
three records; keep identity resolution, semantic completion and speedup distinct.
