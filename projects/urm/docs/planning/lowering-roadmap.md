# Lowering coverage and remaining work

Status: current-state record. All three lowering families are implemented and
measured; this document records what each family computes today and the
engineering barriers that remain. Earlier coverage claims were exploratory
hypotheses, not the supported architecture matrix.

## Family status

The three kernel contracts are [softmax](../kernels/softmax-attention.md),
[linear/delta](../kernels/linear-delta.md), and
[sparse delta](../kernels/sparse-delta.md). Each is implemented, with measured
evidence in the [master coverage table](../validation/master-table.md) and the
[upstream comparison table](../validation/upstream-comparison.md).

| Family | Native status | Measured evidence | Remaining barrier |
|---|---|---|---|
| K1 softmax reduction | Native online-softmax candidate covers MHA/MQA/GQA and masked/sparse variants | Kernel-slice parity + paired overhead vs pinned FlashAttention; native profile on 4 rows | Full-layer, cache and end-to-end training/inference qualification |
| K2 linear/delta recurrence | Native candidate for **diagonal** recurrence (HGRN / Mamba-1 class) only | Diagonal native qualification is `correct_below_target` (~55% slower than FLA's chunked kernel) | Native **matrix-state** (gated-delta) lowering is a measured generation gap; chunk the diagonal scan to close the performance gap |
| K3 sparse delta state | Native sparse-state candidate for Sparse Delta Memory | Native K3 output/state/all six gradients match the corrected equation reference; 17–44% faster than pinned SDM on the measured fixture | Product-key routing and full-layer composition remain external |

Representational coverage is broader than native generation: all 62 covered
recipes lower into a canonical core and match their independent equation, but
the native generator reaches a subset of them. See
[representation coverage](../validation/representation-coverage.md) and
[native coverage](../validation/native-coverage.md) for the honest split.

## Evidence levels

Every mapping is proposed, derived, implemented, or validated. A derivation proves
a mathematical correspondence under stated assumptions; a validated implementation
also establishes its numerical, gradient, state and execution-mode contract.
Passing one backend or shape does not certify another backend or configuration.

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
this record. Qualification is per architecture and mode rather than a blanket
percentage.

## Remaining work packages

| Package | Concrete deliverable | Exit gate |
|---|---|---|
| Native K2 matrix state | A native gated-delta matrix-state lowering | The `k2-gated-delta-recurrence` workload qualifies against FLA |
| K2 diagonal performance | Chunk the native diagonal scan | `k2-diagonal-recurrence` meets its frozen slowdown budget |
| Full-layer integration | Projections, frontends, caches per named row | Each advertised architecture/mode passes its mandatory gates |
| Release matrix | Automated frozen campaign, support report and reproducible artifacts | Every advertised architecture/mode/device passes its mandatory gates |

The source of truth for targets is the [named register](coverage.md), for design
changes the [axes contract](../compiler/generality-axes.md), and for run admission
the [parity campaign](../validation/parity.md).
