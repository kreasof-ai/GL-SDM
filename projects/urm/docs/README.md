# URM documentation

This tree separates **what URM must mean**, **what the current code runs**, and **what historical experiments measured**. The three execution families are K1 streamed reduction, K2 compact fixed-address state, and K3 indexed mutable state. They are not three universal GPU binaries. Architecture names and model layer arrangements stay outside `src/urm`.

## Read in this order

1. [Compiler charter](compiler/compiler-charter.md): ownership, semantic IR, verified rewrites, provider admission and fail-closed execution.
2. [K1](kernels/softmax-attention.md), [K2](kernels/linear-delta.md), and [K3](kernels/sparse-delta.md): the currently written equation contracts and their limits.
3. [Generality axes](compiler/generality-axes.md): typed extensions to prove before broadening a family.
4. [76-architecture composition ledger](planning/architecture-composition.md): external call graphs, unresolved internal axes and per-ID closeout record. It is a target, not a coverage claim.
5. [Verified direction sweep](planning/direction-sweep.md): per-architecture verified direction (composition / combinator / axis / external-only) against pinned source, with the ledger corrections it surfaced.
6. [Roadmap](planning/roadmap.md): ordered work, file ownership and completion gates.
7. [Architecture-to-URM integration instructions](planning/architecture-urm-integration.md): review of external-only mixer modules, reusable semantics, and acceptance checks.
8. [Runtime contract](runtime/execution.md): serialized plans, provider ABI and state sessions.
9. [Evidence rules and current status](validation/evidence.md): what can be claimed now, how to measure, and how historical results are labeled.

## Machine records and generated pages

The [80-row source register](../benchmarks/architecture-coverage.json) is the catalog of source identity and preserved fragment comparisons; four rows are outside mixer scope. The [rendered coverage index](planning/coverage.md) is generated from it by `benchmarks/coverage_register.py`. The [alignment](validation/alignment.md) and [inference-throughput](validation/inference-throughput.md) tables are generated views of preserved artifacts. They are historical evidence, not a statement that the present public graph path runs complete source models.

No planning note or generated table overrides the compiler charter or an exact kernel equation. A proposed lowering must decline until its descriptor, independent references, plan binding and mode-specific evidence exist.
