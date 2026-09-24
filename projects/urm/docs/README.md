# URM compiler documentation

Active docs contain the specifications, current-state records and acceptance
gates for URM. Construction contracts say what a lowering must satisfy; the
validation records say what has been measured.

Read in this order:

1. [Compiler charter](compiler/compiler-charter.md): architectural invariants.
2. [Architecture](compiler/architecture.md): frontend, IR and component ownership.
3. [Kernel generation](compiler/kernel-generation.md): verified lowering pipeline.
4. [Unified mixer compiler](compiler/unified-mixer.md): executable K1/K2/K3 semantic and anchor boundary.
5. [Runtime execution](runtime/execution.md): capabilities, binding and providers.
6. Kernel contracts: [softmax attention](kernels/softmax-attention.md),
   [linear/delta recurrence](kernels/linear-delta.md), and
   [sparse routed delta](kernels/sparse-delta.md).
7. [Named coverage register](planning/coverage.md): architecture-by-architecture comparisons.
   [Production replacement matrix](planning/production-matrix.md) freezes the
   mandatory workloads URM-native kernels must qualify against upstream.
   [Unification audit](planning/unification-audit.md) records source findings;
   [generality axes](compiler/generality-axes.md) specifies the required IR extensions.
8. [Lowering coverage and remaining work](planning/lowering-roadmap.md): current family status and open barriers.
9. [Acceptance requirements](validation/acceptance.md) and
   [parity plan](validation/parity.md): numerical, integration and performance gates.
   [Representation coverage](validation/representation-coverage.md) records the
   evidence separating K1/K2/K3 expressibility from native generation.
   [Master table](validation/master-table.md) is the per-recipe record of the
   unified generator's reach - what URM computes with its own kernels versus
   what it can only dispatch to upstream.
   [Master coverage table](validation/master-table.md) is the consolidated
   product-evidence record: all 62 covered recipes dropped into the frozen ~100M
   decoder LM, native vs upstream, across training and inference MFU/MBU,
   throughput, parity, KL divergence and peak memory.
   Supporting evidence records, each regenerated from committed artifacts:
   [gradient alignment and decode KL](validation/alignment.md) and
   [inference throughput and MFU](validation/inference-throughput.md).

Kernel contracts are architecture-independent. Names such as `sparse_delta`
describe mathematical operations; named models belong in presets, adapters and
comparison evidence. Shared routing or tensor layouts do not imply equivalent
update rules.

Add active docs only for construction contracts, necessary kernel derivations or
acceptance gates.
