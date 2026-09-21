# Full-version compiler construction

Active docs contain the specifications and gates needed to build URM. Historical
milestones, adapter freezes, tuning reports and benchmark protocols are retained
in the [research archive](../archive/README.md).

Read in this order:

1. [Compiler charter](compiler/compiler-charter.md): architectural invariants.
2. [Architecture](compiler/architecture.md): frontend, IR and component ownership.
3. [Kernel generation](compiler/kernel-generation.md): verified lowering pipeline.
4. [Unified mixer prototype](compiler/unified-mixer.md): executable K1/K2/K3 semantic and anchor boundary.
5. [Runtime execution](runtime/execution.md): capabilities, binding and providers.
6. Kernel contracts: [softmax attention](kernels/softmax-attention.md),
   [linear/delta recurrence](kernels/linear-delta.md), and
   [sparse routed delta](kernels/sparse-delta.md).
7. [Named coverage register](planning/coverage.md): architecture-by-architecture comparisons.
   [Production replacement matrix](planning/production-matrix.md) freezes the
   mandatory workloads URM-native kernels must qualify against upstream.
   [Upstream comparison table](validation/upstream-comparison.md) consolidates
   coverage, parity, and dispatch overhead against pinned upstream sources.
   [Unification audit](planning/unification-audit.md) records source findings;
   [generality axes](compiler/generality-axes.md) specifies the required IR extensions.
8. [Construction roadmap](planning/lowering-roadmap.md): implementation milestones.
9. [Acceptance requirements](validation/acceptance.md) and
   [parity plan](validation/parity.md): numerical, integration and performance gates.
   [Representation coverage](validation/representation-coverage.md) records the
   evidence separating K1/K2/K3 expressibility from native generation.

Kernel contracts are architecture-independent. Names such as `sparse_delta`
describe mathematical operations; named models belong in presets, adapters and
comparison evidence. Shared routing or tensor layouts do not imply equivalent
update rules.

Add active docs only for construction contracts, necessary kernel derivations or
acceptance gates. Archive experiment reports and superseded plans with provenance.
