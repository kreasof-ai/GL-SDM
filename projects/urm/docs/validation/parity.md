# Numerical and performance parity plan

This is the construction acceptance policy, not a report of achieved parity.
The [coverage matrix](../planning/coverage.md) identifies the workloads to qualify.

## Named upstream comparison campaign

The [architecture register](../../benchmarks/architecture-coverage.json) is the
input backlog. It is not yet an executable benchmark runner: callable identities,
supported modes and tolerance profiles deliberately remain unresolved until audited.
Implement the following runner stages through the existing benchmark utilities.

1. **Resolve:** identify the precise architecture variant and all compatible
   available implementations: original authors, pinned FLA, vendor/library and
   local baselines where applicable. Record missing sources and unsupported modes.
   A repository name is insufficient; locate the exact forward/backward callable.
2. **Freeze:** emit a per-implementation record with repository commit, callable
   source hash, dependency versions, license, capability probe, precision and
   state policy, hardware, execution mode and any algorithm/chunk options. Preserve
   old accepted pins alongside new candidates. Never benchmark moving `main`.
3. **Derive:** map the full operator or layer graph to typed URM operations.
   Record the equation, state bundle, effects, missing axis/primitives, VJP and
   rewrite obligations. An external anchor may execute it while native work remains
   incomplete, but is not native compiler coverage.
4. **Qualify:** run independent references and gradient/state checks per mode,
   then test compiler binding. Pin tolerance values and mandatory workloads before
   examining candidate performance. Resolve ambiguities rather than relaxing gates.
5. **Compare:** run all qualified compatible baselines, the same call through URM,
   and the native lowering. Select the fastest semantically valid comparator on
   an independent discovery run, freeze it, then use fresh paired confirmation.
   Retain results against every baseline; do not use a deliberately slow variant
   as the sole reference or select a winner from confirmation noise.
6. **Publish:** retain per-architecture/per-mode/per-device results and blockers.
   Report expression, adapter support, native correctness and parity separately.
   Compiler configuration must not choose a numerically failing fast path.

### Required fixture dimensions

Use supported architecture dimensions, not one forced universal shape. Start from
upstream representative shapes plus matched URM model fixtures. Suggested context
probes are 1, 128, 1024, 4096 and 16384, subject to documented support and memory
limits. Include short/mid/long context, small/throughput batches, awkward tail
lengths, supported head sharing, and representative key/value/state dimensions.
Sparse fixtures add uniform, skewed, repeated and collision-heavy routes at several
route densities. Recurrent fixtures add long decay histories, multiple chunk sizes,
nonzero initial state and stream continuation. Parameter/expert fixtures include
balanced and imbalanced traffic and dispatch/combine cost.

Training requires forward, backward, saved/recomputed state and complete optimizer
steps. Prefill includes all required routing/cache construction. Decode includes
cache or state updates at multiple existing context lengths, with per-token latency
and throughput reported separately. A model fixture preserves its own architecture;
architecture A is not required to train identically to architecture B.

Run targeted smoke/gradient cases on changes, representative fixtures before merging,
and the full frozen matrix on release hardware. Avoid an arbitrary full Cartesian
product: commit the mandatory case list per architecture before measurements, and
record coverage of each dimension. Unsupported cases remain visible with reasons.

### Result record and release rule

Each result identifies `architecture_id`, frontend/variant, baseline identity,
source hashes, semantic contract, lowering/rewrite IDs, axis dependencies, schedule,
device/provider, shapes/layouts, precision, mode, required gradients, numeric report,
raw timing blocks, confidence interval, memory, model trajectory and environment.
Statuses distinguish `identity_unresolved`, `semantic_blocked`, `upstream_unavailable`,
`numeric_failed`, `correct_below_parity`, `inconclusive`, and `parity_qualified`.

CUDA/A10G is the existing evidence base, not universal hardware certification.
Add Hopper/Blackwell, ROCm and other providers as separately qualified targets when
hardware and kernels are available. A source portability claim is not a run result.
Use per-architecture mandatory-case gates; an aggregate geometric mean may summarize
the campaign but cannot erase failures. Publish the exact qualified support subset.

## Baselines and measurement levels

| Family | Transparent reference | Competitive baseline | Main implementation work |
|---|---|---|---|
| Softmax | Dense stable softmax and independent gradients | Pinned compatible SDPA/attention kernel | Online-softmax tiling, fused backward, head sharing, decode cache |
| Linear/delta | Token recurrence plus chunk differential check | Pinned matching gated-delta/linear kernel | Stable chunk solve, transition handling, state fold and recurrent decode |
| Sparse delta | Existing NumPy recurrence/chunk/VJP references | Pinned upstream sparse-memory implementation on matching semantics | Correct decay and VJP, chunk boundary state, routing/coefficient cost and workspace |

Measure four paths separately: transparent reference, direct competitive baseline,
URM adapter invoking that same baseline, and native URM lowering. Adapter parity
measures integration overhead; it cannot establish native kernel parity.

## Gate 1: semantic equivalence

Freeze routes, normalization, masks, update order, initial/final state, precision
and execution mode. Record exact baseline revision, algorithm and chunk settings.
If per-token BF16 commits differ from chunk-boundary commits, establish a separate
numerical contract; do not relax the old gate and call it unchanged parity.
Resolve contradictory documentation against reference code and targeted examples.

## Gate 2: numerical and training equivalence

Validate float64 oracle equations and finite-difference/independent VJP checks.
Then freeze dtype-specific absolute and relative tolerances before GPU comparison;
reuse existing frozen tolerances where the contract is unchanged. Report elementwise
violations and maximum errors for outputs, final states and every gradient. Cosine
similarity is diagnostic, not the sole gate.

Include zero/empty cases allowed by the contract, collisions, long decay histories,
partial chunks, multiple chunk sizes, state continuation and both output/state
losses. The timed backend, compilation mode and schedule must be exactly the
validated ones. Missing gradients or omitted state work disqualify the path.

## Gate 3: executable compiler parity

Pass frontend-to-plan-to-runtime tests. Confirm serialized settings drive launches,
capability declines are honest, and training does not select a forward-only anchor.
Run a short matched training trajectory with identical initialization, batches,
optimizer, precision, accumulation and state-reset policy. Compare parameter and
optimizer state elementwise; whole-model cosine alone is insufficient.

## Gate 4: performance parity

Initial target: candidate/baseline latency ratio at most **1.05** on each mandatory
workload, separately for complete training steps, prefill and decode. Apply the
same target to adapter overhead, reported separately from native performance.
This is a project target, not a guaranteed property of the equations.

Use at least three fresh-process repetitions with paired randomized AB/BA timing,
warmup and compilation excluded, and retained raw samples. Require both the median
ratio and the paired-bootstrap 95% upper confidence bound to be at most 1.05.
If intervals are too wide, report inconclusive and gather more data under the
same frozen protocol. Predeclare workload-specific memory budgets and report peak
allocated/reserved memory; exceeding a budget fails that workload. Do not average
away a mandatory workload regression.

Benchmark state kernels, route-to-state execution and full model separately.
Profile projections, routing, coefficients, solves, state folding, backward and
optimizer costs. Fix the largest measured end-to-end cost before tuning an isolated
operation. Use the same required gradients and state outputs in both paths.

Keep the old frozen upstream configuration visible. Add tuned baseline configurations
as separately named comparisons; do not silently change chunk sizes. Match current
environment/device measurements. Report useful model FLOPs separately from extra
dense implementation work; MFU is diagnostic and never substitutes for latency.

## Implementation sequence and exit criteria

1. Softmax: bind existing adapter, then native online-softmax forward/backward;
   qualify initial causal workload before extending masks and cache support.
2. Linear/delta: verify exact ordering, implement oracle and scalar-decay chunk
   path plus recurrent decode; qualify state/gradient parity before extensions.
3. Sparse delta: repair decay, complete VJP and boundary dependencies; qualify
   collision stress before chunk/sparsity optimization and full-model integration.
4. For every family, record semantic and numerical gates, adapter ratio, native
   stage ratio, full-workload ratio, memory and provenance in the support matrix.

A failing performance gate leaves a correct implementation usable but explicitly
below parity. A numerical failure leaves it experimental. Neither result is hidden
by an aggregate success label or by a faster unrelated family.
