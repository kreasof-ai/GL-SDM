# Numerical and performance parity

This is the acceptance policy and the current parity status. The
[coverage register](../planning/coverage.md) identifies the workloads; measured
results live in the [master coverage table](master-table.md) (model level) and
the master coverage table (which recipes URM computes natively).

## Current status

All 76 mixer-relevant register rows have measured kernel-upstream parity and
paired profiling evidence against pinned sources. At model level, 61 of 62
covered recipes compile and run natively in the frozen ~100M decoder LM (one,
`titans_linear_memory_core`, exceeds the sweep's serial-kernel budget and is
recorded as not measurable). These are kernel-slice and model-slice results, not
full-layer or production qualification: projections, frontends, caches and
full-layer integration remain open per row.

## Comparison campaign

The [architecture register](../../benchmarks/architecture-coverage.json) records
each row's pinned comparator, lowering, parity status and remaining work. Each
qualified comparison runs the following stages through the existing benchmark
utilities:

1. **Resolve:** identify the precise architecture variant and all compatible
   available implementations: original authors, pinned FLA, vendor/library and
   local baselines where applicable. Record missing sources and unsupported modes.
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
6. **Publish:** retain per-architecture/per-mode/per-device results and blockers.
   Report expression, adapter support, native correctness and parity separately.

### Fixture dimensions

Use supported architecture dimensions, not one forced universal shape: short/mid/long
context, small/throughput batches, awkward tail lengths, supported head sharing, and
representative key/value/state dimensions. Sparse fixtures add uniform, skewed,
repeated and collision-heavy routes at several route densities. Recurrent fixtures
add long decay histories, multiple chunk sizes, nonzero initial state and stream
continuation.

Training requires forward, backward, saved/recomputed state and complete optimizer
steps. Prefill includes all required routing/cache construction. Decode includes
cache or state updates at multiple existing context lengths, with per-token latency
and throughput reported separately. A model fixture preserves its own architecture;
architecture A is not required to train identically to architecture B.

### Result record and release rule

Each result identifies `architecture_id`, frontend/variant, baseline identity,
source hashes, semantic contract, lowering/rewrite IDs, axis dependencies, schedule,
device/provider, shapes/layouts, precision, mode, required gradients, numeric report,
raw timing blocks, confidence interval, memory, model trajectory and environment.
Statuses distinguish `identity_unresolved`, `semantic_blocked`, `upstream_unavailable`,
`numeric_failed`, `correct_below_parity`, `inconclusive`, and `parity_qualified`.
The register also uses `not_applicable` for pinned catalog items classified outside
the sequence-mixer kernel boundary.

CUDA/A10G is the existing evidence base, not universal hardware certification. Use
per-architecture mandatory-case gates; an aggregate geometric mean may summarize the
campaign but cannot erase failures. Publish the exact qualified support subset.

## Baselines and measurement levels

| Family | Transparent reference | Competitive baseline | Main implementation work |
|---|---|---|---|
| Softmax | Dense stable softmax and independent gradients | Pinned compatible SDPA/attention kernel | Online-softmax tiling, fused backward, head sharing, decode cache |
| Linear/delta | Token recurrence plus chunk differential check | Pinned matching gated-delta/linear kernel | Stable chunk solve, transition handling, state fold and recurrent decode |
| Sparse delta | Existing NumPy recurrence/chunk/VJP references | Pinned upstream sparse-memory implementation on matching semantics | Correct decay and VJP, chunk boundary state, routing/coefficient cost and workspace |

Measure four paths separately: transparent reference, direct competitive baseline,
URM adapter invoking that same baseline, and native URM lowering. Adapter parity
measures integration overhead; it cannot establish native kernel parity.

## Gates

1. **Semantic equivalence.** Freeze routes, normalization, masks, update order,
   initial/final state, precision and execution mode. Record exact baseline revision,
   algorithm and chunk settings. Resolve contradictory documentation against
   reference code and targeted examples.
2. **Numerical and training equivalence.** Validate float64 oracle equations and
   finite-difference/independent VJP checks. Freeze dtype-specific absolute and
   relative tolerances before GPU comparison. Report elementwise violations and
   maximum errors for outputs, final states and every gradient. Include zero/empty
   cases, collisions, long decay histories, partial chunks, multiple chunk sizes,
   state continuation and both output/state losses.
3. **Executable compiler parity.** Pass frontend-to-plan-to-runtime tests. Confirm
   serialized settings drive launches, capability declines are honest, and training
   does not select a forward-only anchor. Run a short matched training trajectory
   with identical initialization, batches, optimizer, precision, accumulation and
   state-reset policy; compare parameter and optimizer state elementwise.
4. **Performance parity.** Initial target: candidate/baseline latency ratio at most
   **1.05** on each mandatory workload, separately for complete training steps,
   prefill and decode, with the same target applied to adapter overhead reported
   separately. Use at least three fresh-process repetitions with paired randomized
   AB/BA timing, warmup and compilation excluded, and retained raw samples. Require
   both the median ratio and the paired-bootstrap 95% upper confidence bound to be
   at most 1.05. Predeclare workload-specific memory budgets and report peak
   allocated/reserved memory; exceeding a budget fails that workload.

A failing performance gate leaves a correct implementation usable but explicitly
below parity. A numerical failure leaves it experimental. Neither result is hidden
by an aggregate success label or by a faster unrelated family.
