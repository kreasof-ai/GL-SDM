# Benchmark and acceptance protocol

## Dependency policy

- NumPy: `numpy>=1.26,<3`. The validated GPU line (and every committed
  artifact) uses **1.26.4**; the CPU oracle and schema tests also run on
  numpy 2.x. Both ends of the range are exercised in clean environments before
  each release commit, so `pyproject.toml` never demands a version the
  validated environment does not have (or vice versa).
- Triton: `triton>=3.4,<3.5` in the `gpu` extra (the validated line).
- Ruff is pinned (`ruff==0.16.4`) for reproducible lint/format results;
  import order (`I` rules) is part of `ruff check`.
- flash-linear-attention is pinned to exactly `0.5.2` for the gated delta-rule
  comparator. The expected pin and the installed version are recorded as
  separate fields, and an incompatible installed version is rejected at
  dispatch time instead of being relabeled as the pin
  (tests/test_upstream_version_contract.py).
- Original Sparse Delta Memory is an external Git checkout pinned to
  `183e7df809131b80ad4393741029d0f20fc3640b`, because upstream has no Python
  package manifest. The adapter verifies the checkout revision and cleanliness
  plus Torch 2.8.0, Triton 3.4.0, CUDA, and SM80+ before dispatch. Its CC-BY-NC
  4.0 source is never vendored; see `docs/sparse-delta-memory.md`.
- z3-solver is pinned to exactly `4.15.3.0` in the optional `solver` extra.
  It is never a core dependency: the full suite passes without the extra
  (solver-dependent tests skip; the compiler falls back to its documented
  deterministic cost heuristic). Artifacts record the installed version.

## Artifact provenance protocol

Every committed benchmark artifact records: the exact `git` revision whose
clean tree produced it, whether the tree was dirty (`dirty_tree`), the full
benchmark command, a SHA-256 hash of the benchmark configuration, the
installed solver version (when relevant), and a hash of the constraint-model
summaries. The workflow is two-commit:

1. commit the implementation;
2. run every benchmark on that exact clean implementation commit;
3. commit artifacts and documentation.

An artifact whose revision points at code other than the code that produced
it is invalid and must be regenerated.

## Comparison levels

Every covered operation is measured at four levels when available:

1. **Semantic oracle:** slow NumPy or explicit scan used only for correctness.
2. **Framework baseline:** straightforward PyTorch implementation with visible
   operations.
3. **Optimized upstream:** FlashAttention, FlexAttention, FLA, Mamba,
   MegaBlocks, SonicMoE, the original Sparse Delta Memory repository, or another
   pinned primary implementation.
4. **URM lowering:** generated or selected specialized backend from `MixerSpec`.

An upstream adapter at level 4 is labeled an external anchor, not a native
lowering. Upstream production kernels define comparison points and may serve as
temporary external anchors. Native URM lowerings are generated from URM-owned,
typed mixer skeletons and must not depend semantically on FA/FLA/SDM/Mamba
library APIs.

## Correctness gates

- Route indices and tie behavior match the oracle for deterministic cases.
- Forward outputs meet dtype-specific absolute and relative tolerances.
- Backward gradients match a float32 framework baseline where differentiation
  applies.
- Mask, causal alignment, top-k normalization, capacity policy, and collision
  policy are identical.
- Transactional paths read one frozen version and publish exactly one merged
  commit.
- Deterministic mode is bitwise stable on the same software and hardware stack.

Tolerance values belong in each adapter because fp32, fp16, bf16, fp8, recurrent
scans, and atomic reductions have different error envelopes.

## Performance protocol

- Record hardware, clocks/power mode, driver, runtime, compiler, package commit,
  dtype, tensor layout, and all semantic flags.
- Separate cold compile/autotune time from warm steady-state measurements.
- Warm up until allocations and compilation are complete.
- Prefer CUDA/HIP events or the backend's synchronized benchmark utility over
  unsynchronized wall-clock timing.
- Report median, p95, and dispersion across repeated samples.
- Report tokens/s, effective GB/s, useful FLOP/s, peak allocated bytes, and
  temporary bytes when meaningful.
- Include routing density, token/expert balance, page hit rate, collision count,
  and bytes moved for sparse operations.
- Run forward, backward, prefill, and decode regimes separately.
- A label containing `training` does not imply backward timing. Forward-only
  training-mode cases must say `forward_only`; differential backward evidence
  is reported separately unless a backward region is actually measured.

## Initial shape grid

The committed `benchmarks/cases.toml` is the machine-readable source. The first
coverage set is:

- Attention: batch 1 and 8; 16 heads; head dimension 64 and 128; sequence 128,
  2K, 8K, and 32K; causal and non-causal.
- MoE: 1K, 8K, and 32K tokens; hidden dimension 1K and 4K; 8, 64, and 128
  experts; top-2 and top-8; uniform and skewed routing.
- Memory: 1K to 1M addresses initially; top-4, top-8, and top-32; page size 64,
  128, and 256; uniform, Zipfian, and recurrent-reuse traces.
- Recurrence: sequence 128 to 32K; state dimensions 16, 64, and 128; recurrent
  and chunk-parallel execution.
- Routed reduction: decode, prefill, memory-reuse, and non-power-of-two cases;
  route widths 2 to 32 initially; value dimensions 128 to 4096; fp16, bf16, and
  fp32. These named cases are directly executable by
  `benchmarks/routed_reduce.py`.

## Triton measurement entrypoint

The first GPU benchmark is documented in
[Triton backend preparation](triton-backend.md). It compares the transparent
PyTorch and Triton implementations from identical precomputed routes, captures
cold compilation separately, and emits JSON conforming to
`benchmarks/result-schema.json`.

## Profiling entrypoints

- `benchmarks/measure_device_limits.py`: measures the denominators used by all
  utilization numbers on the same device and software stack - sustainable HBM
  bandwidth (best of copy/fill/read kernels), FP32 CUDA-core peak (TF32-off
  SGEMM), and BF16 tensor-core peak. Vendor datasheet figures are stored as
  reference only and never serve as MFU/MBU denominators.
- `benchmarks/profile_roofline.py`: per committed case and mode it reports
  wall time, aggregate GPU kernel time with a per-kernel breakdown, host
  dispatch share, useful algorithmic TFLOP/s (documented FLOP model, not
  instruction counts), FP32 CUDA-core MFU for routed reduction (tensor-core
  peaks are prohibited there because those kernels do not use MMA), static
  analytic traffic bounds clearly separated from measured counters, MBU of the
  static bound against measured bandwidth, route statistics, an atomic
  contention indicator, registers per thread, theoretical occupancy, and
  eligibility flags that mark host-bound small cases as ineligible for
  normalized utilization. Nsight Compute fields are emitted as explicit
  `not_available` records when counter permissions are missing.
- `benchmarks/compare_results.py`: constraint checker comparing two result
  directories - correctness within committed tolerances, triton p95 within a
  calibrated allowance (3% base, torch-drift + 2%, or a 15% floor for
  host-bound sub-millisecond cases), and peak memory within +2% or 1 MiB.
- `benchmarks/dense_attention.py`: four-level dense causal attention
  comparison (semantic oracle, SDPA math, pinned FlashAttention upstream
  direct, and the same call behind the URM adapter) over the committed
  attention shape grid. Unsupported configurations are recorded as
  `not_applicable` with a reason, never as zero performance. Timed regions
  contain no input cloning or allocation; direct-versus-adapter samples are
  paired and interleaved; overhead is reported as a paired-fraction
  distribution with a bootstrap CI; wall and device-span latencies are
  recorded separately.
- `benchmarks/gated_delta_rule.py`: four-level FLA gated delta-rule
  comparison (fp32 recurrent oracle within a documented 2048-token budget,
  eager PyTorch recurrence, pinned FLA chunk/fused-recurrent direct calls,
  and the same calls behind the typed URM adapter) across separate prefill
  and token-by-token decode regimes. Reports cold first call, median/p95
  forward and backward, tokens/s, useful FLOP/s from a documented recurrence
  model, static-analytic MBU, MFU against the measured BF16 tensor-core peak
  (host-bound decode cases report absolute microseconds and dispatch share
  instead), peak/temporary memory, paired direct-versus-adapter overhead,
  and final-state materialization cost. Nsight Compute fields remain
  explicit `not_available` on this host.
- `benchmarks/compilation_matrix.py`: builds semantic programs from every
  canonical preset, enumerates rewrite/lowering candidates under an explicit
  training intent, compiles what is supported, records structured decline
  reasons for what is not, and emits `results/compiler/compilation-matrix.json`
  (escape-hatch count stays zero; architecture and schedule parameters
  serialized separately). Coverage metrics are named for what they measure:
  `routing_skeleton_compile_rate` (coarse routing skeleton maps to compiled
  routed reduction - this does NOT mean dense attention, MoE, or sparse
  attention is fully compiled), `full_architecture_compile_rate` (complete
  family detail lowered today), `native_lowering_rate`,
  `upstream_adapter_rate`.
- `benchmarks/sparse_delta_memory.py`: compares NumPy and transparent PyTorch
  semantics with the pinned original SDM calls both directly and behind the
  typed URM adapter. It covers read-only, prefill, decode/cache, writes,
  cross-token collision stress, training, and A10G-sized capacity cases;
  retains exact routes, final-state checks, raw AB/BA wall/device samples,
  throughput, allocator peaks, analytical traffic, cold timing, call identity,
  and complete upstream provenance. Callable identity is computed from the
  stored bound object, instance, and function and artifact generation aborts on
  mismatch. Untimed fp32/bf16 differential backward evidence starts from the
  compiler-visible write/read scores and covers exact product-key addresses,
  Softmax, ordered state evolution, and gradients for both scores, initial
  memory, values, beta, and log-decay. Schema version 2 requires these fields.
  The artifact partitions substantial workloads from tiny host-bound
  read/decode cases, reports the latter in absolute microseconds and percentage,
  and explicitly makes no mature-kernel gate claim without a predeclared
  eligibility decision. Its schema is
  `benchmarks/sparse-delta-memory-result-schema.json`.
The native SparseStateMixer grid is frozen in
`benchmarks/sparse_state_mixer_cases.toml`. It compares identical precomputed,
certified routes at kernel-only scope; pipeline route-production timing remains
explicitly not applicable until a native selector exists. Confirmation uses
three fresh processes, randomized paired AB/BA samples, bootstrap upper bounds,
drift sentinels, raw samples, and clean provenance. See
`docs/sparse-state-mixer.md` for fixed semantic/numerical envelopes.

- `benchmarks/routed_epilogue_selection.py`: solver-guided schedule selection
  for the routed-scale epilogue. Runs the full documented pipeline -
  candidates, constraint model, Z3 feasibility + bounded lexicographic
  optimization, independent verification, exhaustive-sweep agreement check -
  then measures the legal fused schedule grid on GPU (fwd+bwd medians via
  CUDA events), captures compile feedback (registers/thread, shared memory),
  and reports legality accuracy, solve time, pruned candidates, and empirical
  regret of the Z3-selected versus heuristic schedules. Emits
  `results/compiler/solver/routed-epilogue-selection.json`.
- `benchmarks/routed_epilogue_stability.py`: fresh-process discovery diagnostic
  over the full schedule set. It retains raw samples and operating conditions,
  reports rank correlation/top-k overlap and bootstrap regret, and emits
  `results/compiler/solver/routed-epilogue-stability.json`. Its candidate set
  is exploratory; marginal confidence-interval overlap is never reported as
  confirmatory equivalence.
- `benchmarks/routed_epilogue_confirmation.py`: canonical deployment
  confirmation over a frozen shortlist/reference. It uses randomized paired
  AB/BA blocks, drift sentinels with fail-closed retry exhaustion,
  full-precision child evidence, cross-run provenance/configuration checks,
  and a hierarchical-bootstrap upper slowdown bound. Only schedules whose 95%
  upper bound is at most the declared 2.5% margin enter the equivalent set.
  Emits `results/compiler/solver/routed-epilogue-confirmation.json`.
- `benchmarks/placement_selection.py`: solver-guided expert/page placement on
  simulated 2x2 / 2x4 meshes against capacity, ownership/replication,
  colocation and anti-affinity constraints; lexicographic objectives
  (max load, critical path, bytes, peer pairs, deterministic tie-break);
  compared against round-robin, greedy load balancing, and a brute-force
  optimum on tiny instances; every returned plan independently verified.
  Emits `results/compiler/solver/placement-selection.json`.
- `benchmarks/unsat_diagnostics.py`: runs the representative impossible
  problems (training with forward-only anchor, tile/vector mismatch,
  shared-memory overrun, unsupported dtype, unplaceable item, replication
  beyond mesh, deterministic merge with atomic-only anchors, transactional
  update without commit lowering, push dispatch without return path) and
  commits their unsat cores mapped to concise messages as
  `results/compiler/solver/unsat-diagnostics.json`. Raw solver formulas are
  never the primary diagnostic.
- `benchmarks/routed_scale_epilogue.py`: materialized versus fused plans for
  `output[q,d] = row_scale[q] * routed_reduce(...)`. Compares the trusted v1 +
  scale plan against the compiler-generated fused epilogue anchor; host-bound
  and GPU-bound shapes measured separately; wall and device-span recorded
  separately; launch counts and traffic deltas are documented analytic models;
  a full-row-tile schedule variant is measured and retained whether accepted
  or rejected. Numerical differences are recorded by dtype against eager
  references with assert_close semantics.

## Program-level targets

These are engineering gates for each family as it enters active scope, not
claims that every deferred family was completed by the closed
routed-reduction/compiler-validation tranche:

| Gate | Target |
| --- | --- |
| Semantic coverage | Dense attention, block-sparse attention, top-k MoE, recurrent mixer, parameter-token mixer, and sparse transactional memory all represented without arbitrary tensor escape hatches |
| Correctness | All reference and adapter tests pass; no silent semantic fallback |
| Dense attention overhead | URM dispatch/lowering is within 5% median latency of the selected upstream kernel on covered steady-state shapes |
| Other mature kernels | Within 10% median latency of the selected upstream implementation, or a documented reason to keep a specialized path. Host-bound decode regimes report absolute microseconds and dispatch share instead of utilization ratios |
| Memory locality | Page grouping reduces measured HBM bytes/token on at least one realistic trace family without changing routing results |
| Write path | One deterministic merge/commit per transaction with collision and version metadata |

A family that cannot meet the overhead target may remain a first-class specialized
backend. URM fails only if the contract repeatedly requires unrelated escape
hatches or prevents competitive lowering across the target set.
