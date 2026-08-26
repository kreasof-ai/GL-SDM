# URM GPU profiling, refinement, and production-comparison report

Follow-on to `triton-optimization-report.md`, from commit `2bd1f8a`. The
routed-reduction v1 semantic contract remains untouched: no changes to
normalization, collision behavior, fp32 accumulation, validation behavior,
layouts, or gradient semantics.

## 0. Hardware / software identity

| Item | Value |
| --- | --- |
| GPU | NVIDIA A10G (SM 8.6), 23028 MiB, driver 595.91.07 |
| Python / PyTorch / Triton | 3.12.13 / 2.8.0 (cu129) / 3.4.0 |
| FlashAttention upstream | flash-attn 2.8.3, pinned release wheel cu12torch2.8cxx11abiTRUE-cp312 |
| Nsight Compute | 2025.2.1 installed; **counters not permitted** on this host |
| Lint/format | ruff 0.16.4 (now pinned) |

Exact validated versions: `benchmarks/validated-environment.json`.

## Phase 1 - stabilization

- Reproduced final routed-reduction state: 45→57 tests pass, sanitizer
  memcheck 0 errors, baseline-vs-final constraint checker clean.
- **Tooling reproducibility fixes**
  - `gpu` extra now pins `triton>=3.4,<3.5` (the validated line).
  - The direct-launch fast path is capability-gated (`_fast_launch_capable`):
    static probe for `warmup`, `CompiledKernel.__getitem__`, and launch-hook
    knobs; first real use is guarded so any API drift permanently falls back
    to the standard `JITFunction.__getitem__` launcher. Fallback verified to
    produce bitwise-identical outputs.
  - `dev` extra pins `ruff==0.16.4`; project reformatted so
    `ruff format --check` is reproducible.
  - `compare_results.py` summary no longer claims "p95 within +3%"; it now
    describes the calibrated allowance actually applied.
- Dependency versions recorded; hardware support intentionally unchanged.

## Phase 2 - MFU/MBU and roofline profiling

New tooling:

- `benchmarks/measure_device_limits.py` -> `results/device-limits.json`
  - Measured sustainable HBM bandwidth: **513.3 GB/s** (best of copy/fill/read
    kernels on 1 GiB buffers; vendor spec 600 GB/s is recorded as reference
    only and never used as a denominator).
  - Measured FP32 CUDA-core peak (TF32-off SGEMM): **23.1 TFLOP/s** - the only
    legal MFU denominator for routed reduction, whose kernels accumulate via
    FP32 CUDA-core ops rather than MMA instructions.
  - Measured BF16 tensor-core peak: **66.2 TFLOP/s** - used only for
    MMA-based dense attention comparators.
- `benchmarks/profile_roofline.py` + schema `benchmarks/profiling-schema.json`
  reports per case x mode: wall vs aggregate-GPU-kernel time vs host dispatch,
  launch counts, per-kernel breakdown, useful algorithmic TFLOP/s (documented
  model formulas: fwd `2QKD`, grad-weights `2QKD`, grad-values `2QKD`,
  backward ~`4QKD`; explicitly not instruction counts), FP32 CUDA-core MFU,
  static analytic byte bounds kept separate from measured traffic, MBU of that
  static estimate against measured sustainable bandwidth, route statistics,
  atomic contention indicator, registers/thread and theoretical occupancy from
  Triton metadata, and eligibility gating.
- Nsight Compute counters (measured DRAM bytes, L2 hit rate, achieved
  occupancy, scheduler stalls) are reported as explicit `not_available` with
  the blocking reason (`ERR_NVGPUCTRPERM`: container lacks CAP_SYS_ADMIN;
  driver `RmProfilingAdminOnly=1`). No fabricated zeros anywhere.
- Small host-bound cases are marked ineligible for normalized utilization;
  their wall/GPU/dispatch split is still reported.

Committed artifacts: `results/profiling/*.profiling.json` (10 case-modes +
summary), validated by tests in `tests/test_profiling_schema.py`.

### Roofline snapshot (final code)

| Case-mode | wall ms | GPU ms | dispatch | MFU% (FP32 core) | static-MBU |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke fwd/bwd | 0.041 / 0.374 | 0.002 / 0.009 | 96-98% | ineligible | ineligible |
| decode_top2 fwd/bwd | 0.046 / 0.379 | 0.001 / 0.012 | 97-98% | ineligible | ineligible |
| prefill_top8 fwd | 0.157 | 0.151 | 4% | 15.4% | 4.36* |
| prefill_top8 bwd | 4.350 | 4.316 | 1% | 1.1% | 0.94 |
| memory_top32 fwd | 0.052 | 0.015 | 72% | ineligible | ineligible |
| memory_top32 bwd | 0.716 | 0.701 | 2% | 0.8% | 1.05 |
| non_power_of_two fwd/bwd | 0.047 / 0.340 | 0.002 / 0.011 | 95-97% | ineligible | ineligible |

*prefill-forward static-MBU > 1 means the gather upper bound exceeds what DRAM
could serve in the kernel time: most duplicate-row reads are absorbed by L2
(S=256 rows = 2 MiB working set). This quantifies why measured counters would
be needed for exact traffic and why the analytic bound is labeled static.

## Phase 3 - evidence-driven refinement

**Hypothesis test.** The naive claim "skewed prefill-backward is dominated by
hot-row grad-values atomic serialization" was verified with controlled
dose-response experiments (identical shapes/bytes/instructions, only route
distribution varied):

| Route distribution (S=256) | hottest row share | gv kernel time | atomic-add rate |
| --- | ---: | ---: | ---: |
| uniform | 0.45% | 1485 us | 180.7 G/s |
| skew^2 | 6.1% | 1914 us | 140.3 G/s |
| skew^3 | 15.6% | 2913 us | 92.1 G/s |
| skew^4 (committed case) | 24.8% | 4025 us | 66.7 G/s |

Collision concentration alone costs ~2.7x at identical traffic; the committed
case matches the in-situ profile exactly (4016 us). The grad-values buffer is
4 MiB (L2-resident), so the limiter is on-chip RMW throughput at contended
addresses, not DRAM (static-MBU 0.94 confirms under-utilized DRAM).

**Accepted change**: full-row tiles for the per-query grad-values kernel when
`value_dim >= 1024` (BLOCK_D = min(4096, next_pow2(D)), 16 warps): removes
masks and repeated grad_output fragment loads, keeps one wide atomic stream
per program. Prefill backward 5186 -> **4347 us median (16.2% latency
reduction, a 1.19x speedup)**; every other
committed case unchanged within noise; correctness/memory/p95/sanitizer gates
re-run clean. Config confined to the per-query path after a first attempt
regressed small-Q per-route launches (rejected intermediate retained in
`results/p3-gv-fullrow/`).

**Rejected candidates** (all measured end-to-end):
- K-wide 2-D block atomics (wide-issue): 14.2 ms vs 4.0 ms - register pressure
  without issue-rate gain.
- Within-query duplicate folding (16% duplicate routes found): 7.1 ms -
  compiler serializes the O(K^2) fold plus masked atomics.
- K-split grid decomposition: no improvement beyond BLOCK_D effect alone.
- Cross-query aggregation/sorting: out of scope by contract (deterministic
  backward must be introduced as its own documented capability).

## Phase 4 - dense causal attention production comparator

First four-level slice per docs/baselines.md, semantics matched exactly
(causal alignment, BHSD boundary layout, GQA `heads % kv_heads == 0`,
scale `1/sqrt(head_dim)`, dropout disabled, bf16):

1. oracle: explicit fp32 softmax-reduce (evaluated where the S^2 fp32 matrix
   fits an 8 GiB budget; larger sequences are `not_applicable` by memory
   estimate, never zero);
2. SDPA math backend (same budget rule);
3. pinned FlashAttention upstream called directly (flash-attn 2.8.3);
4. `UrmDenseCausalAttentionAdapter` invoking the same upstream call.

Plus SDPA-flash and its adapter variant as a second optimized pair.

### Methodology correction (this iteration)

The first committed comparator cloned Q/K/V inside every timed iteration and
derived adapter overhead from two independent median timings. Both choices
were wrong for a dispatch-bound comparison and were fixed:

- Q/K/V and the output gradient are preallocated leaf tensors; nothing is
  cloned, generated, or allocated inside a timed region. Backward builds a
  fresh graph from those leaves, applies a fixed preallocated output
  gradient, and clears gradients outside the timed region.
- Direct-versus-adapter samples are paired and interleaved (A/B, B/A order
  alternating per pair) on one clock; overhead is reported as the
  distribution of per-pair fractions with a percentile bootstrap CI for the
  median, not as a ratio of two independent medians.
- Every sample records wall latency (end-to-end, host dispatch included) and
  CUDA-event device span separately.
- FlashAttention identity is resolved dynamically from installed package
  metadata (`direct_url.json` provenance), no hardcoded wheel tag.
- SDPA-flash calls run inside an SDPA-context restricted to
  FLASH_ATTENTION, which raises instead of silently falling back;
  backend-selection evidence is recorded per result row.

### Adapter overhead results (v2 methodology, 30 pairs/case)

Values in this section are derived programmatically from the committed
artifact `results/attention/dense-causal.json` (`tests/test_artifact_schemas.py`
recomputes them, so docs and artifacts cannot drift apart):

| Shape class | FA direct -> URM (median paired) | SDPA-flash -> URM |
| --- | --- | --- |
| seq >= 2048 steady state, fwd+bwd (24 case-modes) | **-0.03% .. +2.32%** | -0.50% .. +1.06% |
| seq 128 (dispatch-bound, ~0.14-0.5 ms) | +0.52% .. +5.95% median paired | same order |

Every steady-state case passes the <=5% median gate with margin. The largest
observed bootstrap-CI upper bound among steady-state rows is +4.53% on a
+1.53% median (b1/d64/s2048 forward); the worst steady-state median is +2.32%
(b1/d128/s2048 forward). seq=128 shapes are explicitly marked
dispatch-bound: their absolute deltas are tens of microseconds and co-host
noise dominates (the negative "overhead" rows show the adapter's saved-flag
SDPA path measuring *faster* than per-call context construction - clock and
host noise, both sides identical kernels).

FlashAttention reaches ~96% of the measured BF16 tensor-core peak at
b8/h16/d128/s32768 forward (63.4 TFLOP/s useful) and ~54% at s=2048 -
consistent with wave quantization at shorter sequences.

Artifacts: `results/attention/dense-causal.json` (schema v2), schema
`benchmarks/attention-result-schema.json`, tests in
`tests/test_dense_attention_adapter.py`.

## Phase 5 - FLA gated delta-rule four-level comparator

New family per docs/fla-gated-delta-rule.md (frozen contract, upstream pin
flash-linear-attention==0.5.2 / GitHub tag v0.5.2, PyPI install, MIT license,
called externally). Levels:

1. explicit fp32 recurrent oracle (token budget 2048; 14 larger cases are
   `not_applicable`, never zero);
2. transparent eager PyTorch recurrence (differentiable);
3. direct pinned FLA: chunk prefill / fused-recurrent token-by-token decode;
4. the same operations behind the typed `UrmGatedDeltaRuleAdapter`.

Upstream facts honored and verified in the installed package: the
fused-recurrent decode kernel ships **no backward** (`NotImplementedError`),
so decode is forward-only and all backward measurements target the chunk
path. Chunked prefill and step-by-step recurrent decode agree to
reassociation level (max |diff| <= 0.0156 output / 0.0071 state across all
evaluated cases).

### Correctness results (tests/test_gated_delta_rule_adapter.py)

Forward output and final state vs the fp32 oracle for bf16 and fp16;
initial/final-state semantics (carried vs zero vs absent); q/k/v/g/beta and
initial-state gradients vs the eager baseline (atol 5e-2, rtol 2e-2,
dtype-scaled); T=1 decode equals chunk prefill; non-power-of-two lengths
(1, 7, 37, 100); gate limits (g=0 no decay, g=-20 full decay) and beta
boundaries (beta->0 suppresses writes leaving pure decay, beta=1 exact write
with L2-normalized k); multi-batch/head plus GVA grouping (HV>H via upstream
repeat_interleave semantics); bitwise-deterministic repeated forward.
Committed benchmark artifact: zero tolerance failures across 40 measured
cases.

### Direct FLA versus URM adapter (30 interleaved pairs per case)

| Regime | Paired median overhead | Notes |
| --- | --- | --- |
| Prefill forward, steady state (s >= 2048, 20 case-modes) | **-0.74% .. +2.26%** | gate <=10% met everywhere |
| Prefill forward, s=128 small shapes | +1.5% .. +2.9% | dispatch-bound but already inside the gate |
| Prefill backward (32 case-modes) | median +0.17%, range -2.0% .. +2.2% | chunk path |
| Decode, token-by-token t=256 loops | +4.9% .. +8.6% relative | host-bound; see below |

Decode is host-bound: one fused-recurrent kernel launch per token
(~155-225 us/token end-to-end wall on this co-hosted CPU, device span near
zero). The adapter adds ~12-13 us per token for typed validation and
dispatch - reported as absolute microseconds (+4.9..8.6% relative), which is
the honest lens for host-bound regimes; MFU/MBU are not meaningful there and
are marked ineligible via the >20% dispatch-share rule.

### Decode versus prefill

Token-by-token decode costs ~155 us/token (b1/k64/v64) to ~235 us/token
(b8/k128/v128) versus chunked prefill throughput of 0.54-9.6M tokens/s:
prefill amortizes the recurrence over T tokens per kernel launch while
decode pays launch+dispatch per token. This quantifies the incentive to
batch or fuse decode steps before any native URM recurrent kernel work.

### Utilization interpretation

At the largest GPU-bound prefills the chunk kernel reaches ~8.5 TFLOP/s
useful (MFU ~12.9% of the measured BF16 tensor-core peak) and static-analytic
MBU ~17% against measured sustainable bandwidth. The recurrence moves far
more bytes than its FLOP model implies and the analytic bound excludes
intermediate chunk traffic, so these denominators bound rather than explain
performance; measured DRAM counters remain unavailable on this host
(ERR_NVGPUCTRPERM).

Final-state materialization adds at most ~64 us (b8/s2048/k128v128, 8 MiB
state) and is within noise elsewhere.

Unsupported configurations: b8/s32768/k128/v128 prefill (zero and carried)
exceed 22 GiB on this A10G during chunk-backward graph construction and are
recorded as `not_applicable` with reasons, never as failed performance.

Artifacts: `results/fla-gated-delta-rule/benchmark.json` (schema v1),
schema `benchmarks/gated-delta-rule-result-schema.json`.

## Phase 5 - reacceptance check (post-report correction)

The accepted `gv-fullrow` state was re-measured from a clean tree
(`results/p0-reacceptance/`) and checked against both the original baseline and
the previously shipped `results/final` directory with `compare_results.py`:

- vs `results/baseline`: no violations; prefill-backward triton now 1.41x.
- vs `results/final`: no violations; prefill-backward improvement reproduces
  (5186 -> 4346 us median = 16.2% reduction, 1.19x speedup); every other case
  unchanged within the calibrated allowances.
- One host-bound wobble (`non_power_of_two` backward median 270 -> 313 us,
  p95 improved 411 -> 351 us) was re-run twice from identical code
  (`results/p0-reacceptance-rerun/*.r{1,2}.json`): medians 252 / 293 / 313 us
  and p95 322-351 us on the same binary path. This is co-tenant timing noise,
  not a reproducible regression; no incremental p95 gate tripped.

## Limitations

1. No Nsight Compute counters on this host: DRAM bytes, L2 hit rate, achieved
   occupancy, and stall breakdowns remain `not_available`. Re-run
   `profile_roofline.py` on a CAP_SYS_ADMIN-enabled host to fill them in; the
   harness already queries metric availability dynamically.
2. Atomic-order nondeterminism (documented v1 property) means grad-value
   comparisons carry ordering wiggle; tolerances live in the adapters/tests.
3. The attention comparator measures URM *dispatch* overhead only; routed
   reduction does not compete with attention kernels and no such claim is
   made. The same holds for the FLA comparator: URM calls the pinned FLA
   kernels externally and adds validation/dispatch only.
4. Host-bound timing noise on this co-hosted machine is +-10-15% below 1 ms;
   paired interleaved sampling cancels most of it for A/B comparisons, but
   single-sided small-case medians still wobble (see the Phase 5 reacceptance
   reruns).
5. Decode backward does not exist upstream (fused-recurrent is
   forward-only), so recurrent training cost is measured on the chunk path
   only.

## Next recommendation

With dense attention and gated delta-rule recurrence both covered at four
levels with measured adapter overhead, the next family per docs/baselines.md
is sparse memory: an external adapter to the original Sparse Delta Memory
Triton/CUDA implementation (pinned revision, outputs and address traces as
the baseline), followed by GL-SDM page-local gather/merge. A native URM
recurrent lowering stays deferred until a second upstream point (FLA 0.6.x or
a Mamba-3 kernel) can be compared against the same frozen contract; if decode
dispatch cost matters before then, batching decode steps behind one adapter
call removes most of the measured ~12-13 us/token integration cost without
any new kernel.

## Addendum - compiler architecture iteration

A follow-on iteration (see git history after this file's last result update)
established the compiler layer those next steps plug into:
docs/compiler-charter.md, docs/coda-retrospective.md, the verified rewrite
system under `src/urm/compiler/`, the routed-reduction row-scale epilogue
prototype (`results/compiler/routed-scale-epilogue/`), the simulated
communication planner, and the preset compilation matrix
(`results/compiler/compilation-matrix.json`). Headline attention numbers were
re-derived from the committed artifact during the same pass: worst
steady-state FA-direct paired median is **+2.32%** and the worst bootstrap-CI
upper bound is **+4.53%** (`tests/test_artifact_schemas.py` pins docs to
artifacts). The SDM-adapter recommendation stands, now with placement/state
planning available to consume it.
