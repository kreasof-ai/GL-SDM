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
per program. Prefill backward 5186 -> **4347 us median (-19%)**; every other
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
scale `1/sqrt(head_dim)`, dropout disabled, bf16, warm steady state, cold
first call recorded separately):

1. oracle: explicit fp32 softmax-reduce (evaluated where the S^2 fp32 matrix
   fits an 8 GiB budget; larger sequences are `not_applicable` by memory
   estimate, never zero);
2. SDPA math backend (same budget rule);
3. pinned FlashAttention upstream called directly (flash-attn 2.8.3);
4. `UrmDenseCausalAttentionAdapter` invoking the same upstream call.

Plus SDPA-flash and its adapter variant as a second optimized pair. Backend
evidence: FA identity recorded from the package; SDPA forced to FLASH-only via
backend flags around each call (save/force/restore); oracle error <= 0.0156
max-abs in bf16 across all evaluated cases; GQA forward/backward covered by
tests against an explicit oracle.

Grid: batch {1,8} x heads 16 x head_dim {64,128} x seq {128, 2048, 8192,
32768}, causal, bf16 - all 16 combinations ran on A10G (no OOM exclusions).

### Adapter overhead (median, steady-state shapes)

| Shape class | FA direct -> URM | SDPA-flash -> URM |
| --- | --- | --- |
| seq >= 2048 (all 12 cases) | -0.9% .. +1.4% | -1.0% .. +0.4% |
| seq 128 (dispatch-bound) | +2..14% (~10 us absolute) | +/- noise (~100 us calls) |

**Working gate met**: URM dispatch is within 5% median latency of the same
dense-attention implementation on every covered steady-state shape; tiny
seq=128 shapes are explicitly marked dispatch-bound rather than treated as
gate failures.

### Efficiency findings

- FlashAttention reaches 85-94% of the *measured* BF16 tensor-core peak at
  s >= 8192 (62 TFLOP/s useful at b8/h16/d128/s32768 fwd) and ~42-67% at
  s=2048 - consistent with known wave-quantization behavior at shorter
  sequences.
- SDPA-flash tracks FA within ~2-4% (its backward is ~7% slower than FA's at
  large S).

Artifacts: `results/attention/dense-causal.json`, schema
`benchmarks/attention-result-schema.json`, tests in
`tests/test_dense_attention_adapter.py`.

## Limitations

1. No Nsight Compute counters on this host: DRAM bytes, L2 hit rate, achieved
   occupancy, and stall breakdowns remain `not_available`. Re-run
   `profile_roofline.py` on a CAP_SYS_ADMIN-enabled host to fill them in; the
   harness already queries metric availability dynamically.
2. Atomic-order nondeterminism (documented v1 property) means grad-value
   comparisons carry ordering wiggle; tolerances live in the adapters/tests.
3. The attention comparator measures URM *dispatch* overhead only; routed
   reduction does not compete with attention kernels and no such claim is
   made.
4. Host-bound timing noise on this co-hosted machine is +-10-15% below 1 ms;
   constraint checks use a torch-drift-calibrated allowance.

## Next recommendation

Single next kernel family: **Flash Linear Attention (FLA) gated delta-rule /
chunked linear-attention scan**, reached through the same four-level harness
(oracle = explicit recurrent reference; framework = eager PyTorch scan;
upstream = pinned FLA; URM = adapter overhead measurement). Rationale: the
recurrence family is the largest uncovered semantic group in the baseline
catalog, FLA publishes pip wheels, and its chunk-parallel form shares the
gather-reduce structure this repo already profiles well. Sparse-memory
(SDM-style page-local gather/merge) should follow once recurrence lands.
