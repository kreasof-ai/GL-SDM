# Benchmark and acceptance protocol

## Comparison levels

Every covered operation is measured at four levels when available:

1. **Semantic oracle:** slow NumPy or explicit scan used only for correctness.
2. **Framework baseline:** straightforward PyTorch implementation with visible
   operations.
3. **Optimized upstream:** FlashAttention, FlexAttention, FLA, Mamba,
   MegaBlocks, SonicMoE, the original Sparse Delta Memory repository, or another
   pinned primary implementation.
4. **URM lowering:** generated or selected specialized backend from `MixerSpec`.

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
  `not_applicable` with a reason, never as zero performance.

## Working milestone targets

These are engineering gates, not paper claims:

| Gate | Target |
| --- | --- |
| Semantic coverage | Dense attention, block-sparse attention, top-k MoE, recurrent mixer, parameter-token mixer, and sparse transactional memory all represented without arbitrary tensor escape hatches |
| Correctness | All reference and adapter tests pass; no silent semantic fallback |
| Dense attention overhead | URM dispatch/lowering is within 5% median latency of the selected upstream kernel on covered steady-state shapes |
| Other mature kernels | Within 10% median latency of the selected upstream implementation, or a documented reason to keep a specialized path |
| Memory locality | Page grouping reduces measured HBM bytes/token on at least one realistic trace family without changing routing results |
| Write path | One deterministic merge/commit per transaction with collision and version metadata |

A family that cannot meet the overhead target may remain a first-class specialized
backend. URM fails only if the contract repeatedly requires unrelated escape
hatches or prevents competitive lowering across the target set.
