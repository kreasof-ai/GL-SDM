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
