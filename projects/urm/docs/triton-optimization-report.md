# Routed-reduction Triton backend optimization report

Scope: routed-reduction v1 semantic contract (`docs/triton-backend.md`) preserved
throughout. No changes to layouts, dtypes, fp32 accumulation, output dtype,
index-bounds policy, or gradient semantics. Route generation was not fused.

## 1. Hardware / software identity

| Item | Value |
| --- | --- |
| GPU | NVIDIA A10G, compute capability 8.6, 23028 MiB |
| Driver | 595.91.07 (CUDA 13.2 driver API) |
| Platform | Linux-6.12.95-124.187.amzn2023.x86_64 (x86_64) |
| Python | 3.12.13 (/opt/conda) |
| PyTorch | 2.8.0 (CUDA 12.9 build) |
| Triton | 3.4.0 |
| CUDA toolkit headers | cuda-cudart-dev-12-9 12.9.79 (installed for Triton host glue) |
| Compute sanitizer | 2025.2.1.0 (memcheck) |

Environment fix before any benchmarking: the host had no CUDA toolkit headers
(`cuda.h`), so Triton's `cuda_utils.c` glue failed to compile. Installed
`cuda-cudart-dev-12-9` via apt and linked its include directory into
`/usr/local/include`. This is a host provisioning fix; no repo behavior change.

## 2. Pre-optimization acceptance gates

1. Preflight: `PYTHONPATH=src python benchmarks/triton_preflight.py --require-ready` → ready.
2. Full suite: 45 passed (13 Triton GPU tests: fp16/bf16/fp32 × int32/int64 ×
   K ∈ {1,3,8,32} × D < = > tile × non-power-of-two shapes).
3. compute-sanitizer memcheck over the whole GPU test file: **0 errors**.
4. Result JSONs validate against `benchmarks/result-schema.json` (100 files).

### Compilation failure fixed first

Triton 3.4 rejects storing a shape-`(1,)` block through a scalar pointer
(`ValueError: Value argument cannot be block type if pointer argument is not a
block`) in `_routed_reduce_grad_weights_kernel`. Fixed by accumulating the dot
product into a scalar (`tl.zeros((), tl.float32)`); math and dtype identical.
This was required before the committed backward test could run at all.

## 3. Baseline (unmodified code, median/p95 µs, steady state)

Cold compile/first-call time is recorded separately in every JSON
(`cold_compile_or_first_call_ms`); all numbers below are CUDA-event steady
state with index validation disabled after trusted input generation.

| Case | Mode | torch med | triton med | triton p95 |
| --- | --- | ---: | ---: | ---: |
| smoke (Q32,K8,D128,f16) | fwd | 120.8 | 108.9 | 136.2 |
| smoke | bwd | 472.9 | 423.6 | 468.1 |
| decode_top2 (Q1,K2,D4096,bf16) | fwd | 122.3 | 108.8 | 127.9 |
| decode_top2 | bwd | 492.4 | 440.1 | 468.5 |
| prefill_top8 (Q8192,S256,D4096,bf16,skew) | fwd | 11739.0 | 425.1 | 429.7 |
| prefill_top8 | bwd | 36521.6 | 6109.4 | 6114.7 |
| memory_top32 (Q4096,S65536,K32,D256,f16,recurrent) | fwd | 1429.8 | 116.9 | 147.7 |
| memory_top32 | bwd | 4725.2 | 1102.6 | 1122.3 |
| non_power_of_two (Q257,S997,K7,D190,fp32) | fwd | 94.9 | 109.8 | 141.2 |
| non_power_of_two | bwd | 419.3 | 424.8 | 542.0 |

Baseline profiling split: small-Q cases were ~98% host dispatch (kernel ≈1–2 µs
vs ≈110–320 µs wall); prefill/memory were GPU-bound.

## 4. Accepted changes (each re-ran the full 5×2 case matrix + gates)

### C1 — Host/dispatch fast path (results/c1-host-fastpath)
- Memoized capability checks keyed by exact tensor metadata (shapes, strides,
  dtypes, devices, contiguity). Cache misses run the original validation path
  and raise identically; hits skip object construction only.
- Removed runtime stride arguments from all kernels: v1 requires row-major
  contiguous inputs (`require_row_major` runs upstream), so offsets are derived
  from constexpr shapes. Fewer args → cheaper binding and address math.
- No-grad fast path: `routed_reduce` skips autograd plumbing when no input
  requires grad (observable behavior identical).
- Direct cached-kernel launch through Triton's `CompiledKernel.__getitem__`
  runner, keyed by (jit fn, device, grid, warps, constexprs, dtypes,
  16-byte pointer-alignment bits), falling back to standard JIT dispatch when
  launch hooks are active or on key miss (compilation stays excluded from
  steady-state timing by the harness design).
- Effect: decode-shape forward host cost 111→38 µs.

### C2 — Forward GPU regime tuning (results/c2b/c2c-forward-gpu)
- `EVEN_D` constexpr elides dimension masks when `VALUE_DIM % BLOCK_D == 0`.
- Launch table: large-Q regimes (queries ≥ 1024) use `num_warps=2`; others 4.
  Offline sweep showed warps=2 + mask elision gives 2.7× kernel-time win on
  prefill gather workloads (more resident programs per SM).
- Backward keeps its own heuristic (`_backward_launch_parameters`); an
  intermediate variant that let forward's table leak into backward regressed
  prefill backward 27% and was rejected (results/c2-forward-gpu retained).

### C3 — Backward restructure + config table (results/c3b-backward)
- New per-query grad-values kernel: one program per (query, d-tile) loads
  `grad_output` once and loops routes, instead of reloading it per route.
  Atomic adds remain per-route fp32, so duplicate-route correctness and the
  documented nondeterminism are unchanged. Selected for queries ≥ 1024;
  small-Q keeps the original per-route kernel (per-query regressed np2/smoke).
- Backward config table: BLOCK_D=512/warps=8 for value_dim ≥ 1024,
  BLOCK_D=256/warps=4 otherwise (from offline sweep).

### C4 — Relaxed atomic ordering (results/c4b-relaxed-atomics)
- `tl.atomic_add(..., sem="relaxed")` on both grad-values kernels. Atomicity
  and accumulation are unchanged (fp32 adds, order already nondeterministic);
  cross-kernel visibility still guaranteed by stream ordering.
- grad-values: −14% (memory regime), −4% (prefill regime).

## 5. Rejected experiments (measured, then discarded)

| Candidate | Measurement | Verdict |
| --- | --- | --- |
| Block-K broadcast gather (2-D K×D tile) forward | slower/neutral vs mask-elision loop on every regime (e.g. memory 29.3 vs 23.3 µs) | rejected |
| Software pipelining (`num_stages=2..4`) on route loops | zero effect on gather loops; no effect on atomic-bound loops (3962.9/3962.7/3962.5 µs) | rejected |
| Split-K grad-weights (partials [routes, tiles] + reduce kernel) | prefill 2021 vs 1861 µs; memory 343 vs 269 µs; also adds workspace memory | rejected |
| Per-query grad-values for small-Q | np2/smoke backward −6…−13% (fewer programs, serial K-loop) | rejected via regime split |
| Forward table applied to backward launches | prefill backward 6100→8361 µs | rejected (c2 intermediate) |
| CUDA-graph launch capture | rejected without implementation: static-address assumptions conflict with general tensor-identity contract; noted as possible future capability | rejected |

## 6. Final results (results/final, median/p95 µs, speedup vs baseline)

| Case | Mode | torch | triton final | baseline triton | speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| smoke | fwd | 125.8 | 38.3 | 108.9 | **2.84×** |
| smoke | bwd | 492.1 | 308.6 | 423.6 | **1.37×** |
| decode_top2 | fwd | 130.3 | 41.7 | 108.8 | **2.61×** |
| decode_top2 | bwd | 513.4 | 306.1 | 440.1 | **1.44×** |
| prefill_top8 | fwd | 11739.1 | 156.4 | 425.1 | **2.72×** |
| prefill_top8 | bwd | 36532.4 | 5186.3 | 6109.4 | **1.18×** |
| memory_top32 | fwd | 1431.6 | 39.5 | 116.9 | **2.96×** |
| memory_top32 | bwd | 4725.0 | 715.8 | 1102.6 | **1.54×** |
| non_power_of_two | fwd | 95.4 | 39.0 | 109.8 | **2.82×** |
| non_power_of_two | bwd | 386.5 | 269.8 | 424.8 | **1.57×** |

Constraints at final state:
- Correctness: 45/45 tests pass; sanitizer memcheck 0 errors; triton-vs-torch
  max-abs errors unchanged within documented atomic-order wiggle.
- p95: no violation against the calibrated noise model (host-bound cases show
  ±10–15% same-code spread on this shared host; GPU-bound cases <1%).
- Peak allocated memory: byte-identical to baseline on every case/mode
  (no workspace was introduced).

Final gates re-run on shipped code: pytest 45 passed, ruff clean, sanitizer 0
errors, schema validation of all result JSONs.

## 7. Remaining bottlenecks

1. **Prefill backward grad-values atomics (~3.9 ms of 5.2 ms)**: skewed routes
   hammer hot rows; RMW serialization at L2 dominates. A conflict-free
   schedule (sorting/segmentation) is explicitly reserved as a separate
   deterministic capability in the v1 doc and was out of scope.
2. **Small-Q wall time floor (~30–45 µs)**: now dominated by Python/Triton
   dispatch (~18 µs/launch floor measured with a no-op kernel), allocator, and
   autograd engine. CUDA-graph capture would cut most of the remainder but
   needs a scoped contract decision (static addresses per capture).
3. **Host-bound measurement noise**: ±10–15% run-to-run on sub-millisecond
   cases (co-hosted machine); GPU-bound cases reproduce within ~0.5%.
4. **High-K grad-weights bandwidth**: per-route programs re-read `grad_output`
   K times; the split-K alternative that amortizes it measured slower due to
   partial-buffer traffic. A fused tile-resident multi-route design may help
   but requires register-pressure management beyond this campaign.

## 8. Artifacts

- `results/baseline/` — unmodified forward/backward JSON per committed case.
- `results/c1-host-fastpath/`, `results/c2-forward-gpu/` (rejected
  intermediate), `results/c2b-forward-gpu/`, `results/c2c-forward-gpu/`,
  `results/c3-backward/` (rejected intermediate), `results/c3b-backward/`,
  `results/c4-relaxed-atomics/`, `results/c4b-relaxed-atomics/`,
  `results/final/` — complete matrices retained before/after each step.
- `benchmarks/compare_results.py` — constraint checker (median hill-climb
  target; correctness/p95/peak-memory gates; torch-backend drift used as the
  session noise reference).
