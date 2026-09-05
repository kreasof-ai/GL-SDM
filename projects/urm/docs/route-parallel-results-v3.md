# Ordered URM route parallelism and resident state: review evidence

## Recommendation

**Proceed to independent integration review, with the resident-state prototype as the leading candidate. Do not integrate yet.** Both isolated candidates pass the requested native/oracle checks and have credible mode-specific projections below 1.05, including the conservative sensitivity calculation. The unchanged model still has frozen checksum failures and additional full-state diagnostic failures. These negatives, the oracle/upstream contract conflict, and the eager wrapper's extra temporary allocation must remain explicit review items. No further kernel tuning is needed to establish this screening result.

These are isolated training/prefill prototypes, **not production integration**.
The unchanged model and all previous negative artifacts remain intact. No triangular
chunk integration, value-tile search, architecture change, SDPA substitution, or
decode change was made. Both prototypes retain ordered token traversal and every
required BF16 state-store boundary, including the backward read-gradient store
before the write VJP. Selected write/read histories remain 192 MiB per invocation.

## Source and environment

All final v3 correctness, capture, isolated timing, resource, model-baseline, and
projection artifacts identify clean source
`19d008d7ab9b01d43641464552ee9c0f731fd470`.

The [review verification manifest](../results/pretraining-step/route-parallel-review-manifest-v3.json)
binds the final artifacts and fixtures by SHA256 and records source, launch,
initial-condition, and audit-consistency checks.

Implementation history:

- `d65b615`: isolated CUDA global-route and explicitly indexed shared-state kernels,
  forward/backward wrappers, tests, and experiment harness.
- `b36152e`: initialized native resource queries and complete actual-launch audit fields.
- `f69d315`: captured frozen-model operands, complete public-autograd timing,
  strengthened all-input VJPs, and reference-only lossless int64 address adaptation.
- `19d008d`: fixed and regression-tested the profiler's actual interval API.

Measured environment: NVIDIA A10G, SM86, **80 SMs**, 22,589 MiB reported device
memory, Python 3.12.13, PyTorch 2.8.0, Triton 3.4.0, CUDA/NVCC 12.9,
driver 595.91.07; OMP/MKL/PyTorch CPU threads = 1. Device UUID, platform,
compiler flags, binary hashes, launches, and source revisions are serialized.
The earlier commentary's generic 72-SM assumption was incorrect; the artifacts
record 80. No timings or projections depend on that mistaken count.

The frozen model remains microbatch 1, 12 heads, width 768, 12 layers,
T=1024, accumulation 4, 4096 slots, and 64/64 routes. Thus the state operator is
P=12,T=1024,S=4096,D=64,W=R=64. D tile remains **4**.

Read sources and mechanism rationale are in the
[experiment protocol](route-parallel-experiment.md): M²RNN paper and implementation
at `384ed0a7bd82ced1f40609603dd541cac5416844`, and ATMA at
`fcefd2d75f9db73c16850343e20b57b3b729b3ea`.
Their dense transitions, tensor-core results, inference kernels, and speedups
are not URM performance evidence. No dense backward history was copied:
URM's full BF16 history would be 6 GiB **per layer invocation**.

## Numerical evidence

[Correctness artifact](../results/pretraining-step/route-parallel-correctness-v3.json):
79 cases, 410 actual eager/compiled comparisons. Both candidates pass every
native/per-token-oracle check, with zero tolerance violations and no graph breaks.
Every candidate case runs both modes; compiled native covers production shapes
and T=129 rounding stress. Compiled paths each record one graph.

Coverage includes repeated, mixed and disjoint routes, BF16 rounding stress,
nonzero initial memory, both read timings, boundary lengths through 1024, and
three actual model captures. Joint losses depend on readings and final memory.
All six differentiable inputs are checked: initial memory, write weights,
values, beta, log decay, and read weights. Focused tests additionally use
unnormalized cotangents at D=4/64, zero selected write weights, unused output
cotangents, and an exact backward-cast cancellation regression.

Maximum errors across the two candidates and modes:

| Dtype | Reference | Forward/final-state max abs | All-input VJP max abs | Failed candidate comparisons |
|---|---|---:|---:|---:|
| FP32 | Native | 8.94e-7 | 1.03e-8 | 0 |
| FP32 | Per-token oracle | 7.10e-6 | 2.61e-8 | 0 |
| FP32 | Pinned upstream | 9.97e-5 | 2.69e-7 | 0 |
| BF16 | Native | 2.44e-4 | 1.91e-6 | 0 |
| BF16 | Per-token oracle | 4.89e-4 | 7.63e-6 | 0 |
| BF16 | Pinned upstream, separate | 0.125 | 4.28e-4 | 16 |

Frozen forward tolerances (atol/rtol): oracle FP32 `2e-5/2e-5`,
upstream FP32 `.02/.003`, BF16 both `.02/.02`.
Backward: FP32 `3e-5/3e-4`, BF16 `.03/.03`.
No tolerance or floating-point oracle was changed.

The preserved T=129 stress case is explicit in the artifact, for both read modes:
oracle/native **1.0**, upstream **0.875**. Their allowed intervals
**[0.96,1.04]** and **[0.8375,0.9125]** are disjoint. The 16 candidate/upstream
failures are the BF16 rounding-stress comparisons. They are reported separately,
not removed or used to redefine production semantics. Any acceptance-contract
change still needs a separate proposal and user decision.

Tests: 26 focused regressions, 23 authority/accounting/asset checks, and the
39 existing native/upstream GPU regressions passed. The final 49-test focused/
authority rerun is retained in the [validation log](../results/pretraining-step/route-parallel-validation-tests-v3.log).
Memcheck, racecheck,
initcheck, and synccheck logs retain zero errors/hazards for the unchanged CUDA
kernel. Additional full-slice race checks cover both index widths, dtypes, and
read timings. These sanitizer logs are supporting evidence from the earlier
implementation stage; the CUDA arithmetic has not changed. Final numerical,
resource, timing, and model authority is the v3 evidence at `19d008d`.

## Matched operands and complete-stage timing

[Captured operands](../results/pretraining-step/route-parallel-captures-v3.json)
come from a separate unchanged-model replay, with selection fixed before timing:
optimizer step 0, accumulation microbatch 3, layer 5 (zero based), seeds
1701/2903/4409. Incoming state is nonzero. Read/write overlap is approximately
99.45–99.48%; independently random read/write routes were not assumed representative.
Initial parameter, AdamW, batch, and persistent-state hashes match both actual
model backends and modes. Each timing pair uses the same serialized operands and
incoming cotangents. Independent replays need not be bit-identical; for example,
the v2/v3 seed-1701 forward operands matched exactly while incoming reading
cotangents differed by at most 1.19e-7. Pairing is within a fixed capture. One predeclared layer/microbatch is sampled per seed, not every model invocation; the unchanged candidate model grid remains necessary after review.

[Timing artifact](../results/pretraining-step/route-parallel-performance-v3.json).
Medians of the three per-seed medians, in milliseconds:

| Mode | Implementation | Forward ms | Backward ms | Complete autograd ms | Incremental peak MiB |
|---|---|---:|---:|---:|---:|
| Eager | Native | 28.485 | 58.940 | 87.369 | 220.14 |
| Eager | Route-global | 2.833 | 3.913 | 7.095 | 413.64 |
| Eager | Route-resident | 2.228 | 2.836 | 5.389 | 413.64 |
| Compiled | Native | 28.574 | 58.929 | 87.243 | 217.09 |
| Compiled | Route-global | 2.884 | 3.960 | 6.736 | 221.64 |
| Compiled | Route-resident | 2.270 | 2.883 | 5.035 | 221.64 |

Forward/backward columns are separate low-level phase measurements; the complete
column measures the **actual public autograd call**, not their arithmetic sum.
The raw artifact also retains manual-combined and CUDA-span timings, all samples,
and memory peaks. Ten warmups and twenty measurements are used per phase/mode/seed.
Runtime copies, initial/final state movement, history traffic, gradient-buffer
zeroing, casts, allocation, synchronization, and host/autograd orchestration are
included. Incoming cotangents and unchanged route preparation are stage inputs;
diagnostic scans/hashes/snapshots and build/compilation are outside timed calls.
No new candidate preprocessing is required.

Relative to isolated native, complete-stage speedups are **12.31× global / 16.21× resident in eager**, and **12.95× / 17.33× compiled**. Residency adds **1.32× eager / 1.34× compiled** over the otherwise route-parallel global implementation. Token order is unchanged in all three. Thus the experiment distinguishes substantial current-implementation cost from the ordered recurrence itself; it does not establish candidate model performance. Incremental peaks include outputs, saved histories and scratch above the already loaded common inputs, not whole-model memory.

The eager candidates have a measured extra **192 MiB** temporary allocation:
PyTorch materializes zero cotangents for the two non-differentiable history outputs.
The compiled path removes them. This is consistent with the exact peak delta,
the wrapper code, and PyTorch's documented
[non-differentiable-output gradients](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.function.FunctionCtx.mark_non_differentiable.html)
and [default gradient materialization](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.function.FunctionCtx.set_materialize_grads.html).
This cost was **not fixed or subtracted after measurement**. It is a bounded
wrapper follow-up for review, particularly for the frozen full-model memory gate.

## Launch, residency, and resource evidence

[Resource/launch artifact](../results/pretraining-step/route-parallel-resources-v3.json)
and its adjacent assembly directory contain the native PTX/cubins/SASS, CUDA SASS,
and NVCC resource log. Actual eager and compiled forward/backward launches match
serialized schedules. CUDA audit fields include grid, thread count, dynamic shared
bytes, shape, read timing, residency flag, and element/index widths.

| Implementation | Threads/CTA | Registers/thread F/B | Shared bytes/CTA | Spills/local bytes | Resource occupancy ceiling |
|---|---:|---:|---:|---:|---:|
| Native | 64 | 64 / 64 | 0 | 0 | 66.7% |
| Route-global | 256 | 39 / 40 | 128 | 0 | 100% |
| Route-resident | 256 | 40 / 38 | 32,768 + 128 | 0 | 50% |

These are resource ceilings, **not measured achieved occupancy**. All use
192 CTAs. On the recorded 80-SM device, the grid-wide average declared-warp ceilings
are 10% native and 40% for either CUDA candidate. A residency penalty cannot be
deduced solely from the per-SM ceiling.

Shared state is explicitly addressed as `state[slot*4 + local_d]`, with indexed
shared loads/stores visible in generated code, one initial full-slice load, and
one final write. It is not inferred from the size of a Triton tensor.
The global-to-resident comparison retains the same route-parallel arithmetic,
ordered recurrence, histories, and reduction scheme.

Native-to-route-global also keeps selected rows in registers instead of reloading
them and reduces log-decay gradient atomics across routes. Thus its gain is not
attributed solely to route parallelism, and the whole CUDA gain is not attributed
solely to residency. Backward improvements are measured, not inferred from forward.

[Hardware-counter attempts](../results/pretraining-step/route-parallel-counters-v3.json)
fail with `ERR_NVGPUCTRPERM` for all three implementations. Achieved occupancy,
cache hit rates, DRAM traffic and stall counters are unavailable on this host.
Driver privileges were not changed. Static instruction counts/logical accesses
are not presented as measured memory traffic.

## Unchanged-model authority and persistent state

[Full baseline artifact](../results/pretraining-step/route-parallel-model-baseline-v3.json)
contains all twelve fresh processes: both backends, both modes, three seeds,
five correctness / ten warmup / twenty measured steps, and four accumulation
microbatches. Each compiled lane has one actual graph with no breaks or
recompilation. Initial parameters, optimizer state, batches, and persistent state
are matched and checked. Internal gradients remain separate eager replays.

The additional full-state envelope is explicitly
`abs(candidate-reference) <= .02 + .02*abs(reference)`. It is diagnostic only.
Original GPU-produced checksum means and the historical `2e-6` gate remain separate.
Each microbatch/layer comparison reports finiteness, maximum absolute and RMS
error, violation counts, worst coordinates and paired values. Temporary full
snapshots are consumed only after these audits; the serialized metrics and
checksum records are retained. No user or prior result files were removed.

All **2,880 tensor comparisons are finite**. Each row below comprises 720
microbatch/layer comparisons (three seeds × five steps × four microbatches ×
twelve layers), or 2,264,924,160 paired elements.

| Comparison | Tensors failing additional envelope | Violating elements | Historical checksum failures | Max absolute error | Max per-tensor RMS |
|---|---:|---:|---:|---:|---:|
| Native vs upstream, eager | 660 | 35,645 | 7 | 0.087891 | 0.001186 |
| Native vs upstream, compiled | 656 | 39,212 | 0 | 0.079529 | 0.001182 |
| Compiled vs eager, native | 647 | 35,126 | 10 | 0.081299 | 0.001182 |
| Compiled vs eager, upstream | 651 | 34,694 | 2 | 0.077637 | 0.001168 |

There are 2,595 tensor comparisons where the checksum passes but the additional
elementwise diagnostic fails. The discrepancies are localized: 144,677 violating
elements among 9,059,696,640 paired elements, approximately 0.00160%.

The worst native/upstream eager element is seed 4409, step 4, microbatch 3,
layer 11, coordinate **[7,3534,58]**: native **0.0**, upstream **-0.087890625**,
allowance **0.0217578113**. The compiled worst is seed 4409, step 4, microbatch 1,
layer 11, **[7,1016,58]**: native **-0.0132446289**, upstream **-0.0927734375**,
allowance **0.0218554679**. All remaining worst coordinates and paired values
are in the artifact.

Original actual-execution backend correctness gates, ordered by seeds
1701/2903/4409: eager **pass/pass/fail**, compiled **pass/pass/pass**.
Cross-mode original gates are **fail/pass/fail for each backend**.
The final eager seed's largest checksum error is **2.93586e-6**, exceeding the
unchanged 2e-6 gate. Previous authority-v2 failures are not overwritten by
different pass/fail outcomes in a fresh run.

Separate eager mixer-input-gradient replays have minimum cosine at least
0.998886 and maximum element error below 8.95e-8. Their aggregate replay reports
also include checksum checks; those flags are not substituted for actual
compiled correctness. In particular, compiled-process eager replays fail
aggregate checks for seeds 2903/4409 while the actual compiled backend comparisons
pass. Both records are preserved.

These baseline differences are **not caused by a candidate in the model**:
neither candidate was installed. A passing mean checksum must not be described
as elementwise state equivalence. Conversely, the new elementwise diagnostic
is not silently promoted into a changed acceptance gate. Previous negative
authority-v2 and triangular artifacts remain unchanged.

## Mode-specific Amdahl screen

[Projection artifact](../results/pretraining-step/route-parallel-projections-v3.json)
uses
`R_predicted = R_baseline * ((1-f) + f/s)`.
Each mode uses its own fresh model ratio and measured device-kernel fraction.
Only actual native state kernels are credited as replaceable; candidates are
charged their **complete public-autograd stage**. Native copy/zeroing/cast and
other non-kernel costs stay in the remainder. The larger isolated native timing
is not substituted for the model's state-kernel cost.

| Mode | Measured baseline R | Measured f | Prediction at 3× | Required s for 1.05 | Global prediction | Resident prediction |
|---|---:|---:|---:|---:|---:|---:|
| Eager | 2.42194 | 0.830457 | 1.08106 | 3.14574 | **0.61389** | **0.56503** |
| Compiled | 2.31139 | 0.834987 | 1.02474 | 2.88664 | **0.56587** | **0.51929** |

The actual model's median state-kernel cost per invocation is **70.205 ms eager**
and **70.475 ms compiled**. Against the complete candidate costs, the effective
screening speedups are **9.90× / 13.03× eager** and **10.46× / 14.00× compiled**
(global/resident). These are intentionally smaller than the isolated-native
speedups. Both projections pass the requested screen; approximately 3× is not
a universal acceptance rule.

A deliberately conservative sensitivity check uses the baseline ratio's hierarchical-CI upper end, the smallest observed state fraction and kernel cost, and the largest observed candidate complete-stage time. Predicted ratios remain **0.629 global / 0.578 resident eager**, and **0.581 / 0.533 compiled**. This mixed-extrema sensitivity is **not a joint confidence interval**.

This is screening evidence, not measured candidate end-to-end performance.
The `<=1.05` screen is not automatic acceptance: the frozen full grid retains
its stricter geometric-mean/CI, per-process median, p95, numerical and memory
gates. SDPA remains a separate architectural control. Single-token decode is not
benchmarked or redirected to resident state.

## Reproduction and retained attempts

Use a **clean worktree at `19d008d`**, not the later artifact/report commit.
Run from `projects/urm` with the verified FineWeb cache and pinned upstream
checkout at `183e7df809131b80ad4393741029d0f20fc3640b`. The container must match the
recorded environment. Run GPU timing and profiling serially.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=src:benchmarks:/tmp/urm-sdm-upstream.wKxbSo
export LIBRARY_PATH=/opt/conda/lib

python -m pytest tests/test_route_parallel_experiment.py
python -m pytest tests/test_pretraining_authority.py tests/test_pretraining_accounting.py tests/test_benchmark_assets.py
python -m pytest tests/test_sparse_state_mixer_gpu.py tests/test_sparse_state_mixer_upstream_gpu.py
python benchmarks/route_parallel_capture.py \
  --output results/pretraining-step/route-parallel-captures-v3.json
python benchmarks/route_parallel_experiment.py correctness \
  --captures results/pretraining-step/route-parallel-captures-v3.json \
  --output results/pretraining-step/route-parallel-correctness-v3.json
TORCHINDUCTOR_CACHE_DIR=/tmp/urm-route-resource-audit-19d008d \
python benchmarks/route_parallel_experiment.py resources \
  --output results/pretraining-step/route-parallel-resources-v3.json
TORCHINDUCTOR_CACHE_DIR=/tmp/urm-route-performance-f69d315 \
python benchmarks/route_parallel_experiment.py performance \
  --captures results/pretraining-step/route-parallel-captures-v3.json \
  --correctness results/pretraining-step/route-parallel-correctness-v3.json \
  --output results/pretraining-step/route-parallel-performance-v3.json
python benchmarks/route_parallel_baseline.py \
  --output results/pretraining-step/route-parallel-model-baseline-v3.json
python benchmarks/route_parallel_projection.py \
  --baseline results/pretraining-step/route-parallel-model-baseline-v3.json \
  --performance results/pretraining-step/route-parallel-performance-v3.json \
  --correctness results/pretraining-step/route-parallel-correctness-v3.json \
  --captures results/pretraining-step/route-parallel-captures-v3.json \
  --output results/pretraining-step/route-parallel-projections-v3.json
```

The isolated v3 timing reused the timing-only Inductor cache named above, never an
audit cache; compilation is outside its steady-state boundary. The full-model
runner independently creates fresh process/compiler caches for every child.
Do not use `--development` or `--smoke` as authority.

Retained attempts:

- Numerical-v1 stops after 72 successful native/oracle cases because upstream's
  BF16 gather rejects int32 addresses. The adapter now losslessly converts only
  reference addresses to int64; native/candidate routes and floating arithmetic
  are unchanged.
- v2 isolated correctness and timings are complete and retained. The v2 model run
  failed in a separate post-timing profiler reader (`elapsed` versus `elapsed_us`);
  its failure record explicitly disclaims model authority. Temporary child files
  were cleaned by that failed run.
- v3 regenerates all final evidence from the fixed, tested clean commit.
