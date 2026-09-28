# Optimization handoff

Development is paused at the author's request on 2026-09-27. Resume optimization
on a larger GPU. This document records planned work; none of the optimizations
below have been implemented. The last implementation commit is `bf4d595`.

## Starting point

All four models have 16 distinct layers and no weight loops. The primary
configs use length 2,048, 8,192 tokens/update and microbatch one, accumulating
four independent sequences. GL-SDM has one shared bank, repeats
`local → local → global → local` four times, and uses a 512-token sliding window
and 512-token write chunk. Each chunk reads one frozen snapshot and sums writes
without token/layer averaging. Later chunks consume earlier commits. Keep this
architecture fixed during the first optimization comparisons.

Current precision is **BF16 dense weights, FP32 residuals and FP32 GL-SDM state**.
The BF16 state/residual conversion discussed with the author remains pending.
SDM's upstream bank is already BF16. FP32 accumulation for sensitive arithmetic
can remain even when persistent weights, state and residuals are BF16.

The [current report](results/report.md) records preliminary A10G measurements:

| Model | 6ND MFU | Training peak allocated, GiB | Training peak reserved, GiB |
| --- | ---: | ---: | ---: |
| Transformer | 29.50% | 6.42 | 6.81 |
| GDN2 | 24.63% | 7.08 | 7.46 |
| SDM | 7.10% | 6.93 | 7.19 |
| GL-SDM | 4.90% | 11.45 | 15.41 |

These are whole updates, including backward, clipping and AdamW, with one
warmup and three measured updates. They are not converged training results.
The earlier 17.20 GiB GL-SDM peak included inference, not training alone.
Allocated and reserved memory are distinct; fitting a 12 GB GPU is not proven.
The last complete test run passed 92 tests.

## Why the current implementation is slow

The saved [GL-SDM profile](results/sequence_2048/gl_sdm_memory_profile.json) and
[SDM profile](results/sequence_2048/sdm_memory_profile.json) use the current
configs, batch one and length 2,048. They measure forward/backward only, without
clipping or optimizer updates. Their source fingerprints and timings are
preserved exactly from the diagnostic runs.

GL-SDM totals 471.18 ms of GPU kernel time: FP32 additions account for 45.84%
and fills for 17.79%. SDM totals 203.11 ms under the same profiling procedure.
These instrumented kernel totals are not whole-update wall times or MFU.

Each GL-SDM sequence performs 16 global query reads and 12 write-prediction
reads. Training pads state width 64 to 128, and each native read backward
produces a dense FP32 gradient of shape `[1, 8, 262144, 128]`: **1 GiB per read**.
The profile records 28 such zero fills and repeated full-bank gradient additions.
Functional commits and padding backward add more full-state traffic. These
allocations are not all simultaneously resident.

Four chunks of 512 tokens are **four dependent chunk forwards**, not 2,048
concurrent positions. Within a layer, 512 tokens run in parallel. All 16 layers
execute in order before committing the chunk. Microbatch four would expose
four independent sequences, or 2,048 positions per layer, simultaneously; the
current gradient accumulation processes those sequences separately.

## What URM already provides

Keep URM frozen at the exact commit in
[shared/requirements-urm.txt](../../shared/requirements-urm.txt):
`604bfdf5d2c827266a32ef142ca996cc712d70f0`.

- Native product-key route selection, including score gradients.
- Native reads using supplied routes, including backward.
- BF16 and FP32 sparse-state read support. Our adapter's FP32-only plans and
  `MemoryView` contract are project choices, not a lack of BF16 support in URM.

URM's current read backward returns a dense gradient for its input state. It
does not automatically coordinate gradients from the project's separate layer
reads or implement GL-SDM's shared transaction clock and overlays. Initially
address those costs in GL-SDM while reusing URM routing and reads.

The existing factor-512 routing override permits only eight routes, FP32 scores
and INT32 indices. Retain hardware/dependency checks and restoration of the
support probe. BF16 state reads do not imply that BF16 factor-512 routing is
officially supported. Separate routing/read plans and explicit weight casts
are one option to investigate while retaining the existing routing override.

Do not edit or vendor URM. New generic URM features or a pin update require a
separately scoped URM task and validation of dependent workloads. Avoid adding
an SDM-specific exception to URM or replacing its operations without recording
the reason. Keep the project memory contract reusable by CSDM.

## Implementation order

1. **Reproduce and isolate correctness.** Run the unchanged configs on the new
   GPU and record its software, GPU model and dense BF16 peak denominator.
   Isolate SDM at fixed routes and identical operands before attributing its
   discrepancy to upstream. Our SDM oracle keeps recurrent state in FP32 while
   production stores BF16 state; align storage/rounding policies for the BF16
   comparison and retain a separate mathematical FP32 check. Trace GL-SDM's
   first read, hidden-state and route divergence. Do not loosen tolerances to
   make an unexplained failure pass.
2. **Use BF16 storage and residuals.** Make bank/cache dtype explicit in
   `memory/state.py`, update native read plans and project commits, and preserve
   historical configurations/checkpoints. Keep required FP32 reductions and
   accumulator arithmetic explicit. Test logical width 64 rather than assuming
   padding to 128 remains necessary or faster on the new GPU. Benchmark this
   precision change separately; BF16 storage alone will not remove URM's FP32
   gradient workspaces.
3. **Accumulate selected-row gradients.** Preserve native URM route selection,
   remap selected addresses to compact rows and investigate native supplied-route
   reads over that compact state. Explicitly coordinate gradient contributions
   across reads and versions. Materialize the full learned-bank gradient fewer
   times, ideally once per backward when using the current dense optimizer.
   A normal gather before URM is insufficient if its backward allocates another
   full-bank gradient. Include write proposals and gradients through prior
   commits, not only query reads. This is the main engineering task.
4. **Represent commits as compact overlays.** Keep an immutable base and
   versioned touched-row updates instead of cloning the entire bank per commit.
   Resolve a snapshot read from the correct version, preserve canonical
   collision order and BF16 rounding at commit boundaries, and carry gradients
   back through updates and the learned initializer. Bound overlay growth and
   define when to materialize dense serving state. Reuse the same contract for
   CSDM rather than creating an architecture-only memory wrapper.
5. **Increase batching and reduce launch overhead.** At the same 8,192
   tokens/update, compare microbatches 1, 2 and 4 on the larger GPU. Batch four
   gives 2,048 independent positions per chunk layer. Compile suitable dense
   surrounds and evaluate graph capture only after memory lifetime and shapes
   are stable. Measure local attention too: its work is excluded from 6ND and
   GL-SDM has 12 sliding-attention layers absent from the SDM baseline.

The expected scope is a substantial project memory/autograd rewrite with the
model equations retained, followed by batching work. The rough planning estimate
was 3–7 focused engineering days including validation; it is not a guaranteed
completion time or a promise of 40% MFU. Removing all of the approximately 64%
addition/fill cost would ideally give about 2.8× improvement in profiled GPU
kernel time. Reaching 40% from 4.90% requires about 8× whole-update improvement,
so the remaining stages and new hardware still need measurement.

## Verification and measurement

For each stage, verify outputs, all relevant parameter gradients, shared-bank
write gradients from all four global layers, canonical collision behavior,
future-token causality, split prefill/decode across 512-token boundaries,
absolute sliding-window positions, request isolation and checkpoint resume.
Cover both small dense fixtures and the actual 134,217,728-entry bank. FP32
correctness does not establish BF16 correctness. SDM/GDN2 baseline kernels must
remain the actual pinned upstream implementations, without reference fallback.

Record allocated/reserved memory separately for training, prefill and decode,
plus step-end allocations, full-bank gradient/fill counts, bytes moved and native
URM plans. Compare each change against an unchanged run on the **same new GPU**.
Keep architecture, bank capacity, sequence length and tokens/update fixed;
record microbatch and precision changes explicitly. Preserve 6ND accounting,
its exclusions and the hardware denominator. Do not claim hardware-independent
MFU gains from comparisons against A10G.

Use a fresh result directory for each experiment, with at least three warmups
and ten timed updates for follow-up performance comparisons. Retain failures,
commands, configuration, source fingerprints and upstream pins. A workload
reduction or alternate kernel must be an explicit control, never an OOM fallback.
After correctness passes, rerun the FineWeb-Edu pilot; converged training and
long-context/needle quality evaluation remain outstanding.

## Resume commands and local assets

From the repository root on the new instance:

```bash
python -m pip install -e 'projects/gl-sdm[test,urm]'
python projects/gl-sdm/scripts/setup_sources.py
python -m pytest -q projects/gl-sdm/tests
python projects/gl-sdm/scripts/run_layer_experiments.py --phases benchmark --warmup 3 --iterations 10 --output-dir projects/gl-sdm/results/new_gpu_baseline
```

Install the new GPU's compatible Torch/CUDA environment first. For a GPU absent
from `experiments/metrics.py`, use individual `gl-sdm benchmark` commands with
an explicit `--peak-tflops` for dense BF16 throughput; do not reuse A10G's 70.
Use `gl-sdm verify --verify-batch-size 1 --length 513` with the desired config
to cross a memory commit. Use `scripts/profile_layer_memory.py` for the separate
forward/backward diagnostic.

Git preserves source, configs, logs, measurements, profiles and this plan.
Pilot checkpoints under `projects/gl-sdm/checkpoints/`, token shards under
`data/finewebedu10B/`, installed packages and JIT/source caches are **local and
ignored by Git**. They are not part of the push. The old checkpoints are only
short pilots; reconstruct sources from their pins and regenerate data/pilots
if these local assets are not retained separately. Never resume a length-512
pilot as though it used the current length-2048 configuration.
