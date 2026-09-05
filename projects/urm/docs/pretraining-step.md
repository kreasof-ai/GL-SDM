# Model-level Sparse Memory pretraining-step contract

**Status:** measured model-level regression with one frozen-tolerance
correctness miss; no parity or superiority claim.

This is an independently owned URM decoder language model, not a vendored
nanoGPT or modded-nanoGPT program. The authoritative boundary starts with a
prefetched FineWeb-Edu token batch and ends after a complete AdamW optimizer
update. Every one of the 12 blocks executes learned score, value/gate, and
output projections, its selected sequence mixer, two residual connections,
LayerNorm, a 4x GELU MLP, language-model logits and cross-entropy, backward,
gradient clipping, and FP32-master AdamW. No activation checkpointing is used.

## Frozen primary lane

`benchmarks/pretraining_step.toml` freezes a 124,651,008-parameter decoder:
12 layers, width 768, 12 partitions of value dimension 64, context 1024,
vocabulary 50,304, BF16 model tensors, and FP32 optimizer state. Sparse mixers
use 4,096 slots per partition and the maximum common factor width: 64 writes
and 64 reads. Microbatch 1 makes `batch * partitions = 12`, within the native
`P<=16` envelope; four accumulation microbatches produce 4,096 tokens per
optimizer step. This configuration may not be reduced after measurement. The
separately named diagnostic lane (2 layers, width 256, context 128) is never a
substitute for the primary result.

At each optimizer-step boundary all persistent memories reset to zero. Within
the four accumulation microbatches the forward-final memory becomes the next
microbatch's initial memory, then is detached. This retains realistic state
lifetime without backpropagating across microbatch boundaries. The pinned
upstream backward restores its mutable working state, so its model wrapper
snapshots the forward-final state before backward; the native result is already
functional. Both paths implement the same reset/persist/detach policy.

The token cache is
`karpathy/fineweb-edu-100B-gpt2-token-shards` revision
`a33f75d78c7f74236fb03754ec1eb3cc77507d64`, shard
`edu_fineweb_train_000001.bin`, SHA-256
`6bb7ce7bcac8e11463433767ec3402311c7527c3d8d766e7d65ef86dc4546bb2`.
The 200 MB shard is deliberately ignored by Git. Authoritative timing uses
deterministically selected, prefetched CUDA batches. Memmap materialization and
host-to-device copy are measured in a separate data-loading lane.

## Compared execution

- `upstream_sdm`: exact pinned original-SDM product-key and gated write/read
  implementation at commit `183e7df809131b80ad4393741029d0f20fc3640b`.
- `urm_native`: each layer obtains an executable `CompiledSparseMemoryPlan`
  from the typed compiler. The plan verifies and consumes its serialized route
  and state schedules; the benchmark never constructs the Triton backend.
- `sdpa`: causal PyTorch scaled-dot-product attention, reported only as an
  architectural context control. It is not a Sparse Memory equivalence path.

The pinned upstream PyCapsule kernels cannot be fake-tensor traced directly.
For the fullgraph lane, comparator-only integration glue registers opaque
`torch.library` forward and VJP operators that invoke the pinned upstream
forward/backward implementation. This API stays entirely in the external
adapter layer and never enters semantic IR. Native execution remains independent
of the upstream checkout. The comparator wrapper is first-order and
single-process/non-reentrant; it exists only for this controlled benchmark and
is not advertised as a production execution anchor.

## Correctness and measurement

Before timing, every sparse pair executes five seed-matched eager optimizer
steps. The artifact retains before/after losses, fixed-position logits,
per-block/parameter-group gradient norms, mixer-input gradient cosine and
maximum error, parameter-update deltas, persistent-state checksums, and
non-finite counters. Frozen BF16 tolerances are stored in the TOML. The compiled
production lane additionally requires one graph, zero graph breaks, and zero
recompilations.

Each backend/mode/seed runs in a fresh process with fresh Triton,
Torch-extension, and Inductor caches. After five correctness steps, the process
records first-step cost, completes ten total warmup/settling steps, and measures
20 complete optimizer steps with synchronization only at step boundaries.
Three seeds use randomized upstream/native process order. Raw times, median,
p95, hierarchical bootstrap CI, tokens/s, steps/s, and memory peaks are
retained. A failed performance gate is written as evidence and labelled
regression or parity; it is never hidden by changing the model.

Cold evidence separates Torch import, URM-module import, pinned comparator
import/revision probing and extension build when applicable, model/optimizer
construction, the `torch.compile` wrapper call, and the first complete optimizer
step. The compile wrapper is labelled literally because PyTorch compilation is
lazy; compiled code generation is intrinsically present in that lane's first
optimizer-step measurement rather than misreported as wrapper time.

`benchmarks/profile_pretraining_step.py` emits learned-projection, route,
state, block, logits, loss, backward, clipping, and optimizer ranges into a
Torch CUDA trace. The same ranges are NVTX-visible under the documented `nsys`
command. Profiling is a separate pass and is disabled in authoritative timing.

## FLOPs and memory

The semantic FLOP ledger counts useful trained projections, normalization,
MLP, logits/loss, selected-route Softmax, selected sparse state transitions,
backward, and AdamW. Sorting, top-k comparisons, padding, redundant work, and
recomputation receive no useful-FLOP credit. Sparse score/state work remains
separately attributable even though model MFU uses the repository's measured
66.16599973571093 BF16 tensor-core TFLOP/s denominator. Tokens/s is the primary
architecture-neutral metric.

Memory accounting independently reports parameters, gradients, FP32 master and
moment state, persistent memory, tensors saved for backward, temporary peak,
allocated/reserved peak, and first-step/compilation peak. Peak statistics reset
immediately before every measured step and are read only after synchronization.
The common primary microbatch is 1; larger backend-only batches require a
separate capacity experiment and cannot affect fixed-comparison claims.

## Pre-confirmation optimization record

Model profiling identified the partition-owned state recurrence, not route
production or the outer decoder, as the dominant native stage. The original
`D=64` schedule assigned one long-running value fragment to each of 12
partitions. Controlled two-warp value fragments measured 4,966.6 ms at `D=32`,
4,811.0 ms at `D=16`, 4,382.9 ms at `D=8`, and 4,260.8 ms at `D=4`, versus the
5,704.7 ms original `D=64` baseline. `D=4` was retained. `D=2` measured
4,255.0 ms—only 0.14% below `D=4` in a single settled exploratory step—so it
was rejected as noise rather than paying for twice as many fragments and
scalar-gradient atomics. These are exploratory diagnosis numbers, not
acceptance evidence.

The corresponding pinned-upstream exploratory median was 1,772.9 ms. Therefore
value tiling alone cannot establish model-level parity. If the clean
three-process artifact confirms this gap, the concrete remaining technical
work is a new exact, chunk-parallel ordered-recurrence algorithm; changing the
frozen model or continuing isolated value-tile hill climbing is not an allowed
substitute.

## Reproduction

```bash
export PYTHONPATH=src:/path/to/sparse-delta-memory
export LIBRARY_PATH=/opt/conda/lib:${LIBRARY_PATH:-}
python benchmarks/pretraining_step.py \
  --output results/pretraining-step/confirmation.json
python benchmarks/profile_pretraining_step.py --backend urm_native \
  --output results/pretraining-step/native-profile.json
nsys profile --trace=cuda,nvtx python benchmarks/profile_pretraining_step.py \
  --backend urm_native --emit-nvtx
```

The result must validate against
`benchmarks/pretraining-step-result-schema.json`. A superiority claim requires
all thresholds in the frozen TOML; otherwise the artifact's failing gates are
authoritative.

## A10G result and blocker

The clean three-seed artifact at `results/pretraining-step/confirmation.json`
reports native/upstream geometric-mean optimizer-step ratios of **2.365** in
eager mode (95% hierarchical CI **[2.340, 2.387]**) and **2.245** under
fullgraph compilation (CI **[2.221, 2.273]**). Native medians are
4.164–4.193 seconds eager and 4.149–4.176 seconds fullgraph, versus
1.751–1.778 and 1.833–1.867 seconds upstream. Native peak allocated memory is
lower in both lanes, and every compiled run has one graph, zero graph breaks,
and zero recompilations.

All eager five-step correctness comparisons passed. Fullgraph seed 1701 step
five missed only the frozen normalized persistent-state checksum tolerance:
`2.0007428247481585e-6` versus `2.0e-6`; loss, logits, gradient norms,
parameter updates, mixer-input gradients, and finite checks passed. The
tolerance was not relaxed after measurement, so the artifact truthfully records
`correctness_failure` rather than performance regression alone.

The native profile identifies the ordered state recurrence as the dominant
device range. Together with the D=64/32/16/8/4/2 schedule sweep, this is the
concrete blocker: value tiling improves occupancy but cannot remove the serial
`T * route_width` recurrence. The smallest credible next step is an exact
chunk-parallel recurrence with a proved ordered-collision backward, followed by
the unchanged model grid. No model-width, route-width, context, or tolerance
change is justified by these results.
