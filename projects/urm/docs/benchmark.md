# Training benchmark

The campaign measures ten-step decoder training on finewebedu, with checkpoint
and kernel gates. It does not establish source-model or serving parity; see
[evidence.md](evidence.md). The generated [report](../results/report.md) is the
source of current measurements and exclusions.

## Protocol

The default shape is width 768, nine layers, twelve heads of width 64, sequence
512, and vocabulary 50304. Both the effective optimizer batch and microbatch are
8192 tokens. Both arms compile the surround; Python plan dispatch and unsupported
upstream ops remain eager boundaries. Precision is bf16 autocast with fp32 kernel
accumulation. Timing is synchronized around the measured loop. MFU uses the
approximate parameter/state FLOP estimate and adopted A10G peak of 70 TFLOPS.

Two full optimizer steps warm up the run. Warmup and measured steps use identical
accumulation and gradient clipping. Optimizer parameter roles are the same for
compiled and eager models. Equal-shaped Muon weights are processed in bounded
batches with the same independent Newton-Schulz iterations.

Memory-heavy rows use explicit activation checkpointing in both arms. No OOM
fallback runs enter the production table. OOM by itself does not demonstrate a
memory leak: saved recurrent states can exceed the device's capacity. Every
measurement records step-end allocation and peak allocation so persistent growth
can be distinguished from temporary activation demand. Completed loss graphs are
released before the next forward; correctness gates run after the large timed
model and optimizer are released.

## Running

Run from `projects/urm` with `PYTHONPATH=src:.`:

```sh
python -m train.sweep --out-dir results/sweep
python -m train.upstream --out-dir results/upstream --subprocess
python -m train.report --sweep-dir results/sweep \
    --upstream-dir results/upstream --out results/report.md
```

Each row runs in an isolated process. Cached successes must match the complete
config, measurement version, and source fingerprint. Errors are persisted and
retried on the next invocation. Logs retain commands and all attempted batches.
`--rows` selects a subset. `--eager` is available for both arms.

`--allow-oom-fallback` enables diagnostic retries at smaller microbatches while
preserving `--batch-tokens` through accumulation. These results are excluded from
production comparisons. `--include-reference` runs upstream reference/research
implementations as diagnostics; they cannot enter the production table.

## Gates and comparison

Non-finite loss or gradient norms fail immediately, including during warmup.
Successful JSONs contain every measured loss and memory allocation, the actual
config, environment, and source fingerprint. NaN is not a successful measurement
or valid JSON output.

Checkpoint gates use reduced eager models. Deterministic rows must resume exactly;
relaxed-atomic rows must round-trip state exactly and satisfy the loss-trajectory
criterion. These gates certify resume behavior; full-scale finite training is
checked independently. KL is retained where an upstream comparator is wired.

The report admits only finite, checkpoint-verified production runs with matching
shapes, effective batch, microbatch, execution and precision policies, activation
checkpointing, environment, and source fingerprint. It distinguishes shared
frontends from architecture-family baselines with different mixer-side layers.
Parameter counts remain explicit. Unavailable or failed production kernels have
no substitute reference throughput in the paired table.

## Corrections

Kernel math in `src/urm/` remains unchanged. Two necessary autograd-wrapper
fixes replace captured bias/mask and decay/beta tensors with immutable metadata;
capturing those tensors retained upstream graphs after backward. Full-size
probes show flat step-end allocations after the fix. Frontend/harness corrections include
normalized delta operands, contractive IPLR feedback, bounded low-rank factors,
Based's upstream default of 16 query/key features, batched MoM packing, and block
triangular scheduling through the existing public solve. Strict-past softmax
avoids an all-masked first row with NaN derivatives. K1 operands retain training
autocast precision. Pattention's default mode uses the public native softmax reducer multiplied by
the parameter-token count. Other modes retain the public map/normalizer with
bounded value tiles. AttnRes packs
spatial positions as indexed depth queries. Tucker computes wide scores once
before the public softmax reduction.
Native training for TDA and Differential Attention composes public K1 calls with
an external differentiable merge. This preserves Q/K/V and mixing-weight gradients
that the combined native merge dispatch previously dropped. TDA's identical-path
recipe reduces exactly to one call multiplied by the merge weight.

Pinned production adapters for Comba, GDN2, DeltaProduct, DPLR, RWKV-7, Mamba-2,
and both log-linear rows use the same external frontend as URM. Mamba's pure
Triton SSD package is loaded without its optional CUDA-extension initializer.
RWKV uses a supported small chunk; log-linear attention uses a one-stage Ampere
pipeline, bf16 operands (including level scales), independent-head batches,
and final-scale repetition for the capped bank. The pinned checkouts remain unmodified.
Samba uses that same SSD branch and the same RoPE attention frontend with
production SDPA, avoiding the Mamba-1 slow-path fallback. Raven shares the eight-slot/top-k-two deterministic frontend and calls pinned
chunk GSA with equally duplicated slots. Duplication preserves the softmax-weighted
result and meets the production backward kernel’s minimum of sixteen slots.
TDA leaves its pinned Triton call eager and supplies the contiguous head batches
and output gradients required by its launcher while compiling the surround.
Its adapter also matches native query scaling and selects IEEE fp32 dot products
with one pipeline stage through supported Triton launch options. The registered
differential recipe uses identical paths with lambda 0.5; the production adapter
implements the exact merge as half of one kernel call.
No flash-attn installation or source build is required. Unavailable production
training kernels remain explicit exclusions.
