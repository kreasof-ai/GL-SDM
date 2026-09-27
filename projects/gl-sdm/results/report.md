# GL-SDM kernel results

The optimized **four-step GL-SDM reaches 41.48% MFU** on an NVIDIA A10G.
Eight steps reaches **25.08%**, so the 40% target is met only by the four-step
configuration. Long whole-model reference checks still fail their strict
elementwise limits; this is a performance result with an unresolved precision
limitation, not a claim that GL-SDM is ready to freeze for quality comparisons.

## Complete training steps

All rows use BF16, width 512, vocabulary 50,304, eight heads, 4,096 slots/head,
eight read/write routes, chunk size 1,024, sequence length 2,048 and 8,192
tokens/update with microbatch 4. They time forward, backward, clipping and AdamW.
MFU uses unchanged **6ND** with 57,823,824 unique active parameters and A10G's
70 dense BF16 TFLOP/s. The 2,097,152 learned memory parameters are excluded;
reasoning depth and state work do not inflate the numerator.

| Configuration | CUDA graph | Median step | 6ND MFU |
| --- | --- | ---: | ---: |
| [Four steps, URM](gl_sdm_chunk_r4_benchmark.json) | Yes | 97.88 ms | **41.48%** |
| [Eight steps, URM](gl_sdm_chunk_r8_benchmark.json) | Yes | 161.91 ms | **25.08%** |
| [Four steps, PyTorch control](gl_sdm_chunk_r4_torch_control.json) | No | 125.85 ms | 32.30% |
| [Four steps, URM control](gl_sdm_chunk_r4_urm_control.json) | No | 99.83 ms | 40.67% |

The first two rows have 10 warmups and 20 samples. The control pair has three
warmups and five samples, with identical code/config except the memory backend;
URM improves this matched training step by **1.26×**. GPU measurements run alone.
Step-end allocated memory is exactly flat in each artifact. Whole-benchmark
peak allocation is about 8 GiB; peak reservation is 12.47 GiB for four steps and
15.02 GiB for eight. Graph pools account for additional reserved memory.
There are no OOM retries, smaller-workload substitutions or reference fallbacks.
Four-step URM prefill processes 8,192 tokens in 86.13 ms. Four-request decode
after a 2,048-token context takes 5.97 ms. The matched PyTorch control takes
114.48 ms and 6.70 ms respectively; inference uses the ordinary serving runner.

## What changed

Tokens within a chunk reason in parallel against its immutable starting bank.
A local causal SDPA surround supplies context within the chunk. Reasoning steps
and chunk boundaries remain sequential. Writes become visible only at an
absolute chunk boundary; this changes the write clock from the earlier token
model. The token controls remain available as `gl_sdm_token*` configs. Their
[historical results](token_results.md) are not a matched speed comparison.

The chunk path uses frozen URM's public product-key routing and snapshot-read
operators, including both backwards. Its compiled graph contains two native
operators and zero escapes. Padding logical read width 64 to physical width 128
selects URM's existing vector schedule and avoids fragmented bank-gradient
launches; retrieved width and capacity remain unchanged. URM source and its
[dependency pin](../../../shared/requirements-urm.txt) are unchanged.

The project owns versioned buffered commits, which frozen URM cannot execute.
Fixed-depth write projections are batched after reasoning; PyTorch compiles
dense reasoning, proposal arithmetic and vocabulary loss. Optional CUDA graphs
capture fixed-depth forward/backward. ACT compacts finished tokens and uses the
regular runner. No new architecture-specific Triton kernel was added.

## Correctness and precision

All **66 tests pass**, including the three independent upstream baselines,
causality, absolute chunk boundaries, cache continuation, canonical collision
order, ACT compaction, checkpoints and CUDA-graph optimizer updates. The new
production-bank operand test independently checks URM addresses, weights, reads
and all read gradients against PyTorch at 4,096 slots/head and width 64, including
the padded schedule. Native backward execution is observed explicitly.

The [full-vocabulary smoke reference](gl_sdm_chunk_reference.json) passes its
output and parameter-gradient gates. It records 0.158% BF16 relative L2 output
drift and 0.790% split-continuation drift. BF16 shares production SDPA in this
memory reference; an independent strict FP32 check also exercises dense causal
attention. Changing SDPA/GEMM shapes reproduces approximately the same 0.790%
continuation drift with the pure PyTorch memory backend. BF16 chunk continuation
therefore has an explicit 1% relative L2 bound; output/gradient gates are unchanged.

The [large-bank diagnostic](gl_sdm_chunk_large_bank_precision.json), at width
512, four steps and 1,025 tokens spanning a commit, **fails** its strict output
gate: BF16 relative L2 drift is 0.560%, but maximum logit difference is 0.160.
Every BF16 parameter-gradient gate passes in that run. Its independent FP32
whole-model check also fails, with maximum logit difference 0.0461. The
[shorter-chunk diagnostic](gl_sdm_chunk_short_bank_precision.json) also fails
BF16 output/gradient gates, while its strict FP32 check passes. Sparse route
sensitivity is consistent with these results, but the long-model discrepancy
is unresolved. Failed checks remain failed in their artifacts; passing operand
tests and MFU do not replace a passing whole-model reference gate.
The BF16 trace starts with identical selected addresses and a read difference
of only 1.49e-8; later steps select different addresses. It records the propagation
of that precision difference explicitly.

## Training and inference

[Three real FineWeb-Edu updates](gl_sdm_chunk_validation.json) process 24,576
tokens, reaching about 41.8–41.9% MFU after capture initialization. Validation
CE falls from 10.8258 to 10.7670 nats/token. Strict checkpoint parameters reload
exactly. A trained 1,030-token prefill crosses a commit boundary, split
continuation differs by 0.241% relative L2, and four-token generation completes.
Evaluation at lengths 1,024/2,048 and needle/absent-control execution also complete.
The needle haystack is a validation stream for a protocol smoke check. These
short runs establish execution, not converged quality or retrieval ability.

## Reproduction

PyTorch 2.14.0+cu130, Triton 3.8, A10G; dense TF32 disabled. All new timing and
training artifacts record complete configs, native plans, frozen URM revision
`604bfdf5d2c827266a32ef142ca996cc712d70f0`, and the same source SHA-256
`8440659792d33cb59ba909c36f55a7b99d787b2b45c00a866e445b59c1661122`.

```bash
python -m pytest -q projects/gl-sdm/tests
gl-sdm benchmark --config projects/gl-sdm/configs/gl_sdm_chunk_r4.json \
  --warmup 10 --iterations 20 --output /tmp/gl_sdm_r4.json
gl-sdm benchmark --config projects/gl-sdm/configs/gl_sdm_chunk_r8.json \
  --warmup 10 --iterations 20 --output /tmp/gl_sdm_r8.json
python projects/gl-sdm/scripts/benchmark_chunk_controls.py --output-dir /tmp/gl_sdm_controls
gl-sdm verify --config projects/gl-sdm/configs/gl_sdm_chunk_smoke.json
python projects/gl-sdm/scripts/check_chunk_precision.py --output /tmp/gl_sdm_precision.json
python projects/gl-sdm/scripts/check_chunk_precision.py --chunk-size 16 --length 65 \
  --output /tmp/gl_sdm_short_precision.json
```

The last two commands return nonzero when their unchanged strict gates fail,
after saving the diagnostic. The Transformer remains ordinary PyTorch SDPA,
SDM uses Meta's actual CUDA extensions, and GDN2 uses FLA; none uses URM.
Their earlier [baseline integration results](smoke_validation.json) are preserved.
