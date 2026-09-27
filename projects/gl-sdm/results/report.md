# Sixteen-layer experiments

Development is paused. The [optimization handoff](../OPTIMIZATION_PLAN.md)
records the next work and the requirements for resuming on a larger GPU.

The current configs use **length 2,048 and 8,192 tokens per update**, with a
**512-token local window and write chunk** for GL-SDM. Microbatch one accumulates
four sequences per update. Each GL-SDM sequence has four memory chunks; the
first three chunks' writes are consumed by later chunks.

The latest native-kernel timing uses one warmup and three measured updates on
A10G, including backward, clipping and AdamW. Memory peaks are now separated
by stage. These are preliminary throughput measurements using random tokens,
not another FineWeb-Edu training pilot. BF16 dense weights, FP32 residuals and
the FP32 GL-SDM bank remain the same as in the initial run.

| Model | Training tokens/s | 6ND MFU | Training peak allocated, GiB | Training peak reserved, GiB |
| --- | ---: | ---: | ---: | ---: |
| [Transformer](sequence_2048/transformer_benchmark.json) | 13,916 | 29.50% | 6.42 | 6.81 |
| [GDN2](sequence_2048/gdn2_benchmark.json) | 11,051 | 24.63% | 7.08 | 7.46 |
| [SDM](sequence_2048/sdm_benchmark.json) | 5,583 | 7.10% | 6.93 | 7.19 |
| [GL-SDM](sequence_2048/gl_sdm_benchmark.json) | 3,838 | 4.90% | 11.45 | 15.41 |

At fixed tokens/update, the larger sequence reduces microbatch accumulation
from 16 sequences to four. Increasing GL-SDM's write chunk at the same time
keeps four chunks per sequence and reduces full-bank read gradients/commits
per update. Local attention also performs more work with the larger window.
Both changes contribute to this new workload; this is not an isolated
chunk-size ablation. The 40% MFU target remains unmet.

GL-SDM's allocated training peak is approximately 12.29 decimal GB; its allocator
reserves substantially more. These results do not establish that the current
implementation fits a 12 GB GPU. Step-end allocated memory is flat at 2.375 GiB
across the three measured updates. The earlier 17.20 GiB reported peak included
inference with 16 simultaneous requests and was not a training-only measurement.

A small native-URM integration check at window/chunk 512 and length 1,025
passed commit boundaries, sliding KV continuity, exclusion of tokens beyond
the window and gradients through all four global layers' write projections.
Maximum split-continuation logit error was 8.35e-7. The existing CPU stack tests
also pass; details are in [validation](sequence_2048/validation.json).
These checks do not resolve the full-size BF16 reference failures from the
initial run. SDM's BF16 oracle also retains FP32 state, unlike upstream's BF16
bank; that comparison does not establish an upstream bug.

The latest [GL-SDM](sequence_2048/gl_sdm_memory_profile.json) and
[SDM](sequence_2048/sdm_memory_profile.json) profiles cover one length-2048
microbatch, forward/backward without clipping or optimizer. GL-SDM totals
471.18 ms of GPU kernel time, with 45.84% in FP32 additions and 17.79% in fills;
SDM totals 203.11 ms. These are instrumented diagnostics, not whole-update
timings. They confirm that repeated full-bank gradient traffic remains a major
cost under the current 512-token write clock.

The [initial length-512 report](initial_sixteen_layers.md) retains the 20-update
FineWeb-Edu pilot, full-size reference errors, failed OOM attempts and the
large-bank profile. Those results use the former 128-token window/write clock.
The [earlier tied-weight controls](loop_results.md) remain separate as well.
