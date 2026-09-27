# Initial sixteen-layer measurements

These results use length 512 and GL-SDM window/chunk 128. They are preserved
from the first experiment; the current length-2048 results are in
[report.md](report.md).

All four models use **16 distinct layers with no weight loops**. They completed
native-kernel benchmarks and a 20-update FineWeb-Edu training pilot. GL-SDM has
12 local layers and four global layers, one shared bank, a rolling local window
of 128 and a delayed write chunk of 128. Writes are summed without averaging.

**The 40% MFU target is not met.** GL-SDM measures 1.42% on this larger-bank,
128-token-chunk workload. Its full-size FP32 reference check passes, but the
BF16 check fails. GDN2 and SDM also fail the current default-precision gates.
These timings therefore describe experimental implementations, not a validated
quality comparison. No failed check was bypassed with a reference kernel.

## Workload

The saved artifacts contain the original length-512 configs. Parameter sizes
are listed in the [project README](../README.md#models-and-experiments). SDM and GL-SDM each have
134,217,728 learned bank entries; SDM splits them across 16 banks, while GL-SDM
shares one bank across its four global layers. Transformer and GDN2 use width
768 and FFN width 3,328; the memory models use width 512 and FFN width 2,048.
Capacity is approximately matched. FLOPs are not matched.

Measurements ran serially on one A10G, PyTorch 2.14.0+cu130 and Triton 3.8.
Training uses BF16 dense weights, FP32 residuals, sequence length 512,
8,192 tokens/update and microbatch one. Timings include forward, backward,
gradient clipping and AdamW. There are three warmups and five measured updates.
All models use `PYTORCH_ALLOC_CONF=max_split_size_mb:512`.

MFU estimates **6ND** against 70 dense BF16 TFLOP/s. The sparse bank is excluded
from N, and each physical layer counts once. GL-SDM omits unused write
projections in the terminal training chunk, giving effective N = 148,973,872
for length 512. Attention and sparse-state arithmetic are outside this estimate.
MFU uses total measured time; the step column is the median, so the two need
not be exact reciprocals.

## Throughput

| Model | Median update, ms | Training tokens/s | 6ND MFU | Peak allocated, GiB |
| --- | ---: | ---: | ---: | ---: |
| [Transformer](sixteen_layers/transformer_benchmark_attempt2.json) | 905.0 | 9,071 | 19.23% | 7.03 |
| [GDN2](sixteen_layers/gdn2_benchmark_attempt2.json) | 2,021.0 | 4,055 | 9.04% | 6.73 |
| [SDM](sixteen_layers/sdm_benchmark_attempt2.json) | 3,314.2 | 2,336 | 2.97% | 10.70 |
| [GL-SDM](sixteen_layers/gl_sdm_benchmark_attempt4.json) | 7,366.2 | 1,108 | 1.42% | 17.20 |

The peak column is the maximum across training, prefill and decode, including
16 simultaneous inference requests. These older artifacts do not separate
training peak memory; 17.20 GiB must not be interpreted as training-only memory.
New benchmarks report memory separately for each stage. The saved profile's
8.21 GiB peak covers one microbatch without optimizer state, so it is also not
a complete training-update peak.

Inference measures 16 independent requests with 512-token prompts. Cache
construction is outside timing; decode excludes its preparatory prefill.

| Model | Prefill tokens/s | Decode tokens/s | Median decode batch, ms |
| --- | ---: | ---: | ---: |
| Transformer | 50,344 | 839 | 19.17 |
| GDN2 | 48,151 | 583 | 27.49 |
| SDM | 30,360 | 614 | 25.85 |
| GL-SDM | 10,695 | 679 | 23.62 |

The first GL-SDM inference attempt unnecessarily padded the full request bank.
The next retained the preceding request cache in the timing loop and exhausted
memory during a commit. Inference now reads logical width 64 directly, and the
runner releases each request's cache and logits before constructing the next.
The common allocator setting prevents fragmentation of large state blocks.
Failed attempts remain in the [manifest](sixteen_layers/manifest.json).
The successful run uses the same model, batch and sequence sizes; there is no
OOM fallback. GL-SDM's five training step-end allocations are identical at
2.377 GiB, with peak reserved memory 21.313 GiB across the benchmark stages.

## Training pilot

Each model consumed 163,840 training tokens over 20 updates, with validation
on 8,192 tokens at updates 0, 10 and 20. Checkpoints include optimizer, RNG and
data position. Initial validation loss is 10.8258 nats/token for all models,
consistent with ATMA's zero-initialized output head.

| Model | Validation loss after 20 updates, nats/token | Training time, s |
| --- | ---: | ---: |
| [Transformer](sixteen_layers/transformer_pilot.json) | 8.7441 | 18.39 |
| [GDN2](sixteen_layers/gdn2_pilot.json) | 8.8024 | 41.31 |
| [SDM](sixteen_layers/sdm_pilot.json) | 9.3461 | 66.72 |
| [GL-SDM](sixteen_layers/gl_sdm_pilot.json) | 9.2626 | 148.49 |

This is a pipeline pilot, not converged training or an architectural quality
ranking. The configured 1,000-update budget and long-context/needle evaluations
have not been run. Pilot configs preserve the primary model and workload;
only the update budget and validation interval change.

## Reference checks

These checks use the full parameter and bank sizes, batch one and length 129,
so GL-SDM crosses a memory commit boundary. Existing tolerances are unchanged.
The [verification summary](sixteen_layers/verification_summary.json) includes
commands, errors and native-call evidence.

| Model | BF16 weights / FP32 residuals | FP32 weights / default dot precision |
| --- | --- | --- |
| Transformer | Pass | Pass |
| GDN2 | Fail: reference logits | Fail: reference logits |
| SDM | Fail: relative L2 drift 89.7% | Fail: reference logits |
| GL-SDM | Fail: reference logits | Pass |

The SDM BF16 comparison has a precision-policy mismatch: our sequential oracle
retains recurrent state in FP32, while the native bank is stored in BF16. The
89.7% result is disagreement with that oracle, not an established upstream bug.
It needs an isolated comparison with aligned state-storage/rounding policies
before attributing the discrepancy to the upstream implementation.

GL-SDM's [FP32 check](sixteen_layers/gl_sdm_verify_fp32.json) has maximum logit
error 2.09e-6, gradient error 1.77e-8 and split-continuation error 3.22e-6.
It exercises native URM routing/read backward, project-owned commits,
causality, cache reset and checkpoint restoration. BF16 route sensitivity
remains unresolved; FP32 correctness does not establish BF16 parity.

SDM's [separate FP32 IEEE-dot control](sixteen_layers/sdm_verify_fp32_ieee.json)
passes with maximum logit error 2.68e-6 and gradient error 2.42e-8. Setting
`TRITON_F32_DEFAULT=ieee` retains the actual upstream implementation, but changes
its internal dot precision. The passing control identifies default TF32 dots
as a cause of the FP32 failure; it is **not** the benchmark configuration.
FLA also explicitly requests TF32 in its GDN2 triangular solve on A10G, so
FP32 inputs alone do not imply IEEE arithmetic. GDN2's FP32 failure remains
unresolved, as do the BF16 SDM and GL-SDM failures.

## GL-SDM bottleneck and validation

A [one-microbatch profile](sixteen_layers/gl_sdm_memory_profile.json) attributes
56.2% of GPU kernel time to FP32 additions and 21.8% to FP32 fills. Shape
attribution shows repeated full-bank gradients and accumulation. The current
training adapter pads width 64 to 128; each native read backward allocates a
1 GiB dense bank gradient, although only routed rows contribute.
[URM integration notes](urm_notes.md) distinguish this adapter cost from
possible reusable URM improvements. The frozen dependency and its pin remain
unchanged; the factor-512 support override is recorded explicitly.

Automated tests cover the shared snapshot, unaveraged writes, all global-layer
write gradients, sliding-window continuity, cache lifetime, large native URM
routing/read gradients, support-probe restoration and checkpoint resume.
The final test result and wheel build are recorded in
[validation.json](sixteen_layers/validation.json).

The [earlier tied-weight results](loop_results.md) are historical controls.
Their repeated reasoning weights, smaller banks and different write clock do
not describe this experiment or establish 40–50% MFU for the current model.
