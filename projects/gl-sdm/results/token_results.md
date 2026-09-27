# Historical token GL-SDM results

These measurements describe the per-token implementation at commit `59f7146`.
The current chunk implementation is reported in [report.md](report.md).

# GL-SDM and baseline checks

GL-SDM now uses frozen URM for compiled snapshot reads and project-owned CUDA
kernels for routing, delta proposals and deterministic commits. The supplied
GL-SDM configs enable this path. The Transformer still uses ordinary PyTorch
SDPA, SDM uses Meta’s actual pinned CUDA layer, and GDN2 uses FLA. URM source and
its dependency pin are unchanged.

All 47 tests pass on an NVIDIA A10G with PyTorch 2.14.0+cu130 and Triton 3.8.
They cover outputs and every parameter gradient, causal prefixes, request reset,
split-prefill/decode, strict checkpoints, exact resumed training on CPU/CUDA,
evaluation and generation. Native tests also cover stable route ties, signed
zero, non-power-of-two routes, inactive requests, all write policies, FP32/BF16,
colliding commits, snapshot preservation and compact backward storage.
SDM verification observed both actual upstream CUDA extensions executing.

## Whole-model timings

These are integration workloads: width 64, vocabulary 50,304, batch 2 and length
65. Baselines have two layers; GL-SDM has one tied block and three reasoning
steps. Capacity, parameter counts and compute are not matched for a quality
comparison. Adaptive GL-SDM used all three steps in these short runs; controlled
tests separately verify removal of requests that halt early.

| Model | Active parameters | Learned memory parameters | Training step | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 6,580,288 | 0 | 8.79 ms | 2.84 ms | 2.51 ms |
| SDM CUDA | 6,572,624 | 8,192 | 26.37 ms | 9.51 ms | 3.39 ms |
| GDN2 FLA | 6,606,084 | 0 | 17.66 ms | 6.41 ms | 3.48 ms |
| [GL-SDM adaptive, PyTorch](gl_sdm_adaptive_torch_benchmark.json) | 6,505,829 | 4,096 | 1,373.58 ms | 504.90 ms | 8.05 ms |
| [GL-SDM adaptive, native](gl_sdm_adaptive_urm_benchmark.json) | 6,505,829 | 4,096 | 1,059.76 ms | 384.21 ms | 6.47 ms |
| [GL-SDM fixed, PyTorch](gl_sdm_fixed_torch_benchmark.json) | 6,505,764 | 4,096 | 1,047.64 ms | 401.90 ms | 6.52 ms |
| [GL-SDM fixed, native](gl_sdm_fixed_urm_benchmark.json) | 6,505,764 | 4,096 | 720.20 ms | 272.93 ms | 4.79 ms |

The GL-SDM pairs use identical code and configs except `gl_memory_backend`, with
five samples after two warmups. Training includes forward, backward, gradient
clipping and AdamW. Prefill processes 130 tokens; decode processes two tokens
after a 65-token context. The baseline rows retain their earlier three-sample,
two-warmup measurements. No workload retry or reference substitution is used.

Native GL-SDM improves training latency by **1.30× adaptive / 1.45× fixed** and
prefill by **1.31× / 1.47×** against its current PyTorch controls. All four
GL-SDM step-end allocation traces are exactly flat across five measured steps.
Their artifacts share source SHA-256
`f9d443e7eccda9071aeb301db97ae514b6ff103494bcfd519a099fb2fc64332f`,
include full configs and record the frozen URM revision and compiled read plan.

The decoder remains slow: its causal token loop, ACT controller and dense
reasoner still execute through PyTorch at a small batch size. Native adaptive
and fixed unique-parameter 6ND MFU are 0.00630% and 0.01007%. This formula excludes
repeated applications of tied weights; actual reasoning depth is recorded
separately. These runs establish integration and a measured improvement, not
converged quality, learned adaptive depth or high large-model utilization.

## Memory kernels

The [transaction benchmark](gl_sdm_memory_benchmark.json) uses batch 2, eight
heads, 4,096 slots/head, width 64, eight routes and eight reasoning steps, all
FP32. Every step proposes to the same addresses, testing cross-step collisions.
Reads and proposals use one snapshot, followed by one commit. Initial-state and
all operand gradients are checked against the PyTorch equations.

| One transaction, forward + backward | PyTorch | Native |
| --- | ---: | ---: |
| Median time, 10 samples after 3 warmups | 12.28 ms | 8.19 ms |
| Peak allocated memory | 98.49 MiB | 82.03 MiB |
| Saved operator storage for backward | 1.42 MiB | 0.56 MiB |

This is **1.50× faster** with 16.7% lower peak allocation. Step-end allocations
are flat. Maximum state error is 2.38e-7; the largest operand-gradient error is
3.73e-9 with the recorded cotangents. The timing excludes model projections and
the optimizer and carries no MFU claim.

Forward commits sum deltas in stable proposal order within each address. No
forward atomics or global cumulative-sum subtraction are used, so a large update
at one address cannot erase a small update at another. The read wrapper retains
selected rows instead of the full bank versions URM’s ordinary backward saves.

The full-vocabulary BF16 [native reference check](gl_sdm_native_reference.json)
had zero logit difference, maximum parameter-gradient difference 0.000214 and
maximum split-prefill/decode difference 0.00766, within the existing tolerances.
Accurate FP32 exponentials eliminated divergence caused by approximate
exponentials changing later BF16 routes; checking limits were not loosened.
A [width-512, 4,096-slot check](gl_sdm_large_bank_reference.json), with vocabulary
256 and two fixed steps over 23 tokens, also had zero reference logit difference
and maximum parameter-gradient difference 0.000183.

Native GL-SDM completed three real FineWeb-Edu training steps, checkpoint reload,
four-token generation, validation at lengths 65/129 and the needle/absent-control
protocol. See [training and integration](gl_sdm_native_validation.json) and
[validation CE](gl_sdm_native_eval.json). The needle smoke check uses a validation
stream haystack. These short runs do not establish retrieval quality.

## Upstream SDM precision

Upstream SDM BF16 rounding can change later layers’ discrete top-k routes.
The baseline full-vocabulary check measured 4.07% relative L2 logit drift and
3.12% split-prefill/decode drift, with maximum reference logit difference 0.681.
The explicit BF16 test bounds relative L2 drift at 5%; it does not establish
bitwise equivalence or BF16 determinism.

The separate FP32 check measured 0.00399% relative L2 drift, maximum logit error
0.000200, maximum gradient error 0.00000203 and maximum continuation error
0.0000747. Production BF16 runs execute CUDA sparse inner products and CUDA warp
gathering. This upstream numerical limitation remains separate from GL-SDM’s
FP32 transaction state and deterministic commit tests.

## Reproduction

Use the [configs and commands](../README.md#models-and-experiments), including
`--memory-backend torch` for explicit GL-SDM controls and
[scripts/benchmark_memory.py](../scripts/benchmark_memory.py) for transactions.
Run GPU measurements alone. Keep tokenizer, data, prefixes and seeds fixed.
Resource matching and converged architecture comparisons remain experimental
work. The three baselines’ [integration results](smoke_validation.json) and
[Transformer](transformer_smoke_benchmark.json), [SDM](sdm_smoke_benchmark.json)
and [GDN2](gdn2_smoke_benchmark.json) timings are preserved.

The initial PyTorch GL-SDM artifacts (`gl_sdm_smoke_*` and
`gl_sdm_fixed_smoke_benchmark.json`) remain archived from commit `154a82c`.
The current comparisons above use the four explicitly named backend artifacts.
