# GL-SDM and baseline integration checks

The proposed GL-SDM model and three requested baselines use ATMA's model, checkpoint and
experiment interfaces. The Transformer uses PyTorch SDPA; SDM executes Meta's
pinned CUDA-backed layer; GDN2 executes FLA's pinned layer. URM source is unchanged.

The baseline and GL-SDM checks passed on an NVIDIA A10G with PyTorch 2.14.0+cu130
and Triton 3.8. The suite has 25 tests.
They cover reference outputs and every parameter gradient, causal prefixes,
split-prefill/decode continuation with two independent requests, fresh request
state, strict checkpoint loading, exact resumed-training parameters, the shard
format, evaluation token counts, generation and the needle/control protocol.
SDM verification observed both actual CUDA extensions executing.
GL-SDM tests additionally verify one bank/reasoner across depth, snapshot
identity/version checks, deterministic duplicate-address commits on CPU/CUDA,
commit gradients, preserved snapshots, write-timing controls, stable route ties
and removal of halted requests from actual subsequent reasoner calls.

Each model also completed three training steps on real GPT-2 FineWeb-Edu shards,
checkpoint loading, four-token generation, evaluation at lengths 65 and 129,
and whole-model training/prefill/decode timing. These were **integration runs**:
width 64, vocabulary 50,304, batch 2, training length 65. Baselines have two
layers; GL-SDM has one tied block and up to three reasoning steps. They are
not converged quality results or representative large-model MFU measurements.
The needle smoke check uses a validation-stream haystack, not clean documents.

| Model | Active parameters | Learned memory parameters | Training step | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 6,580,288 | 0 | 8.79 ms | 2.84 ms | 2.51 ms |
| SDM CUDA | 6,572,624 | 8,192 | 26.37 ms | 9.51 ms | 3.39 ms |
| GDN2 FLA | 6,606,084 | 0 | 17.66 ms | 6.41 ms | 3.48 ms |
| GL-SDM adaptive, initial PyTorch | 6,505,829 | 4,096 | 1,352.64 ms | 484.30 ms | 8.01 ms |
| GL-SDM fixed, initial PyTorch | 6,505,764 | 4,096 | 1,111.54 ms | 413.97 ms | 6.84 ms |

Timings are medians of three samples after two warmup steps. Training includes
forward, backward, gradient clipping and AdamW. Prefill processes 130 tokens;
decode processes two tokens after a 65-token context. No smaller-workload retry
or reference substitution is used. JSON artifacts record complete configuration
through the associated `*_smoke.json`, source commits, actual CUDA binary hashes,
total/active/memory parameters, GPU memory and explicit 6ND MFU at 70 TFLOP/s.

## GL-SDM

[The model](../src/gl_sdm/global_model.py) uses one global bank and one tied
reasoner. Each token freezes its memory snapshot, reads it throughout reasoning,
merges weighted delta proposals and commits once before the next token. Its
adaptive variant uses ACT; a fixed-depth control and final-only/every-step write
controls use the same implementation. Prefill splitting preserves the token
transaction boundaries.

The full-vocabulary BF16 fixture had zero reference logit and parameter-gradient
differences. Maximum split-prefill/decode logit difference was 0.00766, within
the BF16 tolerance. Commits sum within each address: they do not subtract a
global cumulative sum, which would lose small updates through cancellation
from unrelated addresses.

GL-SDM completed the same real-data training, reload, generation, validation,
needle/control and benchmark paths as the baselines. The smoke run executed
three steps for every token; this does not demonstrate learned adaptive depth.
Controlled tests show mixed request depths and actual removal of halted requests.

The initial PyTorch token loop is much slower than the upstream production
baselines. These are integration timings, not evidence of a performance gain.
GL-SDM kernels are not yet fused or tuned, and none of these short runs establishes
converged quality or a resource-matched frontier. GL-SDM's bank/state is FP32;
capacity and state bytes differ from the BF16 layer-local SDM baseline.
Its reported 6ND counts unique active parameters and excludes repeated uses of
tied weights; executed depth is reported separately.

## SDM precision

Upstream SDM BF16 rounding can change later layers' discrete top-k routes.
The full-vocabulary reference check measured 4.07% relative L2 logit drift and
3.12% split-prefill/decode drift. Repeated launches can also differ; this fixture
does not establish BF16 determinism. The maximum reference logit difference was
0.681, so these outputs must not be described as bitwise or tightly elementwise
equivalent. The explicit BF16 test bounds relative L2 drift at 5% and records it.

The accompanying FP32 check measured 0.00399% relative L2 drift, maximum logit
error 0.000200, maximum gradient error 0.00000203 and maximum continuation error
0.0000747. It validates the recurrence/adapter separately from BF16 route
sensitivity. FP32 checking uses upstream CUDA sparse inner products and its
FP32 gather path; the production BF16 baseline executes CUDA warp gathering.

This remains an upstream baseline numerical limitation. GL-SDM has its own
FP32 transaction operator and deterministic commit checks, separate from the
upstream baseline's BF16 implementation.

## Artifacts and reproduction

- [Reference and integration results](smoke_validation.json)
- [GL-SDM reference](gl_sdm_smoke_reference.json), [integration results](gl_sdm_smoke_validation.json), [validation CE](gl_sdm_smoke_eval.json)
- [Adaptive GL-SDM timings](gl_sdm_smoke_benchmark.json), [fixed GL-SDM timings](gl_sdm_fixed_smoke_benchmark.json)
- [Transformer timings](transformer_smoke_benchmark.json), [SDM timings](sdm_smoke_benchmark.json), [GDN2 timings](gdn2_smoke_benchmark.json)
- [Transformer validation CE](transformer_smoke_eval.json), [SDM validation CE](sdm_smoke_eval.json), [GDN2 validation CE](gdn2_smoke_eval.json)
- [Model configs and commands](../README.md#models-and-experiments), [source mapping](../PROVENANCE.md)

Use the `*_smoke.json` configs with `gl-sdm verify`, `train`, `infer`, `eval`,
and `benchmark`. For the validation-only smoke evaluation, use an evaluation
config with `clean_dataset: null`; the normal configs enable coherent-document
CE and the induction needle experiment. Keep datasets, tokenizer, seeds and
evaluation prefixes fixed for quality comparisons. The default larger configs
are equal-width starting points; resource matching and converged training
remain experimental work.
