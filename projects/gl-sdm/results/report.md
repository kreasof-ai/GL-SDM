# Baseline integration checks

The three requested baselines are implemented with ATMA's model, checkpoint and
experiment interfaces. The Transformer uses PyTorch SDPA; SDM executes Meta's
pinned CUDA-backed layer; GDN2 executes FLA's pinned layer. URM source is unchanged.

Eight tests passed on an NVIDIA A10G with PyTorch 2.14.0+cu130 and Triton 3.8.
They cover reference outputs and every parameter gradient, causal prefixes,
split-prefill/decode continuation with two independent requests, fresh request
state, strict checkpoint loading, exact resumed-training parameters, the shard
format, evaluation token counts, generation and the needle/control protocol.
SDM verification observed both actual CUDA extensions executing.

Each model also completed three training steps on real GPT-2 FineWeb-Edu shards,
checkpoint loading, four-token generation, evaluation at lengths 65 and 129,
and whole-model training/prefill/decode timing. These were **integration runs**:
two layers, width 64, vocabulary 50,304, batch 2, training length 65. They are
not converged quality results or representative large-model MFU measurements.
The needle smoke check uses a validation-stream haystack, not clean documents.

| Model | Active parameters | Learned memory parameters | Training step | Prefill | Decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 6,580,288 | 0 | 8.79 ms | 2.84 ms | 2.51 ms |
| SDM CUDA | 6,572,624 | 8,192 | 26.37 ms | 9.51 ms | 3.39 ms |
| GDN2 FLA | 6,606,084 | 0 | 17.66 ms | 6.41 ms | 3.48 ms |

Timings are medians of three samples after two warmup steps. Training includes
forward, backward, gradient clipping and AdamW. Prefill processes 130 tokens;
decode processes two tokens after a 65-token context. No smaller-workload retry
or reference substitution is used. JSON artifacts record complete configuration
through the associated `*_smoke.json`, source commits, actual CUDA binary hashes,
total/active/memory parameters, GPU memory and explicit 6ND MFU at 70 TFLOP/s.

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

This remains an upstream baseline numerical limitation. GL-SDM's future
transactional memory must establish its own deterministic snapshot/commit
semantics. Neither GL-SDM's reasoner nor its transaction operator is implemented
by this baseline infrastructure work.

## Artifacts and reproduction

- [Reference and integration results](smoke_validation.json)
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
