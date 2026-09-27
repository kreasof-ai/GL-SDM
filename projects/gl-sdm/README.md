# Project I: Global Liquid SDM

[Read the complete proposal](../../docs/research-program.md#proposal-i).

## Research question

Can a weight-tied model allocate variable computation per token while repeatedly
accessing one globally shared, model-scale sparse delta memory, improving the
quality-compute-capacity frontier?

## Models and experiments

The proposed [GL-SDM model](src/gl_sdm/global_model.py) is registered as
`arch_type: gl_sdm` and runs through the same training, inference, reference,
evaluation and benchmark commands as its baselines.

The baseline suite has exactly three models: a full Transformer using ordinary
PyTorch and SDPA, Meta's upstream CUDA SDM, and FLA's GDN2. Every layer uses the
chosen mixer. Neither GL-SDM nor these baselines depends on URM.

Models follow ATMA's `embed`, `blocks`, `norm`, `proj` structure.
`model(inputs, targets)` returns summed CE, regularization loss and an auxiliary
loss; each block returns hidden states and the two auxiliary losses. ATMA-style
evaluation can traverse the blocks directly. Checkpoints contain `config.json`,
`run_config.json`, `weights.pt` with a `model` state dict, and tokenizer metadata.
Logs retain `ABLATION_CONFIG_JSON`, `ABLATION_CURVE_JSON`, `ABLATION_EVAL_JSON`
and `ABLATION_ERROR_JSON` blocks. See [source mapping](PROVENANCE.md).

Install from the repository root:

```bash
python -m pip install -e 'projects/gl-sdm[test]'
python projects/gl-sdm/scripts/setup_sources.py
```

Upstream checkouts default to `~/.cache/gl-sdm/sources/{sdm,fla}`. Override them
with `GL_SDM_SDM_ROOT` and `GL_SDM_FLA_ROOT`; revisions are checked. SDM needs a
CUDA compiler compatible with the installed PyTorch CUDA version. Set
`GL_SDM_CUDA_HOME` if it differs from the default compiler, and optionally
`TORCH_EXTENSIONS_DIR` to reuse verified compiled extensions. Both actual CUDA
extensions must load; build errors stop execution. A compiler cached at
`~/.cache/gl-sdm/cuda` is detected automatically; extensions default to
`~/.cache/gl-sdm/extensions`.

Put GPT-2 FineWeb-Edu `.bin` shards in `data/finewebedu10B`, or set `train_data`
and `val_data` in a config to absolute paths. The format is ATMA's 256-int32
header followed by uint16/uint32 tokens. All commands run from the repo root:

```bash
python -m pytest -q projects/gl-sdm/tests
gl-sdm verify --config projects/gl-sdm/configs/sdm_smoke.json
gl-sdm verify --config projects/gl-sdm/configs/gl_sdm_smoke.json
gl-sdm train --config projects/gl-sdm/configs/gl_sdm.json \
  --output projects/gl-sdm/checkpoints/gl_sdm --log projects/gl-sdm/runs/gl_sdm.log
gl-sdm infer --checkpoint projects/gl-sdm/checkpoints/gl_sdm \
  --prompt 'The research question is' --tokens 64
gl-sdm eval --checkpoint projects/gl-sdm/checkpoints/gl_sdm \
  --output projects/gl-sdm/results/gl_sdm_eval.json
gl-sdm benchmark --config projects/gl-sdm/configs/gl_sdm.json \
  --output projects/gl-sdm/results/gl_sdm_benchmark.json
```

Replace `gl_sdm` with `transformer`, `sdm` or `gdn2` for the baselines. Add
`--checkpoint <directory>` to `train` to resume with the same config. Use the
`*_smoke.json` configs for short integration runs. Training configs use AdamW
by default; `optimizer: atma_muon` selects ATMA's optimizer arrangement, keeping
the sparse learned SDM memory bank in AdamW. No architecture falls back to a
reference kernel or a smaller workload on failure.

Evaluation reports ATMA's `clean_ppl` and `junk_ppl` as **nats/token**, with true
perplexity in separate fields, plus needle digit accuracy and its absent-needle
control. The clean dataset is streamed and tokenized once for common nested
prefixes. Evaluation aborts on OOM or non-finite values rather than omitting
samples. It chunks the vocabulary head to bound logit memory.

Inference keeps state per request: Transformer KV tensors, upstream SDM memory,
or FLA recurrent and convolution state. `new_cache`, `prefill` and `decode`
provide a common interface. This first runner supports batched generation;
ATMA's paged serving scheduler and CUDA-graph serving engine are not ported.

Benchmarks time complete forward/backward/optimizer steps after warmup, then
prefill and single-token decode at a fixed context length. MFU uses **6ND**,
where N is the active parameter count and D is the processed training tokens;
the sparse learned SDM memory bank is counted separately. State operations are
not added to this figure. A10G's denominator is 70 dense BF16 TFLOP/s; unknown
GPUs require `--peak-tflops`. Equal-width configs are not parameter-matched
quality comparisons. The runner records total, active and memory parameters.

Reference checks compare outputs and every parameter gradient, verify causality,
split-prefill/decode continuation, request reset and strict checkpoint reload.
The BF16 SDM check also runs FP32: chunked and sequential BF16 rounding can
change later top-k routes, so its end-to-end check records relative L2 drift
with a 5% bound rather than claiming bitwise equivalence. References are explicit
test modes and never selected by production runners.

[Smoke results](results/report.md) record the checked paths and their limits.

## GL-SDM model

GL-SDM has one learned memory bank, partitioned into heads, and one reasoner
whose projections, normalization and MLP weights are reused at every reasoning
step. It processes tokens in causal order. All steps for a token read the same
FP32 snapshot. Product-key addressing selects sparse read/write slots; the
reasoner conditions on both the current token and retrieved memory values.

Writes are delta proposals computed against that snapshot. The default
`gl_write_policy: merged` combines them with weights summing to one and commits
once at the token boundary. Duplicate addresses use stable ordering and a sum
within each address before a unique-address update. A commit returns a new
version and preserves the previous snapshot; stale or unrelated buffers are
rejected. Requests have independent states initialized from the same learned
bank. Routing parameters change at optimizer boundaries, not during a token's
transaction. The [memory API](src/gl_sdm/memory.py) is project-owned and reusable
by CSDM.

`gl_reasoning: adaptive` uses ACT halting, a weighted latent output, and a
maximum `gl_max_steps`. Finished requests leave subsequent reasoner calls.
The third training loss is the summed ACT ponder cost, weighted explicitly by
`auxiliary_loss_weight`. Logs report mean, 95th percentile and maximum executed
depth. [The fixed-depth config](configs/gl_sdm_fixed.json) uses the same tied
reasoner and bank, with exactly `gl_max_steps` calls and no halting head.

`gl_write_policy: final` commits only the final step's proposal. `every_step`
applies each weighted proposal immediately and lets later steps read it. These
are write-timing controls within GL-SDM, alongside the three external baselines.
The token boundary keeps results independent of how prefill is split into
chunks; the entire token/reasoning loop lives inside one ATMA-compatible block.

This is the initial PyTorch model and transaction implementation. Its token
loop, routing and memory commits are not yet fused or tuned for large-model
throughput. The configs and smoke comparisons establish working modeling and
correctness, not a quality or performance advantage. Adaptive halting has not
been validated on converged training. Memory capacity, active parameters and
compute must be matched for the architecture study.

The reported 6ND uses unique active parameters. For GL-SDM it excludes repeated
applications of tied weights, so it is not a complete estimate of its executed
FLOPs. Use measured tokens/s and recorded reasoning depth alongside it; no depth
multiplier is used to inflate the reported MFU.

## URM boundary

[URM](https://github.com/kreasof-ai/urm) remains frozen at the revision in
[shared/requirements-urm.txt](../../shared/requirements-urm.txt). Any future use
of its compiler is developed against that external package. This project does
not carry or modify URM source.

## This project owns

- The global sparse address space and read operator.
- Fixed and adaptive weight-tied recurrent reasoning.
- Frozen-snapshot reads and buffered, transactional commits.
- Memory sharing, capacity, routing-contention, and write-timing experiments.
- Architecture-level throughput and HBM-residency measurements.

It does not own multi-timescale consolidation policy or a general mixer compiler.
Those belong to CSDM and URM respectively.

## First milestone

Build the smallest dense or small-memory reference model that can reproduce:

1. the write-every-step versus snapshot-and-commit comparison; and
2. the capacity comparison between layer-local and globally shared memory.

Before scaling, fix one transaction scope, one halting objective, one fixed-depth
control, and the key/versioning policy.

## Acceptance gate

Proceed when the implementation has deterministic snapshot/commit behavior and
shows a reproducible improvement on a resource-matched quality/compute/capacity
frontier. Treat collapsed halting, excessive global contention, or gains that
vanish after bandwidth matching as negative results.

## Expected output

A paper-quality architecture study plus a reusable transactional memory operator
that CSDM can extend and URM can use as a systems workload.
