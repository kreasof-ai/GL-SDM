# Project I: Global Liquid SDM

[Read the complete proposal](../../docs/research-program.md#proposal-i).

## Research question

Can a weight-tied model allocate variable computation per token while repeatedly
accessing one globally shared, model-scale sparse delta memory, improving the
quality-compute-capacity frontier?

## Models and experiments

The baseline suite has exactly three models: a full Transformer using ordinary
PyTorch and SDPA, Meta's upstream CUDA SDM, and FLA's GDN2. Every layer uses the
chosen mixer. These baselines do not depend on URM. The future GL-SDM architecture
and transactional memory operator are separate work; they are not implemented
by calling the SDM baseline "global".

Models follow ATMA's `embed`, `blocks`, `norm`, `proj` structure.
`model(inputs, targets)` returns summed CE, regularization loss and alignment
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
gl-sdm train --config projects/gl-sdm/configs/sdm.json \
  --output projects/gl-sdm/checkpoints/sdm --log projects/gl-sdm/runs/sdm.log
gl-sdm infer --checkpoint projects/gl-sdm/checkpoints/sdm \
  --prompt 'The research question is' --tokens 64
gl-sdm eval --checkpoint projects/gl-sdm/checkpoints/sdm \
  --output projects/gl-sdm/results/sdm_eval.json
gl-sdm benchmark --config projects/gl-sdm/configs/sdm.json \
  --output projects/gl-sdm/results/sdm_benchmark.json
```

Replace `sdm` with `transformer` or `gdn2` for the other baselines. Add
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
