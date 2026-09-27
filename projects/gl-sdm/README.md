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
chosen mixer. These baselines have no URM calls. GL-SDM's chunk path uses frozen URM for routing and snapshot reads, including
backward. PyTorch compiles the dense reasoner and write arithmetic; the project
owns ordered sparse commits.

Models follow ATMA's `embed`, `blocks`, `norm`, `proj` structure.
`model(inputs, targets)` returns summed CE, regularization loss and an auxiliary
loss; each block returns hidden states and the two auxiliary losses. ATMA-style
evaluation can traverse the blocks directly. Checkpoints contain `config.json`,
`run_config.json`, `weights.pt` with a `model` state dict, and tokenizer metadata.
Logs retain `ABLATION_CONFIG_JSON`, `ABLATION_CURVE_JSON`, `ABLATION_EVAL_JSON`
and `ABLATION_ERROR_JSON` blocks. See [source mapping](PROVENANCE.md).

Install from the repository root:

```bash
python -m pip install -e 'projects/gl-sdm[test,urm]'
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
prefill and single-token decode at a fixed context length. MFU is an estimated
**6ND** percentage. For GL-SDM, N is weighted by execution: embeddings, head,
input projection and local context are counted once per token; reasoner weights
are counted per observed token/pass; write projections are counted only in
chunks that execute writes. ACT uses observed depths, not the configured maximum.
This retains the parameter-count approximation (including embeddings, biases
and normalization parameters); it is not an exact GPU instruction count.
Attention and sparse state FLOPs are excluded. The sparse learned bank is counted
separately. `unique_parameter_6nd_pct` preserves the old capacity-normalized
throughput proxy, which is unsuitable for comparing utilization across depths.
A10G's denominator is 70 dense BF16 TFLOP/s; unknown
GPUs require `--peak-tflops`. Equal-width configs are not parameter-matched
quality comparisons. The runner records total, active and memory parameters.

Reference checks compare outputs and every parameter gradient, verify causality,
split-prefill/decode continuation, request reset and strict checkpoint reload.
Chunk GL-SDM shares production SDPA in its BF16 memory reference, keeps the
existing output/gradient limits, and independently checks dense attention and
continuation in FP32. BF16 split continuation has a separate 1% relative L2
bound: changing GEMM/SDPA shapes can change a discrete route even with the
PyTorch memory control. Measured drift is recorded.
The BF16 SDM check also runs FP32: chunked and sequential BF16 rounding can
change later top-k routes, so its end-to-end check records relative L2 drift
with a 5% bound rather than claiming bitwise equivalence. References are explicit
test modes and never selected by production runners.

[Kernel results](results/report.md) record the checked paths and their limits.
The corrected estimate is 50.53% for four reasoning passes and 38.08% for eight;
the eight-pass configuration remains below the 40% target.
Large-bank whole-model reference gates still fail despite passing
read-operand checks. Keep that precision limitation separate from throughput.

## GL-SDM model

GL-SDM has one learned FP32 memory bank, partitioned into heads, and one tied
reasoner. `gl_max_steps` controls fixed reasoning depth or the maximum ACT depth.
ACT removes finished tokens from later dense reasoner calls and returns a
weighted latent output plus a summed ponder loss. The runner weights that loss
with `auxiliary_loss_weight` and records actual depth.

`gl_chunk_size` chooses when writes become visible. The main configs now use
one transaction per chunk; `gl_sdm_token*` retains the per-token control. In
chunk transactions, tokens reason in parallel against the chunk-start snapshot, then
commit together at its absolute boundary. A causal SDPA surround runs once per
chunk to supply within-chunk context. Its KV cache holds only the unfinished
chunk and resets at commit. Chunking changes the model's write clock; it is an
explicit architecture choice, not equivalent to the token control.

Each proposal computes a sparse delta against the frozen snapshot. Merged
writes have reasoning weights summing to one per token, divided by the fixed
chunk size. `final` selects the last reasoning proposal with weight 1/chunk_size.
`every_step` is available only for token transactions: exposing other tokens'
writes during chunk reasoning would violate causality. Commits sum collisions
in canonical address/token/depth order, return a new version and preserve the
previous snapshot. Stale or unrelated buffers are rejected. Requests maintain
independent states initialized from the same learned bank. The
[memory API](src/gl_sdm/memory.py) remains reusable by CSDM.

Prefill and decode use the same absolute chunk boundaries. An unfinished chunk
retains its snapshot, sparse proposals and local KV tensors across calls. Future
tokens cannot affect earlier outputs. When training has no outgoing-state
consumer, the final chunk omits unused writes; all preceding writes and their
gradients remain connected.

## URM integration and optimization

The chunk path composes URM's public product-key route and read operations into
one native graph. Both operators and their backwards execute in frozen URM;
ties choose the highest address, with selected addresses in ascending order.
Token controls retain their earlier smaller-address tie rule and project read
backward. These are explicit differences between configurations.

URM remains pinned by [shared/requirements-urm.txt](../../shared/requirements-urm.txt).
The adapter checks the installed package's exact Git origin and revision. It
pads width-64 read values with zero channels to width 128: this selects URM's
existing vector schedule instead of its four-value gradient fragments. Returned
values and bank capacity retain their logical width. Runtime metadata records
both widths and the actual compiled plan. No URM source or pin is changed.

Fixed-depth writes are projected and routed together after reasoning finishes,
since all their proposals use the same snapshot. `gl_compile: true` fuses dense
reasoner, proposal and vocabulary-loss arithmetic through PyTorch, preserving
BF16 rounding casts. `gl_cuda_graph: true` additionally captures fixed-depth
forward/backward at a fixed batch shape. Gradient clipping, optimizer updates,
data copies and finite checks remain in the measured training step. ACT uses
the regular runner because its active token count varies.

The project still owns sparse commit: frozen URM has no executable lowering for
buffered versioned sparse transactions. The three baselines have no URM calls.
Unsupported native workloads fail; reference execution is an explicit test mode.

The optimized configs keep width 512, vocabulary 50,304, 4,096 slots/head and
eight read/write routes. They use length 2,048, 8,192 tokens/update and microbatch
4. [Four reasoning passes](configs/gl_sdm_chunk_r4.json) and
[eight reasoning passes](configs/gl_sdm_chunk_r8.json) are separate depth controls; their
performance must be reported separately. The `gl_sdm_token*` configs retain the earlier token controls.
The estimated MFU counts tied weights for their repeated execution. Once-only
weights are not multiplied by depth; terminal training chunks omit write work.
The four-pass PyTorch/URM control pair disables CUDA graphs for both; reproduce
it with `scripts/benchmark_chunk_controls.py --output-dir projects/gl-sdm/results`.
See [results](results/report.md) for measured performance and limitations.

```bash
gl-sdm verify --config projects/gl-sdm/configs/gl_sdm_chunk_smoke.json
gl-sdm train --config projects/gl-sdm/configs/gl_sdm_chunk_r4.json \
  --output projects/gl-sdm/checkpoints/gl_sdm_chunk_r4
gl-sdm benchmark --config projects/gl-sdm/configs/gl_sdm_chunk_r4.json \
  --warmup 10 --iterations 20 \
  --output projects/gl-sdm/results/gl_sdm_chunk_r4_benchmark.json
```

Use `gl_sdm_chunk_r8.json` for the original eight-pass reasoning depth. These
passes occur inside each forward/backward, independently of optimizer updates.
A reference check
must span a commit boundary to exercise write gradients; `verify` defaults to
at least chunk_size+1 tokens, or accepts `--length`. Small fixtures independently
check dense memory equations. Full chunk-model checks use vectorized PyTorch
routing/gather and segmented commits, with independent dense causal attention
in FP32, avoiding enormous
one-hot matrices. Converged quality and learned adaptive depth remain unproven.

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
