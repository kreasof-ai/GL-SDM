# Global Liquid SDM

The current experiment compares four models with **16 distinct physical layers,
no weight loops**: a full Transformer, FLA GDN2, upstream CUDA SDM and GL-SDM.
Adaptive per-token depth is deferred. The original research proposal is in
[the research program](../../docs/research-program.md#proposal-i).

## Models and experiments

| Model | Width | FFN intermediate | Static parameters | Learned bank entries | Total parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 768 | 3,328 | 247,341,824 | 0 | 247,341,824 |
| GDN2 | 768 | 3,328 | 260,036,544 | 0 | 260,036,544 |
| SDM | 512 | 2,048 | 148,294,656 | 134,217,728 | 282,512,384 |
| GL-SDM | 512 | 2,048 | 153,447,232 | 134,217,728 | 287,664,960 |

All use vocabulary 50,304 and head dimension 64. Dense weights use BF16 with
FP32 residual accumulation. Parameter capacity is comparable; FLOPs are not
matched. The bank is sparse state, not a dense matrix multiplied by every token.

The Transformer uses ordinary PyTorch and SDPA in every layer. SDM uses Meta's
actual CUDA implementation, with 16 separate banks of 16,384 slots/head. GDN2
uses FLA's actual GatedDeltaNet2, including its short convolution. The three
baselines have no URM calls. Source pins and adaptations are documented in
[PROVENANCE.md](PROVENANCE.md).

## GL-SDM forward

The layer pattern **local → local → global → local** repeats four times:
12 local sliding-window attention/MLP blocks and four global read/MLP blocks.
Every block has its own weights. Only the learned memory and runtime bank are
shared: eight heads, 262,144 slots/head, width 64, eight read/write routes.

1. Freeze the shared bank at the start of each 512-token chunk.
2. Execute layers 1 through 16 once. Each local layer attends to the current
   token and up to 511 preceding tokens, including tokens in earlier chunks.
3. Each global layer reads the same chunk-start snapshot and proposes a delta
   from its updated hidden state. Writes are not visible between global layers.
4. After the chunk finishes, sum token/layer deltas by address and commit once.
   There is **no division by token count or global-layer count**. Learned gates
   control update strength. Collisions are ordered by address, token and layer.

Training processes chunks in sequence and tokens within a chunk in parallel.
The last training chunk omits outgoing writes when there is no state consumer.
Serving retains pending writes across calls, commits at absolute boundaries,
and keeps each local layer's sliding KV history across commits. State is per
request; the learned initializer is never mutated during inference.

Start reading at [model.py](src/gl_sdm/model.py), then
[layers/stack.py](src/gl_sdm/layers/stack.py) and
[layers/global_layer.py](src/gl_sdm/layers/global_layer.py).
The [code guide](src/gl_sdm/README.md) maps memory and experiment operations.
The reusable [memory API](src/gl_sdm/memory/__init__.py) belongs to this project
and remains available to CSDM.

## Frozen URM

GL-SDM compiles product-key routing and snapshot reads, including backward,
through the exact [URM pin](../../shared/requirements-urm.txt). The project owns
chunk scheduling, buffered deltas and deterministic commits. Width-64 reads
are zero-padded to URM's existing width-128 vector schedule during training.
Inference reads width 64 directly and avoids doubling the bank. Logical bank
capacity stays unchanged. No new Triton kernel is introduced for this stack.

URM's declared route support ends at factor extent 256. The production config
explicitly enables `gl_urm_large_route_override`: a temporary compiler support
monkeypatch for **factor 512, eight routes, FP32 scores and INT32 indices**.
It preserves dependency/hardware checks, restores the original method after
compilation, and executes the normal native plan with the actual larger shape.
URM source and its pin are unchanged. Runtime artifacts record this experimental
override and the compiled plan. It is not official upstream shape support.

## Running

Install from the repository root, then prepare the pinned baseline sources:

```bash
python -m pip install -e 'projects/gl-sdm[test,urm]'
python projects/gl-sdm/scripts/setup_sources.py
python -m pytest -q projects/gl-sdm/tests
```

Put ATMA-format GPT-2 FineWeb-Edu token shards under `data/finewebedu10B`.
The four primary configs are `configs/{transformer,gdn2,sdm,gl_sdm}.json`.
They all use length 2,048, 8,192 tokens/update and microbatch one: four sequences
accumulated per update. GL-SDM processes four 512-token chunks per sequence,
so later chunks consume and train the earlier writes. The `*_smoke.json`
configs retain 16 layers with smaller widths and banks, using length 513 to
cross the same write boundary.

```bash
python projects/gl-sdm/scripts/run_layer_experiments.py   --phases verify verify-fp32 benchmark train --warmup 3 --iterations 5 --train-steps 20
```

This runs models serially in separate GPU processes and records every command,
exit status and artifact in `results/sequence_2048/manifest.json`. The suite
uses the same `max_split_size_mb:512` CUDA allocator setting for all models,
preventing small workspaces from splitting multi-GiB state blocks. Failures
retain their logs; the runner never reduces a workload or substitutes a kernel.
Twenty updates are a short pipeline pilot. Use `--train-steps 1000` for the
primary configs' training budget. See [results/report.md](results/report.md)
for measured results and the status of reference checks.

Individual commands use the same ATMA-style model, loss and checkpoint contract:

```bash
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
gl-sdm verify --config projects/gl-sdm/configs/gl_sdm.json   --verify-batch-size 1 --length 513
gl-sdm train --config projects/gl-sdm/configs/gl_sdm.json   --output projects/gl-sdm/checkpoints/gl_sdm
gl-sdm benchmark --config projects/gl-sdm/configs/gl_sdm.json
gl-sdm infer --checkpoint projects/gl-sdm/checkpoints/gl_sdm   --prompt 'The research question is' --tokens 64
gl-sdm eval --checkpoint projects/gl-sdm/checkpoints/gl_sdm
```

`model(inputs, targets)` returns summed CE, regularization and auxiliary loss.
`new_cache`, `prefill` and `decode` share the interface across all models.
Checkpoints include config, weights, tokenizer metadata, optimizer/RNG state
and data position. Logs retain ATMA's structured JSON blocks. Evaluation reports
nats/token, true perplexity and needle digit accuracy; it aborts on missing
samples, non-finite values or OOM.

## Measurement and earlier work

MFU uses an estimated **6ND**, with the sparse initializer excluded from N.
Each physical layer's weights count once; GL-SDM write projections count only
where writes execute. Attention and sparse state arithmetic are excluded, so
this is not exact instruction accounting. A10G's denominator is 70 dense BF16
TFLOP/s. Complete-step timings include backward, clipping and optimizer updates.
Training, prefill and decode speeds are reported separately. Fewer FLOPs do not
guarantee faster execution when sparse memory or state movement dominates.
Benchmarks record peak allocated/reserved memory separately for training,
prefill and decode; the top-level peak is the maximum across those stages.
The saved initial 16-layer measurements used length 512 and window/chunk 128.

The [earlier tied-weight controls](LOOP_CONTROLS.md) and
[their results](results/loop_results.md) are preserved separately. Their four
or eight reasoning passes, chunk size 1,024 and MFU figures do not apply to this
16-layer experiment. Larger-bank support must be promoted to URM through a
separately scoped task with dependent-workload validation before changing its
frozen pin.
