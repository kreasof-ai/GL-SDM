# Reading the code

Start with [model.py](model.py), then
[layers/global_memory.py](layers/global_memory.py) and
[layers/chunk.py](layers/chunk.py). These show the language-model interface,
the registered GL-SDM modules, and the order in which they execute.

```text
gl_sdm/
  model.py          embeddings → blocks → normalization → vocabulary head
  cli.py            train / infer / eval / verify / benchmark commands
  layers/           model components and chunk/token execution schedules
  memory/           snapshots, write proposals, commits and sparse routing
    backends/       frozen URM adapter and project-owned CUDA operators
  baselines/        Transformer block, upstream SDM/FLA adapters and references
  runtime/          compiled tensor arithmetic and training CUDA graphs
  experiments/      data, training, checkpoints, inference, evaluation and MFU
```

## The current GL-SDM forward

For `gl_chunk_size > 1`, the order is:

1. Take the current frozen bank snapshot for this absolute chunk.
2. Apply local causal attention once, with normalization and a residual.
3. Repeatedly apply the tied global reasoner: route a query, read the snapshot,
   add the input condition and projected reading, then apply the residual MLP.
   Fixed depth runs all passes; adaptive depth tracks halt probabilities and
   accumulates a weighted output.
4. Build write deltas against the same snapshot. Fixed-depth projections batch
   across passes; adaptive proposals follow the active tokens.
5. At the chunk boundary, commit the deltas and reset local attention's KV cache.
   An unfinished serving call retains its proposals and local KV tensors.

Local attention is an attention mixer, with no separate local MLP. It runs
before global reasoning, rather than between global passes. The model has one
tied global block and one shared learned bank. This describes the existing
implementation; the reorganization does not change the architecture.

## Where to inspect each operation

| Concern | Implementation |
| --- | --- |
| Modules, configuration and checkpoint parameter names | [layers/global_memory.py](layers/global_memory.py) |
| Local causal attention | [layers/attention.py](layers/attention.py) |
| Global read/write projections | [layers/router.py](layers/router.py) |
| Chunk clock, reasoning and halting | [layers/chunk.py](layers/chunk.py) |
| Dense reasoner equations and compilation adapter | [runtime/dense.py](runtime/dense.py) |
| Frozen views, request cache and learned bank | [memory/state.py](memory/state.py) |
| Reusable read/propose/merge/commit API | [memory/transactions.py](memory/transactions.py) |
| Chunk routing and write construction | [memory/routing.py](memory/routing.py), [memory/chunk_writes.py](memory/chunk_writes.py) |
| Actual URM route/read plans and execution | [memory/backends/urm.py](memory/backends/urm.py) |
| Project-owned ordered commit kernel | [memory/backends/commit.py](memory/backends/commit.py) |
| Sequential token-clock control (`gl_chunk_size=1`) | [layers/token.py](layers/token.py), [memory/backends/token.py](memory/backends/token.py) |
| Reference checks and MFU accounting | [experiments/verify.py](experiments/verify.py), [experiments/metrics.py](experiments/metrics.py) |

URM is installed at the frozen dependency pin; its source is outside this
repository. The chunk path uses URM route/read forward and backward. Ordered
commit, chunk scheduling, projection batching, PyTorch compilation and CUDA
graph capture are project-owned. See [provenance](../../PROVENANCE.md) for their
origins and the current reference-check limitations.

The public `gl_sdm.Model`, `gl_sdm.create_model`, `gl_sdm.memory` API and CLI
commands are retained. Internal imports now use the directories above; old flat
module paths have been removed. Existing checkpoint keys and configuration
fields are retained. New experiment source fingerprints include all Python
subpackages; historical result artifacts retain their original fingerprints.
