# Reading the code

Start with [model.py](model.py), then
[layers/stack.py](layers/stack.py) and
[layers/global_layer.py](layers/global_layer.py). These show the language-model interface,
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

The primary configs select the untied 16-layer stack. Read
[layers/stack.py](layers/stack.py) for its complete execution order:

1. Freeze one shared memory bank at the start of each 128-token chunk.
2. Execute `local → local → global → local` four times, with distinct weights.
3. Local blocks have rolling 128-token attention and an MLP; their KV history
   persists across memory commits.
4. All four global layers read the same snapshot. Each proposes a write from
   its updated hidden state; proposals do not affect other layers in the chunk.
5. Sum token/layer deltas without averaging and commit at the chunk boundary.

[layers/global_layer.py](layers/global_layer.py) is one global read/MLP block
and its write equation. The bank is registered once at `model.bank`, while all
16 physical layers appear in `model.blocks`. The evaluation adapter calls the
model's chunk schedule so it preserves the shared-bank contract. No loop depth
or ACT telemetry applies to this stack.

The earlier tied-weight controls remain in
[layers/global_memory.py](layers/global_memory.py),
[layers/chunk.py](layers/chunk.py) and [layers/token.py](layers/token.py), solely
for the preserved historical configs. See [their guide](../../LOOP_CONTROLS.md).

## Where to inspect each operation

| Concern | Implementation |
| --- | --- |
| Layer pattern, cache and shared bank | [model.py](model.py), [layers/stack.py](layers/stack.py) |
| One global layer and write equation | [layers/global_layer.py](layers/global_layer.py) |
| Local causal attention | [layers/attention.py](layers/attention.py) |
| Global read/write projections | [layers/router.py](layers/router.py) |
| Current chunk clock and layer order | [layers/stack.py](layers/stack.py) |
| Earlier loop/halting controls | [layers/chunk.py](layers/chunk.py) |
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
module paths have been removed. Earlier checkpoints retain their keys and load from their stored configurations.
The new stack registers one `bank.memory` and 16 distinct `blocks` entries. New experiment source fingerprints include all Python
subpackages; historical result artifacts retain their original fingerprints.
