# Benchmark authority repair from 42671f9

The frozen model, FineWeb shard and revision, context, 64/64 routes, optimizer,
seeds, warmup/measurement counts, tolerances, and D=4 production value fragment
are unchanged. All previously committed negative artifacts remain byte-for-byte
intact. New confirmation and profile outputs use `authority-v2` filenames and
new schemas; the old schemas remain available for historical artifacts.

The benchmark audit repairs five authority problems:

1. Correctness executes the requested eager or `compile_fullgraph` callable for
   five optimizer steps. Parameters, complete initial FP32 optimizer tensors,
   and all prefetched batches have SHA-256 identities. A diagnostic-free cold
   step runs first; the model and optimizer are restored and hash-checked before
   correctness. Native/upstream pairs and eager/compiled pairs must match these
   identities. Training losses, logits, parameter-gradient norms, updates, and
   finite checks come from the actual execution. Post-update no-grad evaluation
   is explicitly auxiliary eager evidence. Non-leaf mixer gradients are captured
   in a separately labelled eager replay from the same initial state and batches;
   their original cosine/error gates remain required independently.
   `loss_before` is the mean training loss across accumulation microbatches;
   `loss_after` is the auxiliary eager evaluation on the first batch after the
   update. `compiled_or_eager_first_step` retains the first restored correctness
   step; cold latency is the separate diagnostic-free `first_optimizer_step_ms`.
2. Every accumulation microbatch snapshots each layer's complete persistent
   tensor after backward and persist/detach. Pairwise full-tensor maximum error
   and RMS are retained as diagnostics, alongside per-layer/microbatch checksums.
   The frozen normalized-checksum tolerance applies to the maximum across all
   microbatches. Temporary tensor snapshots are consumed before cleanup, including
   comparisons between modes; no dangling temporary paths enter the final artifact.
3. Steady and cold timing contain the complete training/AdamW step without loss
   `.item()`, checksum reductions, finite scans, or correctness tensor transfers.
   Explicit synchronization occurs only at step boundaries. The saved-activation
   accounting probe and all correctness diagnostics run outside timing.
4. The compiler now serializes the existing D=4/two-warp state schedule, using
   the same pure schedule function as the production launcher. Route forward and
   backward warp counts and pipeline stages are explicit. A separate process
   observes actual compiled Triton kernel constants, grids, warps, and stages
   for forward and backward at the frozen production shape in both execution
   modes, and rejects disagreement with the serialized plan. This audit enables
   launch hooks only in its own process/cache, never in timed children.
5. Profiling puts actual native route generation in the route range, separates
   preparation and the whole pipeline, and captures state forward and backward
   independently. CUDA entries duplicating CPU-range device attribution are
   excluded. This avoids treating preparation as route work or assigning route
   work to state. The profile remains a separate pass from acceptance timing.

`pretraining_projection.project_state_replacement` uses
`R_baseline * ((1 - f) + f / s)`, where `f` is the replaceable fraction of the
native baseline and `s` is a measured **complete-stage** speedup. It reports the
infinite-speedup floor and speedup needed for ratio 1.05. For example, a baseline
ratio 2.365 with `f=0.8` and `s=3` predicts 1.10367 and fails screening for model
integration. Threefold stage speedup is not automatic acceptance. A credible
projection must include preprocessing, backward, allocations/scratch, and
orchestration; device-kernel time alone does not establish it. Only after
numerical acceptance and a credible projection at or below 1.05 may a prototype
enter the unchanged full-model acceptance grid. SDPA remains architectural
context and is excluded from Sparse Memory equivalence and performance gates.

Reproduce from `projects/urm` with the pinned upstream checkout on `PYTHONPATH`:

```bash
PYTHONPATH=src:/tmp/urm-sdm-upstream.wKxbSo LIBRARY_PATH=/opt/conda/lib \
python benchmarks/pretraining_step.py \
  --output results/pretraining-step/confirmation-authority-v2.json
PYTHONPATH=src:/tmp/urm-sdm-upstream.wKxbSo LIBRARY_PATH=/opt/conda/lib \
python benchmarks/profile_pretraining_step.py --backend urm_native \
  --output results/pretraining-step/native-profile-authority-v2.json
```

The confirmation intentionally exits nonzero after saving an artifact if frozen
gates fail. Neither failure nor a missing projection authorizes kernel integration.

## Corrected A10G evidence

`results/pretraining-step/confirmation-authority-v2.json` completed the entire
three-seed, two-mode grid and its separate SDPA controls. Its decision is
`correctness_failure`; allocated-memory gates pass, but native training remains
substantially slower.

| Execution | Native/upstream geometric mean | Hierarchical 95% ratio CI | Direct sparse correctness by seed |
| --- | ---: | --- | --- |
| Eager | 2.432676 | [2.415884, 2.436041] | pass / fail / fail |
| Compiled fullgraph | 2.311383 | [2.301176, 2.318509] | pass / pass / pass |

All compiled sparse processes have one graph, zero breaks, and zero recompilations.
Eager-versus-compiled comparisons pass for upstream in all three seeds; native
misses the unchanged state-checksum gate in seeds 2903 and 4409. Separate eager
internal-gradient replays also retain their own checksum misses. For example,
the actual eager comparison at seed 2903, optimizer step 4, microbatch 3, block 1
has normalized checksum error `2.6427151169627905e-6` against `2e-6`.
Native eager-versus-compiled seed 4409 step 5 reaches
`2.6281632017344236e-6`. These failures are not reassigned to compiled/native
equivalence merely because the process was labelled fullgraph.

`native-profile-authority-v2.json` and its trace record 48 state-forward and
48 state-backward spans: 1010.322 ms forward and 2465.791 ms backward within
a 4163.758 ms profiled optimizer step. The diagnostic replaceable fraction is
0.834850. `state-screening-authority-v2.json` applies the corrected formula to
the eager baseline: an **assumed**, not measured, complete-stage speedup of 3
predicts ratio **1.078730**, above 1.05. The implied speedup requirement is
**3.132959**. This profile-based screen is not an accepted model projection;
there is no measured candidate complete-stage speedup or permission to skip
the numerical and full-model gates.

Validation: 50 targeted compiler, accounting, authority, artifact, semantic, and
GPU tests pass; repository-wide Ruff and whitespace checks pass. The benchmark
failure above is the retained experimental outcome, not a test or schema failure.
