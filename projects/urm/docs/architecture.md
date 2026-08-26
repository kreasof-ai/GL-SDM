# URM architecture target

## Objective

URM tests whether a small, typed operator contract can describe the common
routing skeleton of several mixer families while preserving specialized
lowerings. It is a compiler and kernel research boundary, not a claim that all
mixers have identical mathematics.

As of this iteration URM is explicitly a **semantic-to-execution compiler for
routed sequence models** (see the [compiler charter](compiler-charter.md) and
the [CODA retrospective](coda-retrospective.md)); optimized kernels are its
lowering targets, not its definition.

The fixed skeleton is:

```text
route -> gather -> score/gate -> normalize -> weighted reduce -> optional update
```

## Contract boundary

`MixerSpec` declares common choices that affect scheduling or observable
semantics:

- query and source domains;
- dense, block-sparse, top-k, product-key, or recurrent routing;
- normalization mode;
- absent, recurrent, buffered, or transactional mutation;
- device, SRAM-staged, HBM, or sharded residency;
- determinism, page size, top-k, and collision policy.

Three optional, closed detail records refine that skeleton without admitting
arbitrary tensor programs:

- `ExpertRoutingSpec`: router origin, score activation, Top-k versus threshold
  selection, shared/routed experts, expert and latent widths, capacity policy,
  balancing method, and topology-limited groups;
- `RecurrentSpec`: named recurrence, state layout, decay granularity, edit-gate
  coupling, supported scan modes, and short-convolution use;
- `SparseAttentionSpec`: indexer type, token versus block selection, sharing
  scope, selected-unit budget, block size, and forced-local allocation.

Tensor programs, arbitrary index expressions, expert MLP definitions, and scan
equations are deliberately outside the IR. A backend receives a `MixerSpec` plus
runtime tensors and selects a family-specific lowering.

## Why flavors are compositional

A name such as "MoE" or "sparse attention" does not determine a kernel. The
first expansion set makes the differences explicit:

| Example | Score / indexer | Selection | Weighting / balance | Layout consequence |
| --- | --- | --- | --- | --- |
| DeepSeekMoE 16B | Linear router, pre-Top-k Softmax | 6 of 64 routed + 2 shared | Expert/device auxiliary losses | Fine-grained expert grouped GEMM |
| DeepSeek-V3 MoE | Sigmoid linear router | 8 of 256, at most 4 of 8 node groups | Selected-score L1; routing-only dynamic bias | Dropless, topology-aware dispatch |
| Routing-Free MoE | Per-expert internal norm + ReLU | Independent threshold, dynamic width | No competitive normalization; adaptive density auxiliary objective | Threshold compaction, variable work per token |
| Kimi K3 Stable LatentMoE | Latent linear router + Sigmoid | 16 of 896 + 2 shared | Selected-score L1; Quantile Balancing | Down-project, grouped latent experts, normalize, up-project |
| DeepSeek Sparse Attention | Learned lightning indexer | Top-k tokens | Exact Softmax on selected tokens | Fine-grained gather; one index set shared across heads |
| MiniMax Sparse Attention | Learned block index branch | 16 blocks of 128 | Exact Softmax; forced local block | Contiguous block gather per GQA group |

Load balancing is represented separately from forward routing because it may be
a training loss, a non-gradient bias update, or a quantile update. In
particular, "auxiliary-loss-free" does not mean "no balancing state".

The recurrent catalog similarly keeps a shared ordered-scan shell while
distinguishing Mamba, Mamba-2/SSD, Mamba-3, Gated DeltaNet, Gated DeltaNet-2,
and Kimi Delta Attention. These algorithms differ in state layout, decay
granularity, discretization, and whether erase and write gates are coupled, so
they must select distinct backend capabilities even if they share chunk-scan
infrastructure.

## Semantic families

| Family | URM-common portion | Specialized portion |
| --- | --- | --- |
| Dense attention | Route all sequence positions, score, softmax, reduce | QK scaling, causal semantics, backward kernel |
| Block-sparse attention | Route selected sequence blocks, softmax, reduce | Block-mask construction and tile scheduler |
| MoE | Top-k expert route, gate, gather/scatter | Expert MLP and grouped GEMM |
| Parameter-token mixer | Route parameter blocks and reduce values | Parameter layout and adapter/update rules |
| Linear recurrence / SSM | Typed recurrent state and mutation contract | Ordered scan equation and state transition |
| SDM / GL-SDM | Top-k page route, gather, reduce, buffered write | Original SDM product-key/sparse kernels; GL-SDM overlay composition and commit |

Linear recurrence is intentionally represented by a dedicated routing kind. The
reference evaluator raises `NotImplementedError` for it instead of disguising an
ordered scan as an unordered gather-reduce operation.

Original SDM is also distinct from the proposed GL-SDM transaction: it performs
ordered, in-place gated delta updates over product-key-selected slots. The IR
therefore gives it an `ordered` collision policy and a dedicated backend, while
GL-SDM retains buffered snapshot-and-commit semantics.

## Layering

```text
Model adapters
  PyTorch SDPA / FlexAttention / FLA / Mamba / MoE / original SDM / GL-SDM
                          |
NAS-facing compilation API (planner.UrmCompiler)
  validate -> enumerate candidates -> compile -> cost features
                          |
Semantic routing/state IR + verified rewrites (src/urm/compiler)
  logical domains, routes, effects, locality, registered rules
                          |
Placement, sharding, communication planning (simulated mesh today)
                          |
Trusted execution anchors + constrained visitors
  GEMM | attention | recurrent scan | grouped GEMM |
  routed reduction | page gather/update | collective exchange
                          |
Reference oracle | framework baseline | optimized upstream | URM lowering
                          |
CPU             | CUDA/Triton          | future ROCm/XPU backends
```

The NumPy oracle defines only the v0 generic routing, normalization, reduction,
deterministic ties, and write-collision behavior. It deliberately refuses the
advanced detail records until family-specific semantic oracles are added; a
coarse gather/reduce result must not be presented as equivalence for a named
architecture.

## Compiler coverage (distinct from kernel coverage)

URM reports five coverage axes separately; conflating them hides real gaps.

| Axis | Today |
| --- | --- |
| Semantic coverage | Dense/block-sparse/top-k/threshold/product-key routes over sequence/expert/parameter-block/recurrent-state/memory-page domains; ordered recurrence represented as a typed barrier op; transactional commits with version boundaries; collective intent (`all_reduce`, `all_to_all`, ...) as first-class effects |
| Compiler/reparameterization coverage | Two verified rules: routed-reduction row-scale epilogue folding and delayed row scaling through linear maps; deterministic traces; structured rejections (non-row-wise scales, nonlinear intervening transforms, effect barriers, multi-consumer intermediates) |
| Upstream adapter coverage | FlashAttention 2.8.3 dense causal; FLA 0.5.2 gated delta-rule (prefill + decode) |
| Native backend coverage | One Triton lowering family: routed-reduction v1 forward/backward; plus the experimental compiler-generated row-scale epilogue anchor |
| Distributed planning coverage | Simulated mesh only: deterministic executable plans with grouped exchanges, byte estimates, send/receive counts, commit steps; no multi-device host validation yet |

## Success metrics beyond peak kernel speed

- Semantic-family coverage (families expressible without escape hatches).
- Percentage of architectures requiring escape hatches (target: zero; the
  committed compilation matrix reports the current count).
- Valid-compilation rate over preset architectures (compilation matrix).
- Performance regret versus the best pinned upstream implementation per family
  (paired-overhead methodology from docs/benchmarking.md).
- Avoided materializations and HBM bytes (analytic bound plus measured wall/
  device deltas for the epilogue prototype).
- Communication volume and critical-path estimates from route plans.
- Adapter overhead (dispatch share) per covered family.
- NAS candidates evaluated per unit time (candidate enumeration is metadata-
  level and needs no hardware).
- Reproducibility and correctness-gate pass rate (clean-environment suites,
  artifact-schema tests, deterministic traces).

## First vertical slice

The first frozen GPU operation is routed weighted reduction over precomputed
indices. Its tensor contract, PyTorch baseline, Triton forward/backward kernels,
capability checks, and benchmark harness are described in
[Triton backend preparation](triton-backend.md).

Subsequent slices are:

1. Dense and masked sequence reduction against the NumPy oracle.
2. Stable Top-k and threshold route generation.
3. Product-key-style memory routing plus deterministic transactional write
   merging.
4. PyTorch adapters for SDPA math/flash dispatch and transparent MoE.
5. External adapters to maintained upstream kernels and trace formats.
6. Family-specific recurrent and state-mutation lowerings.

## Non-goals for milestone zero

- A general tensor compiler.
- Training a language model.
- Reimplementing every upstream kernel.
- Distributed expert or memory sharding (simulated planning exists; real
  multi-device execution does not yet).
- Selecting GL-SDM's final address or update algorithm.
