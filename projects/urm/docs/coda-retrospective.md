# CODA retrospective and design mapping

Reference: *CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs*
([arXiv:2605.19269](https://arxiv.org/abs/2605.19269), v2). This document maps
CODA's compiler strategy onto URM. URM adopts **ideas**, not code: the paper's
repository was inspected as a reference only, and no source was copied into
this repository.

## CODA's strategy in one paragraph

CODA treats a Transformer block as a small set of trusted, optimized GEMM
mainloops plus *constrained* prologue/epilogue/side-output programs. Instead of
synthesizing arbitrary kernels, it algebraically moves memory-bound work
(normalization, residual adds, masking, loss heads) into the lifetime of tiles
that are already resident inside a GEMM mainloop, so intermediate tensors stop
being materialized. Compilation becomes a composition/search problem over a
small vocabulary of verified program fragments anchored at GEMMs.

## Concepts adopted by URM

| CODA concept | URM realization |
| --- | --- |
| Trusted optimized mainloops | `ExecutionAnchor` registry; production kernels (FlashAttention, FLA, routed-reduction v1) stay trusted lowering targets |
| Constrained prologue/epilogue programs | Typed `VisitorDescriptor` / `EpilogueSpec` - declarative descriptors interpreted by anchors, never Python callables |
| Move memory-bound work into resident-tile lifetimes via algebra | Verified rewrite rules (`rewrite.py`) with preconditions, equivalence class, numerical envelope, effect preservation |
| Side outputs with bounded lifetime | `SIDE_OUTPUT` visitor kind; obligations recorded in the compilation trace |
| Compilation as composition/search over fragments | Planner enumerates legal candidates per program; accepted and rejected schedules are retained in artifacts |
| Keep what is measured; demote what loses | Experimental anchors are marked `experimental` until differential + performance gates pass; rejected schedules stay in artifacts |

## Concepts deliberately not adopted (yet)

- **GEMM-only anchoring.** CODA centers on one anchor kind. URM must cover
  routing, state mutation, ordered scans, transactional commits, and distributed
  communication, so URM's anchors form an open typed family
  (`AnchorKind`), not a single mainloop template.
- **Tile-local epilogues as the only fusion site.** URM generalizes the fusion
  question to locality levels (`locality.py`): register/lane/tile/block/device/
  mesh. A rewrite may only move work where the locality model allows it.
- **Block-level rewriting of a fixed architecture.** CODA rewrites known
  Transformer blocks. URM compiles architecture *descriptions* (NAS-facing),
  so its rewrites are registered rules over semantic IR nodes rather than
  hand-matched module patterns.
- **CuTeDSL/CUDA-specific code generation.** Not required for this iteration's
  proofs on A10G; generated-kernel capability is demonstrated with Triton
  (routed-reduction epilogue prototype) instead.

## Why CODA-inspired rewriting sits below URM's semantic layer

The delayed-scaling identity proves the split:

```text
Linear(RowScale(x, r), W) <-> RowScale(Linear(x, W), r)
```

In URM this is rule `delay_row_scale_through_linear_matmul`: a verified
contract over semantic ops (`Transform(ROW_SCALE)` feeding `Matmul`), with
rejections for non-row-wise scales, nonlinear intervening transforms, and
effect barriers (mutation/communication between the two ops). The rule knows
nothing about tiles or PTX; anchors decide how "scale in the epilogue" is
physically realized. Conversely, the routed-reduction row-scale epilogue
prototype shows the same principle landing in a real kernel: the planner folds
the scale into the reduction's typed epilogue, the experimental Triton anchor
executes it, and the artifact records avoided materialization bytes and
wall/GPU time against the materialized plan.

## Prototype summary (this iteration)

- Rule set: `fold_row_scale_into_routed_reduction_epilogue` (floating-point
  equivalence; backward CERTIFIED for fp32/fp16/bf16 via tile recomputation -
  gradients cover weights, values AND the row scale - with a resolved
  recomputation obligation) and `delay_row_scale_through_linear_matmul`
  (floating-point equivalence with dtype envelopes; full gradient proof via
  linearity). Neither rule carries a forward-only restriction any more, and
  `exact` equivalence is reserved for bitwise-equivalent execution models.
- Differential results: fp32/bf16/fp16 envelopes hold on GPU
  (`tests/test_compiler_epilogue_gpu.py`, CPU GEMM identity:
  `tests/test_compiler_delayed_scaling.py`).
- Measured comparison: `results/compiler/routed-scale-epilogue/benchmark.json`
  (materialized vs fused; host-bound and GPU-bound shapes separated;
  rejected schedule retained).
- Solver-guided selection of launch schedules for this epilogue:
  `docs/kernel-generation.md` and
  `results/compiler/solver/routed-epilogue-selection.json`.
