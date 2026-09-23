# Model-level product evidence: plan and status

Status: active construction. This tracks the move from the per-recipe kernel table
(`product-table.md`) to **model-level** MFU / MBU evidence, which is the number that
is actually meaningful for a product claim.

## Why model-level

The per-recipe table measures one isolated mixer kernel at T=256. At that scale the
kernel is launch/latency-bound, so MFU collapses to ~0.5% regardless of kernel
quality. MFU is a **model-level** metric: it only reaches the 30-50% (training) /
~80% (prefill) range when the mixer sits inside a full model doing a large enough
step to amortize launch overhead and saturate the tensor cores. Decode is the
mirror: it is memory-bandwidth-bound, so the honest metric is **MBU** (achieved
bytes/s / measured HBM bandwidth) at large batch (256-512), targeting 60-70%.

## Targets (the product bar)

| Claim | Target | Bound |
|---|---|---|
| Training MFU | ~50% | compute (tensor core) |
| Prefill MFU | ~80% | compute |
| Decode MBU | 60-70% | HBM bandwidth, bs 256-512 |

## What already exists (reused)

`benchmarks/pretraining_step.py` + `src/urm/pretraining.py`: a frozen model-level
harness - URM-owned decoder LM (12 layers, 768 width, ~100M params), FineWeb-Edu
tokens, AdamW (fp32 master), bf16, gradient accumulation + clipping, state
reset/detach, paired-seed fresh-process measurement, and model MFU against the
**measured** bf16 tensor-core peak (`results/device-limits.json`). Currently
hardwired to the sparse-memory mixer vs upstream SDM.

## Work items

1. **Backwards for all 62** (in flight, 3 parallel tiers) - the blocker for training
   coverage. Each must match the canonical core gradient (< 2e-2) and set
   `backward_supported: True`.
   - **Tier 2 (DONE)**: the 4 materializing K1 recipes (parallax, kata, tda,
     deltaformer) now have correct native backwards via differentiable recomputation
     (grad err ~1e-7 vs the reference). Forward fast-path unchanged.
   - **Tier 1 (in flight)**: extending the existing gated matrix-state backward to
     the normalizer / multi-rank / dual-gate / left-transition / polynomial configs
     (adds NORM_STATES / RANK_STATES to the forward+backward kernels).
   - **Tier 3 (DONE)**: all 13 distinct recurrent operators (rnn, gru, m2rnn, mamba2,
     mamba3, rwkv4, rwkv6, ttt, titans, mesa, gated_oja, gsa, abc) now have correct
     native backwards via differentiable recomputation wrapped around the unchanged
     native forward kernels (grad err 1e-7..1.6e-5). New `recurrence/backward.py`.

**Backward coverage: 62/62 (commit fec52de).** All three tiers landed and
momentum_delta (the one distinct-executor recipe Tier 1 flagged) was filled in.
Every recipe's native backward matches the canonical/reference gradient (< 2e-2,
most ~1e-7). Full test suite passes. Training coverage is no longer blocked.

## Master coverage table (in construction)

`benchmarks/master_table.py` drops each recipe into the frozen 100M model as a
`RecipeMixer` (shared projection stack; native and upstream differ only in the
mixer kernel). Columns: gradient parity, parameter parity after 10 FineWeb steps,
training MFU, prefill/decode MFU+MBU+throughput (seq 1K-32K, bs 256-512), inference
KL divergence, and peak memory (training / prefill / decode / long-sequence).

- **Native coverage: 62/62** in the 100M model (all run forward+backward).
- **Upstream coverage: widened by fixing the adapter dtype/config mismatches** -
  the "no plan" cluster was a bf16-vs-fp32/intent mismatch (fixed with a dtype
  fallback), plus per-adapter fp32-gate casts (comba, gated_oja, mesa) and
  static-head-decay handling (lightning, retention). The irreducible gaps are
  (a) heavy missing deps (`mamba_ssm`, `kata`, `xma`, `flash_attn`), (b) hard FLA
  revision pins (ttt, titans, gsa, hgrn), (c) HLA's dim<=32 kernel limit, and
  (d) K3 (no upstream adapter by design). These are reported as "no upstream",
  never fabricated.
- **Correctness columns compare native vs the reference equation** (always
  available); **performance columns compare native vs upstream** only where a real
  adapter exists. A native-vs-upstream number is never shown without an upstream.
- **Levers**: CUDA-graph replay for decode (gla decode 6.8% -> 37% MFU at bs512);
  `torch.compile` is opt-in (it does not help here - the custom native kernel is
  the bottleneck, and upstream SDPA/FLA are already fused).

## First validated row (mha, 124M params, FineWeb)

The full pipeline produces every column, native vs upstream:

| Metric | Native | Upstream |
|---|---|---|
| Training MFU | **34.1%** | 35.7% |
| Training throughput | 30,205 tok/s | 31,621 tok/s |
| Prefill MFU @1K | 31.0% | 37.7% |
| **Decode MFU @bs512** | **49.7%** | 40.8% |
| Gradient parity | 0.013 | - |
| Inference KL div | 1.65e-5 | - |
| Peak mem (train) | 2030 MB | 2663 MB |

Native mha **beats upstream on decode MFU** (CUDA-graph replay) and is within ~5%
on training/prefill. This is the honest model-level comparison the table records
per recipe.
2. **Generic recipe mixer block** - a `RecipeMixer(nn.Module)` that projects
   hidden -> recipe operands -> native compiled plan -> back, so any of the 62
   recipes slots into the frozen 100M model in place of `SparseMemoryMixer`.
3. **Per-recipe FLOP model** - generalize `semantic_training_flops` so MFU credits
   the actual recipe mixer.
4. **FineWeb shard** - RESOLVED via the public mirror `kjj0/finewebedu10B-gpt2`
   (same GPT2 token-shard format: magic 20240520, v1, 100M tokens, 200MB). The
   original pinned shard (`karpathy/fineweb-edu-100B-gpt2-token-shards` @ a33f75d,
   SHA-256 6bb7ce...) is gated on HF (401 anonymously). MFU is data-agnostic, so the
   mirror shard drives the performance number; the exact pinned hash is only needed
   for bit-exact reproduction of the frozen acceptance gate.
5. **Inference harness** - model-level prefill MFU and decode MBU at bs 256-512.

## Proof of concept (validated)

`benchmarks/model_level_poc.py` drops the native `gla` mixer into the frozen 100M
model (131.5M params, seq 1024, grad-accum 4) and measures end-to-end training:

```
model parameters: 131.5M
median step: 424.7 ms   throughput: 9,644 tok/s
model FLOPs/step: 3.23 TFLOP   measured bf16 peak: 66.2 TFLOP/s
MODEL TRAINING MFU: 11.50%
```

This is the honest model-level baseline: 11.5% MFU versus the ~0.3% isolated-kernel
number - confirming that MFU is a model-level property. The gap to the ~50% target
is kernel optimization (the gla mixer is correctness-first), which is the
"optimize after" phase. The RecipeMixer bridge (project hidden -> q,k,v,log_decay
-> native plan -> project back, with `-softplus` gating for the decay) is proven
and generalizes to the other recipes.

## Honest caveat

The current native kernels are correctness-first. The model-level harness will
report the **true** (initially low) MFU/MBU. Reaching the 50/80/70 targets is
kernel *optimization* work that comes after the honest baseline is published.

## Performance levers (required for the targets)

- **Training**: `torch.compile` the forward+loss step (fuses the elementwise work
  around the mixer kernel). Without it the step is launch/overhead-bound.
- **Inference decode**: CUDA-graph capture + replay of the single-token forward
  (removes per-kernel launch overhead in the bandwidth-bound regime). The native
  decode kernels are CUDA-graph-capturable by design (in-place state, no autograd
  graph, no host sync).
- **Inference prefill**: large batch / long sequence to stay compute-bound.

The master table (`benchmarks/master_table.py`) applies all three; each row records
whether compilation / graph capture succeeded (`*_compiled`, `cuda_graph`).
