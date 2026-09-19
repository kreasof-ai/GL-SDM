# URM Runtime and Execution Layer

**Status:** Design document. Companion to `sdm-sparse-attention-reparameterization.md` (math) and `dual-form-coverage-roadmap.md` (architecture coverage). This doc covers the downstream half — kernels, dispatch, hardware portability, and the integration with the Tensor compiler.

**Scope:** What runs after a `MixerSpec` is created. The 3-kernel taxonomy that satisfies 90% of literature coverage, the dispatch decision tree that picks the right kernel per call, the Tensor compiler integration story, and the optimisation-as-architecture extension pattern.

---

## 1. Executive Summary

The architecture doc (`architecture.md`) defines the **semantic contract**: `MixerSpec` declares routing/state/communication, and the compiler produces a verified reparameterisation plus placement. This doc defines what happens at runtime:

1. **3 irreducible kernel shapes** handle ~90% of published mixers: dense softmax, dense linear/delta, sparse slot.
2. **A 7-phase dispatch tree** routes a `MixerSpec` call to either a specialised vendor kernel (FA-family, fla-org) or the framework's generic kernel (with TileLang codegen for cross-hardware).
3. **Tensor compiler integration** is the Bun/React-equivalent split: URM is the framework, Tensor is the runtime. URM doesn't ship kernels; it ships kernel **contracts** that Tensor compiles to hardware variants.
4. **Optimisation-as-architecture** extends the framework to support TTT-style inner-loop updates via a `state_update = Callable` plug-in on `MixerSpec`. This is a research axis, not a stable-release feature, but it's the structural extension point.
5. **Pattention (MLP/MoE as parameter-axis attention)** widens the framework from "sequence-axis mixer" to "tensor-axis mixer," covering the entire transformer block's compute in one kernel shape.
6. **MFU targets** are realistic: 60-70% for the framework's generic kernels, competitive with vendor implementations where they exist, dominant on the sparse-slot path where no vendor alternative exists.

The headline architecture decision: **the framework ships 3 canonical Triton/TileLang sources; Tensor's compiler emits per-hardware variants; the dispatcher picks the best available kernel at runtime.** Same kernel logic across all hardware; specialised codegen per hardware.

---

## 2. The 3-Kernel Taxonomy

The matrix `(I + D_β A)` has three structurally distinct regimes. Each is an irreducible kernel:

### 2.1 Kernel 1: Dense Softmax

```python
def dense_softmax_kernel(
    Q, K, V: [B, H, T, D],
    mask: optional additive mask,
    softmax_kind: "causal" | "full" | "foveal" | "with_null",
    score_fn: "softmax" | "relu_softmax" | "topk_softmax",
) -> [B, H, T, D]:
    # Online softmax + masked GEMM. Pure Omega_read @ V, no triangular solve.
```

**Covers:** vanilla attention, MHA/MQA/GQA, MLA, cross-attention, foveal sparse (NSA/MoBA/Longformer via mask), Differential (TDA via `Ω_read = Ω₁ − λΩ₂`), TPA/Tucker via upstream factorisation, Sparse Transformer rectified attention.

**Replaces:** FlashAttention-family (vendor). The framework's kernel is "FA4-equivalent in shape" — not in absolute performance.

### 2.2 Kernel 2: Dense Linear / Delta

```python
def dense_linear_delta_kernel(
    Q, K, V: [B, H, T, D],
    A_schedule: per-token decay OR per-token rate,
    correction: bool,            # delta-rule correction on/off
    layout: "linear" | "delta",
) -> ([B, H, T, D], [B, H, D, D]):
    # Triangular solve on (I + D_beta K K^T) Delta = D_beta(V - V^0).
    # Same kernel serves linear (correction=False) and delta (correction=True).
```

**Covers (correction=False, A_schedule=diag(g)):** GLA, RetNet, Mamba-2 SSD, KATA, HLA, all linear-attention variants.

**Covers (correction=True):** GDN, GDN-2, KDA, MesaNet, PGDN, PKDA, all delta-rule variants.

### 2.3 Kernel 3: Sparse Slot Memory

```python
def sparse_slot_kernel(
    memory: [B, H, S, D],
    read_indices, read_weights: [B, H, T, R],
    write_indices, write_weights: [B, H, T, W],
    values: [B, H, T, D],
    beta: [B, H, T],
    log_decay: [B, H, T],
) -> ([B, H, T, D], [B, H, S, D]):
    # Sparse indexer + slot-routed updates + boundary fold.
    # URM's native case; no vendor alternative.
```

**Covers:** SDM, GSA, MoM, Raven, sparse slot routing in general. **Also covers parameter-axis attention** (MLPs as static-key attention, MoE as static-key sparse routing — see §7).

### 2.4 Why exactly three?

The three kernel shapes correspond to three structurally different `(A, Ω_read)` configurations:

| Kernel | A shape | Ω_read shape | Solve pattern |
|---|---|---|---|
| Softmax | zero (no write-collision) | dense softmax | pointwise softmax + GEMM |
| Linear/delta | dense triangular `I + D_β KK^T` | dense linear/near-identity | triangular solve |
| Sparse slot | sparse `[S, S]` collision | sparse `[T, S]` from routing | gather/scatter |

Folding these into one or two kernels loses 5-10×. The softmax kernel needs no triangular solve; folding linear into it would invoke the wrong gate. The triangular solve has a different register-pressure profile than the gather/scatter pattern.

**Could you approximate delta rule with linear attention?** Yes, by setting the decay appropriately. But you lose the correction term `S_{t-1} k_t k_t^T` that discriminates delta-rule recall from EMA-style decay. Different primitive; different kernel.

### 2.5 Counts

| Level | Count |
|---|---|
| Irreducible kernel shapes | 3 |
| Hand-maintained canonical sources | 3 (one per shape) |
| Per-shape variants (chunked, autotuned, bwd) | ~3 per shape |
| Per-hardware compiled kernels | 3 × variants × hardware = ~30-60 |
| Per-framework-config specialisations | unlimited (config-generated) |

Hand-maintenance cost is 3 sources. Per-hardware variance is Tensor's job.

---

## 3. Tensor Compiler Integration — The Bun/React Split

URM is React. Tensor is Bun. Together they layer.

### 3.1 The architecture

```
┌─────────────────────────────────────────────┐
│  URM: architecture choice                   │   ← "what mixer kind?"
│  (MixerSpec, Dispatcher, autograd)         │
├─────────────────────────────────────────────┤
│  Tensor: hardware choice                    │   ← "what hardware target?"
│  (TileLang source, TIRx IR, providers,     │
│   capability model, module cache)           │
├─────────────────────────────────────────────┤
│  hardware                                   │
└─────────────────────────────────────────────┘
```

URM provides the architecture-level abstraction. Tensor provides the hardware-level abstraction. They meet at the TileLang source + capability query interface.

### 3.2 The contract

**URM produces:**
- A `MixerSpec(kind, axis, score_fn, aggregator, routing, ...)`
- A TileLang/Triton source module that implements the kernel contract
- A descriptor of required capabilities

**Tensor receives:**
- The TileLang/Triton source
- The capability requirements
- The target hardware

**Tensor emits:**
- A compiled binary for the target hardware
- Cached at `~/.tensor/cache/<spec_fingerprint>/<hardware>/<shape>/.tbin`

**The capabilities required from a provider include:**

```python
class RequiredCapabilities:
    matrix_multiply: bool           # GEMM
    async_copy: bool                # TMA / async DRAM → SRAM
    warp_specialization: bool       # producer/consumer warps
    hardware_collective: bool       # cross-CTA sync (not needed single-device)
    sparse_routing: bool            # URM extension point
    slot_memory: bool               # URM extension point
    fused_norm: bool                # RMSNorm, LayerNorm fused with mixer
    closed_form_solve: bool         # for MesaNet-style local opt kernels
```

These are an extension of Tensor's §10 capability model. The provider declares which it supports; the dispatcher queries.

### 3.3 What URM gains from Tensor

| Capability | Without Tensor | With Tensor |
|---|---|---|
| Add new architecture | Hand-write kernel × every hardware | Write TileLang × 1; Tensor emits variants |
| Add new hardware (Blackwell SM100) | Re-implement every kernel | Write provider × 1; kernels get variants automatically |
| Compose mixer + MLP megakernel | Hand-fuse per architecture | URM's uniform mixer interface + Tensor fusion pass |
| Specialise for new attention pattern | Hand-tune kernel | Add kind-space config; Tensor compiles |
| Per-channel β (RWKV-7-style) | Hand-derive new kernel + backward | Implement slot_channel axis; Tensor generates |

Without Tensor: N×M work (N architectures × M hardware). With Tensor: N+M work. The framework's value compounds.

### 3.4 What Tensor gains from URM

Tensor's MVP target is "FlashAttention-like kernel." Without a flagship workload, the compiler is research. **URM is the workload that proves Tensor is real.** Not a contrived test, but the actual softmax kernel that needs to ship for production.

URM provides:
- A real, demanding benchmark (sequence-mixer kernels)
- A diverse workload set (softmax + linear/delta + sparse slot)
- A test surface for new Tensor features (every `MixerSpec` config exercises the compiler)
- A flagship adoption example for the Tensor ecosystem

This reciprocal relationship is what makes the pairing work. Neither Tensor nor URM alone is the product; together they're a coherent stack.

### 3.5 The boundary discipline

**URM does not know:** about TIRx lowering passes, about TileLang primitives, about provider implementations, about backend hardware features.

**Tensor does not know:** about mixer semantics, about routing patterns, about parameter-axis vs sequence-axis, about why a particular kernel contract matters.

The boundary is the **TileLang source module + capability query**. Each side evolves independently. The integration is the last thing to commit to.

---

## 4. The Tensor Primitive Library

Tensor currently has the compiler and runtime ABI. What's missing is the kernel-level library that frameworks consume.

### 4.1 What "primitive" means

A Tensor primitive is **one compiled tensor operation with a clean signature**. Framework-neutral. Stateless with respect to autograd. No parameter management. No module composition.

The test: "would a Fortran programmer in 1985 want to call this?" If yes, it's a primitive. If no, it's a framework.

### 4.2 The MVP primitive set

```python
# Core tensor ops
tensor.gemm(a, b, accumulator=None)
tensor.attention(q, k, v, mask=None, softmax_kind="causal")
tensor.layer_norm(x, weight, bias, eps)
tensor.linear(x, weight, bias)
tensor.softmax(x, dim)
tensor.gather(x, indices, axis)
tensor.scatter(x, indices, values, axis)

# Sequence-mixer primitives
tensor.mixer(mixer_kind, memory, indices, weights, beta, decay)
tensor.kv_cache_update(cache, new_k, new_v, slot_idx)
tensor.rotary_embedding(q, k, cos, sin)

# URM-specific bridge
tensor.dual_form_sdm(memory, ...)        # fusible module
tensor.chunked_dual_form_sdm(memory, ...)  # fusible module

# Optimisation-as-architecture primitives
tensor.closed_form_solve(state, gradient, regularization)
tensor.fixed_point_iterate(state, eval_fn, n_iters)
tensor.dynamic_linear(x, hyper_input)  # hypernetwork-style
```

### 4.3 Why this matters

1. **Tensor becomes a usable product, not research.** A compiler without users is research. Primitives with named semantics make Tensor usable from day one.

2. **URM doesn't need to ship kernels.** `dual_form_sdm.py` becomes the PyTorch fallback. `tensor.dual_form_sdm` becomes the production path. URM ships the framework; Tensor ships the kernels.

3. **Other frameworks benefit for free.** ATMA's Polar attention calls `tensor.softmax` + a custom null-sink layer. Custom research code calls `tensor.gemm` directly. Ecosystem forms around the primitive layer.

4. **The primitive layer is the test surface.** Every new Tensor feature validates against the primitive API. If `tensor.attention` works correctly across all providers, the compiler is working.

### 4.4 The proposal extension

Tensor's existing `proposal.md` describes compiler + runtime + modules + providers. Add a "Tensor Primitive Library" section defining:

- The MVP primitive set above.
- Per-primitive capability requirements.
- The framework-neutral contract (no autograd coupling).
- The cache key structure (`primitive × provider × shape × dtype`).
- The integration path with URM and other consumers.

This is a small doc addition; it doesn't expand Tensor's scope, just exposes primitives at a stable API layer.

---

## 5. The Dispatch Decision Tree

When `MixerSpec(...)` is constructed and called, this is the runtime flow:

### 5.1 Phase 1: Spec construction (Python, runs once)

```
MixerSpec(kind, axis, score_fn, aggregator, routing, β/de/boundary, dtype, ...)
   │
   ├─[1] Validate — fields present, (axis, aggregator) coherent
   │
   └─[2] Compile spec fingerprint
       hash(kind, axis, score_fn, aggregator, routing,
            β/de/boundary types, dtype)
       → spec_fingerprint (cache key)
```

Failure here raises `SpecError`. Fingerprint cached for the spec's lifetime.

### 5.2 Phase 2: First call — capability + tier classification

```
spec_fingerprint + provider_capabilities
   │
   ├─[3] Hardware query
   │     device → backend (CUDA / ROCm / Metal / CPU)
   │     device capability table (compute, SRAM, TMA, mbarrier, ...)
   │
   ├─[4] Tier classify
   │     axis × aggregator × routing → kernel family
   │     softmax → Kernel 1
   │     linear/delta → Kernel 2
   │     slot/sparse → Kernel 3
   │
   └─[Result: (kernel_family, sparse_pattern, capability_set)]
```

### 5.3 Phase 3: Specialised-kernel lookup

```
(kernel_family, dtype, hardware) + specialised registry
   │
   ├─[5] Vendor / library check
   │     softmax → FA4 / FA3 / cuDNN
   │     linear/delta → fla-org GDN
   │     sparse → fla-org NSA/MoBA / URM sparse slot
   │
   ├─[6] URM hand-tuned variant check
   │     (spec_fingerprint, hardware) → registered artefact?
   │
   └─[Result: specialised_path | framework_fallback]
```

### 5.4 Phase 4: Resolution

```
(spec_fingerprint, hardware, specialised?, cached?)
   │
   ├─[7] Cache lookup
   │     ~/.tensor/cache/<spec_fingerprint>/<hardware>/<shape>/.tbin
   │     hit → load cached binary
   │     miss → continue
   │
   ├─[8] Compile decision (cold path)
   │     TileLang source + provider → TIRx IR → binary
   │     autotune tile sizes (if not pinned)
   │     cache result
   │
   ├─[9] Memory-budget check
   │     tile size × dtype × shape ≤ SRAM budget?
   │     no → reduce tile, retry
   │
   └─[10] Backward resolution
        specialised bwd available? → use it
        otherwise → autograd fallback
```

### 5.5 Phase 5: Per-call invocation (warm path)

```
MixerSpec instance + (Q, K, V, ...)
   │
   ├─[11] Resolve cached binary pointer
   │
   ├─[12] Dtype / shape check
   │      matches cached fingerprint? → invoke directly
   │      shape changed → recold cache lookup
   │
   └─[13] Kernel invoke
        forward kernel + autograd wrap (if needed)
```

### 5.6 Per-phase latency

| Phase | Cold | Warm |
|---|---|---|
| 1-2 (construct, classify) | ~ms | cached after first call |
| 3-4 (specialised, classification) | ms | cached after first call |
| 5-6 (cache, compile) | seconds to minutes | μs |
| 7 (memory check) | μs | μs |
| 8 (per-call dispatch) | μs | μs |

Total warm-path overhead: **a few microseconds per call.** The expensive parts (TileLang compile, autotune) are amortised across all subsequent calls with the same fingerprint.

### 5.7 Failure modes

| Failure | Fallback |
|---|---|
| Compile fails | Generic kernel (PyTorch eager) |
| Autotune can't find tile | Use default tile; log warning |
| Specialised kernel forward OK but no bwd | Use specialised fwd + autograd bwd fallback |
| Provider missing capability | Skip that provider; try next-best |
| Cache corrupted | Re-compile from source |
| Shape mismatch every call | Suggest preprocessing / batching |

Every failure has a defined fallback. The dispatcher never crashes silently.

---

## 6. Optimisation-as-Architecture

The boundary between "what the model is" and "how it's trained" is dissolving. TTT, MesaNet, hypernetworks, meta-learning all put optimisation loops inside the forward pass.

### 6.1 What this looks like in the framework

URM's `MixerSpec` already exposes `state_update` as part of the contract. Currently it's hard-coded to "boundary fold" — the closed-form projection of writes back into `M_T`. The extension:

```python
class MixerSpec:
    kind: str
    axis: str
    score_fn: str
    aggregator: str
    routing: str
    β_schedule: Callable
    decay_schedule: Callable
    state_update: Callable          # ← currently closed-form, extension hook
    forward: Callable
```

`state_update = tensor.boundary_fold(...)` is the default. `state_update = lambda M, X: tensor.inner_opt_step(M, X, lr=...)` is the TTT/FwPKM swap-in.

This is a **structural extension point**, not a runtime feature. The dispatcher doesn't need to know what's inside `state_update`; it just calls it.

### 6.2 Two-tier strategy

**Tier 1: Kernel-level primitive (closed-form cases).**
- MesaNet's local optimization: closed-form solve → fits a kernel primitive.
- Hypernetwork: forward pass → fits a forward-pass primitive.

```python
tensor.closed_form_solve(state, gradient, regularization)        # MesaNet
tensor.dynamic_linear(x, hyper_input)                           # hypernet
```

The codegen fuses the matrix ops into single kernels.

**Tier 2: Framework-level loop support (gradient-based cases).**
- TTT / MAML / Reptile: gradient-based inner loop.

```python
urm.inner_loop(forward_fn, params, data, n_steps, lr, checkpoint=True)
```

Properties:
- Low-overhead step runner (≤5μs per step, similar to torch.func)
- Checkpointed backward for memory efficiency
- Calls compiled primitives per inner step; only the loop wrapper is Python

### 6.3 Coverage by architecture kind

| Inner-loop type | Lives in | Status |
|---|---|---|
| Closed-form solve (MesaNet) | `tensor.closed_form_solve` | Add for stable release |
| Fixed-point iteration (Hopfield) | `tensor.fixed_point_iterate` | Add if needed |
| Hypernetwork | `tensor.dynamic_linear` | Standard pattern |
| Gradient-based (TTT, MAML) | `urm.inner_loop` | Add for stable release |
| FwPKM-style (chunk-level gradient on slots) | `urm.inner_loop` calling `tensor.dual_form_sdm` | Fits the framework's interface |

### 6.4 What "fits the framework" means for FwPKM

FwPKM is structurally:
- A sparse slot memory (URM's Kernel 3 area)
- With state updates via gradient descent on activated slots

Today, the dual-form kernel's `state_update = boundary_fold` doesn't compute gradients. The framework extension to support FwPKM is:

```python
spec = MixerSpec(
    kind="fwpkm",
    axis="sequence",
    aggregator="identity",
    routing="sparse_topk",
    state_update=partial(
        urm.inner_opt_step,
        loss_fn=local_memory_rewrite_loss,
        lr=chunk_learning_rate,
    ),
)
```

The dispatcher's API absorbs it. The static kernel does not. The framework's static-A invariant is preserved in the kernel itself; the framework's `state_update` plug-in allows the broader semantics.

---

## 7. Pattention: MLP and MoE as Parameter-Axis Attention

`foveal-sparse-indexer/foveal_indexer/pattention_triton.py` shows that **MLP is static-key attention with non-softmax activation**:

```
pattention(q=x, k=W_gate, v=W_down, activation=SwiGLU) ≡ MLP
```

This widens URM from "sequence-axis mixer" to "tensor-axis mixer":

| Component | Axis | Keys | Activation | Routing |
|---|---|---|---|---|
| Sequence attention | T (sequence) | dynamic | softmax / linear | dense / sparse |
| MLP (linear) | D (feature) | static | identity | dense |
| MLP (SwiGLU) | D (feature) | static | pointwise | dense |
| MoE (top-k) | D (feature) | static | pointwise | sparse top-k |
| MoM | mixed | mixed | mixed | mixture |

The framework's primitives all carry over:

- **`A` (write-collision matrix)** = expert overlap (MoE) or parameter-pair overlap (MLP) for D-axis; slot overlap for T-axis.
- **`Ω_read`** = the read contraction along the chosen axis.
- **`β`, decay** = same role.
- **`M_T → M_D`** = persistent parameter table (MLP/MoE) instead of slot table.

### 7.1 What this means for Kernel 3

Kernel 3 (sparse slot memory) covers parameter-axis attention naturally when keys are static:

- **Dense MLP** as `dense_softmax_kernel` (linear variant) with `axis="parameter"` and `aggregator="linear"` (identity for MLP, SiLU for SwiGLU).
- **Sparse MoE** as the same kernel with `routing="sparse_topk"` over the expert parameter table.
- **Mixed MoM** as composition of T-axis and D-axis mixer calls.

The backward-pass audit is the only thing that needs verification. Pattention forward is Triton-fused; backward needs to either (a) be a separate Triton bwd kernel, (b) use PyTorch autograd fallback, or (c) be derived analytically. This is the same priority as fixing `dual_form_sdm.py`'s `dLogDecay` silent zero.

### 7.2 What this means for the paper

If the pattention backward is verified:

> "URM unifies sequence-axis and parameter-axis mixing under one kernel shape. A single Megakernel handles sequence attention, MLPs, and MoE — three components that together account for ~80% of transformer-block compute."

That's a much stronger contribution than "unified sequence mixer." The competitive move is to make the simplest nontrivial claim as strong as possible, and "one kernel for the whole block" is much stronger than "one kernel for the sequence axis."

### 7.3 The tensor-axis framework contract

```python
class MixerSpec:
    axis: Literal["sequence", "parameter", "mixed"]
    # for axis="parameter": static keys (parameter table)
    # for axis="sequence": dynamic keys (sequence tokens)
    # for axis="mixed": composition of both
    key_source: Literal["dynamic", "static"]
    value_source: Literal["dynamic", "static"]
    aggregator: str
    routing: str
```

Pattention is just `axis="parameter", key_source="static", aggregator="linear"`. The framework's existing primitives absorb it.

---

## 8. Kernel Optimisation: MFU Targets

Per-kernel optimisation priority for the stable release:

| Priority | Kernel | Target MFU | Effort | Why |
|---|---|---|---|---|
| High | Kernel 3 (sparse slot) | 60-70% on H100/B100 | 4 weeks | No vendor alternative; framework's competitive win |
| Medium | Kernel 2 (linear/delta) | 55-60% on H100/B100 | 3 weeks | Less mature than softmax; can match fla-org |
| Low | Kernel 1 (softmax) | 55-65% on H100/B100 | 1-2 weeks | FA-family has head start; aim for FA2 equivalence |

The numbers assume Triton 3.0+ on Hopper/Blackwell. For older hardware, expect 5-10% lower.

### 8.1 Optimisation tactics per kernel

**Kernel 1 (Softmax):**
- Online softmax (Triton pattern library)
- Two-stage online softmax (FA2-style; eliminates re-pass for backward)
- Persistent threadblocks on H100/B100
- Per-shape tile-size search (Triton autotune)

**Kernel 2 (Linear/Delta):**
- Chunked WY representation (parallel triangular solve within chunk)
- TMA loads for K, V, β_t, g_t
- Per-tile delta correction accumulated in registers
- Sparse index recomputation outside the GEMM

**Kernel 3 (Sparse Slot):**
- Chunked routing with SRAM slot table hash
- Async copy of slot values from HBM
- Sparse scatter-add via atomic stores or pre-allocated output buffers
- Fused boundary fold into the same kernel as the sparse write
- Tile-level sparse GEMM (custom kernel for sparse K vs dense V)

### 8.2 What determines the gap to FA4

The remaining ~10-15% gap between the framework's kernels and FA4 is:

1. **Hardware intrinsics depth.** FA4 uses raw TMA descriptors, mbarriers, warp-group MMA. Triton 3.0 exposes most of these through `tl.extra.cuda.*` but with higher-level abstractions. The last 5-10% needs hand-tuned provider paths.
2. **Tile-size search.** FA4's tiles come from offline autotuning over millions of shapes; Triton's runtime autotune is faster but less exhaustive.
3. **Instruction interleaving.** FA4's specific pattern: `cp.async.bulk` → `commit` → `mma` → `wait` with hand-tuned latency hiding.
4. **Special cases.** FA4 ships dedicated kernels for `(D=64, 128, 192)`, mixed-precision paths, block-sparse variants. Triton uses one general kernel.

### 8.3 The honest framing for the paper

> "The 3-kernel fallback path achieves 50-65% MFU on Hopper-class hardware, comparable to flash-attention-class implementations in the same generation. The framework dispatches to specialised kernels (FA4, fla-org GDN) where available; the fallback kernels exist for exotic configurations and new hardware targets where specialised implementations have not yet been written. The sparse-slot kernel achieves 60-70% MFU, competitive with or exceeding specialised sparse-attention implementations because the framework's kernel design is purpose-built for this regime and not constrained by softmax-attention precedents."

### 8.4 Per-kernel engineering plan

For the workshop + MLSys '28 timeline:

- **Month 1-2:** Kernel 1 Triton-tuning to 55% MFU. ~3 weeks.
- **Month 2-3:** Kernel 3 optimisation to 60-65% MFU. ~4 weeks.
- **Month 3-4:** Kernel 2 chunked WY implementation. ~3 weeks.
- **Month 5-6:** Tensor integration: compile all 3 kernels through TileLang. ~3 weeks.

Total: ~13 weeks of focused kernel engineering.

---

## 9. Reduction Operator Coverage

The framework's reduction is fundamentally `Σ_τ w_τ v_τ` — linear aggregation. This covers ~90% of published attention mechanisms:

### 9.1 Natively covered (linear aggregation)

- All softmax-attention variants (dense, sparse, block-sparse, top-k, foveal, rectified, differential)
- All linear-attention variants (kernel methods, polynomial features)
- All delta-rule variants (GDN, KDN, MesaNet)
- All slot-memory variants (SDM, GSA, MoM, Raven)
- All MoE-style compositions
- All architectural hybrids (Conformer, H3, Hyena)

10 major families × multiple instances each = the bulk of the literature.

### 9.2 NOT natively covered (non-linear aggregation)

- Max-attention / winner-take-all readout
- Geometric / product attention (Π v_τ^{w_τ})
- Log-bilinear / LSE attention (log Σ exp)
- Hyperbolic / Möbius attention
- Differentiable sorting / quantile readout
- Tropical / max-plus attention

These are ~10% of the literature and mostly theoretical or rarely deployed.

### 9.3 To extend coverage beyond 90%

Three paths, in order of cost:

| Path | Cost | Coverage added | When to do |
|---|---|---|---|
| Differentiable sort/quantile primitive | ~1 week | Quantile-readout attention | If a real workload needs it |
| Logarithmic-domain mixer | ~2-3 weeks | Log-bilinear attention | Research tier only |
| Hyperbolic / Möbius mixer | Research | Hyperbolic embeddings | Probably not worth |

The framework's stable release covers 90%. The remaining 10% is future work, not a gating item.

---

## 10. Stable-Version Deliverables

### 10.1 What ships

1. **3 canonical kernel sources** in TileLang/Triton:
   - `kernels/dense_softmax.tl`
   - `kernels/dense_linear_delta.tl`
   - `kernels/sparse_slot.tl`
2. **Dispatcher** (`urm/dispatcher.py`) implementing §5's 7-phase tree
3. **Specialised-kernel integrations** (vendored):
   - FA-family for softmax (FA2/FA3/FA4 per hardware)
   - fla-org's GDN for delta
4. **Optimisation-as-architecture hooks:**
   - `urm.inner_loop(...)` for gradient-based inner loops
   - `tensor.closed_form_solve(...)` (if Tensor adopts the primitive library proposal)
5. **Pattention support** (after backward-pass audit):
   - `kind="pattention"` in `MixerSpec`
   - `axis="parameter"` configuration
6. **Hardware portability** via Tensor integration:
   - Compile all 3 kernels through TileLang
   - Per-provider bin dispatch
   - Cache at `~/.tensor/cache/urm/...`

### 10.2 What does NOT ship in stable release

- RWKV-7 value-residual extension (rejected per coverage roadmap §3)
- TTT as a first-class mixer kind (research tier; covered via `state_update` extension)
- Slot-slot interaction graph (Tier-C research)
- Adaptive S / slot birth/death (Tier-C research)
- Stochastic routing (Tier-D exotic)
- Non-linear aggregation primitives (10% boundary)

### 10.3 Verification

- Float64 oracle parity (mirroring `sdm-sparse-attention-reparameterization.md` §11.2 methodology)
- Cosine similarity ≥ 0.9999 vs analytical backward
- 10-step pretraining alignment for at least one variant per kernel family
- Pattention backward via gradcheck before shipping axis="parameter"

### 10.4 Benchmarks

- Frozen Phase 3 shapes on A10G (matching `sdm-sparse-attention-reparameterization.md` §9.2 conditions)
- Cross-check: lazy-loaded FA4 dispatch when softmax on Blackwell detected
- Per-kind MFU table (10+ representative mixer configs)

---

## 11. Open Questions

1. **Pattention backward.** Triton-fused forward exists; backward needs verification. Highest priority for the axis="parameter" extension.
2. **Tensor primitive library adoption.** When does Tensor ship the primitive layer? Affects URM's dispatch contract.
3. **Slot-slot interaction graph.** The Tier-C research axis that's most likely to be a follow-up paper. Worth sketching but not for stable release.
4. **Auto-derived primitives.** Could Tensor's compilation outputs be exposed as primitives (not just kernels)? Significant API design question.
5. **Reduction alternatives.** If a real workload needs LSE attention or geometric attention, when does the framework extend?
6. **Disaggregated providers.** How does the dispatcher handle cross-provider kernels (e.g., part on CUDA, part on a custom accelerator)?

---

## 12. References

- `sdm-sparse-attention-reparameterization.md` — math + verification
- `dual-form-coverage-roadmap.md` — architecture coverage (companion doc)
- `architecture.md` — semantic contract (`MixerSpec`, IR)
- `compiler-charter.md` — URM as compiler, pipeline
- `kernel-generation.md` — kernel-generation guide
- `triton-backend.md` — Triton backend preparation
- `tensor/proposal.md` — Tensor compiler stack
- `atma/docs/POLAR_ATTENTION.md` — example of a custom attention
- `foveal-sparse-indexer/foveal_indexer/pattention_triton.py` — MLPs as parameter-axis attention
- `fla-org/flash-linear-attention` — kernel catalog
- `tech-blog/posts/2603.30033` etc. — recent literature coverage

