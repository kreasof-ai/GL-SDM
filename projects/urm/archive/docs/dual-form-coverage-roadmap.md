# Dual-Form Framework: Coverage, Enumeration, and Future Axes

**Status:** Design roadmap companion to `sdm-sparse-attention-reparameterization.md`. Sweeps existing architectures against the framework's §10.3 unification table, identifies what fits and what doesn't, characterises the kind-space of admissible mixers, lays out the hardware-portability strategy, and proposes fresh axes the framework's structure makes possible.

**Companion doc:** `sdm-sparse-attention-reparameterization.md` (the math and the empirical verification). This doc assumes that math is in scope and focuses on coverage and forward direction.

---

## 1. Executive Summary

The dual-form framework claims a "unified sequence mixer" anchored on four rows: dense Transformer attention (A=0, dense softmax Ω), Foveal Sparse Attention (A=0, block-sparse Ω), Gated DeltaNet / Titans (dense A, dense Ω), and SDM (sparse A, sparse Ω). Three follow-up questions drive this doc:

1. **Coverage.** Out of ~35 kernels in `fla-org/flash-linear-attention`, which fit under one of these rows (or a trivial extension), and which are structurally out?
2. **Yield.** The framework's grammar admits a finite *kind-space* of mixer families; only ~35 have been implemented. What is the unexplored surface, and what does enumerating it buy?
3. **Hardware.** Modern Blackwell-tuned kernels (FlashAttention 4) win on hardware-specific features (TMA, async barriers, warp-group MMA). Does the framework's generalisation impose a permanent tax, and if so, where does the framework still win?

**Headline conclusions:**

- **~30 of ~35 fla kernels fit** under one of the §10.3 rows or a 1-line extension. The boundary cases (RWKV-7, TTT, CAT) have structural reasons; the framework should not chase them.
- **Mamba-1 and Mamba-2 both fit Tier 4** (linear-attention family) as degenerate cases — earlier classification of Mamba-2 as "not covered" was an overstatement. Mamba-2's SSD block-diagonal causal mask is a positional mask on Ω_read, not a structural break.
- **The framework's kind-space is finite but mostly unexplored.** Roughly 1k–10k admissible mixer families vs. 35 implemented. Systematic enumeration is tractable; the question is *yield*, not feasibility.
- **Hardware portability is at the interface layer, not the kernel layer.** The framework's value is the `MixerSpec` API and dispatcher; specialized kernels (FA4, fla-org's GDN/NSA/etc.) plug in underneath. The generic dual-form kernel is the fallback for exotic configs and new hardware.
- **Three new axes are 1-line extensions** (slot-channel, slot-locality, coordinated routing) and one is a new primitive (recursive chunking). These give the stable-version showcase material that is genuinely not in the literature.

---

## 2. Coverage of Existing Architectures

Sweep of `fla-org/flash-linear-attention` (`fla/ops/` + `fla/layers/`, ~35 distinct kernels) against the framework's four rows plus extensions.

### 2.1 Tier 1 — covered as-is (named in §10.3)

| fla kernel | Dual-form row |
|---|---|
| `attn` (FlashAttention path) | Dense Transformer Attention (A=0, dense softmax Ω) |
| `gated_delta_rule` (GDN), `gdn2` (GDN-2) | Gated DeltaNet / Titans (dense A) |
| `delta_rule` (vanilla DeltaNet backend) | Gated DeltaNet / Titans |
| `titans` | Gated DeltaNet / Titans |

### 2.2 Tier 2 — trivial extension of one row

These are softmax-attention variants where the change is on the indexer side or the read mask, not the recurrence structure:

| fla kernel | Maps onto | Note |
|---|---|---|
| `abc` | Dense softmax, A=0 | Bounded-memory cutoff on Ω; no recurrence change |
| `mla` | Dense softmax, A=0 | Compressed KV is upstream of Ω_read |
| `nsa` | Block-sparse softmax, A=0 | Top-p block selection on K |
| `moba` | Block-sparse softmax, A=0 | Same row as NSA |
| `forgetting_attn` (FoX) | Dense softmax, A=0 | Element-wise forget gate on Ω |
| `log_linear_attn` | Dense softmax, A=0 | Forget mask `g_τ = exp(Σ g_j)`; identical decay-factorisation as SDM |
| `dsa` | Dense softmax, A=0 | DeepSeek-style attention |
| `deltaformer` | Hybrid composition | Softmax row + delta row composed; two `MixerSpec` calls |
| `gsa` | SDM sparse row | Slot-based routing, sparse A + sparse Ω |
| `mom` | SDM sparse row (mixture of experts) | Each memory is its own sparse A/Ω |
| `raven` | SDM sparse row | Sparse memory routing |

### 2.3 Tier 3 — dense-A row (Gated DeltaNet / Titans slot)

All reduce to `S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T`, algebraically identical to the §13.2 WY representation:

`delta_net`, `gated_delta_rule`, `gdn2`, `kda`, `gated_delta_product`, `momentum_delta_rule`, `generalized_delta_rule`, `gated_oja_rule`, `precond_gated_delta_rule` (PGDN), `precond_kda` (PKDA), `rodimus`, `comba`, `mesa_net`, `titans`.

The framework's `Δ = D_β(V − V⁰)` triangular solve is exactly the WY representation these kernels compute. The "preconditioned" variants put a non-identity `P` left-multiplying `(I + D_β A)` — a row-scale extension.

### 2.4 Tier 4 — linear-attention family, A=0 with non-softmax Ω

No softmax, no write collision (A=0 because writes don't overwrite):

`linear_attn`, `lightning_attn`, `simple_gla`, `retention` (RetNet kernel), `gla`, `based`, `rebased`, `lightnet`, `hgrn`, `hgrn2`, `rwkv4`, `rwkv6`.

These are dense-A=identity variants. `Ω_read = ϕ(q)·ϕ(k)^T` (no softmax), `A = diag(g)` with `g` per-token decay. Triangular solve degenerates to element-wise scaling.

**Mamba-1 and Mamba-2 belong here too.** Earlier classification (in earlier drafts of this analysis) placed Mamba-2 in Tier 7, but this was incorrect: Mamba-1's diagonal SSM is structurally `A = 0` with element-wise `Ω_read = identity` per channel; Mamba-2's SSD duality shows it is `A = 0` with `Ω_read` carrying a block-diagonal causal positional mask. Both fit Tier 4.

**Mamba-3** is the same family as Mamba-2 (multi-head SSM, structured diagonal A); it fits Tier 4 with the same caveat — needs verification that the multi-head structure doesn't break `A = 0`.

### 2.5 Tier 5 — hybrid compositions

These are not new primitives but compositions of Tier 1–4 plus FFN/state-space pieces:

| fla kernel | Composed of |
|---|---|
| `yoco` | global FFN cache + local softmax; not a single mixer |
| `samba` | Mamba layer + attention layer; both Tier-1 |
| `attnres` | attention with residual; not a kernel |

### 2.6 Tier 6 — covered with one small framework extension

| fla kernel | Missing piece | Cost |
|---|---|---|
| `wall_attn` | Per-head diagonal gate on A (length-generalizing) | 1 column on D_β |
| `parallax` | Windowed A (intra-window writes collide) | Mask in triangular solve |
| `path_attn` (PaTH) | Position-rotation on Ω_read | Absorbed into q/k projections |
| `rwkv6`, `rwkv4` | None — fit Tier 4 | Zero |

### 2.7 Tier 7 — NOT covered by the dual-form framework

These have a structural mismatch with the framework's `(A, Ω_read, β, decay, M_T)` contract:

| fla kernel | Why it doesn't fit |
|---|---|
| **`rwkv7`** | Value residual `v'_t = v_t − S_{t-1}·k_t` requires sequential state knowledge in the RHS, breaking the parallel WY form. Input-dependent `δ_t = sigmoid(r'_t·k_t)` couples reads into writes, violating the routing-decoupling invariant. See §3. |
| **`ttt`** (Test-Time Training) | Inner-loop optimisation modifies `A` itself per step via gradient descent. Not representable in a single mixer call — meta-mixer. |
| **`cat`** (Compress-and-Attend) | Variable compression ratio breaks the `[P, T, D]` activation contract. |
| **`deltaformer`** (as single primitive) | If composed of two dual-form calls (softmax row + delta row), fits; as a unified primitive, doesn't. |

**Mamba-3 caveat:** Should be verified. If it remains in the SSM family with structured diagonal A, it fits Tier 4 with a positional mask. If Mamba-3 introduces state-shape changes that break `A = 0`, it moves to Tier 7.

---

## 3. Why GDN-2 Fits but RWKV-7 Doesn't

GDN-2 is widely cited as a degenerate case of RWKV-7. The question is: what specifically does GDN-2 *remove* such that it fits inside the dual-form's dense-A row?

### 3.1 What GDN-2 has (fits dense-A row)

| GDN-2 piece | Dual-form role |
|---|---|
| State `S_t ∈ R^{d×d}` per head | Matrix-valued slot table (dense-A row, same as WY rep) |
| Scalar `β_t` per token | `D_β` diagonal in the triangular system |
| Static per-channel decay `w` | Folded into initial-state projection `V⁽⁰⁾` |
| `S_t = S_{t-1}(I − β_t k_t k_t^T) + β_t v_t k_t^T` | Standard delta; dense `A = I + D_β K K^T`, parallel triangular solve |
| Read `y_t = S_t q_t` | `Ω_read = identity`, degenerate cross-attention |

All four pieces match the framework's contract: **writes depend only on `(K, V)`, reads are a separate pass over the state**.

### 3.2 What RWKV-7 turns on

1. **Value residual** — `v'_t = v_t − S_{t-1}·k_t` instead of raw `v_t`. The write becomes `S_t = S_{t-1}·diag(w_t) + δ_t·v'_t·k_t^T`. The RHS now contains `S_{t-1}`, which depends on `Δ[:t−1]` — what you're solving for. The WY representation no longer has a closed-form parallel form.

2. **Input-dependent `δ_t = sigmoid(r'_t·k_t)`** — the learning rate is a function of the read tensor `R`. The write-collision matrix becomes `A = I + D_δ(R,K)·K K^T` — `A` is no longer determined by the indexer's write side, it's a function of the read weights too. Violates the routing-decoupling invariant.

3. **Time-varying decay `w_t`** — `diag(w_t)` is now per-token and input-dependent. The "precomputed" `A` becomes a function of the input.

4. **Vector `δ_t` (per-channel)** — extension of `D_β` to per-channel. Structurally absorbable as a lift from `D_β ∈ R^{T×T}` to `D_δ ∈ R^{T×T×C}`. Doesn't break the contract on its own.

### 3.3 What specifically breaks the dense-A row contract

- **(1) breaks the parallel solve.** Value residual makes the system a coupled recurrence without clean WY form.
- **(2) breaks routing decoupling.** `A` becomes a function of the read projection; writes depend on reads.
- **(3) breaks A-as-static-input.** `A` is no longer determined at mixer-call time from the indexer's outputs.

**GDN-2 is RWKV-7 with (1), (2), (3) switched off.** Per-channel `δ` (4) is the cheap extension; the other three are the structural breaks.

### 3.4 Recommendation

Don't extend the framework to cover RWKV-7. The framework's win is the WY representation; RWKV-7's value residual loses it for the write-collision computation. The architectural decoupling invariant (routing ↔ execution) is a strength, not a limitation. Extending toward RWKV-7 is extending against the grain of where the field is moving (length-extrapolation work like Polar, KDA's softmax-priors, MesaNet's TTT-as-attention).

For RWKV-7 users, compose: run fla-org's RWKV-7 kernel side-by-side with URM's mixer. Both are at the same MFU level; the framework's "architecture-independent execution" invariant is preserved by *not* trying to subsume RWKV-7.

**Flip to "yes, extend" only if:**
- RWKV-7 becomes the production target.
- The framework needs to credibly claim coverage of every architecture in fla for adoption reasons.
- A user demonstrates a hybrid model that mixes RWKV-7 with SDM/softmax in a way that the dual-form + fla side-by-side handles poorly.

---

## 4. The Kind-Space of Admissible Mixers

A dual-form mixer is a tuple along five axes:

| Axis | Possible values |
|---|---|
| Indexer | dense identity · dense softmax top-k · sparse top-p · block-sparse contiguous · product-key codebook · 16D foveal · per-head routing · per-token routing |
| Write structure `A` | zero · dense `KK^T` · sparse slot-overlap · block-windowed · diagonal-gated · low-rank · input-dependent |
| Read structure `Ω_read` | dense softmax · block-sparse softmax · identity · element-wise · block-diagonal causal · with null sink · with input-dep temperature |
| β, decay schedules | scalar · per-head · per-channel · input-dep · static · learned once |
| Boundary state `M_T` | discarded · persistent slot table · folded into KV · rank-R matrix · neural-net state |

With ~10 choices per axis, the **kind-space** is roughly `10^5 ≈ 100k` mixer families. The literature has implemented ~35 of them. The gap between "implemented" and "admissible" is real.

### 4.1 Why enumeration is tractable

- **Kinds are finite; instances are infinite.** Continuous parameters give infinitely many instances within each kind, but the kinds themselves are countable.
- **Coherence filters reduce drastically.** Filters like "A non-zero only with write-side state", "Ω_read row-bounded", "β in `[0,1]` when interacting with state fold", "boundary state finite-rank" bring 100k kinds down to 1k–3k plausible.
- **Same kernel, different config.** Each kind is a `(A, Ω, β, decay, boundary)` tuple applied to the existing `DualFormSDMFunction`. No new kernel code — the implementation cost is the configuration, not a new autograd function.

### 4.2 The interesting unexplored combinations

A few that are not in fla or ATMA today and would test novel hypotheses:

1. **Sparse-write × dense-softmax-read.** Top-W slot writes, but reads use full softmax `QK^T` over K·V. Inverts NSA's block-sparse read with dense compression. Tests write-sparsity / read-expressivity asymmetry.

2. **Sparse-write × linear-read** (no softmax, no delta). Top-W slots get `w_t v_t`; reads are `q_t M_t k_t`-style without normalisation. Doesn't exist because fla pairs linear with linear.

3. **Slot-table with value residual.** RWKV-7's `v'_t = v_t − M_{t-1}·k_t` semantics on a sparse `S × D` slot matrix. Small structural extension; tests whether value-residual helps slot memories.

4. **Block-windowed delta-A.** Parallax-style local delta on a slot table. Combines with full softmax reads for "local-write, global-read".

5. **Per-channel β with sparse routing.** Each token picks W slots but write rate varies per channel. Tests fine-grained gating on top of coarse routing.

6. **Input-dependent β from a null-sink scoring rule.** Polar's `temp_i = 1 + softplus(len_gain)·log n` generalised to slot routing — adaptive top-p that sharpens as the slot table fills.

### 4.3 The yield question

Open research question: of the ~1k–3k plausible kinds, what fraction turn out to be Pareto-good (better loss at comparable or better MFU than the closest existing baseline)?

If the yield is 1–5%, that's 10–150 novel mixer families worth benchmarking. The framework's value is in making this enumeration essentially free — the same kernel serves each kind via configuration.

---

## 5. Hardware Portability and the Dispatch Layer

Modern Blackwell-tuned kernels (FlashAttention 4) exploit hardware features that a generalised kernel cannot:

- **TMA** (Tensor Memory Accelerator) for async global → shared memory loads.
- **`mbarrier`** async barriers for cross-warp synchronisation without busy-wait.
- **Warp-group MMA** (different from SM80/SM89 MMA).
- **Distributed shared memory** across clusters.
- **Producer/consumer warp specialization** — half the warps issue loads, half compute.

Every one of these is specific to a single data-movement pattern. A kernel parameterised over (sparse/dense × softmax/linear/delta × input-dep/static × …) can't specialise warps because all the patterns are present at the JIT-compile boundary. The compiler falls back to lowest-common-denominator codegen.

**The framework can avoid this tax by separating interface from kernel implementation.**

### 5.1 The architecture

```
┌─────────────────────────────────────────────┐
│  MixerSpec API (architecture-agnostic)      │   ← URM's contribution
│  DualFormSDMFunction / chunked_dual_form_sdm│
├─────────────────────────────────────────────┤
│  Dispatch (arch × hardware → kernel)        │   ← cache after first call
├─────────────────────────────────────────────┤
│  Kernel library                             │
│   ├── softmax → FA4 (Blackwell)             │
│   ├── softmax → FA3 (Hopper)                │
│   ├── softmax → FA2 (Ampere)                │
│   ├── delta-rule → fla GDN kernel (any)     │
│   ├── sparse → fla NSA / MoBA kernel        │
│   ├── linear → fla GLA / RetNet kernel      │
│   └── generic → chunked dual-form (fallback)│   ← what exists today
└─────────────────────────────────────────────┘
```

The `DualFormSDMFunction` is the *fallback* — runs when no specialised kernel exists. It is the research path and the exotic-config path, not the production path for `mixer_kind=softmax_attention`.

### 5.2 What this means for the stable version

The stable release should ship:

1. **Interface layer:** `MixerSpec`, `DualFormSDMFunction`, chunked variant, autograd contract — version-pinned.
2. **Dispatch layer:** runtime selection by `(architecture, hardware, head_dims)`. Cached after first call.
3. **Kernel library:**
   - Hand-tuned Blackwell path for softmax (FA4 or a clean reimplementation that uses TMA + warp specialisation).
   - fla-org integrations for delta / linear / sparse — vendored as backends.
   - The generic dual-form kernel as the fallback for exotic configs and new hardware.
4. **Documentation:** explicit "for production softmax on Blackwell, use kernel X; for sparse routing research, use the generic kernel."

### 5.3 When the generic kernel is still the right call

- **Exotic combinations** that no specialised kernel exists for (sparse-write × dense-softmax-read, slot-table with value residual, etc.).
- **New hardware where specialised kernels haven't been written yet.** Mid-cycle hardware (B300, Rubin) often lacks specialised softmax kernels for 6–12 months. The generic kernel is the hedge.
- **Mixed-precision edge cases** that specialised kernels don't cover (block-quantised K/V with FP32 accumulation).
- **Research iteration speed.** Changing β schedule or `A` pattern should not require writing a new kernel.

### 5.4 The honest scope

The framework saves you from re-doing the *model code* each hardware generation, not the *kernel integration*. When Hopper launched, FA3 had to be integrated; when Blackwell launched, FA4 had to be integrated; when TMA changes, the integration updates. That's ongoing work, but it's a known, bounded surface.

---

## 6. Fresh Axes from the Framework's Structure

The framework's primitives admit new axes that the literature hasn't sampled. Sorted by cost to implement.

### 6.1 Tier A — 1-line framework extensions (stable-release candidates)

**1. Slot-channel axes.** Promote memory from `S × D` to `S × K × D` — a channel index inside each slot. The indexer selects `(slot, channel)` pairs.

Unlocks:
- Per-channel β (one decay schedule per channel) — RWKV-7's vector `δ` done properly.
- Channel-aware sparsity — different channels with different routing patterns.
- Mixture-of-channel-reads — query reads a weighted combination across channels, not just slots.

This is what RWKV-7's vector `δ` should have been. One extra index, no new math.

**2. Slot-locality (metric-based routing).** Add `slot_positions: R^{S × d_pos}` and a distance function. The indexer becomes "select the W slots nearest to `q_t`."

Unlocks:
- k-NN style memory (retrieval-augmented flavour, but as a mixer).
- LSH-based routing for hardware efficiency.
- Compositional generalisation — slots at test time can be added without retraining the router.

No existing fla kernel does this as a mixer primitive. Closest is product-key SDM but without explicit geometry.

**3. Coordinated routing.** The indexer output becomes a function of all heads' assignments, not per-head.

Two flavours:
- **Coverage constraint**: heads must collectively cover the slot space (no two heads write to the same slot). Forces diversity.
- **Exclusive routing**: winner-take-all per slot (slot can only be written by one head).

MoM is close (experts are exclusive), but per-slot coordination at the head level hasn't been done. The framework just needs a constraint layer on the indexer output.

### 6.2 Tier B — new primitive, moderate cost

**4. Recursive chunking.** Promote the chunked dual-form from flat to hierarchy: chunks-of-chunks-of-chunks, each level with its own `(A, Ω_read, β)`.

Unlocks:
- Hierarchical attention (Hourglass / HMMT but with slot tables at each level).
- Variable cost per level — cheap at coarse, expensive at fine.
- Length-extrapolation because routing happens at multiple granularities.

Existing architectures do single-level chunking (NSA, MoBA) or two-level (Longformer) but not recursive.

**5. Time-scale decomposition.** Within a single mixer call, maintain N parallel memories with different decay rates `g_1 > g_2 > ... > g_N`. Output is a learned combination.

Unlocks:
- Short-term + long-term memory in one mixer.
- Fourier-style decomposition of the temporal signal.
- The "polarisation" idea (decay over multiple scales instead of one).

### 6.3 Tier C — architectural shifts (research, not stable release)

**6. Slot-slot interaction graph.** Slot table gets an adjacency structure — slots are nodes, edges allow information flow between slots independent of token routing. The framework's `A` (token-token collision) becomes augmented by `G` (slot-slot graph).

The slot memory itself becomes recurrent. Closest analog is graph attention networks, but as a mixer primitive rather than a structural prior. Strongest "next-next-axis" candidate — would let the framework subsume graph-attention architectures.

**7. Adaptive S (slot birth/death).** Slot count varies per sequence or per chunk. New slots created when routing becomes competitive; slots destroyed when unused for K steps.

CAT does this at the token level (compress). Doing it at the slot level lets the framework adapt memory capacity to sequence complexity without retraining.

**8. Read-coupled write (RWKV-7 style).** Already covered in §3. Costs the parallel-solve invariant.

### 6.4 Tier D — exotic, low-yield

**9. Complex/quaternion-valued slots.** Slot contents have phase + magnitude. Math generalises cleanly; the literature on complex-valued SSMs hasn't shown clear wins.

**10. Stochastic routing with REINFORCE-style gradient.** Sample slot indices from a learned distribution. Probably too unstable for production.

### 6.5 Stable-version recommendation

Take **two from Tier A** (slot-channel + slot-locality) and **one from Tier B** (recursive chunking or time-scale decomposition). Three fresh axes that:

- Are not in fla or ATMA today.
- Cost little to implement (the framework's primitives absorb them).
- Each opens a real new design space.
- Each gives a paper-able result when paired with the right benchmark.

The other tiers are research directions to flag in the doc but not implement now. Slot-slot interaction graph (Tier C) is the strongest "next-next-axis" — flag for a future doc.

---

## 7. Stable-Version Deliverables

Concrete artefacts for the showcase release:

### 7.1 Tier-A axes (this release)

1. **`urm/mixer/slot_channel.py`** — slot-channel memory, per-channel β, per-channel routing. Implementation: ~200 lines.
2. **`urm/mixer/slot_locality.py`** — metric-based routing, k-NN + LSH backends. ~250 lines.
3. **`urm/mixer/recursive_chunking.py`** — multi-level chunked dual-form. ~300 lines.

### 7.2 Verification

- Float64 oracle parity for each new axis (matching the §5 / §11 verification methodology).
- Cosine similarity ≥ 0.9999 vs analytical backward.
- 10-step pretraining-step alignment for at least one Tier-A variant per §9.5 pattern.

### 7.3 Benchmarks

- Frozen Phase 3 shapes on NVIDIA A10G (matching §9.2 conditions).
- Cross-check: lazy-loaded FA4 dispatch when softmax on Blackwell is detected.

### 7.4 Documentation

- One doc per Tier-A axis: math formulation, verified equivalence, benchmark.
- This coverage roadmap as the umbrella design doc.

### 7.5 What NOT to ship in this release

- RWKV-7 value-residual extension (rejected per §3).
- Slot-slot interaction graph (Tier C — research).
- Adaptive S (Tier C — research).
- Stochastic routing (Tier D — exotic).

---

## 8. Open Questions

1. **Mamba-3 verification.** Does the multi-head SSM structure break `A = 0`? Needs reading the paper's recurrence form closely.
2. **PaTH (path_attn).** Is the position-rotation truly absorbable into q/k projections, or does it require a new primitive?
3. **Yield estimate for kind-space enumeration.** Pilot enumeration of ~50 plausible kinds to estimate the Pareto-good fraction. Would inform whether the framework's design-space win is real or hypothetical.
4. **Dispatch-layer interface design.** Concrete API between `MixerSpec` and pluggable backend kernels (FA4, fla-org). This is the integration contract that determines how easily new specialised kernels can be plugged in.
5. **Slot-slot interaction graph primitive.** If pursued, what's the right form? `A` tensor of shape `[T, T, S, S]` (token-token × slot-slot) is one option; a slot-graph attention is another.

---

## 9. References

- `sdm-sparse-attention-reparameterization.md` — the math + verification this roadmap extends.
- `compiler-charter.md` — URM Invariants 1, 2, 7 (§10.1 references these).
- `fla-gated-delta-rule.md` — prior coverage work on FLA's GDN.
- `fla-org/flash-linear-attention` — the kernel catalog swept in §2.
- `atma/docs/POLAR_ATTENTION.md` — example of a custom attention with magnitude channel; analysed in the prior chat thread.
- `lingua/sparse_delta_memory/memory_ops.py` (Meta SDM, pinned `183e7df`) — the §13 alignment reference.

