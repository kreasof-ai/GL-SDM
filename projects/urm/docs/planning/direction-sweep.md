# Verified direction sweep — 76 mixer-relevant architectures

**Status:** verified against the pinned upstream sources at `/tmp/urm-comparator-pins/` (see `benchmarks/provision_comparators.py`) on 2026-09-24, at HEAD `0d83495`. This document is the *verified decision map* that operationalizes the [architecture-composition ledger](architecture-composition.md). Every row was read from pinned source with file:line evidence; the machine-readable per-row records are preserved under `results/validation/direction-sweep/row-*.json`. **This is still a construction target, not executable coverage** — "direction" is the *planned* route for each architecture, gated by the completion order in the [roadmap](roadmap.md).

## The decision rule (refined by the sweep)

Each row is assigned exactly one direction:

- **`composition-now`** — the mixer equation *is exactly* one of the closed contracts (U1.S softmax attention; U2.A/U2.D canonical linear-delta; U3.D sparse-delta). Only an external modeling module + recipe is needed; **zero new core semantics**. Head maps (MHA/MQA/GQA), latent/factorized projections, feature maps, masks, depth/parameter source domains, and external FFT convolutions are all external `E[...]` operators.
- **`composition-k2-gated`** — the mixer equation is exactly U2.A or U2.D, blocked only on `B[K2_graph_execution]` (the K2 graph binder does not yet exist). No new core semantics beyond Gate 1.
- **`combinator`** — a *family-generic* typed graph structure that takes mixer calls as arguments: differential pairing `Diff(F,G,λ)=F−λG`, higher-order nesting `Nest(F,G)`, or routed-expert composition (MoE-shaped). Admitted as IR/compiler graph structure, **never** as a provider/kernel branch. The first plan is unfused multiple calls + typed merge; fusion is a later proven rewrite.
- **`axis`** — the equation differs from the closed contracts by a *closed typed property* that is reusable (multi-client now, or hypothetically composable). Descriptor + independent NumPy/Torch references **first**; a native physical branch only after a two-client admission record and a measured regime.
- **`external-only`** — the semantics do not transfer to any other (even hypothetical) architecture, or would tax GPU performance on existing paths. Kept as external comparator/reference; never enters core.

## Verified direction for each of the 76 rows

### `composition-now` (16 rows) — zero new core semantics

| ID | Architecture | Verified note |
|---|---|---|
| 001 | MHA | U1.S equal-head; head-map is a descriptor field |
| 002 | MQA | U1.S all→one KV head; head-map field |
| 003 | GQA | U1.S explicit grouped head map |
| 004 | MLA | U1.S over externally expanded latent heads; compressed-cache equality is a recorded blocker |
| 014 | BitAttention | BitLinear projection external; mixer is plain U1.S |
| 047 | SDM | Router matches `R.PK` in structure **but** tie-policy is backend-dependent `torch.topk` (not HIGHEST_ADDRESS) and route weights use a scaled softmax — `composition-now` **with that recorded caveat** |
| 053 | Samba | attention/retention/GLA configs only; `mamba_swa_mlp`/`use_mamba` gated on the Mamba axis (A8) |
| 054 | AttnRes | U1.S over the **depth** domain (not sequence); external residual lifetime |
| 057 | Pattention | U1.S over the **parameter** domain; normalizer is a parameter-domain nonlinear variant, not vanilla softmax |
| 066 | CAT | compressed ∪ local block attention; architectural token identity preserved |
| 069 | TPA | factorized QKV external; mixer is plain U1.S |
| 070 | Tucker | factorized QKV external; mixer is plain U1.S |
| 075 | Conformer | U1.S noncausal + external relative-position bias |
| 076 | H3 | external causal FFT convs; **does not count** as native three-kernel mixer coverage |
| 077 | Hyena | external implicit filter + FFT conv; no compact U2 realization established |
| 078 | Hopfield | iterated one-step U1.S association; external query-refresh edge |

### `composition-k2-gated` (14 rows) — blocked only on `B[K2_graph_execution]`

| ID | Architecture | Verified note |
|---|---|---|
| 015 | Linear attention | U2.A numerator `||` U2.A denominator + external ε division |
| 016 | Lightning | U2.A + external layer-index **head** decay (confirmed head scope, not channel) |
| 017 | RetNet | U2.A + static per-head decay; **no data-dependent gate in the pinned source** — sweep disproved the A10 hypothesis |
| 018 | Simple GLA | U2.A + data-dependent head-scalar gate (supplied as operand, still head scope) |
| 019 | GLA | U2.A + channel-diagonal gate |
| 020 | Based | external Taylor-2 feature map + U2.A numerator ∥ denominator |
| 021 | ReBased | external squared-dot feature map + U2.A numerator ∥ denominator |
| 022 | LightNet | GLA channel-decay U2.A + external log-cumsum-exp key normalization |
| 024 | HGRN2 | GLA channel-decay U2.A; `state_v_first` is purely physical layout |
| 025 | DeltaNet | exact U2.D, c=1, β, G=I |
| 026 | Gated DeltaNet | U2.D + external head-scalar decay (−exp(A_log)·softplus) |
| 028 | KDA | U2.D + key-channel decay + explicit 1/√K read-scale field |
| 036 | Rodimus | U2.A + channel decay + source 1/√64 read scale (no `fla/ops/rodimus/`; mixer is `chunk_gla`) |
| 052 | YOCO | U2.A self-decoder + U1.S cross-decoder over shared KV (also inherits K2 gate-VJP concerns) |

### `combinator` (6 rows) — family-generic typed graph structure

| ID | Architecture | Verified note |
|---|---|---|
| 048 | ABC | two additive U2.A (channel-decayed) calls + external interstage slot softmax |
| 049 | GSA | same shape as ABC; difference is gate provenance (logsigmoid/8, s=1−exp(g)) |
| 050 | Raven | external top-k router injected as dense gates into the 049 call graph |
| 051 | MoM | external softmax top-k router + pack/unpack around per-memory U2.D + external scatter-add merge (MoE-shaped) |
| 067 | Differential | flagship **Diff**: two U1.S calls + weighted difference (λ); V2 paired-head fusion is a later proven rewrite |
| 074 | HLA | flagship **Nest**: `z,s1=U2.A(q,k,k)` then `y,s2=U2.A(q,z,v)`; masked second-order unnormalized causal case only |

### `axis` (32 rows) — closed typed property, descriptor + references first

| ID | Architecture | Axis |
|---|---|---|
| 005 | NSA | A2 indexed softmax traversal / route ABI (clients 006/007/071/072) |
| 006 | MoBA | A2 indexed softmax traversal / route ABI |
| 007 | DSA | A2 indexed softmax traversal / route ABI |
| 008 | FoX | A13 score/reducer algebra (cumulative-log-gate score; shared with 011) |
| 009 | Log-linear | A4 hierarchical chunks / level state (shared with 046) |
| 010 | PaTH | typed causal triangular transform + A13 cumulative-log-gate bias |
| 011 | Wall | A13 score/reducer algebra (channel-decay score modulation) |
| 012 | Parallax | A13 shared-softmax multi-statistic reducer |
| 013 | DeltaFormer | typed causal triangular transform (strict-past value correction) |
| 023 | HGRN | vector-state channel-gated transition (vector K2) — may collapse to external-only without a 2nd client |
| 027 | GDN2 | A8 distinct erase/write transition factors |
| 029 | Gated DeltaProduct | A8 ordered rank-R multi-delta |
| 030 | Momentum DeltaNet | A8 coupled multi-state (memory + momentum) |
| 031 | Gen. delta IPLR | A8 low-rank I+LRᵀ transition |
| 032 | Gen. delta DPLR | A8 low-rank D+LRᵀ transition |
| 033 | Gated Oja | A8 alternate rank-1 correction orientation |
| 037 | Comba | A8 dual-key delta |
| 040 | RWKV-4 | max-shifted normalized vector state (before-update read) |
| 041 | RWKV-6 | A1 slot/channel gate + before-update bonus-read timing |
| 042 | RWKV-7 | A8 read/write coupling (input-precomputable coefficients) |
| 043 | Mamba-1 | A8 read/write coupling + per-token diagonal gate scope |
| 044 | Mamba-2 / SSD | A8 semiseparable boundary-state transition |
| 045 | Mamba-3 | A9 complex/block-real state (rotary phase) + coupled four-state bundle |
| 046 | LogLinearMamba2 | A4 hierarchical chunks (dyadic level state) |
| 056 | FwPKM | inner-optimization (typed inner optimizer state/update; write law does not map to K3) |
| 063 | BDH | read-timing axis (strict-past additive read-before-write) |
| 064 | POLAR | A13 polar direction/count reducer |
| 065 | Foveal | A13 polar reducer (shared with 064) + A2 locality routing |
| 068 | TDA | A13 threshold-ReLU-power reducer + Diff combinator for the two-branch merge |
| 071 | Longformer | A2 indexed K1 schedule (local ∪ global) |
| 072 | Sparse Transformer | A2 indexed K1 schedule (exact blocks) |
| 073 | KATA | A13 positive squared-group-dot normalized reducer |

### `external-only` (8 rows) — never enters core

| ID | Architecture | Reason |
|---|---|---|
| 034 | PGDN | nonlinear ATK preconditioner; state-dependent write coefficients |
| 035 | PKDA | same ATK preconditioner over KDA gates |
| 038 | MesaNet | two covariance states + 30-iteration regularized per-token CG solve |
| 039 | Titans | architecture-specific inner-loss/optimizer loop |
| 055 | TTT | architecture-specific inner gradient-descent loop (TTT-Linear vs TTT-MLP) |
| 060 | RNN | closed nonlinear tanh vector transition; no affine scan |
| 061 | GRU | closed nonlinear gated vector transition; no affine scan |
| 062 | M2RNN | closed nonlinear matrix transition; no affine scan |

## Ledger corrections found by the sweep

These are verified against source and **have been applied** to `architecture-composition.md`:

1. **005 NSA** — the branch merge is a *gated weighted sum* (`addcmul` with `g_slc/g_cmp/g_swa`), not an unweighted `||`. Record `E[weighted merge]`.
2. **010 PaTH** — the final score also adds a FoX-style cumulative log-gate bias (`A + gc_i − gc_j`, `naive.py:94`); this is a dependency on the 008 A13 axis, not a silent fold.
3. **012 Parallax** — `p1` uses a single max shift over `s1` only; a two-pass U1.S fallback cannot legally share the max/denominator through the current U1.S contract. Record this.
4. **013 DeltaFormer** — strict softmax uses a clamped denominator (`clamp_min 1e-20`), so fully-masked row `t=0` yields all-zero probs; this edge policy must be in the descriptor.
5. **016 Lightning** — decay scope is **head** (per-head scalar `g_gamma`), not channel.
6. **017 RetNet** — decay is a *fixed* per-head constant and static causal mask, **not** data-dependent; no A10 axis needed.
7. **018 Simple GLA** — gate is data-dependent (logsigmoid of a projection) but still head scope (supplied as operand).
8. **047 SDM** — router matches `R.PK` structurally but tie-policy is backend-dependent `torch.topk` (not HIGHEST_ADDRESS) and route weights use a scaled softmax; record the gap.
9. **048 ABC / 049 GSA** — both stages are *additive* U2.A (channel-decayed), **not** U2.D; no retrieval-correction term appears.
10. **053 Samba** — the Mamba branch is the selective-SSM law (A8), not U2.A; `mamba_swa_mlp`/`use_mamba` are gated on the Mamba axis.
11. **057 Pattention** — the reducer is a parameter-domain nonlinear normalizer (exp+L1 / gelu_l2_norm), not vanilla softmax.
12. **063 BDH** — strict-past additive **read-before-write**; canonical U2.A is after-update, so BDH is a distinct read-timing descriptor.
13. **065 Foveal** — additionally depends on `B[indexed_K1_schedule]` for efficient sparse traversal (dense mask is a correctness oracle only).
14. **068 TDA** — the threshold-ReLU-power reducer is non-softmax; the A13 axis must be admitted before any native branch, and it is used inside a Diff-like two-branch merge.

## What the sweep proved about the overall strategy

- **30 rows are pure composition** (16 `composition-now` + 14 `composition-k2-gated`): they need *zero* new core semantics — only the K2 graph binder and honest external modeling modules. This is the coverage backbone.
- **6 rows are combinators** (Diff/Nest/routed-expert): admitting these two family-generic graph structures unlocks a combinatorial space (differential-GDN, differential-Mamba, higher-order attention, etc.) with *no* new kernels.
- **32 rows need a true axis**: each is a closed typed property with named clients. They enter as descriptor + references first; a native branch only after the two-client admission record and a measured regime.
- **8 rows stay external-only** (inner optimizers, nonlinear solves, closed nonlinear recurrences): they do not transfer, and forcing them in would re-create the kernel-branching mess.

The verified order of work is therefore: **Gate 0/1 foundation → composition sweep (30 rows, zero core branches) → combinator admission (Diff, Nest) → axis admission in two-client order (indexed-K1, Mamba selective-SSM, A13 score/reducer, A8 low-rank, A4 hierarchical)**.
