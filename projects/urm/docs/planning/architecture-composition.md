# Architecture composition construction contract

**Status:** construction target for the 76 mixer-relevant catalog IDs at HEAD `4cb35c5` (2026-09-25). The authoritative semantic and promotion rules are the [compiler charter](../compiler/compiler-charter.md#backend-branch-admission), [generality axes](../compiler/generality-axes.md), and [K1](../kernels/softmax-attention.md), [K2](../kernels/linear-delta.md), [K3](../kernels/sparse-delta.md) equations. Each ID resolves its pinned comparator revision, source path, supported modes and measured fragment scope in the [register](../../benchmarks/architecture-coverage.json). **A call graph below is a construction target, not executable coverage.**

**Current claim ceiling:** none of these 76 has a qualified complete source-model module on the current public graph path. Only 14 K1 and one K3 named fragments are live graph recipes, and no K2 graph call binds end to end. The retained historical fragment tests cannot be promoted into model coverage by this document. Rows with `B[...]` are unresolved for the named capability; even rows without an axis blocker still require the closeout record below. This ledger is an implementation decision map, **not** 76 independent source-equation proofs or a guarantee of near-native performance.

## Non-negotiable ownership rule

1. Every **architecture** has an external model module under `architectures/` (or a model-facing application outside `src/urm`). It owns layer order, parameters, ordinary operators, cache policy, and the public URM call sites. A JSON manifest may bind configuration and source provenance; JSON does not make a model-specific execution branch legal in core. The 15 current named graph fragments are external callable compositions, not 15 new core kernels. `src/urm/frontend/recipes.py` may load only name-agnostic typed fragments and must not register architecture names.
2. A public call enters core only through a **closed semantic descriptor** with role-indexed operands, logical domains, state bundle, read/write timing, effects, numerical policy, intended modes and gradients. The compiler must either emit a complete serialized executable region or decline. It may not infer semantics from a recipe name, Python tensor name, upstream callable, unknown transition string, or an ignored JSON field. Logical routes precede placement; physical addresses and communication are selected later.
3. `B[x]` below is a **blocker**, not an instruction to add a backend `if`. Resolve it in order: transcribe the pinned source equation and its state/cache effects; provide independent NumPy and Torch references; define a closed reusable descriptor and legal cross-axis combinations; show two structurally independent client graphs; prove any rewrite including VJP and precision envelope; then admit a physical branch only if dependency, memory, numerical or measured cost requires it. Until then keep the source-specific implementation as an external comparator. A descriptor/reference may exist before any native branch.
4. A complete source-model claim requires the external module, parameter and state mapping, all required modes, reference and upstream parity, and paired end-to-end cost. A kernel-fragment result has scope `fragment`, even if it uses a native URM call. Library/reference fallback is labeled explicitly and cannot count as native.

## Ledger notation and exact existing equations

`E[name]` is an operation or layer **outside** `src/urm` (projection, convolution, FFT, normalization, routing objective, cache allocation, MLP, etc.). `R.PK` is the current closed `SparseRouteSelectionSpec` pure product-key route operation: pairwise-additive factor score, top-k, softmax route weights, ascending canonical addresses and highest-address tie policy. A source router with different semantics is `E[route]` or `B[route descriptor]`, never silently `R.PK`. `U1`, `U2`, `U3` are calls through the **same public URM frontend** and provider contract; they are not names of source-specific kernels. `;` is an ordered data/state edge; `||` means branches whose results are explicitly merged by the external model or a separately admitted generic graph operation. `B[axis]` forbids compiling that axis as though it were already covered. An `E` stage may itself call a separately qualified ordinary operator; it is not a hidden arbitrary tensor callback inside URM IR.

Only these equation abbreviations are already written as URM-owned kernel contracts:

- `U1.S(q,k,v;map,scale,mask,bias)` means `softmax(scale*q*kᵀ+bias over visible keys)*v`, with explicit query-to-KV head map, all-masked behavior and position offsets. The *current native envelope* is narrower than this full contract. `U1.S[R]` means the identical equation restricted to exact route set `R`; it additionally requires `B[indexed_K1_schedule]` before **efficient** sparse coverage. A dense boolean mask may be a correctness oracle only.
- `U2.A(q,k,v;G,M0)` and `U2.D(q,k,v,β;G,M0)` are the additive (`c=0`) and delta (`c=1`) cases in [linear-delta.md](../kernels/linear-delta.md), with **after-update** read, explicit initial/final matrix state and VJP. Gate shape, normalization, denominator state, precision and read timing are not inferred. `U2` is **not yet executable through the current graph binder**; every U2 row has the common `B[K2_graph_execution]` prerequisite. No other K2 law is silently identified with A or D.
- `U3.D(routes,w,q,v,β,g;M0)` is the selected-slot decayed-delta law in [sparse-delta.md](../kernels/sparse-delta.md), with explicit route provenance, within-token uniqueness, ordered collisions, selected-slot decay, read timing, commit boundary, initial/final state and VJP. A different sparse write law is `B[K3_write_law]` until proved.

The graph expressions below are deliberately **partial when the source equation is not fully established**: `B[...]` is part of the row's required result, not a placeholder to guess from its architecture name. Shared blockers can be discharged once for multiple rows. A proposed `B` axis is an IR/reference candidate; it is **not** permission for a physical backend branch.

## 001–014: attention constructions

| ID | External call graph, in execution order | Mandatory blocker before full row claim |
|---|---|---|
| 001 MHA | `E[QKV/position] ; U1.S(equal-head,source mask) ; E[output/cache]` | `B[full_layer_and_cache_modes]` |
| 002 MQA | `E[QKV/position] ; U1.S(all query heads→one KV head) ; E[output/shared-KV cache]` | `B[shared-head gradients_and_decode]` |
| 003 GQA | `E[QKV/position] ; U1.S(explicit grouped head map) ; E[output/grouped-KV cache]` | `B[grouped-head gradients_and_decode]` |
| 004 MLA | `E[latent KV and position projections] ; U1.S(explicit latent-to-head view) ; E[output/latent cache]` | `B[prove expansion/cropping and compressed-cache equality]`; no expanded-cache performance claim |
| 005 NSA | `E[compress/index/branch merge] ; (U1.S[compressed] || U1.S[local] || U1.S[selected routes])` | `B[exact branch equations]`, `B[indexed_K1_schedule]`, `B[route-and-compression cost]` |
| 006 MoBA | `E[block scores/top-k] ; U1.S[local ∪ selected blocks] ; E[output]` | `B[block route ABI/ties]`, `B[indexed_K1_schedule]`; top-k cost included |
| 007 DSA | `E[indexer/objective/top-k token routes] ; U1.S[selected tokens] ; E[output]` | `B[token route ABI/selection gradient]`, `B[indexed_K1_schedule]` |
| 008 FoX | `E[QKV/forget gates] ; B[K1 cumulative-log-gate score] ; E[output/cache]` | No softmax-only or pairwise-bias materialization claim; exact gate VJP and stable tiled formula required |
| 009 Log-linear attention | `E[QKV/dyadic levels] ; B[K2 hierarchical level state/read] ; E[output/cache]` | Exact 64-token boundary and arbitrary partial-chunk state equations; no additive-K2 substitution |
| 010 PaTH | `E[QKV,w/β/g,short convolution] ; B[typed causal triangular transform] ; U1.S(transformed operands) ; E[output]` | Triangular operator VJP/traffic; no projection-only equivalence |
| 011 Wall | `E[QKV/per-channel gates] ; B[K1 channel-decay score] ; E[optional gate/sink/window/cache]` | Stable interval decay and optional branches; no scalar-bias approximation |
| 012 Parallax | `E[QKV/secondary query r] ; B[K1 shared-softmax multi-statistic reducer] ; E[correction/output]` | Prove `E[v]*(1+E[r·k])-E[(r·k)v]` and reducer VJP; repeated K1 passes are an explicit costed fallback |
| 013 DeltaFormer | `E[QKV/β] ; B[strict-causal triangular value correction] ; U1.S(corrected V) ; E[output]` | Preserve strict-past correction and gradients; no ordinary attention-only claim |
| 014 BitAttention | `E[BitLinear QKV and its gradient policy/RoPE] ; U1.S ; E[output/cache]` | Full quantized projection and cache parity; no BitLinear code in K1 |

## 015–046: state constructions

| ID | External call graph, in execution order | Mandatory blocker before full row claim |
|---|---|---|
| 015 Linear attention | `E[feature-map Q/K,V] ; U2.A(numerator) [|| U2.A(denominator if normalized)] ; E[epsilon division/output]` | Denominator state, epsilon, read timing and `B[K2_graph_execution]` |
| 016 Lightning | `E[projections/layer-index decay] ; U2.A(head decay) ; E[output]` | Exact decay schedule, state continuation, `B[K2_graph_execution]` |
| 017 RetNet | `E[projections/rotary/multiscale bank] ; parallel U2.A(decay per bank) ; E[bank combine/norm]` | Independent states and source combine, `B[K2_graph_execution]` |
| 018 Simple GLA | `E[QKV/head gate] ; U2.A(head decay) ; E[output]` | Gate VJP and low-precision chunk path, `B[K2_graph_execution]` |
| 019 GLA | `E[QKV/key-channel gate] ; U2.A(channel-diagonal G) ; E[output]` | `B[channel-decay native envelope]`, `B[K2_graph_execution]` |
| 020 Based | `E[QKV/Taylor-2 feature map] ; U2.A(numerator) || U2.A(denominator) ; E[division/output]` | Prove feature map and denominator equivalence; cost expanded state |
| 021 ReBased | `E[QKV/squared-dot feature map] ; U2.A(numerator) || U2.A(denominator) ; E[division/output]` | Prove exact polynomial feature and epsilon policy; cost expanded state |
| 022 LightNet | `E[projections/short conv/key normalization/gates] ; U2.A(channel decay) ; E[gated output]` | All frontend states and `B[channel-decay native envelope]` |
| 023 HGRN | `E[input/gate projections] ; B[K2 vector gated transition] ; E[norm/output]` | Read/update order and vector-state VJP; matrix-A is not assumed |
| 024 HGRN2 | `E[Q/K/input/gates] ; U2.A(channel decay,value-first layout) ; E[layer norm/output]` | Layout is physical only if equation preserved; `B[channel-decay native envelope]` |
| 025 DeltaNet | `E[QKV/β/key norm] ; U2.D(no decay) ; E[output]` | Initial/final-state and all key/gate VJPs; `B[K2_graph_execution]` |
| 026 Gated DeltaNet | `E[QKV/β/decay gate] ; U2.D(explicit decay) ; E[output/cache]` | Source-specific decay ordering and ATMA decode equivalence; `B[K2_graph_execution]` |
| 027 GDN2 | `E[QKV/erase/write/decay gates] ; B[K2 distinct erase-write transition] ; E[norm/output]` | Distinct factor gradients; no scalar-β delta equivalence by naming |
| 028 KDA | `E[QKV/A_log,dt,bias/norm] ; U2.D(key-channel decay,read scale=1/√K) ; E[gated output]` | `B[channel-decay delta native envelope]`, exact gate and cache ABI |
| 029 Gated DeltaProduct | `E[rank-R factor and gate generation] ; B[K2 ordered multi-delta transition] ; E[output]` | Independent factor VJPs, rank/cost bound; no single-delta collapse |
| 030 Momentum DeltaNet | `E[Q/K/p/α and optional norms] ; B[K2 coupled memory+momentum states] ; E[output]` | Preserve both states and default source modes; no one-matrix alias |
| 031 Generalized delta IPLR | `E[QKV/left/right factors] ; B[K2 I+LRᵀ transition] ; E[output]` | Rank growth and boundary state, all factor VJPs |
| 032 Generalized delta DPLR | `E[QKV/diagonal+left/right factors] ; B[K2 D+LRᵀ transition] ; E[output]` | Diagonal distinct from rank factors; no IPLR default |
| 033 Gated Oja | `E[QKV/gates] ; B[K2 value-channel decay+Oja correction] ; E[output]` | Exact update ordering and backward envelope |
| 034 PGDN | `E[QKV/gates/ATK parameters] ; B[K2 memory+evolving metric and nonlinear preconditioning] ; E[output]` | State-dependent coefficients, metric VJP; no affine scan assertion |
| 035 PKDA | `E[QKV/KDA gates/ATK parameters] ; B[K2 channel-decayed delta+evolving metric] ; E[output]` | Metric and channel gates kept distinct; no PGDN name alias |
| 036 Rodimus | `E[short conv/QKV/gates/norm] ; U2.A(channel decay,source read scale) ; E[output]` | Source scale, nonzero state and V-first layout; `B[channel-decay native envelope]` |
| 037 Comba | `E[dual-key projections/gates] ; B[K2 delta with independent predict/write keys] ; E[output]` | Separate key gradients; no tied-key rewrite |
| 038 MesaNet | `E[Q/K normalization/regularizer] ; B[K2 two covariance states+regularized per-token solve] ; E[output]` | Solver tolerance/iterations, implicit VJP and nonzero-state gradients |
| 039 Titans | `E[model memory hierarchy/outer attention/projections] ; B[closed typed inner-loss/update region] ; E[block output]` | No callback, no bare affine-K2 relabel; optimizer state and outer VJP |
| 040 RWKV-4 | `E[time/channel mixing/projections] ; B[K2 max-shifted normalized vector state] ; E[output gate]` | Stable rescaling and cache continuation; no raw additive-state substitution |
| 041 RWKV-6 | `E[time mix/projections] ; B[K2 channel decay+static bonus read] ; E[output gate]` | Bonus read timing and final-state cotangent distinction |
| 042 RWKV-7 | `E[time/value mixing/factor generation] ; B[K2 diagonal+rank-one transition] ; E[output gate]` | Rank/chunk closure and source read order |
| 043 Mamba-1 | `E[projection/short conv/activation] ; B[K2 diagonal selective SSM] ; E[skip/output gate/cache]` | Input-precomputable coefficients, scan/step equivalence |
| 044 Mamba-2 | `E[projection/short conv/dt] ; B[K2 semiseparable transition with boundary state] ; E[skip/output gate/cache]` | Cross-chunk carry; no block-mask equivalence |
| 045 Mamba-3 | `E[projection/A,dt,trapezoid frontend] ; B[K2 rotary phase+SSM+prior-K/V bundle] ; E[output/cache]` | SISO and MIMO separate, four-state VJP/cache |
| 046 LogLinearMamba2 | `E[Mamba-2 frontend/dt/level transforms] ; B[K2 hierarchical level state of 009] ; E[output/cache]` | Partial-chunk streaming and source layer, not only aligned chunks |

## 047–057: indexed memories and multi-call models

| ID | External call graph, in execution order | Mandatory blocker before full row claim |
|---|---|---|
| 047 SDM | `E[product-key feature projections] ; R.PK ; U3.D(certified route result) ; E[output/state ownership]` | Prove the pinned source router exactly matches `R.PK`; route provenance, overlap VJP, collision/read timing and full router cost; no SDM selector in core |
| 048 ABC | `E[QKV/slot gates] ; B[K2 stage-1 slot summary] ; E[slot softmax] ; B[K2 stage-2 value summary] ; E[output]` | Transcribe both source state equations and their coupling; two calls cannot be replaced by one generic K2 by name |
| 049 GSA | `E[QKV/slot gates] ; B[K2 stage-1 slot summary] ; E[slot softmax] ; B[K2 stage-2 value summary] ; E[output]` | Compare ABC/GSA state and gate semantics before any shared descriptor; both states and VJPs |
| 050 Raven | `E[router/top-k/feature map/decays] ; (049 exact two-stage GSA call graph) ; E[norm/output]` | Source Raven routing and GQA; no Raven backend path |
| 051 MoM | `E[router/top-k/pack] ; parallel U2.D(gated delta per selected memory) ; E[merge/shared memory/output]` | Route ownership, variable stream lengths, expert state/cache and router gradients; no ungated one-stream equivalence |
| 052 YOCO | `E[RoPE/gates/projections] ; U2.A(Simple-GLA self-decoder) ; E[shared KV/cache construction] ; U1.S(cross-decoder) ; E[output]` | Both state/cache regimes and layer order; no Simple-GLA fragment as YOCO claim |
| 053 Samba | `E[config-derived ordered Block list] ; each Block = E[norm] ; selected mixer call(s): Mamba→B[K2 selective SSM], attention→U1.S, retention→U2.A, or GLA→U2.A ; E[residual/optional MLP]` | Preserve the source's **two** arrangement modes: mixer selection per layer, or `mamba_swa_mlp` running Mamba then attention in one Block. Record exact config, order and both cache types; `samba_attention_core` is only one attention fragment |
| 054 AttnRes | `E[collect previous residual states/depth parameters] ; U1.S(depth as source domain,source normalization) ; E[residual update/next layer]` | Exact depth score/normalization, live-state lifetime and cross-layer VJP; do not treat depth as sequence state |
| 055 TTT | `E[QKV/inner objective and model schedule] ; B[typed inner optimizer state/update] ; E[outer output/cache]` | TTT-Linear versus TTT-MLP, higher-order gradients and no arbitrary callback |
| 056 FwPKM | `E[product-key address search/entropy] ; B[K1 selected-logit score/reducer] ; E[fast-weight write/optimizer/cache]` | Read-only fragment does not establish write; promote `U3` only after its exact write law/collision/timing equals a reusable K3 descriptor |
| 057 Pattention | `E[construct/reparameterize parameter tokens] ; U1.S(parameter domain,source scale) ; E[GELU/L2/output/model replacement]` | `B[K1 parameter-domain ABI]`, parameter-token gradients and complete replaced layers; no claim that one attention contraction equals MLP or MoE |

Catalog rows **058 SwiGLU MLP** and **059 top-k MoE** are not mixer calls. Keep their ordinary pointwise/GEMM and route/grouped-GEMM graphs outside this ledger; the 80-row register marks them not applicable to K1/K2/K3 parity.

## 060–078: nonlinear, alternate reduction and long-convolution rows

| ID | External call graph, in execution order | Mandatory blocker before full row claim |
|---|---|---|
| 060 RNN | `E[input/recurrent projections] ; B[K2 closed nonlinear vector transition] ; E[output/cache]` | Activation, weight sharing, packed lengths and VJP; no affine scan assumed |
| 061 GRU | `E[reset/update/candidate projections] ; B[K2 closed GRU gate transition] ; E[output/cache]` | Exact gate order and recurrent VJP; no RNN-name branch |
| 062 M2RNN | `E[matrix-memory projections/gates] ; B[K2 closed nonlinear matrix transition] ; E[output/cache]` | Matrix state shape, packed lengths and full gradients |
| 063 BDH | `E[encoder/value projections/RoPE] ; B[K2 strict-past additive read-before-write] ; E[norm/MLP/dropout/residual]` | Exact strict-past timing; U2.A is after-update and cannot be used unchanged |
| 064 POLAR | `E[QKV/canonical convolution] ; B[K1 polar direction/count reducer] ; E[output gate/projections]` | Score/reducer identity, neutral elements and VJP; no softmax substitution |
| 065 Foveal | `E[geometric/local/remote route selection] ; B[K1 indexed polar reducer] ; E[projections/count/output]` | Reuse 064 reducer plus exact route ABI; include selection cost and gradients |
| 066 CAT | `E[chunk compression/adaptive and separator tokens] ; U1.S[compressed history ∪ local block] ; E[decoder output/cache]` | Architectural token identity and source mask; no schedule-only chunk substitution |
| 067 Differential | `E[Q1/K1,Q2/K2,V,λ frontend] ; (U1.S_1 || U1.S_2) ; E[weighted difference/per-head norm/output]` | Exact V1 λ and cache; V2 paired-head one-call rewrite separately proved with VJP and cost |
| 068 TDA | `E[Q1/K1,Q2/K2,V,β/threshold/λ] ; (B[K1 threshold-ReLU-power reduction]_1 || same_2) ; E[weighted difference/output]` | Exact source normalization and threshold gradient; U1.S softmax is invalid |
| 069 TPA | `E[CP-factorized QKV/RoPE] ; U1.S ; E[output/factorized cache]` | Source factorized projection and cache; no expanded-cache comparison |
| 070 Tucker | `E[Tucker QKV factors] ; U1.S ; E[output/factorized cache]` | Factorization gradients and cache; no Tucker-named K1 |
| 071 Longformer | `E[QKV/global/padding mask] ; U1.S[local ∪ global] ; E[encoder output]` | Global-query semantics and `B[indexed_K1_schedule]`; local-only fragment insufficient |
| 072 Sparse Transformer | `E[QKV/source all/local/strided/fixed pattern] ; U1.S[exact blocks] ; E[output]` | `B[indexed_K1_schedule]`; compare optimized source traversal, not dense masked SDPA |
| 073 KATA | `E[QKV/grouping] ; B[K1 positive squared-group-dot normalized reducer] ; E[model output/cache]` | Exact source score, denominator and VJP; optional polynomial U2 rewrite needs proof **and** state-size cost |
| 074 HLA | `E[QKV feature frontend] ; z,s1=U2.A(q,k,k;s1) ; y,s2=U2.A(q,z,v;s2) ; E[optional final normalization/output]` | Exact masked second-order **unnormalized** causal case only; both cache states and VJPs. Higher/asymmetric/decayed variants are separate blocked equations |
| 075 Conformer | `E[QKV/relative-position frontend] ; U1.S(noncausal,exact bias) ; E[convolution/FFN/norm/residual]` | Relative-position variant and full encoder; no attention-only Conformer claim |
| 076 H3 | `E[projections] ; E[causal FFT-conv_1] ; E[multiply] ; E[causal FFT-conv_2] ; E[gate/output]` | Source head layouts, step/cache and FFT cost. A U2 state-space rewrite is **blocked** until exact realization and cheaper schedule are proved |
| 077 Hyena | `E[implicit filter generation/short conv/projections] ; E[order-specific causal FFT-conv/gates] ; E[output/cache]` | No compact general U2 realization established; do not count FFT path as native three-kernel mixer coverage |
| 078 Hopfield | `E[patterns/projections/scaling] ; repeat source association steps { U1.S(exact static/dynamic mask) } ; E[norm/output]` | Iteration/stopping, learned scaling and pattern gradients; one-step fragment only |

Catalog rows **079 MAML** and **080 Reptile** are optimizer/training algorithms outside sequence-mixer qualification. They may become external training loops, never K2 merely because they update state.

### Worked configuration rule: Samba (053)

The pinned [Samba `Block` source](https://github.com/microsoft/Samba/blob/617c7a0f8c/lit_gpt/model.py) has more than one arrangement. With `mamba_swa_mlp`, **the same block** runs Mamba, then attention, with separate normalizations and residual additions. Otherwise it selects a mixer per layer: `attn_layer_pos` overrides the periodic Mamba choice; the periodic Mamba/RetNet/GLA tests have source-defined precedence, and an unmatched layer uses causal attention. The optional MLP follows the mixer. A correct external module must materialize this **configuration-derived ordered block list** and bind distinct SSM/retention/GLA states and attention KV caches. The `samba_attention_core` fragment cannot be promoted to `complete_model_graph` by relabeling it; it lacks the other branch and the schedule. A source revision or configuration change requires a new graph/parity record, not a compiler condition on `name == "samba"`.

## Explicit reuse/admission matrix

The following are the only **candidate** shared mechanisms implied by the ledger. “Candidate” means source equations and references still gate promotion. Multiple IDs in a row show possible independent clients, **not** automatic proof that their exact equations coincide. Two clients with a renamed source equation do not satisfy the admission rule.

| Generic axis or operation | Potential independent clients | What must be proved before a native branch |
|---|---|---|
| Indexed softmax traversal and route ABI | 005/006/007/071/072; 066 as a different compressed/local graph | Exact route/visibility semantics, safe provenance, gradients, real sparse work and route-inclusive cost |
| Stable time/channel score modulation | 008/011 | Source score factors and numerically stable tiled interval evaluation; do not merge scalar and channel gates by name |
| Positive/non-softmax K1 reducer | 064/065 share polar; 068/073 are different reducer equations | Closed score/normalizer descriptors, neutral elements, all-masked behavior, VJP; a shared *physical* path is optional |
| Channel-decayed additive/delta state | 019/024/028/036 and 027 as a distinct update | Source gate shapes, read order, state/gradient equality; efficient long prefill |
| Low-rank state transition | 031/032/042; 029 has ordered multi-update rather than the same equation | Typed rank, diagonal contribution, composition bounds, precision and VJP |
| Coupled/multiple state ownership | 030/034/035/038/045/048/049/074 | State bundle ABI and gradients; do **not** infer shared transition algebra merely from multiple states |
| Inner optimization | 039/055; 056 only after write/optimizer equation audit | Closed optimizer/loss descriptors, higher-order gradients and checkpoint policy; no Python callback |
| Cross-call output combination | 067/068; 048/049 have interstage normalization; 074 has dependent K2 calls | Typed data/state edges, coefficient gradients and lawful fusion; two calls may remain two launches |
| Indexed mutable state | 047; 056 only if its fast-weight write law maps to a typed K3 rule | Logical route, collision, read/write timing, commit effects, provenance and VJP |

**Physical schedule decision:** start with an unfused external call graph. A compiler rewrite may fuse or reassociate only with a registered equivalence proof, dtype/backward envelope and complete cost delta. The base plan remains selectable. If a proposed axis has only one source client or its equation is still opaque, keep it outside core and record the row as **external/reference only**. “Representable,” “reference-executable,” “native,” and “within performance gate” are four different outcomes.

## Required closeout record for each ID

The ledger alone cannot close an architecture. Before changing a row's claim level, produce one versioned evidence record keyed by `architecture_id`, not a free-text assertion. It must contain:

| Field | Required content / rejection rule |
|---|---|
| `source` | Pinned repository revision, exact source file/function and invoked model unit. A kernel adapter is not the model unit. |
| `external_graph` | Executable model entry point, ordered ordinary operators, complete layer schedule, every public URM call, state/cache edges, train/prefill/decode paths. For Samba, the ordered layer list must contain both Mamba and attention branches. |
| `semantic_signatures` | For each public call: closed K1/K2/K3 descriptor hash; role-indexed operands and outputs; logical domains; masks/routes/head mapping; read/write/collision/commit effects; shapes, dtypes, accumulation/casts; initial/final state; gradient requirements. Rejected or ignored JSON fields invalidate the record. |
| `rewrite_proofs` | For every normalization, fusion, feature-map substitution or state-space equivalence: rule ID, preconditions, exact versus floating-point class, dtype envelope, independent forward/VJP tests, saved-state policy and traffic delta. No proof means use the unfused source graph. |
| `plan` | Per-region provider tier, exact serialized binding and schedule, reference/native/library declines, placement and route provenance. Every graph node must be executed, lawfully fused, or cause compile failure. |
| `measurement` | Separate fragment, generic decoder and full source-model boundaries; paired source and URM results by mode/shape/dtype/device, numerical errors for outputs, parameters and state, peak memory, launch/materialization/route cost and uncertainty. A source-mode absence is recorded as unavailable. |
| `verdicts` | Four independent fields: `represented`, `reference_executable`, `native_qualified`, `performance_qualified`, each with exact supported envelope. Never derive one field from another. |

Reject the evidence record if an architecture name changes core dispatch, a model branch is absent, any `B[...]` remains unresolved for a claimed native mode, the plan invokes an unrecorded fallback, or a fragment result is presented as source-model parity. A core branch is added only after the separate two-client axis admission record is complete. This makes the ledger operational: a reviewer can inspect the exact source graph, semantic request, compiled plan and measured boundary for each of the 76 rows.
