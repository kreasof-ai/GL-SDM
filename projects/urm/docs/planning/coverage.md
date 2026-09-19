# Named architecture coverage and comparison register

This is the production construction backlog, not a claim of current support.
Inclusion commits us to resolve the mapping and attempt a fair comparison. Each
architecture has a comparator or an explicit source-resolution blocker. None is
silently dropped because it does not fit the first three kernels.

The [machine-readable register](../../benchmarks/architecture-coverage.json) records
IDs, source revisions, proposed lowerings, mode qualification and remaining work.
Revisions captured on 2026-09-19 are audit/comparison candidates; they do not replace
older frozen acceptance pins. Every run must additionally pin its actual callable,
dependencies, shapes and tolerances. No new GPU parity is claimed here.

K1 = [softmax](../kernels/softmax-attention.md), K2 =
[linear/delta](../kernels/linear-delta.md), K3 =
[sparse delta](../kernels/sparse-delta.md). A `+` or extension denotes compiler work
that still needs derivation and implementation, not a configuration already supported.

## Source register

- **fla**: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2); pin `864a87f6ce5b`.
- **flash**: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554); pin `1bda8f9290cd`.
- **mamba**: [state-spaces/mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa); pin `e9594ce1c732`.
- **sdm**: [facebookresearch/sparse-delta-memory](https://github.com/facebookresearch/sparse-delta-memory/tree/183e7df809131b80ad4393741029d0f20fc3640b); pin `183e7df80913`.
- **pattention**: [Haiyang-W/TokenFormer](https://github.com/Haiyang-W/TokenFormer/tree/4d56c73f407635e62f6df16b97dc897b4477129e); pin `4d56c73f4076`.
- **xma**: [open-lm-engine/accelerated-model-architectures](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844); pin `384ed0a7bd82`.
- **bdh**: [pathwaycom/bdh](https://github.com/pathwaycom/bdh/tree/2b0d7a45b058d4309c84a10e0768d541fe18bdc2); pin `2b0d7a45b058`.
- **samba**: [microsoft/Samba](https://github.com/microsoft/Samba/tree/617c7a0f8c71f1b7cb6180b86f9543d146f5c66f); pin `617c7a0f8c71`.
- **fwpkm**: [SakanaAI/fast-weight-product-key-memory](https://github.com/SakanaAI/fast-weight-product-key-memory/tree/b1c8e234b523d70245fa197eed4b80a985c413a8); pin `b1c8e234b523`.

ATMA is a local comparator at its recorded revision; hash dirty source files before
benchmarking. `unresolved` is a visible blocker, never permission to substitute an
unrelated baseline. FLA inventory establishes source availability only. Selected
equation inspections and corrections are in the [audit](unification-audit.md).

## Wave 1: Core native closure

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-001: Transformer MHA | [flash](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1 | Online softmax and complete Q/K/V backward; compare matching direct attention kernels |
| arch-002: Transformer MQA | [flash](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1 | Shared KV layout and gradient reduction; cache decode workload |
| arch-003: Transformer GQA | [flash](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1 | Explicit head mapping and KV gradient accumulation |
| arch-015: Linear attention | [fla / `fla/ops/linear_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/linear_attn) | K2 additive | Feature maps and denominator state; prefill and recurrent decode |
| arch-025: DeltaNet | [fla / `fla/ops/delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/delta_rule) | K2 delta | Stable triangular solve; initial/final-state and key gradients |
| arch-026: Gated DeltaNet | [fla / `fla/ops/gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_delta_rule) | K2 delta | Decay-before-retrieval; full backward and recurrent inference |
| arch-047: Sparse Delta Memory | [sdm / `lingua/sparse_delta_memory`](https://github.com/facebookresearch/sparse-delta-memory/tree/183e7df809131b80ad4393741029d0f20fc3640b/lingua/sparse_delta_memory) | K3 delta | Repair chunk decay VJP and boundary folding; preserve declared rounding policy |

## Wave 2: Structured variants and composite memories

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-004: MLA | [fla / `fla/layers/mla.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/mla.py) | K1+composition | Preserve latent KV and positional branches; do not inflate cache for a false parity win |
| arch-005: NSA | [fla / `fla/ops/nsa`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/nsa) | K1+route/compression | Compression selection and local branches; include indexer and auxiliary routing work |
| arch-006: MoBA | [fla / `fla/ops/moba`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/moba) | K1+route | Block scoring selection and causal boundaries; sparse traversal |
| arch-007: DSA | [fla / `fla/ops/dsa`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/dsa) | K1+route | Indexer objective and selected attention; full-layer comparison |
| arch-008: Forgetting Transformer / FoX | [fla / `fla/ops/forgetting_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/forgetting_attn) | K1+gate | Log-forget score bias; stable normalization and gate gradients |
| arch-016: Lightning Attention | [fla / `fla/ops/lightning_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/lightning_attn) | K2 additive candidate | Derive exact decay/normalization and tile schedule |
| arch-017: RetNet / retention | [fla / `fla/ops/retention`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/retention) | K2+frontend | Multiscale decay positional transforms and layer normalization |
| arch-018: Simple GLA | [fla / `fla/ops/simple_gla`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/simple_gla) | K2 candidate | Freeze gate granularity then chunk and recurrent paths |
| arch-019: GLA | [fla / `fla/ops/gla`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gla) | K2 diagonal transition | Channel-wise gates; stable decay factorization and all gate gradients |
| arch-020: Based | [fla / `fla/ops/based`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/based) | K2+feature map | Feature expansion and denominator; count feature-map execution cost |
| arch-021: ReBased | [fla / `fla/ops/rebased`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rebased) | K2+feature map | Preserve learned feature-map normalization and gradients |
| arch-022: LightNet | [fla / `fla/layers/lightnet.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/lightnet.py) | K2+frontend audit | Extract exact feature/gate composition; layer-level parity |
| arch-023: HGRN | [fla / `fla/ops/hgrn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/hgrn) | K2 vector-state extension | Vector state layout and gate activation; no unnecessary matrix expansion |
| arch-024: HGRN2 | [fla / `fla/layers/hgrn2.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/hgrn2.py) | K2+frontend audit | State expansion and gate semantics; include projection costs |
| arch-028: KDA | [fla / `fla/ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/kda) | K2 transition extension | Channel gate placement and triangular solve; normalization and backward |
| arch-040: RWKV-4 | [fla / `fla/ops/rwkv4`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv4) | K2+stable normalized state | Numerical rescaling denominator and time mixing; no generic delta substitution |
| arch-041: RWKV-6 | [fla / `fla/ops/rwkv6`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv6) | K2+read correction | Time-dependent transition and read correction; frontend time shift |
| arch-048: ABC | [fla / `fla/ops/abc`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/abc) | K2+slot softmax composition | Two recurrent summaries and slot normalization; not a simple bounded attention mask |
| arch-049: GSA | [fla / `fla/ops/gsa`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gsa) | K2+slot softmax composition | Two state matrices and interposed softmax; no SDM delta substitution |
| arch-071: Longformer | unresolved | K1 sparse-mask fixtures | Resolve separate original implementations and mask contracts before benchmarking |
| arch-072: Sparse Transformer | unresolved | K1 sparse-mask fixtures | Resolve separate original implementations and mask contracts before benchmarking |

## Wave 3: Generalized state, routing and axis coverage

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-009: Log-linear attention | [fla / `fla/ops/log_linear_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/log_linear_attn) | extension audit | Derive structured mask and state; do not assume ordinary softmax semantics |
| arch-010: PaTH attention | [fla / `fla/ops/path_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/path_attn) | K1+transform recurrence | Preserve recurrent transformation and its gradients before softmax |
| arch-011: Wall attention | [fla / `fla/ops/wall_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/wall_attn) | extension audit | Audit wall-specific score/state equations and supported head dimensions |
| arch-012: Parallax | [fla / `fla/ops/parallax`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/parallax) | extension audit | Derive positional/state transformation; do not assume a window mask suffices |
| arch-013: DeltaFormer | [fla / `fla/ops/deltaformer`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/deltaformer) | K1+K2 composition audit | Verify whether fused interaction matches a composition; time full coupled layer |
| arch-014: BitAttention | [fla / `fla/layers/bitattn.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/bitattn.py) | K1+precision audit | Audit quantization and gradient surrogate; compare identical precision contract |
| arch-027: GDN2 | [fla / `fla/ops/gdn2`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gdn2) | K2 extension audit | Derive actual recurrence instead of treating it as a renamed GDN |
| arch-029: Gated DeltaProduct | [fla / `fla/ops/gated_delta_product`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_delta_product) | K2 multi-update | Ordered products of updates per token; rank and update ordering |
| arch-030: Momentum DeltaNet | [fla / `fla/ops/momentum_delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/momentum_delta_rule) | K2 augmented state | Carry momentum state and derive block transition/VJP |
| arch-031: Generalized delta IPLR | [fla / `fla/ops/generalized_delta_rule/iplr`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/generalized_delta_rule/iplr) | K2 low-rank transition | Independent left/right factors; chunk representation already exists upstream |
| arch-032: Generalized delta DPLR | [fla / `fla/ops/generalized_delta_rule/dplr`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/generalized_delta_rule/dplr) | K2 low-rank transition | Diagonal-plus-low-rank transition; separate factor gradients |
| arch-033: Gated Oja rule | [fla / `fla/ops/gated_oja_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_oja_rule) | K2 extension audit | Derive exact Oja update and additional state dependence |
| arch-034: PGDN | [fla / `fla/ops/precond_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/precond_gated_delta_rule) | K2+preconditioner state | ATK state transform and gradients; not a scalar row rescaling |
| arch-035: PKDA | [fla / `fla/ops/precond_kda`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/precond_kda) | K2+preconditioner state | Preconditioner plus channel gates and saved-state policy |
| arch-036: Rodimus | [fla / `fla/layers/rodimus.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/rodimus.py) | K2+frontend audit | Audit update and gating composition; full-layer comparison |
| arch-037: Comba | [fla / `fla/ops/comba`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/comba) | K2 extension audit | Derive full state equation and non-shared update factors |
| arch-038: MesaNet | [fla / `fla/ops/mesa_net`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/mesa_net) | K2/solve extension audit | Explicit solve or optimization state and differentiability; inspect actual implementation |
| arch-039: Titans | [fla / `fla/ops/titans`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/titans) | typed update composition | Pin operator variant; distinguish memory kernel from complete Titans architecture |
| arch-042: RWKV-7 | [fla / `fla/ops/rwkv7`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv7) | K2 low-rank extension | Audit generalized state transition and value/time mixing; compare chunk and recurrent upstream |
| arch-043: Mamba-1 | [mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | K2/SSM+convolution | Selective state update discretization local convolution and cache ABI |
| arch-044: Mamba-2 / SSD | [mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | K2/semiseparable+convolution | Inter-chunk state and diagonal blocks; preserve SSM discretization |
| arch-045: Mamba-3 | [mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | K2/SSM multi-state | SISO/MIMO rotary phase and previous-key/value state; separate mode gates |
| arch-046: LogLinearMamba2 | [fla / `fla/layers/log_linear_mamba2.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/log_linear_mamba2.py) | K2+structured composition | Audit structured mixing plus full Mamba layer and cache |
| arch-050: Raven | [fla / `fla/layers/raven.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/raven.py) | GSA+route/frontend | Audit actual GSA use and router/decay options; compare full layer |
| arch-051: Mixture of Memories / MoM | [fla / `fla/layers/mom.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/mom.py) | route+state composition | Memory selection dispatch merge and gradient coverage for each routed memory |
| arch-052: YOCO | [fla / `fla/layers/yoco.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/yoco.py) | multi-layer/cache composition | Cache sharing and self/cross-decoder composition; not only an isolated mixer |
| arch-053: Samba | [samba](https://github.com/microsoft/Samba/tree/617c7a0f8c71f1b7cb6180b86f9543d146f5c66f) | K1+SSM+MLP composition | Benchmark complete hybrid block and caches against pinned original model |
| arch-054: Attention Residuals / AttnRes | [fla / `fla/ops/attnres`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/attnres) | depth-axis reduction | Layer-axis dependencies normalization and gradient accumulation |
| arch-057: TokenFormer / Pattention | [pattention](https://github.com/Haiyang-W/TokenFormer/tree/4d56c73f407635e62f6df16b97dc897b4477129e) | parameter-axis contraction | Exact activation/normalization and parameter gradients; no automatic SwiGLU equivalence |
| arch-058: SwiGLU MLP | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | parameter-axis grouped contractions | Two up projections SiLU product and down projection; fused forward/backward |
| arch-059: Top-k MoE | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | route+grouped expert compute | Dispatch capacity expert MLP combine and backward; include permutation costs |
| arch-064: POLAR attention | atma / `docs/POLAR_ATTENTION.md` | score/reduction extension audit | Freeze local implementation and magnitude/normalization channel; full backward |
| arch-065: Foveal sparse attention | atma | K1+geometric route audit | Resolve exact local callable and selection semantics; routing cost included |
| arch-067: Differential Attention | unresolved | multiple K1+combine | Pin exact variant lambda parameterization and normalization; do not conflate names |
| arch-068: TDA | unresolved | multiple K1+combine | Pin exact variant lambda parameterization and normalization; do not conflate names |
| arch-069: TPA | unresolved | factorized projection+K1 audit | Pin architecture and factorization; cache expansion cost included |
| arch-070: Tucker attention | unresolved | factorized projection+K1 audit | Pin architecture and factorization; cache expansion cost included |

## Wave 4: Nonlinear updates and unresolved identities

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-055: TTT | [fla / `fla/ops/ttt`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/ttt) | typed inner-update extension | Pin linear/MLP variant inner loss update schedule and higher-order gradients |
| arch-056: FwPKM | [fwpkm](https://github.com/SakanaAI/fast-weight-product-key-memory/tree/b1c8e234b523d70245fa197eed4b80a985c413a8) | sparse route+typed inner update | Fast weight/key update objective chunk boundaries and optimizer state |
| arch-060: RNN | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | nonlinear recurrent extension | Preserve nonlinear transition; audit iterative solver convergence and exactness |
| arch-061: GRU | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | nonlinear recurrent extension | Reset/update gates and full training path; do not use affine scan without proof |
| arch-062: M2RNN | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | nonlinear recurrent extension | Pin upstream operator and solver tolerances; compare state and all gradients |
| arch-063: BDH | [bdh](https://github.com/pathwaycom/bdh/tree/2b0d7a45b058d4309c84a10e0768d541fe18bdc2) | graph/state composition audit | Extract reference update and graph semantics before proposing a lowering |
| arch-066: CAT / Compress-and-Attend | unresolved | compression+K1 composition | Disambiguate named paper/version and official implementation; shape variation is not alone a rejection |
| arch-073: KATA | unresolved | identity and equation audit | Archived names need exact paper/repository resolution before assigning family |
| arch-074: HLA | unresolved | identity and equation audit | Archived names need exact paper/repository resolution before assigning family |
| arch-075: Conformer | unresolved | convolution+mixing composition audit | Split exact architecture fixtures after source resolution; not all linear aggregation is attention |
| arch-076: H3 | unresolved | convolution+mixing composition audit | Split exact architecture fixtures after source resolution; not all linear aggregation is attention |
| arch-077: Hyena | unresolved | convolution+mixing composition audit | Split exact architecture fixtures after source resolution; not all linear aggregation is attention |
| arch-078: Hopfield | unresolved | iterative update composition audit | Identify model and objective; fixed-point and meta-optimizer semantics are different |
| arch-079: MAML | unresolved | iterative update composition audit | Identify model and objective; fixed-point and meta-optimizer semantics are different |
| arch-080: Reptile | unresolved | iterative update composition audit | Identify model and objective; fixed-point and meta-optimizer semantics are different |

## What counts as coverage

Report frontend expression, external-adapter execution, native execution and
performance parity separately. Each target needs kernel, full-layer and model
results where those boundaries exist. A kernel does not implicitly cover a
router, convolution, positional transform, normalization, cache or inner optimizer.

Qualify training, prefill and decode separately. Unsupported upstream modes must
be recorded with evidence as `upstream_unavailable`, never counted as passes.
Wave 1 is the first release gate; subsequent waves are full-version construction
work. Every row exits with qualification or a published blocker and concrete
implementation task. Blocked rows remain in the backlog and outside advertised
support. New upstream variants add comparisons rather than overwrite old baselines.

Use the [parity plan](../validation/parity.md) and
[generality axes](../compiler/generality-axes.md) to build each row.
