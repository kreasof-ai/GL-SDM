# Source architecture index

Generated from the [machine register](../../extra/architecture-coverage.json). This is the source identity and construction backlog. The [architecture catalog](../catalog.md) maps the registered rows; [evidence rules](../evidence.md) define what each status can claim. Inclusion is not model support.

Of 80 catalog rows, 76 are mixer-relevant and 4 are outside mixer scope. 19 rows map to live public-graph fragments and 57 retain the `pending_graph_migration` label. These figures do not qualify a complete source model. No K2 graph recipe is live. Parity and performance evidence for the current rows is the training-harness campaign (`results/sweep/`, `results/upstream/`, `results_report.md`).

Proposed lowerings in the register are hypotheses until closed descriptors, references and public plans qualify them.

## Wave 1: Core native closure

| ID | Architecture | Proposed fragment | Comparator | Prototype |
|---|---|---|---|---|
| arch-001 | Transformer MHA | K1 | flash-attention `1bda8f9290` | kernel_prototype_only |
| arch-002 | Transformer MQA | K1 | flash-attention `1bda8f9290` | kernel_prototype_only |
| arch-003 | Transformer GQA | K1 | flash-attention `1bda8f9290` | kernel_prototype_only |
| arch-015 | Linear attention | K2 additive | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-025 | DeltaNet | K2 delta | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-026 | Gated DeltaNet | K2 delta | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-047 | Sparse Delta Memory | K3 delta | sparse-delta-memory `183e7df809` | kernel_prototype_only |

## Wave 2: Structured variants and composite memories

| ID | Architecture | Proposed fragment | Comparator | Prototype |
|---|---|---|---|---|
| arch-004 | MLA | K1+composition | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-005 | NSA | K1+route/compression | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-006 | MoBA | K1+route | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-007 | DSA | K1+route | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-008 | Forgetting Transformer / FoX | K1+gate | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-016 | Lightning Attention | K2 additive candidate | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-017 | RetNet / retention | K2+frontend | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-018 | Simple GLA | K2 head-decayed additive state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-019 | GLA | K2 key-channel-decayed additive state | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-020 | Based | K2+feature map | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-021 | ReBased | K2+feature map | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-022 | LightNet | K2 GLA state recurrence plus frontend | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-023 | HGRN | K2 vector-state extension | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-024 | HGRN2 | K2+frontend audit | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-028 | KDA | K2 transition extension | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-040 | RWKV-4 | K2+stable normalized state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-041 | RWKV-6 | K2 key-channel-decayed matrix state plus static bonus read correction | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-048 | ABC | K2 two-stage slot attention | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-049 | GSA | K2 two-stage slot attention | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-050 | Raven | K2 shared GSA kernel | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-052 | YOCO | K2 shared Simple GLA kernel | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-071 | Longformer | K1 sparse-mask fixtures | longformer `caefee668e` | pending_graph_migration |
| arch-072 | Sparse Transformer | K1 masked softmax for exact all/local/strided/fixed source masks | sparse_attention `c53f3bdbf6` | kernel_prototype_only |

## Wave 3: Generalized state, routing and axis coverage

| ID | Architecture | Proposed fragment | Comparator | Prototype |
|---|---|---|---|---|
| arch-009 | Log-linear attention | extension audit | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-010 | PaTH attention | K1+transform recurrence | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-011 | Wall attention | K1 per-channel decay attention | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-012 | Parallax | extension audit | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-013 | DeltaFormer | K1 causal softmax after strict-causal triangular K2 value correction | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-014 | BitAttention | K1 causal softmax attention after BitLinear Q/K/V projections and RoPE | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-027 | GDN2 | K2 extension audit | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-029 | Gated DeltaProduct | K2 multi-update | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-030 | Momentum DeltaNet | K2 augmented state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-031 | Generalized delta IPLR | K2 low-rank transition | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-032 | Generalized delta DPLR | K2 low-rank transition | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-033 | Gated Oja rule | K2 value-channel-decayed Oja state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-034 | PGDN | K2 ATK-preconditioned gated delta state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-035 | PKDA | K2 key-channel ATK-preconditioned delta state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-036 | Rodimus | K2 BF16 key-channel GLA with V-first state and source-default 1/sqrt(K) read scale | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-037 | Comba | K2 head-decayed dual-key delta state | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-038 | MesaNet | K2 dual covariance-state recurrence plus regularized key-space solve | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-039 | Titans | typed update composition | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-042 | RWKV-7 | K2 low-rank extension | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-043 | Mamba-1 | K2/SSM+convolution | mamba `e9594ce1c7` | pending_graph_migration |
| arch-044 | Mamba-2 / SSD | K2/semiseparable+convolution | mamba `e9594ce1c7` | pending_graph_migration |
| arch-045 | Mamba-3 | K2 SISO rotary angle accumulator with trapezoidal four-state SSM recurrence | mamba `e9594ce1c7` | pending_graph_migration |
| arch-046 | LogLinearMamba2 | K2+structured composition | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-051 | Mixture of Memories / MoM | K2 route-dispatched gated-delta memory core | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-053 | Samba | Samba no-PE attention branch with QKV/output projections plus causal K1 | Samba `617c7a0f8c` | kernel_prototype_only |
| arch-054 | Attention Residuals / AttnRes | depth-axis reduction | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-057 | TokenFormer / Pattention | parameter-axis contraction | TokenFormer `4d56c73f40` | kernel_prototype_only |
| arch-058 | SwiGLU MLP | parameter-axis grouped contractions | accelerated-model-architectures `384ed0a7bd` | — |
| arch-059 | Top-k MoE | route+grouped expert compute | accelerated-model-architectures `384ed0a7bd` | — |
| arch-064 | POLAR attention | K1 online polar reduction | atma `28bb3de8af` | pending_graph_migration |
| arch-065 | Foveal sparse attention | K1 block-sparse polar reduction over supplied routes | atma `28bb3de8af` | kernel_prototype_only |
| arch-067 | Differential Attention | multiple K1+combine | unilm `50224e3872` | pending_graph_migration |
| arch-068 | TDA | multiple K1+combine | TDA `cd8ddc9d5b` | pending_graph_migration |
| arch-069 | TPA | factorized projection+K1 audit | TPA `c276c80d5a` | kernel_prototype_only |
| arch-070 | Tucker attention | factorized projection+K1 audit | Tucker-Attention `c3e3d3cec9` | pending_graph_migration |

## Wave 4: Nonlinear updates and newly resolved targets

| ID | Architecture | Proposed fragment | Comparator | Prototype |
|---|---|---|---|---|
| arch-055 | TTT | typed inner-update extension | flash-linear-attention `864a87f6ce` | pending_graph_migration |
| arch-056 | FwPKM | K1 fused softmax/value reduction over exact product-key selected logits | fast-weight-product-key-memory `b1c8e234b5` | pending_graph_migration |
| arch-060 | RNN | K2 nonlinear recurrent state | accelerated-model-architectures `384ed0a7bd` | pending_graph_migration |
| arch-061 | GRU | K2 nonlinear recurrent state | accelerated-model-architectures `384ed0a7bd` | pending_graph_migration |
| arch-062 | M2RNN | K2 nonlinear recurrent state | accelerated-model-architectures `384ed0a7bd` | pending_graph_migration |
| arch-063 | BDH | K2 strict-past rotary linear attention | bdh `2b0d7a45b0` | pending_graph_migration |
| arch-066 | CAT / Compress-and-Attend | compression+K1 composition | flash-linear-attention `864a87f6ce` | kernel_prototype_only |
| arch-073 | KATA | K1 normalized positive attention with grouped squared-dot scores | KATA `f93fe75750` | pending_graph_migration |
| arch-074 | HLA | K2 masked second-order causal attention using the HLA paper streaming summaries | HLA `484fef2bb4` | pending_graph_migration |
| arch-075 | Conformer | K1 bidirectional softmax attention | espnet `2950325ea6` | kernel_prototype_only |
| arch-076 | H3 | K2 two-stage causal FFT convolution with H3 multiplicative query/key/value composition | H3 `5c4d06b579` | pending_graph_migration |
| arch-077 | Hyena | K2 implicit-filter causal FFT convolution with source short-convolution, gating and projections | safari `02220c69d2` | pending_graph_migration |
| arch-078 | Hopfield | K1 unscaled softmax association with one retrieval update | hopfield-layers `f56f929c95` | kernel_prototype_only |
| arch-079 | MAML | iterative update composition audit | maml `a7f45f1bcd` | — |
| arch-080 | Reptile | iterative update composition audit | supervised-reptile `8f2b71c67a` | — |

## Source register

Each comparator is pinned to an exact revision; a resolved identity does not imply an executable comparator, and source blockers stay attached to the pinned row. ATMA is a local comparator at its recorded revision.

| Key | Repository | Pin |
|---|---|---|
| atma | https://github.com/kreasof-ai/atma | `28bb3de8af` |
| bdh | https://github.com/pathwaycom/bdh | `2b0d7a45b0` |
| conformer | https://github.com/espnet/espnet | `2950325ea6` |
| differential | https://github.com/microsoft/unilm | `50224e3872` |
| fla | https://github.com/fla-org/flash-linear-attention | `864a87f6ce` |
| flash | https://github.com/Dao-AILab/flash-attention | `1bda8f9290` |
| fwpkm | https://github.com/SakanaAI/fast-weight-product-key-memory | `b1c8e234b5` |
| h3 | https://github.com/HazyResearch/H3 | `5c4d06b579` |
| hla_higher_order | https://github.com/yifanzhang-pro/HLA | `484fef2bb4` |
| hopfield | https://github.com/ml-jku/hopfield-layers | `f56f929c95` |
| kata | https://github.com/ayghri/KATA | `f93fe75750` |
| longformer | https://github.com/allenai/longformer | `caefee668e` |
| mamba | https://github.com/state-spaces/mamba | `e9594ce1c7` |
| maml | https://github.com/cbfinn/maml | `a7f45f1bcd` |
| pattention | https://github.com/Haiyang-W/TokenFormer | `4d56c73f40` |
| reptile | https://github.com/openai/supervised-reptile | `8f2b71c67a` |
| safari | https://github.com/HazyResearch/safari | `02220c69d2` |
| samba | https://github.com/microsoft/Samba | `617c7a0f8c` |
| sdm | https://github.com/facebookresearch/sparse-delta-memory | `183e7df809` |
| sparse_transformer | https://github.com/openai/sparse_attention | `c53f3bdbf6` |
| tda | https://github.com/snap-research/TDA | `cd8ddc9d5b` |
| tpa | https://github.com/tensorgi/TPA | `c276c80d5a` |
| tucker | https://github.com/ScSteffen/Tucker-Attention | `c3e3d3cec9` |
| xma | https://github.com/open-lm-engine/accelerated-model-architectures | `384ed0a7bd` |

## What counts as coverage

Represented, reference-executable, native-qualified, performance-qualified and source-model-qualified are separate verdicts. A fragment never covers its router, frontend, cache or full layer by implication. Training, prefill and decode qualify separately. See the [evidence protocol](../evidence.md) and [generality axes](../compiler/generality-axes.md).
