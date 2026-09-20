# Unified mixer kernel compiler prototype

Status: executable kernel-level prototype. One typed entry point lowers the
three physical mixer families K1 softmax reduction, K2 recurrent state, and K3
sparse delta state through URM's semantic validator, candidate enumerator,
intent checks, and trusted anchor resolver. The reference executor supports
CPU/GPU autograd. K1, selected FLA K2 paths, the pinned Mamba-1, Mamba-2 and
Mamba-3 SISO library adapters, the FwPKM selected-read Triton reducer,
the pinned Samba no-PE attention branch,
and H3's two-stage K2 FFT-convolution adapter,
Momentum DeltaNet, MesaNet, Titans' chunked
associative memory, Gated Oja, COMBA,
PGDN, PKDA, ABC/GSA, the native diagonal Triton scan, and K3 have accelerated
dispatch paths. K2
simple-GLA and GLA acceleration also covers forward-only one-token decode.

```python
from urm.compiler.unified_mixer import (
    MixerIntent,
    compile_frontend_mixer,
    compile_mixer,
    named_mixer_recipe,
)
from urm.presets import GATED_DELTANET

recipe = named_mixer_recipe("gated_delta_net")
plan = compile_mixer(recipe, intent=MixerIntent.TRAINING)
result = plan(query=q, key=k, value=v, beta=beta, log_decay=log_decay)
loss = result.output.square().mean() + result.final_state.square().mean()
loss.backward()

frontend_plan = compile_frontend_mixer(GATED_DELTANET)
```

`compile_mixer()` returns a deterministic, serializable plan and the complete
URM `CompilationResult` used to select its anchor. Plans record the base
candidate, compiler dispatch step, semantic family, backend, and compile-time
dtype. Runtime checks enforce the primary input dtype and each executor checks
its shape, routes, state, and device contract. Compiling a `MixerRecipe` records
its architecture IDs, equation scope, and required external stages. A recipe is
not a claim that an entire named architecture is implemented.
`compile_frontend_mixer()` lowers supported existing `MixerSpec` values and
gives an explicit decline for unrepresented expert, transactional, complex
state and recurrent equations.

## Shared semantics

`UnifiedMixerSpec` is a closed descriptor. It declares the physical family,
state layout, update rule, feature map, decay granularity, transition type,
normalization, read timing, and optional attention scale/bias. It does not accept
tensor callbacks or backend-specific launch values. Runtime tensors are checked
against fixed rank and axis layouts before execution.

### K1: softmax reduction

The BTHD-layout reference computes

```text
Y[b,t,h,:] = sum_s softmax_s(Q[b,t,h] K[b,s,h_kv]^T * scale + bias + mask) V[b,s,h_kv,:]
```

It handles shared KV heads (MQA/GQA), causal prefill, cached decode alignment,
boolean/additive masks, and an optional additive score bias. Sparse attention
can use an explicit mask for semantic checks; this path still computes dense
scores and does not implement a sparse indexer or sparse traversal.

### K2: recurrent state

Matrix state is `[B,Hv,K,V]`; Q/K are `[B,T,H,K]` and V is `[B,T,Hv,V]`.
`Hv` must be an integer multiple of `H`. The compiler supports additive and
delta updates, optional ELU+1/ReLU/softplus feature maps, query/key denominator
state for additive linear attention, head or key-channel log decay, fixed per-head decay schedules, and
input-conditioned factored bilinear state transitions. Multiple update factors
per token are applied in declared order.

For a delta update, each ordered update is

```text
R_j = k_j^T S
S   = S + k_j (beta_j * (v_j - R_j))^T
```

The token transition is applied before retrieval, and output can read state
before or after the current token's updates. Head decay multiplies the whole
state for a head; key-channel decay scales its key rows. Factored transitions
apply `A_t S B_t^T` before the token updates. The executor does not silently
reduce a factored transition to scalar decay.

The diagonal SSM layout accepts `x[B,T,C]`, input/read gates, and log transition
coefficients. Its core is `s_t = exp(a_t) * s_(t-1) + b_t * x_t`, followed by
`y_t = sum_n c_t * s_t + D * x_t`. A step-size form represents selective
continuous-time SSMs as `s_t = exp(dt_t * A) * s_(t-1) + dt_t * B_t * x_t`.
The native Triton path supports both equations in float32. The pinned Mamba-1
library adapter accepts source-layout `u[B,C,T]`, `delta[B,C,T]`, `A[C,N]`,
and variable `B/C[B,N,T]` gates. These are state scans only; convolution,
projections, activation and cache composition remain external stages.
The Mamba-2 SSD adapter accepts `x[B,T,H,P]`, `dt[B,T,H]`, per-head `A[H]`,
grouped `B/C[B,T,G,N]`, and optional initial matrix state `[B,H,P,N]`. It matches
the pinned `mamba_chunk_scan_combined` call when `D`, `z`, and `dt_bias` are
absent and `dt_softplus` is false. Projections, short convolution, time-step
bias and softplus, skip/output gate and full cache/chunk-boundary composition
remain external.
The Mamba-3 SISO recipe tracks the rotary angle accumulator, SSM matrix, previous
rotated key and previous value. Each token applies `alpha = exp(adt)`,
`beta = (1 - sigmoid(trap)) * dt * alpha`, and
`gamma = sigmoid(trap) * dt`, then updates the matrix from previous and current
key/value outer products before reading with the rotated query. The BF16 profile
covers zero initial states and omits the skip and output gate; MIMO/TileLang and
model-side projections and parameter generation remain separate work.
The MesaNet BF16 K2 recipe keeps separate key-key and key-value matrix states,
applies the source log decay and beta-weighted key outer products, then solves
`(H_kk + diag(lambda)) q_star = q` at each token and reads `q_star` from
`H_kv`. Its independent reference uses a direct linear solve; the library
adapter calls pinned FLA `chunk_mesa_net` with 30 conjugate-gradient iterations.
Q/K normalization and lambda production remain frontend inputs. Zero initial
states are profiled; nonzero initial-state gradients and streaming decode remain
unqualified.
The Titans FP32 K2 recipe keeps a matrix memory and momentum state, freezes the
inner reconstruction target for each 16-token chunk, applies the source theta,
alpha and eta updates, and returns the normalized memory readout. Its reference
is a tokenwise recurrence; the library adapter calls pinned FLA's eager
`chunk_titans_linear_ref(use_chunk=True)`. This is an associative-memory
operator slice, not a fused GPU kernel or the complete Titans block.

### K3: sparse delta state

K3 consumes explicit read/write addresses and normalized weights. It decays only
the rows selected for writing, retrieves the write value, forms a gated residual,
scatters the update, and reads either the unmodified pre-update state or the
updated state according to the spec. Rows outside the write routes remain
unchanged. Tokens execute in sequence order. Addresses must be unique within
each token; collisions across tokens are handled by that order. The reference
commits to memory storage dtype at each token boundary.

## Execution backends

| Backend | Selection | Limits |
|---|---|---|
| `reference` | K1/K2/K3 | PyTorch eager operations with autograd; correctness and prototyping |
| `library` | K1 uses PyTorch SDPA plus named pinned FLA/ATMA adapters including MoBA, FoX, PaTH, Parallax, Wall, DeltaFormer, AttnRes, POLAR and Foveal; K2 uses pinned FLA linear-attention, HGRN, Retention, Lightning Attention, simple-GLA, GLA, delta-rule, gated-delta, GDN-2, MesaNet, Titans, Mamba-1 selective-scan, Mamba-2 SSD, Mamba-3 SISO, Gated Oja, COMBA, PGDN, PKDA, ABC and GSA anchors | K1 SDPA dispatches to Flash SDPA for supported shapes and dtypes. The MoBA parity/profile fixture builds its pinned FlashAttention dependency for BF16 D=32 causal and noncausal calls. FLA linear attention, un-decayed delta, and gated delta use fp16/bf16 chunk prefill/training. HGRN uses float32 recurrent execution; Retention and Lightning Attention use static per-head decay in float32 chunk execution. Simple GLA/GLA use float32 recurrent/chunk execution; Simple GLA and GLA also have forward-only fp16/bf16 one-token decode. GDN-2, Mamba-1 and Mamba-2 use their exact pinned source operators in float32. MesaNet is qualified for BF16 B1/T64/H2/K=V=16 with 30 CG iterations and zero initial states. Titans uses FLA's eager PyTorch chunk operator at FP32 B1/T64/H1/D=16; it is not a fused GPU kernel. Mamba-3 SISO is qualified at BF16 B1/T64/H2/K=V=16 with zero initial states. COMBA, PGDN, PKDA, Gated Oja, ABC and GSA use pinned fp16/bf16 chunk operators. K2 upstream adapters require their exact source pins |
| `native` | URM Triton diagonal SSM scan for K2 and sparse-state anchor for K3 | CUDA is required. Diagonal SSM currently supports float32. K3 supports float32/bfloat16 state, ordered unique routes, normalized route weights and native shape limits. The diagonal scan passes equation parity but is not speed-qualified against Mamba's CUDA scan |

Plans record the selected semantic family, anchor and backend. The K3 native
anchor covers route-to-state execution; route score construction and selection
remain separate inputs. Unsupported native/library requests fail with a
diagnostic instead of falling back silently. K1 causal prefill preserves the
SDPA causal fast path, while one-token cached decode avoids building a causal
mask; other cached alignments use an explicit bottom-right mask.

Every family now builds a typed `SemanticProgram` and passes it through
`UrmCompiler`. The unified plan pins an explicit trusted anchor override and
checks that the resulting dispatch step matches the executor binding. The K2
diagonal and K3 native executors check their runtime contracts before dispatch.
The K3 native executor additionally checks the runtime shape and serialized launch
schedule before dispatch, then reuses that checked binding by semantic shape.
These atomic mixer programs currently expose the
general compiler's base candidate; no K1/K2-specific algebraic rewrite rules
have passed the rewrite evidence gates.

## Named kernel recipes

| Recipe | Coverage rows | Kernel-level equation | Work outside the kernel |
|---|---|---|---|
| `mha`, `mqa`, `gqa` | arch-001–003; arch-014 uses the MHA K1 core after BitLinear projections and RoPE | Causal softmax attention | Q/K/V projections, positional transforms, cache ABI; BitAttention quantization and gradient surrogate |
| `pattention_core` | arch-057 | Parameter-token softmax attention with the source token-count multiplier | Parameter-token generation, GELU/L2 normalization variants, routed tokens and complete MLP/MoE mappings |
| `attnres_depth_core` | arch-054 | Residual-depth softmax with source RMS-normalized keys and a learned query; its library anchor invokes pinned FLA fused AttnRes | Transformer-layer scheduling and cross-layer gradient aggregation |
| `mla_attention_core` | arch-004 | Causal K1 attention after latent-KV expansion and RoPE concatenation; V padding is cropped after attention | Latent projections, compressed cache ABI and complete MLA layer |
| `fox` | arch-008 | FoX causal softmax from per-token log-decay gates; reference uses the pairwise cumulative-gate bias and the library anchor invokes pinned FLA `parallel_forgetting_attn` | Forget-gate production, projections, full layer and mode qualification |
| `moba_selected_attention_core` | arch-006 | MoBA local causal attention plus selected earlier blocks from caller-supplied routes | Block scoring, route selection and full-layer/cache integration; the library adapter invokes the pinned FLA sparse traversal |
| `bdh_attention_core` | arch-063 | Rotary strict-past unnormalized attention expressed as an additive K2 key-value state recurrence | BDH encoder/value projections, normalization, gated MLP, dropout, residual graph and full-model modes |
| `mom_selected_memory_core` | arch-051 | Headwise gated-delta recurrence with source Q/K L2 normalization and V-first matrix state | MoM routing/top-k, variable route packing, dispatch/merge, expert projections/convolutions and full layer/cache composition |
| `cat_attention_core` | arch-066 | CAT interleaved causal mask: prior compressed tokens plus the current local block | Chunk compression, adaptive/separator tokens, rotary transform, Q/K/V projections and complete decoder layer |
| `differential_attention_core` | arch-067 | Two causal softmax reductions with the Differential Transformer V1 subtraction coefficient; Microsoft V1 output/gradient errors 4.47e-8/3.64e-11 and median plan time -59.5%/-51.1% | Differential Q/K/V projections, RoPE, learned lambda vectors, per-head RMSNorm and output projection |
| `tda_attention_core` | arch-068 | Two normalized causal threshold-rectified score reductions with the source power and lambda; pinned Triton adapter exact, equation output max error 9.40e-5 and median plan overhead +2.6%/+1.8% | Threshold parameter production, Q/K/V projections, variable-length masks and full layer |
| `tpa_attention_core` | arch-069 | TPA CP-factorized Q/K/V, RoPE, causal K1 and output projection; exact FP32 output/gradient parity and +2.0%/-0.9% median overhead | Decoder cache, training schedule and full-model integration |
| `tucker_attention_core` | arch-070 | Tucker factorized-query softmax attention; BF16 output exact, maximum gradient error 1.16e-10 and +6.0%/+0.8% median overhead | Full Tucker projection frontend and cache expansion |
| `longformer_attention_core` | arch-071 | Bidirectional sliding-chunks local-window K1; output exact, maximum gradient error 2.84e-14 and +3.1%/-1.8% median overhead | Global-token and padding-mask branches, projections, full encoder and cache |
| `sparse_attention_core` | arch-072 | K1 attention under caller masks; pinned all/local/strided equations and fixed-mode layout/callback pass output/gradient checks, with source-vs-SDPA timing for each | Optimized TensorFlow 1 `BlocksparseTransformer` traversal still lacks a runnable comparator |
| `kata_attention_core` | arch-073 | Causal normalized-positive attention from summed squared group dot products; pinned BF16 Triton output/gradients exact and +5.3%/+1.4% median overhead at B4/T1024/H8/D64/M4 | KATA projections, variable-length path, GQA and cache integration |
| `conformer_attention_core` | arch-075 | ESPnet bidirectional MultiHeadedAttention SDPA path including Q/K/V and output projections; exact FP32 output/parameter gradients and +5.8%/+2.7% median overhead | Relative-position variants, Conformer convolution, FFN and full encoder residual graph |
| `hopfield_attention_core` | arch-078 | One unscaled softmax association update (`update_steps_max=0`); output error 7.57e-10, max gradient error 4.10e-12 and -67.0%/-45.0% median overhead | Iterative retrieval, learned scaling, pattern normalization and static-memory modes |
| `fwpkm_memory_read_core` | arch-056 | Fused Triton K1 softmax/value reduction over caller-produced selected product-key logits; exact pinned IDW/dot-product output and gradients, default IDW +9.0%/-0.6% median overhead | Fast-weight writes, address entropy, optimizer, chunk and cache behavior |
| `samba_attention_core` | arch-053 | Samba_421M_nope CausalSelfAttention at 12 heads and width 1536; exact output/gradient parity and -5.8%/-2.0% median overhead | Mamba/GLA/Retention alternatives, hybrid Block, cache, short-convolution/rotary variants, normalization and MLP |
| `h3_ssm_fft_core` | arch-076 | H3 head_dim=1 K2 adapter for both causal FFT convolutions, skip paths and q/k/v multiplication; output/gradients within configured tolerances and profiled against pinned `H3.forward` | head_dim>1, fused FFT convolution, recurrent cache and step inference |
| `hyena_fftconv_core` | arch-077 | Pinned order-2 Hyena implicit-filter causal FFT convolution; exact FP32 output/gradient parity and -4.6%/-3.3% median overhead including filter generation, short convolution, gating and projections | Higher-order filter routing, low-precision fused FFT and streaming/cache |
| `hla_second_order_core` | arch-074 | Exact masked second-order HLA summary scan from pinned paper Equation (3.3); Triton FP32 forward/reverse scan, output/gradient parity and -30.8%/-46.5% median overhead vs the dense equation at T=1024/H=8/D=16 | Higher/asymmetric/decayed HLA, chunk-parallel scheduling, and cache |
| `nsa_selected_attention_core` | arch-005 | Causal K1 attention under precomputed NSA block routes/counts | NSA compression, indexer, auxiliary branches and full layer |
| `dsa_attention_core` | arch-007 | Causal softmax attention under precomputed DSA token routes | DSA indexer objective, route selection, projections and full layer |
| `parallax_attention_core` | arch-012 | Causal Parallax correction `E[v] * (1 + E[r·k]) - E[(r·k)v]`; reference equation is explicit and the library anchor invokes pinned FLA `parallel_parallax` | Model-side secondary-query construction, positional transforms, full layer and mode qualification |
| `wall_attention_core` | arch-011 | Causal per-channel-decay Q/K score modulation; reference builds `scale * Σ(q_i k_j exp(cumsum(g)_i-cumsum(g)_j))`, and the library anchor invokes pinned FLA `parallel_wall_attn` | Gate production, optional scalar gate/sink/window branches, decode/cache and full layer |
| `linear_attention` | arch-015 | Feature-mapped additive state plus optional denominator | Architecture-specific feature map/projections |
| `based_attention_core` | arch-020 | Normalized causal Taylor-2 attention through additive polynomial feature state | Q/K/V projections and surrounding layer |
| `rebased_attention_core` | arch-021 | Normalized causal squared-dot attention through additive quadratic feature state | Q/K/V projections and surrounding layer |
| `retention_core` | arch-017 | Head-decayed additive matrix state | Multiscale bank and layer normalization |
| `lightning_attention_core` | arch-016 | Static per-head-decayed additive matrix state | Layer-index decay schedule and Q/K/V projections |
| `lightnet_gla_core` | arch-022 | Key-channel-decayed GLA recurrence after LightNet feature transforms | Projections, optional short convolutions, key normalization/gate construction and gated output layer |
| `simple_gla` | arch-018, arch-052 | Head-decayed additive matrix state; used by YOCO GatedRetention after its rotary/gate frontend | Gate and projection generation; YOCO shared-KV/cross-decoder cache composition |
| `gla` | arch-019 | Key-channel-decayed additive matrix state | Gate and projection generation |
| `hgrn_ssm_core` | arch-023 | Vector-state gated recurrence with a unit input/read gate | Projection and input gate production, normalization and output projection |
| `hgrn2_ssm_core` | arch-024 | Key-channel-decayed GLA matrix-state recurrence | HGRN2 projections and gates, value-first state layout and layer normalization |
| `delta_net`, `gated_delta_net` | arch-025–026 | Rank-one delta update with optional head decay | Input projections, source-specific key normalization, and gates |
| `gdn2_core` | arch-027 | Channel-decayed matrix state with separate key erase and value write gates | Gate and projection production, normalization and full layer |
| `kda_core` | arch-028 | Key-channel-decayed delta state with `1/sqrt(K)` query read | A_log/dt/bias gate generation and full-layer composition |
| `gated_delta_product_core` | arch-029 | Head-decayed ordered rank-R delta updates with `1/sqrt(K)` query read | Per-update projection/gate generation; pinned FLA chunk supports FP16/BF16 |
| `gated_oja_core` | arch-033 | Value-channel-decayed Oja update with key residual correction and a matrix state | Q/K/V/gate projections, gate production, and full-layer training/prefill/decode integration |
| `comba_core` | arch-037 | Head-decayed delta update with separate prediction and write keys | COMBA feature projections and full-layer training/prefill/decode integration |
| `pgdn_core` | arch-034 | Head-decayed gated delta with a second diagonal ATK metric state and nonlinear key preconditioning | ATK parameter generation, grouped value heads, and full-layer training/prefill/decode integration |
| `pkda_core` | arch-035 | Key-channel-decayed KDA with a second diagonal ATK metric state and nonlinear key preconditioning | KDA gate frontend, ATK parameter generation, and full-layer training/prefill/decode integration |
| `deltaformer_attention_core` | arch-013 | Strict-causal lower-triangular value correction followed by causal softmax attention | Q/K/V/beta frontend, full coupled layer, and training/prefill/decode qualification |
| `rodimus_gla_core` | arch-036 | BF16 key-channel GLA recurrence with V-first state and source-default 1/√K read scaling | Rodimus projections/gates, short convolution, normalization, output projection, nonzero cache state, and full-layer mode qualification |
| `abc_core`, `gsa_core` | arch-048–050 | Two-stage slot attention with per-slot decay, value summaries, and a softmax over slot reads; Raven dispatches to the GSA core after routing | Slot/gate and Q/K/V frontend, projections, and full-layer integration; GSA GQA is reference-covered but not source-profiled on the current GPU |
| `generalized_delta_iplr_core`, `generalized_delta_dplr_core` | arch-031–032 | Additive key-value write after explicit left low-rank `(I + beta alpha^T)` or diagonal-plus-low-rank transition | Source factor/projection generation, full layer, and mode-specific cache integration |
| `rwkv6_memory_core` | arch-041 | Key-channel-decayed state with the RWKV-6 static bonus read correction; the library anchor invokes pinned FLA `fused_recurrent_rwkv6` | Time-mix frontend, projections, output gate and final-state cotangent (not consumed by pinned upstream backward) |
| `momentum_delta_core` | arch-030 | Coupled fast-weight and momentum matrix-state recurrence; pinned FLA BF16 chunk adapter, with q/k/p normalization and p-times-alpha disabled | Default normalization modes, frontend transforms, projections, decode/cache and full layer |
| `mesa_net_core` | arch-038 | BF16 dual covariance state with per-token regularized key-space solve; equation max errors 1.22e-4 output, 2.83e-4 final state and 1.53e-5 input gradients, with exact pinned adapter parity and +3.8%/+1.2% median plan overhead | Nonzero initial-state gradients, streaming decode, model-side Q/K normalization and lambda construction, complete layer and architecture modes |
| `titans_linear_memory_core` | arch-039 | Chunked associative-memory update with a learned linear reconstruction loss; independent tokenwise equation at FP32 B1/T64/H1/D=16 has 3.05e-5 output, 1.43e-5 final-state, and at most 9.31e-4 input-gradient errors; exact pinned adapter and -0.9%/+0.1% median plan overhead | Outer attention, projections and gate production, memory hierarchy, complete block, and full-model cache modes; upstream callable is eager PyTorch rather than a fused kernel |
| `rwkv4_memory_core` | arch-040 | Stable max-shifted WKV recurrence with per-channel `(alpha, beta, eps)` state | Time-mix projections, channel mixing and output gate |
| `rwkv7_transition_core` | arch-042 | Additive KV write after `diag(exp(w)) + b a^T`, with unit-scale `r` read | Model-side factor production, projections, output gate and state/cache integration |
| `mamba1_ssm_core` | arch-043 | Diagonal selective state scan | Convolution, projections, activation and layer cache |
| `mamba2_ssm_core` | arch-044 | Decayed matrix-state recurrence core | Convolution, projections, chunk schedule and boundary state |
| `mamba3_siso_core` | arch-045 | Rotary angle accumulator with SSM matrix and previous K/V states; independent recurrence errors are 7.82e-5 output, 1.09e-2 maximum state and 2.61e-3 maximum input gradient; pinned adapter exact and +2.7%/+0.3% median plan overhead | MIMO/TileLang, input/output projections, A/dt/trap frontend, nonzero cache state, full layer and mode qualification |
| `log_linear_attention_core` | arch-009, arch-046 | Dyadic level-scaled matrix attention with 64-token chunk state | Mamba-2 projections and dt/level transforms, partial-chunk streaming cache and full layer |
| `sparse_delta_memory` | arch-047 | Ordered sparse-slot decayed delta update | Product-key route score construction and selection |

Coverage rows marked `kernel_prototype_only` in
`benchmarks/architecture-coverage.json` identify these exact equation cores.
The tests run forward and backward reference passes for every named kernel
recipe, exercise K1 mixed-precision SDPA gradients and causal prefill/decode
alignment, compare K2 FLA linear, delta, and gated-delta forward and gradients,
exercise FLA simple-GLA/GLA float32 chunk forward/backward and one-token decode
against the reference, verify MesaNet's equation, final states, and all input
gradients against pinned FLA, compare Mamba-3 SISO output and four states against
the pinned Triton source and independent recurrence, and execute K3's compiler-verified Triton forward and
backward path where supported. BDH's strict-past K2 recurrence matches the pinned
`Attention.forward` output and query/value gradients; its compiler library adapter
invokes that same pinned source method. MoM's per-routed-memory K2 recipe matches
pinned FLA output/state/gradient behavior with its L2-normalization and V-first
state flags. Architecture-specific training/prefill/decode
qualification and `native_parity_status` remain unchanged because full upstream
layer compositions and their exact comparator workloads are separate. Other rows
remain proposed or blocked until their additional state, routing, convolution,
nonlinear update, parameter-axis or model composition is implemented and
independently qualified.

## Qualification boundary

The three families use separate execution kernels because their equations
differ. K1 can use PyTorch's fused Flash SDPA dispatcher, supported K2 subsets
use chunked prefill or fused one-token recurrent kernels, and K3 uses the URM
Triton state kernel. K2 factored transitions, rank-R token updates, and
head/key-channel-decayed delta still execute through the eager reference path.
The native diagonal SSM scan supports float32, but is not speed-qualified
against Mamba's CUDA implementation. Simple GLA and GLA use additive updates with head and
key-channel gates respectively; float32 chunk prefill/training, BF16 GLA chunk
training for the Rodimus core, and low-precision one-token decode paths are
accelerated. Full-layer coverage remains open, and broader low-precision chunk
prefill/backward coverage, including Simple GLA, remains unqualified.
These paths establish usable kernel prototypes; they do not qualify full model
layers or all schedule/dtype combinations. Upstream parity and performance gates
are qualified only for rows with a linked pinned profile artifact.

## Paired upstream evidence

The factorized attention runner in
`benchmarks/unified_mixer_factorized_attention.py` records pinned TPA, Tucker,
Longformer, KATA, Conformer and Hopfield comparisons. TPA profiles its full T6
attention call; Tucker and Longformer cover their fused/local attention kernels;
KATA covers grouped normalized-positive Triton attention; Conformer covers its
bidirectional SDPA self-attention with projections; Hopfield covers one
association update with projections. The artifacts are
[`tpa-k1.json`](../../results/unified-mixer/tpa-k1.json),
[`tucker-k1.json`](../../results/unified-mixer/tucker-k1.json),
[`longformer-k1.json`](../../results/unified-mixer/longformer-k1.json),
[`kata-k1.json`](../../results/unified-mixer/kata-k1.json),
[`conformer-k1.json`](../../results/unified-mixer/conformer-k1.json), and
[`hopfield-k1.json`](../../results/unified-mixer/hopfield-k1.json). All six
measured profiles pass output/gradient parity and the 10% paired median latency
gate for both forward and forward/backward at their recorded shapes. These are
source-specific slices; the coverage register keeps full-layer and mode work
separate.

`benchmarks/unified_mixer_fwpkm.py` profiles the pinned FwPKM `retrieve_values`
call against its product-key route plus URM's fused K1 selected-read reducer.
The default IDW case passes output/input-gradient parity and the 10% median
overhead gate at FP32 B4/T256/H2/D=64/V=64/topk=8. Fast-weight writes and
chunk/cache update behavior are outside this read profile.

`benchmarks/unified_mixer_samba.py` profiles the exact pinned
`Samba_421M_nope` attention layer at B1/T512/H12/D128, including source QKV and
output projections. Output and all input/parameter gradients match exactly;
median plan overhead is -5.8% forward and -2.0% forward/backward. The enclosing
hybrid Block and its Mamba/cache path remain outside this comparison.

`benchmarks/unified_mixer_h3.py` profiles pinned H3 at FP32 B1/T128/D64/state16
with head_dim=1 and the source PyTorch FFT path. URM independently reproduces
both causal convolutions and their skip/multiplicative composition; output and
input/parameter gradients pass, and both timing modes pass the 10% median gate.
Head_dim>1 and inference cache/step remain outside this prototype.

`benchmarks/unified_mixer_flash.py` compares MHA, MQA and GQA plans with
`flash_attn_func` from FlashAttention revision
`1bda8f9290cd48d030f1516f0e680cd464ef3554`. On the A10G, BF16 causal 64-token
fixtures with four query heads and 32-wide features, output and input gradients
match exactly for MHA/MQA; GQA has a 0.00049 maximum output error and sub-
`5e-7` gradient errors. The compiler plan is forced to PyTorch's Flash SDPA
backend. Across 21 paired samples, forward median overhead is 1–5% and
forward/backward median time is 10–11% lower than the direct source call.

The FlashAttention checkout did not have a prebuilt extension for this Torch
and CUDA runtime. The profile uses an extension built from the exact pinned
kernel sources, narrowed to BF16 causal head dimension 32. The artifact records
the source hashes and extension hash; this qualifies the measured slice only.

`benchmarks/unified_mixer_fla.py` compares the library-backed compiler plan with
direct callables from FLA revision
`864a87f6ce5be8828bef81eb22baafd41937cdf2`. The saved A10G profile covers
BF16 linear attention, DeltaNet and Gated DeltaNet; FP32 Simple GLA, GLA,
LightNet's GLA core and HGRN2 at batch 1, sequence 64, four heads and 32-wide key/value dimensions; FP32
Lightning Attention at sequence 1024; and FP32 Retention at sequence 4096. It
checks output, final state and input gradients, then alternates 21 direct/compiled
measurements for forward and forward/backward. All nine kernel cases pass parity
and both median-overhead modes stay within 10%. Retention and Lightning also
compare their layer-facing wrappers with the prepared-schedule kernel calls; their
outputs and final states match exactly. Because both timing sides invoke the same
pinned FLA kernel, these results measure compiler-plan overhead, not a new kernel
speedup. The artifact records cold compilation separately. Full-layer projection,
normalization and schedule costs, broader dtypes and other execution modes remain
unqualified.

The separate HGRN profile in `benchmarks/unified_mixer_hgrn.py` compares the
compiler plan with `fused_recurrent_hgrn` from the same FLA revision. At FP32
batch 1, sequence 1024 and width 1024, output, final-state and input-gradient
errors are zero. Across 21 paired samples, median overhead is 1.1% for forward
and 0.1% for forward/backward. This measures plan overhead around the same
recurrent kernel; input projections, gate production, normalization and output
projection remain outside the kernel slice.

The runtime identity check accepts either the frozen FLA 0.5.2 package/module
pair or the exact source revision above. It records both the imported module
version and installed distribution version because the pinned checkout reports
module version 0.6.0 while the environment's distribution metadata says 0.5.2.
Low-precision FLA chunk training with odd key or value dimensions is rejected
with a clear contract error; the pinned upstream backward kernels require even
dimensions for those dtypes.

ABC and GSA compare BF16 batch 1, sequence 128, 2 heads, K=V=32 and 16 slots
with pinned `chunk_abc` and `chunk_gsa`. The independent equations have maximum
output errors 4.88e-4/3.91e-3, final-state errors 7.42e-5/1.53e-3 and supported
input-gradient errors no larger than 4.51e-7/7.63e-6; both adapters match the
source exactly. Median plan overhead is +3.9%/+1.6% for ABC and +4.6%/+2.9%
for GSA in forward/forward-backward. The output-only source backward omits
initial-state gradients. Pinned GSA GQA faults on the available A10G, so its
source profile uses equal query/KV heads; the independent equation supports
GQA. Slot/gate frontends and complete layers remain outside this comparison.

`benchmarks/unified_mixer_sdm.py` compares the unified native K3 plan with the
direct `SparseDeltaMemory.gated_write_read` operator from SDM revision
`183e7df809131b80ad4393741029d0f20fc3640b`. The saved A10G profile covers
batch 1, 16 tokens, 256 slots, 32-wide values and four read/write routes in
FP32 and BF16. Writes repeatedly collide across tokens while read routes are
disjoint. Output, final state and all floating input gradients pass the pinned
comparison tolerances. Native K3's paired medians are 17–20% lower for forward
and 41–44% lower for forward/backward over the direct source operator, with
21 pairs per mode and dtype. The measurements include route validation and
compiler-plan dispatch, but exclude product-key score generation and layer
projections.

The SDM source extension was built with the matching CUDA 13 compiler, runtime
headers and CCCL package before paired timing. A separate native-versus-reference
regression now covers repeated writes and overlapping reads/writes for both
before-update and after-update reads in FP32 and BF16; output, state and all six
input gradients pass. Against the pinned SDM custom backward on the same FP32
overlap fixture, output and state pass, but maximum input-gradient errors are
1.69e-4 for read weights, 1.36e-4 for write weights and 1.90e-4 for log decay,
above the frozen 3e-5 absolute / 3e-4 relative limits. The qualified SDM profile
therefore retains disjoint read routes; overlapping-route parity with that
upstream VJP remains blocked by the source comparator. The reproducible parity
diagnostic is saved in
[`sdm-k3-overlap-diagnostic.json`](../../results/unified-mixer/sdm-k3-overlap-diagnostic.json);
it records the native/reference pass, upstream component errors and the fact
that it contains no timing measurements.

`benchmarks/unified_mixer_mamba.py` compares the compiler's Mamba-1 library plan
with `selective_scan_fn` from Mamba revision
`e9594ce1c732d97440f0332fdc43170a2294dbfa`. The saved A10G profile uses FP32,
batch 1, sequence 1024, 256 channels and state width 64. Output, final state and
all input gradients match exactly for 21 paired samples. Median plan overhead is
4.2% for forward and 6.3% for forward/backward, inside the 10% gate. The library
adapter invokes the same pinned CUDA operator as the direct path, so the timing
measures compiler dispatch overhead rather than a new kernel speedup. The
separate URM Triton diagonal scan passes targeted equation and gradient parity,
but remains slower than the pinned CUDA scan and is not the selected Mamba
profile backend. Convolution, projections, activation gate and cache ABI remain
outside this kernel comparison.
