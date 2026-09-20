# Named architecture coverage and comparison register

This is the production construction backlog, not a claim of full architecture support.
Inclusion commits us to resolve the mapping and attempt a fair comparison. Each
architecture has a pinned comparator, a documented execution blocker, or an
explicit non-mixer classification. None is
silently dropped because it does not fit the first three kernels.

The [machine-readable register](../../benchmarks/architecture-coverage.json) records
IDs, source revisions, proposed lowerings, mode qualification and remaining work.
Revisions captured through 2026-09-20 are audit/comparison candidates; they do not replace
older frozen acceptance pins. Every run must additionally pin its actual callable,
dependencies, shapes and tolerances. Paired GPU kernel-slice comparisons are now
recorded across the pinned FlashAttention, FLA, Mamba, SDM and TokenFormer source
groups. The FlashAttention comparison covers MHA, MQA and GQA at BF16, causal sequence
64 and head dimension 32 in [`results/unified-mixer/flash-k1.json`](../../results/unified-mixer/flash-k1.json).
Their outputs and gradients pass against the exact FlashAttention source; forward
median plan overhead is 1–5%, and forward/backward is 10–11% faster. MLA is
profiled separately in [`results/unified-mixer/mla-k1.json`](../../results/unified-mixer/mla-k1.json)
after its latent expansion and RoPE concatenation, with a 16-wide V padded to
32 for the kernel call and cropped afterward. Output and input gradients match
exactly; median plan overhead is +7.7% forward and -9.4% forward/backward.
Those `flash-k1.json` measurements qualify the library/SDPA plan only. URM's
separate native tiled online-softmax implementation is measured in
[`results/unified-mixer/native-k1.json`](../../results/unified-mixer/native-k1.json)
against pinned FlashAttention revision
`1bda8f9290cd48d030f1516f0e680cd464ef3554`. On an A10G with BF16 B1/T64,
Hq=4, D=V=32, native output and Q/K/V gradients pass for MHA, MQA and GQA.
Twenty-one paired CUDA-graph samples pass the 10% kernel gate: native forward
is about 24% faster and forward/backward is 29–39% faster by paired medians.
The ordinary per-call path remains 118–128% slower for forward and 29–36%
slower for forward/backward because Python dispatch and
allocation dominate these short cases. The locally narrowed FlashAttention
extension supports only equal-length causal D=32 prefill, so decode and other
head dimensions have no upstream profile. Full-layer, cache and end-to-end
training/inference qualification remain open.
The FLA comparison covers nine K2 recipes in
[`results/unified-mixer/fla-k2.json`](../../results/unified-mixer/fla-k2.json).
All nine cases pass output, final-state and input-gradient parity against the
exact FLA revision, and both timing modes stay within the 10% median-overhead gate.
Retention and Lightning Attention also compare their layer-facing wrapper with
the precomputed-schedule kernel call; outputs and final states match exactly.
The separate HGRN profile in
[`results/unified-mixer/hgrn-k2.json`](../../results/unified-mixer/hgrn-k2.json)
also passes output, final-state and input-gradient parity at FP32 batch 1,
sequence 1024 and width 1024; both timing modes pass the 10% overhead gate.
The K3 profile in [`results/unified-mixer/sdm-k3.json`](../../results/unified-mixer/sdm-k3.json)
passes the same parity components for FP32 and BF16 against pinned SDM. On the
measured A10G fixture, native K3 is 17–20% faster for forward and 41–44% faster
for forward/backward. That fixture has repeated write collisions and disjoint
read routes. A separate regression verifies native K3 output, state and all six
input gradients against the corrected equation reference with overlapping read
and write routes, in both read-timing modes and FP32/BF16. Pinned SDM output and
state also pass that FP32 overlap case, but its custom backward exceeds the
frozen gradient tolerance for read weights, write weights and log decay. The
component errors and tolerances are recorded in
[`sdm-k3-overlap-diagnostic.json`](../../results/unified-mixer/sdm-k3-overlap-diagnostic.json).
The upstream-comparison limitation is specific to its VJP on overlapping routes.
The Mamba-1
profile in [`results/unified-mixer/mamba-k2.json`](../../results/unified-mixer/mamba-k2.json)
compares FP32 selective scan at batch 1, sequence 1024, 256 channels and state
width 64. Output, final state and input gradients match the pinned CUDA extension
exactly; paired overhead is within 7% for forward and forward/backward. That
compiler library anchor invokes the same pinned source operator, so the timings
measure plan overhead rather than a new kernel's speed. These qualify kernel
paths against direct upstream operators; they do not qualify full architecture
layers or the other register rows. The Mamba-2 SSD profile compares FP32 at
batch 1, sequence 256, 4 heads, head width 16 and state width 64. Output, final
state and all six input gradients match the pinned source exactly; paired median
overhead is +4.4% forward and -0.8% forward/backward. Its adapter also invokes
the pinned source operator, so this measures plan overhead. Mamba-3 SISO compares
the rotary angle accumulator and four-state trapezoidal recurrence at BF16
B1/T64/H2/K=V=16 against pinned `mamba3_siso_combined`. The independent
equation has 7.82e-5 output, 1.09e-2 maximum final-state and 2.61e-3 maximum
input-gradient errors; the compiler adapter is exact. Paired median plan
overhead is +2.7% forward and +0.3% forward/backward. MIMO/TileLang, nonzero
initial states, projections, full-layer and cache integration remain open.
MesaNet compares the
BF16 dual-state recurrence and regularized solve at B1/T64/H2/K=V=16 against
pinned `chunk_mesa_net`; the independent equation has 1.22e-4 output, 2.83e-4
final-state and 1.53e-5 maximum input-gradient errors, while the library adapter
is exact. Paired median plan overhead is +3.8% forward and +1.2%
forward/backward. Nonzero initial-state gradients, streaming decode, model-side
normalization/lambda construction and full-layer integration remain open.
Titans compares its pinned `chunk_titans_linear_ref(use_chunk=True)` memory
operator at FP32 B1/T64/H1/D=16 with chunk size 16. The independent tokenwise
equation has 3.05e-5 output, 1.43e-5 final-state and at most 9.31e-4
input-gradient error (3e-3 absolute / 1e-2 relative gradient tolerance); the
pinned adapter is exact for output, state and all gradients. Across 21 paired
measurements, median plan overhead is -0.9% forward and +0.1%
forward/backward. The FLA Titans callable is an eager PyTorch chunk operator,
not a fused Titans GPU kernel; the profile measures wrapper/plan overhead.
Outer attention, projections, memory hierarchy and full-model/cache modes remain
outside this memory-core slice.
TTT-Linear compares the pinned BF16 chunk operator at B1/T64/H2/D=16, chunk
size 16, with nonzero matrix and bias states. The independent FLA naive
equation passes configured output/state/gradient tolerances (maximum errors:
1.55e-2 output, below 8e-7 state, and 1.56e-2 gradient); the pinned chunk
adapter matches outputs, both states and all gradients exactly. Across 21
paired measurements, median plan overhead is +3.1% forward and -0.2%
forward/backward. The profile measures compiler-plan overhead around the
upstream chunk operator. TTT's MLP variant, full layer, higher-order gradients
and cache modes remain outside this slice. The pinned FLA GDN-2 comparison adds a separate-gate recurrence
case at FP32 batch 1, sequence 64, 2 heads, key/value width 16. Output, final
state and all seven input gradients match exactly; plan overhead is +4.4%
forward and +3.0% forward/backward. Its gates and full-layer projections remain
outside the measured operator.
The XMA nonlinear recurrence profiles compare fixed-length FP32 B1/T64/H=1/D=16
cores with the exact pinned Triton operators. RNN, GRU and M2RNN equation
references match XMA's torch backend within 3e-6 output/state/gradient
tolerances; their Triton adapters also match output, final state and all input
gradients within 3e-6. Across 21 paired measurements, median plan overhead is
RNN +3.7%/+0.4%, GRU +3.2%/+1.5% and M2RNN +3.9%/+0.9% for forward and
forward/backward. These timings measure compiler-plan overhead around the XMA
kernels. Multi-head replication, packed variable-length support, gradient
clipping and model integration remain open.
The ATMA K1 profiles cover the direction and magnitude outputs at FP32
B1/T64/H2/D=16. POLAR's equation and adapter match output, magnitude and all
input gradients exactly; median plan overhead is +4.2%/+0.6% forward and
forward/backward. Foveal also matches exactly with page size/local window 16
and supplied remote routes; median overhead is +5.8%/+0.4%. These are timings
around the pinned Triton kernels. Full-layer projections and Foveal route
generation remain external.
ATMA's separate K2 decode kernel is now bound as a forward-only compiler
anchor. At FP32 B4/T1/H8/K=V=128, output and updated slot state match the
independent recurrence within 4.5e-8 and the compiler adapter matches the
pinned ATMA kernel exactly. Across 21 paired CUDA graph replays, median plan
overhead is -0.7%, passing the 10% gate. Eager Python dispatch adds 22.7% on
this short step, so this performance qualification applies to the CUDA-graph
serving path documented by ATMA; eager low-batch dispatch remains outside the
gate. Q/K/V and gate projections, RMSNorm, output projection and the full
global block remain external. See
[`atma-gated-delta-decode-k2.json`](../../results/unified-mixer/atma-gated-delta-decode-k2.json).
MoBA's supplied-route BF16 B1/T128/H2/D=32, block-size-32 profile matches the
pinned `parallel_moba` output and gradients exactly. The narrow FlashAttention
build enables both causal local attention and noncausal routed-block calls;
its source revision, kernel hashes, dispatcher hash and extension hash are
recorded in [`results/unified-mixer/moba-k1.json`](../../results/unified-mixer/moba-k1.json).
Across 21 paired samples, compiler-plan overhead is +1.1% forward and +0.6%
forward/backward. The compiler's library adapter invokes the same pinned FLA
operator; this is plan overhead, not a new sparse kernel speedup. Block scoring,
route selection and full-layer/cache integration remain external. All 76
mixer-relevant architecture rows now have measured kernel-upstream parity and
profiling evidence; four catalog rows are classified outside mixer scope.
The FLA Based and ReBased profiles compare FP32 batch 1, sequence 64, 2 heads
and key/value width 8. Their outputs and Q/K/V gradients match exactly. Plan
overhead is +6.7%/+2.4% forward/forward-backward for Based and
+6.2%/+0.7% for ReBased. Both compiler adapters invoke their pinned FLA
operator; these timings measure plan overhead around those source kernels.
The LogLinear profile compares the dyadic attention core at FP32 batch 1, sequence 70, 2 value heads, key width 64 and value width 16. The reference equation passes output, all seven chunk-state fields and all five input gradients against pinned `chunk_log_linear_attn` (maximum absolute errors: 2.7e-5 output, 1.5e-4 state, and 1.5e-3 gradient). Plan overhead is +1.8% forward and -0.5% forward/backward. This qualifies the 64-token chunk core for both Log-linear attention and the LogLinearMamba2 kernel slice; the Mamba frontend, full layer, streaming decode and variable-length cache remain open.
The KDA profile compares FP32 batch 1, sequence 64, 2 heads, key width 64 and value width 16. Its reference recurrence passes output, final-state and all six input-gradient comparisons against pinned `chunk_kda` (maximum errors: 2.0e-4 output, 1.4e-4 state, below 3e-7 gradients). Plan overhead is +2.1% forward and -1.4% forward/backward. A_log/dt/bias gate construction and the complete KDA layer remain external.
The Gated DeltaProduct profile compares BF16 batch 1, sequence 64, 2 heads,
key width 16, value width 8 and update rank 2. The ordered reference recurrence
matches pinned `chunk_gated_delta_product` within 3.91e-3 output, 4.31e-4 state
and 2.1e-5 across its six active gradients. The FLA adapter matches that direct
call exactly. Paired plan overhead is +4.4% forward and -0.4% forward/backward.
Per-update projection/gate production and full-layer integration remain open.
The generalized delta profiles cover the additive IPLR and DPLR equations with
source-factor tensors supplied at the kernel boundary. IPLR uses the FP32 recurrent
operator at batch 1, sequence 256, 2 heads, K=V=32; equation errors are
8.3e-5 output, 1.03e-4 final state and at most 2.7e-6 across input gradients.
Its adapter matches pinned FLA exactly. DPLR uses BF16 chunk execution at the
same shape; equation errors are 9.77e-4 output, 2.78e-4 final state and below
5e-7 across input gradients, while its adapter also matches exactly. Both use
21 paired measurements; their median plan overheads are +7.3%/+2.5% (IPLR) and
+6.6%/+0.9% (DPLR) for forward/forward-backward. The low-rank factors are runtime
inputs; their model-side production and complete layers are not included.

RWKV-4 is measured against the pinned recurrent operator at FP32 batch 1,
sequence 1024 and 512 channels. The max-shifted scalar-state reference has
4.47e-8 output, 4.77e-6 final-state and 4.33e-7 maximum input-gradient error;
the library adapter is exact. Median plan overhead is +2.1% forward and +2.1%
forward/backward. The compiler now keeps its `(alpha, beta, eps)` state, while
time-mix projections, channel mixing and output gates remain external.

RWKV-7 is measured against FLA's `chunk_rwkv7` at BF16 batch 1, sequence 256,
2 heads and K=V=32. It uses the source DPLR transition and unit-scale `r` read;
the reference equation has 7.81e-3 max output, 2.79e-4 final-state and 7.63e-6
maximum input-gradient error, and the pinned adapter is exact. Median plan
overhead is +6.0% forward and +1.7% forward/backward. Frontend factor production,
projections, output gate and cache integration remain external.

AttnRes also passes in [`results/unified-mixer/attnres-k1.json`](../../results/unified-mixer/attnres-k1.json): the K1 equation has 3.91e-3 max output error and 3.82e-6 max gradient error, while its pinned adapter matches output and gradients exactly. Median plan overhead is +8.9% forward and +3.4% forward/backward at BF16 depth 8, B1/T256 and width 128. The transformer-level schedule and cross-layer gradient aggregation remain outside the kernel slice.

TokenFormer Pattention is measured in its `softmax` normalization mode in [`results/unified-mixer/pattention-k1.json`](../../results/unified-mixer/pattention-k1.json). At BF16 B1/T128 with 256 parameter tokens and width 32, max output error is 4.88e-4 and max input-gradient error is 1.53e-5. The K1 plan is 0.8% faster forward and 22.3% faster forward/backward by paired medians. GELU/L2 modes, parameter initialization and routed experts are not covered.

RWKV-6 is measured against pinned `fused_recurrent_rwkv6` at FP32 B1/T512, 2 heads and K=V=32. The reference equation max errors are 1.49e-8 output, 1.49e-7 final state and below 2e-13 across input gradients; the compiler adapter is exact. Median overhead is +5.8% forward and +0.9% forward/backward. The upstream backward does not consume a final-state cotangent, so gradient parity uses output loss; frontend time-mix and projections remain external.

PaTH attention is measured against pinned `parallel_path_attn` at BF16 B1/T128, 4 query/2 key-value heads and K=V=32. The independent chunkwise triangular-transform equation has 9.77e-4 maximum output error and below 4.77e-7 maximum input-gradient error; the compiler adapter matches exactly. Paired median overhead is +3.8% forward and +1.5% forward/backward. Model projections, w short convolution, optional q/k normalization and cache/decode remain open.

Momentum DeltaNet is measured against pinned `chunk_momentum_delta_rule` at BF16 B1/T256, 2 heads and K=V=32, with normalization and p-times-alpha disabled and a stable 0.10–0.12 update gate. The equation has 4.9e-4 maximum output error, 1.82e-3/4.86e-4 final-state errors and at most 2.45e-4 input-gradient error; the compiler adapter matches exactly. Paired median overhead is +0.1% forward and -0.4% forward/backward. Default q/k/p normalization, p-times-alpha, frontend transforms, projections and the full layer remain to be qualified.

Gated Oja is measured against pinned `chunk_gated_oja_rule` at BF16 B1/T256,
2 heads and K=V=16, with FP32 value-channel log-decay and beta inputs. The
independent recurrence has 6.11e-5 maximum output error, 4.61e-4 final-state
error and at most 5.74e-7 input-gradient error; the compiler adapter matches
exactly. Paired median overhead is +3.1% forward and +1.9% forward/backward.
Model-side projections, gate production, prefill/decode integration and the full
layer remain open.

ABC and GSA are measured against pinned `chunk_abc` and `chunk_gsa` at BF16
B1/T128, 2 heads, K=V=32 and 16 slots. Independent recurrence maximum output
errors are 4.88e-4 for ABC and 3.91e-3 for GSA; maximum two-state errors are
7.42e-5 and 1.53e-3. Supported input-gradient errors are at most 4.51e-7 and
7.63e-6, and each compiler adapter matches the pinned operator exactly. Paired
median overheads are +3.9%/+1.6% for ABC and +4.6%/+2.9% for GSA, forward and
forward/backward respectively. These output-only comparisons follow the source
backward contract, which omits initial-state gradients. The GSA comparator uses
equal query/KV heads because its grouped-query value-state kernel faults on the
available A10G; the independent reference supports grouped heads, which remain
unqualified against that source kernel. Slot/gate frontends and full layers
remain external.

The source audit confirms Raven calls the same GSA chunk/recurrent operators;
its precomputed router and decay tensors use the arch-049 kernel profile at
equal query/KV heads. YOCO GatedRetention calls the Simple GLA operator after
its rotary and gate frontends; its kernel maps to the FP32 Simple GLA profile at
B1/T64, H=4 and K=V=32, where output, state and gradients match exactly and
median overhead is +7.5%/+3.3%. Raven's route generation and YOCO's shared-KV,
cross-decoder/cache and output stages are still open.

The DSA comparison pins the selected-token mask and compares the attention
operator at BF16 batch 1, sequence 128, 2 heads, key width 32, value width 16
and top-k 16. The K1 equation matches FLA's `naive_dsa` output and active
gradients exactly; the SDPA backend has 7.81e-3 maximum output error and
7.63e-6 maximum input-gradient error. The 21-pair median time is 58.5% lower
for forward and 60.1% lower for forward/backward against the naive callable.
This measures supplied-route attention; the DSA indexer objective and learned
token selection remain external.

The NSA comparison measures the BF16 selected-block branch at batch 1, sequence
128, 16 query heads, 1 KV head, K=V=32 and block size 32, using one or two
precomputed causal routes per token. The K1 equation has 7.81e-3 maximum output
error and at most 7.63e-6 input-gradient error against pinned
`parallel_nsa`; 21-pair median time is 30.0% lower forward and 41.8% lower
forward/backward. The compression, indexer, other NSA branches and full layer
remain outside this selected-only comparison.

FoX is measured against pinned FLA `parallel_forgetting_attn` at BF16 batch 1,
sequence 128, 4 heads and K=V=32 with per-token log-decay gates. The independent
pairwise-bias equation has 7.81e-3 maximum output error and 1.53e-5 maximum
input-gradient error; the library adapter matches the source exactly. Across
21 paired samples, plan overhead is +5.0% forward and +0.0% forward/backward.
Gate production, projections, full-layer integration and other modes remain
unqualified.

Parallax is measured against pinned FLA `parallel_parallax` at BF16 batch 1,
sequence 128, 2 heads and K=V=32. Its explicit secondary-query correction
equation has 7.81e-3 maximum output error and 1.53e-5 maximum input-gradient
error; the library adapter matches the source exactly. The 21-pair median plan
overhead is +5.7% forward and +0.2% forward/backward. Model-side secondary-query
construction, positional transforms, full-layer integration and other modes
remain unqualified.

Wall attention is measured against pinned FLA `parallel_wall_attn` at FP32
batch 1, sequence 64, 2 query heads/1 KV head, K=32 and V=16. The independent
per-channel decay score equation has 4.77e-7 maximum output error and
1.17e-9 maximum input-gradient error; the library adapter matches exactly.
Across 21 paired samples, plan overhead is +4.1% forward and +0.5%
forward/backward. The optional scalar gate, sink, window, decode/cache and full
layer paths remain unqualified.

All architecture labels now have a resolved source identity or an explicit
non-mixer classification. All 76 mixer-relevant rows have measured equation or
source-kernel parity and performance evidence; four entries
are outside K1/K2/K3 mixer scope.
CAT now resolves to FLA's Compress And Attend Transformers
implementation at the pinned FLA revision. Its BF16 K1 profile uses the exact CAT
causal `BlockMask` for two decoder blocks plus the final special-token pair at
B1/T22/H2/D=16. The SDPA plan has exact output parity with source FlexAttention,
maximum input-gradient error 1.49e-8, and independent equation output error
9.77e-4. Across 21 paired samples, relative median latency is -2.6% forward and
-12.0% forward/backward. Compression and frontend work remain outside the measured
attention core. Differential Attention is measured against Microsoft's pinned
V1 source, separating its lambda and normalization contract from V2. On the
preprojected FP32 B1/T64/H2/D=16 K1 slice, output error is 4.47e-8 and maximum
input-gradient error is 3.64e-11; the plan's paired median latency is 59.5%
lower forward and 51.1% lower forward/backward. RoPE, learned lambda production,
per-head normalization, output projection and cache modes remain outside this
slice.
TDA is measured against the official pinned Triton implementation at FP32
B1/T64/H2/D=32, beta 0.7 and lambda 0.35: the source adapter is exact, while
the independent threshold equation has 9.40e-5 maximum output error and at most
4.94e-8 input-gradient error. Across 21 pairs, plan overhead is +2.6% forward
and +1.8% forward/backward. The other unmeasured rows retain their pinned comparator and
explicit kernel or architecture work. The register gives every row separate
kernel-upstream parity and profiling statuses. BDH is now measured as a K2
strict-past rotary attention core against its pinned `Attention.forward` call.
Across 21 paired samples, its independent equation has 2.98e-8 maximum output
error and at most 2.91e-11 query/value gradient error. The pinned adapter is
exact; median plan overhead is +4.5% forward and +1.9% forward/backward. The
full BDH projection, normalization, gated-MLP, dropout and residual graph remains
outside the measured core. MoM now has a BF16 per-routed-memory K2 profile: its
equation has 9.77e-4 output, 4.97e-4 final-state and 5.07e-7 maximum gradient
errors; the pinned adapter is exact, with median plan overhead +0.8% forward and
+1.4% forward/backward. Router scores, top-k dispatch, route packing/merge and
memory-expert composition remain external. The new factorized-attention profiles
cover TPA, Tucker, Longformer and KATA. KATA uses a BF16 B4/T1024/H8/D=64/M=4
fixture ([profile](../../results/unified-mixer/kata-k1.json)): pinned outputs and gradients are exact, the independent equation has
9.77e-4 maximum output error, and median plan overhead is +5.3% forward/+1.4%
forward/backward. The profile uses the pinned KATA Triton forward and backward;
model projections, variable-length batches and cache integration remain open.
Conformer standard bidirectional self-attention is measured through ESPnet's
pinned `MultiHeadedAttention` with its SDPA path and complete Q/K/V/output
projections at FP32 B2/T1024/H4/D=32. Output and all input/parameter gradients
match exactly; overhead is +5.8%/+2.7% ([profile](../../results/unified-mixer/conformer-k1.json)). Relative-position, convolution and FFN
parts of the full Conformer layer remain outside this slice. Hopfield's pinned
single retrieval update (`update_steps_max=0`, scaling 1.0) matches output within
7.57e-10 and gradients within 4.10e-12 at FP32 B2/T256/H4/D=32; median overhead
is -67.0%/-45.0% ([profile](../../results/unified-mixer/hopfield-k1.json)). Iterative retrieval and other Hopfield modes remain open.
FwPKM's pinned `retrieve_values` also matches output and gradients for both
product-key dot-product and IDW scoring. Its default IDW profile at FP32
B4/T256/H2/D=64/V=64/topk=8 passes the paired gate at +9.0% forward and -0.6%
forward/backward ([profile](../../results/unified-mixer/fwpkm-k1.json)). The
online fast-weight update and chunk/cache contract remain outside this read core.
H3's pinned `H3.forward` matches through both K2 causal FFT convolutions,
skip paths and the multiplicative Q/K/V interaction at FP32 B1/T128/D64/state16
with head_dim=1. Output error is 7.45e-9 and maximum gradient error is
7.28e-12; median plan overhead is -4.8%/-2.7% ([profile](../../results/unified-mixer/h3-k2.json)).
Head_dim>1 and inference cache/step remain open.
Hyena's pinned order-2 `HyenaOperator.forward` matches the unified K2 causal FFT
convolution exactly at FP32 B1/T128/D32/filter-width 16, including gradients
through the source implicit-filter generator, short convolution, gating and
projections. Its paired median latency is 4.6% lower forward and 3.3% lower
forward/backward ([profile](../../results/unified-mixer/hyena-k2.json)). Higher
orders, low-precision fused FFT and streaming/cache remain open.
HLA's masked second-order streaming summary kernel matches the exact recurrence
in pinned paper Equation (3.3), including input gradients. At FP32
B1/T1024/H8/D=16, its Triton forward/reverse scan is 30.8% faster forward and
46.5% faster forward/backward than a dense transcription of that equation
([profile](../../results/unified-mixer/hla-k2.json)). The pinned HLA repository
contains the paper but no executable operator; higher orders, asymmetric/decayed
variants, chunk-parallel scheduling and cache remain open.

The unified compiler has executable kernel-level prototypes for all 76
mixer-relevant rows. Those
records use `prototype_status: kernel_prototype_only`: they identify a reusable
equation core and list the surrounding architecture work that remains. They do
not change `mapping_status`, `native_parity_status` or any architecture mode
qualification. Every named prototype recipe now passes the compiler pipeline
and has forward/backward reference coverage. The register lists its selected
compiler anchors and representative test. Accelerated paths cover SDPA
attention (including TPA, Tucker, Longformer local windows, KATA normalized
positive attention, Conformer self-attention and one-step Hopfield association),
ATMA POLAR/Foveal reduction cores, FLA linear/delta/gated-delta subsets, one-token Simple GLA/GLA decode,
float32 Simple GLA/GLA chunk training, static-head-decay Retention and Lightning
Attention, the BitAttention K1 core, LightNet GLA, FLA HGRN/HGRN2, DeltaFormer,
Rodimus GLA, MesaNet, Titans' chunked memory operator, the TTT-Linear chunk
adapter, the FwPKM selected-read reducer, XMA RNN/GRU/M2RNN nonlinear recurrences,
the pinned Mamba-1/Mamba-3 SISO scan adapters, H3 and Hyena FFT mixers, and URM
native sparse state. A native Triton diagonal scan also passes targeted Mamba
equation and gradient parity, but is not the profile-qualified Mamba backend.
See the [unified mixer prototype](../compiler/unified-mixer.md).

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
- **sparse_transformer**: [openai/sparse_attention](https://github.com/openai/sparse_attention/tree/c53f3bdbf6225be0582f0357072e82b13c69be7d); pin `c53f3bdbf622` (legacy TensorFlow/blocksparse runtime unavailable).
- **hla_higher_order**: [yifanzhang-pro/HLA](https://github.com/yifanzhang-pro/HLA/tree/484fef2bb40d4ed58f7656e545cf5ef64c40c962); pin `484fef2bb40d` (project page only; no operator code).
- **conformer**: [espnet/espnet](https://github.com/espnet/espnet/tree/2950325ea62c8052f448aaf11affdabe169ec8ab); pin `2950325ea62c`.
- **h3**: [HazyResearch/H3](https://github.com/HazyResearch/H3/tree/5c4d06b5795405170387c80998b58d76179a8a1a); pin `5c4d06b57954`.
- **safari**: [HazyResearch/safari](https://github.com/HazyResearch/safari/tree/02220c69d247e5473616cd053a443ad99fd2559b); pin `02220c69d247`.
- **hopfield**: [ml-jku/hopfield-layers](https://github.com/ml-jku/hopfield-layers/tree/f56f929c95b77a070ae675ea4f56b6d54d36e730); pin `f56f929c95b7`.
- **maml**: [cbfinn/maml](https://github.com/cbfinn/maml/tree/a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49); pin `a7f45f1bcd74` (training algorithm, not a mixer).
- **reptile**: [OpenAI supervised-reptile](https://github.com/openai/supervised-reptile/tree/8f2b71c67a31c1a605ced0cecb76db876b607a7a); pin `8f2b71c67a31` (training algorithm, not a mixer).

ATMA is a local comparator at its recorded revision; hash dirty source files before
benchmarking. A resolved paper or repository identity does not imply an executable
comparator. Source blockers stay attached to the exact pinned row and are never
replaced with an unrelated baseline. FLA inventory establishes source availability only. Selected
equation inspections and corrections are in the [audit](unification-audit.md).

## Wave 1: Core native closure

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-001: Transformer MHA | [flash](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1 | Online softmax and complete Q/K/V backward; compare matching direct attention kernels |
| arch-002: Transformer MQA | [flash](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1 | Shared KV layout and gradient reduction; cache decode workload |
| arch-003: Transformer GQA | [flash](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1 | Explicit head mapping and KV gradient accumulation |
| arch-015: Linear attention | [fla / `fla/ops/linear_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/linear_attn) | K2 additive | Feature maps and denominator state; prefill and recurrent decode |
| arch-025: DeltaNet | [fla / `fla/ops/delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/delta_rule) | K2 delta | Stable triangular solve; initial/final-state and key gradients |
| arch-026: Gated DeltaNet | [fla / `fla/ops/gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_delta_rule), plus [ATMA `kernel/gated_delta_triton.py`](https://github.com/kreasof-ai/atma) | K2 delta | FLA chunk training/prefill plus ATMA's in-place slot-table decode are profiled; full-block frontend, low-batch eager dispatch, and full backward remain outside these kernel slices |
| arch-047: Sparse Delta Memory | [sdm / `lingua/sparse_delta_memory`](https://github.com/facebookresearch/sparse-delta-memory/tree/183e7df809131b80ad4393741029d0f20fc3640b/lingua/sparse_delta_memory) | K3 delta | Native overlap VJP matches the equation reference in FP32/BF16 and both read timings. Pinned SDM's FP32 custom VJP exceeds tolerance when read/write routes overlap; its qualified profile uses disjoint reads. Product-key routing and full-layer composition remain external |

## Wave 2: Structured variants and composite memories

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-004: MLA | [fla / `fla/layers/mla.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/mla.py), [FlashAttention kernel](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1+composition | The attention-stage profile uses BF16 B1/T64, 4 query/2 KV heads, qk dim 32 and v dim 16 padded to 32. Output and gradients match exactly; median plan overhead +7.7%/-9.4% forward/forward-backward. Latent projections, full MLA layer and compressed cache ABI remain open |
| arch-005: NSA | [fla / `fla/ops/nsa`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/nsa) | K1 selected-block core | Supplied-route BF16 slice at batch 1, sequence 128, 16 query/1 KV heads, K=V=32, block size 32: output error 7.81e-3, max gradient error 7.63e-6; 21-pair median time improves 30.0%/41.8% vs pinned `parallel_nsa` forward/forward-backward. Compression, indexer, auxiliary branches and full layer remain open |
| arch-006: MoBA | [fla / `fla/ops/moba`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/moba), nested [FlashAttention](https://github.com/Dao-AILab/flash-attention/tree/1bda8f9290cd48d030f1516f0e680cd464ef3554) | K1+route | Supplied-route BF16 B1/T128/H2/D=32, block size 32: exact output/gradient parity and +1.1%/+0.6% median compiler-plan overhead; reproducible narrow comparator build in `benchmarks/build_flash_attn_moba.py`. Block scoring, route selection and full-layer/cache integration remain external |
| arch-007: DSA | [fla / `fla/ops/dsa`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/dsa) | K1 supplied-route mask | BF16 precomputed-route attention at batch 1, sequence 128, 2 heads, K=32, V=16, top-k=16. The mask equation matches FLA `naive_dsa` exactly; SDPA max output error is 7.81e-3 and max gradient error 7.63e-6. Median time improves 58.5%/60.1% vs the naive reference for forward/forward-backward; the learned indexer and full DSA layer remain external |
| arch-008: Forgetting Transformer / FoX | [fla / `fla/ops/forgetting_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/forgetting_attn) | K1 gated causal attention | BF16 batch 1, sequence 128, 4 heads, K=V=32: pairwise-gate equation max errors 7.81e-3 output and 1.53e-5 gradients; pinned adapter exact; 21-pair median overhead +5.0%/+0.0% forward/forward-backward. Gate production and full layer remain open |
| arch-016: Lightning Attention | [fla / `fla/ops/lightning_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/lightning_attn) | K2 static head-decayed additive state | Layer-index decay schedule and Q/K/V projections; full-layer comparison |
| arch-017: RetNet / retention | [fla / `fla/ops/retention`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/retention) | K2 static head-decayed additive state | Multiscale head schedule, positional transforms and layer normalization |
| arch-018: Simple GLA | [fla / `fla/ops/simple_gla`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/simple_gla) | K2 head-decayed additive state | FP32 B1/T64, H=4, K=V=32 output/state/gradient parity is exact; median overhead +7.5%/+3.3%. YOCO uses this kernel too. Low-precision modes and full-layer frontends remain open |
| arch-019: GLA | [fla / `fla/ops/gla`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gla) | K2 key-channel-decayed additive state | Qualify fp16/bf16 chunk prefill/backward and full-layer projection/gate composition against the pinned caller |
| arch-020: Based | [fla / `fla/ops/based`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/based) | K2 additive state with Taylor-2 feature lift | Q/K/V projections and surrounding architecture layers |
| arch-021: ReBased | [fla / `fla/ops/rebased`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rebased) | K2 additive state with squared-dot feature lift | Q/K/V projections and surrounding architecture layers |
| arch-022: LightNet | [fla / `fla/layers/lightnet.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/lightnet.py) | K2+frontend audit | Extract exact feature/gate composition; layer-level parity |
| arch-023: HGRN | [fla / `fla/ops/hgrn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/hgrn) | K2 vector-state extension | Vector state layout and gate activation; no unnecessary matrix expansion |
| arch-024: HGRN2 | [fla / `fla/layers/hgrn2.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/hgrn2.py) | K2+frontend audit | State expansion and gate semantics; include projection costs |
| arch-028: KDA | [fla / `fla/ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/kda) | K2 key-channel delta | Complete A_log/dt/bias gate frontend, optional Q/K normalization and full layer; qualify prefill/decode |
| arch-040: RWKV-4 | [fla / `fla/ops/rwkv4`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv4) | K2 stable scalar state | FP32 equation parity at batch 1, sequence 1024 and 512 channels: 4.47e-8 output, 4.77e-6 final state, and at most 4.33e-7 input-gradient error; pinned adapter exact. Median overhead +2.1%/+2.1% forward/forward-backward; time-mix frontend and layer remain external |
| arch-041: RWKV-6 | [fla / `fla/ops/rwkv6`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv6) | K2 key-channel decay plus bonus read | FP32 B1/T512, 2 heads, K=V=32. Equation max errors 1.49e-8 output, 1.49e-7 final state and below 2e-13 gradients; pinned adapter exact. Median overhead +5.8%/+0.9% forward/forward-backward. Upstream backward does not consume a final-state cotangent; time-mix frontend and projections remain external |
| arch-048: ABC | [fla / `fla/ops/abc`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/abc) | K2 two-stage slot attention | BF16 B1/T128, 2 heads, K=V=32, 16 slots. Equation max errors: 4.88e-4 output, 7.42e-5 state and 4.51e-7 supported gradients; pinned adapter exact. Median overhead +3.9%/+1.6%. Upstream output-only backward omits initial-state gradients; slot frontend and full layer remain open |
| arch-049: GSA | [fla / `fla/ops/gsa`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gsa) | K2 two-stage gated slot attention | BF16 B1/T128, 2 equal query/KV heads, K=V=32, 16 slots. Equation max errors: 3.91e-3 output, 1.53e-3 state and 7.63e-6 supported gradients; pinned adapter exact. Median overhead +4.6%/+2.9%. Upstream output-only backward omits initial-state gradients; GQA faults on the available A10G; slot/gate frontend and full layer remain open |
| arch-071: Longformer | [allenai/longformer](https://github.com/allenai/longformer/tree/caefee668e39cacdece7dd603a0bebf24df6d8ca) | K1 sliding-chunks local window | FP32 B1/T2048/H4/D32/window32 source output parity exact, gradient error 2.84e-14; median overhead +3.1%/-1.8%. Global tokens, padding masks, projections and full encoder remain open |
| arch-072: Sparse Transformer | [openai/sparse_attention](https://github.com/openai/sparse_attention/tree/c53f3bdbf6225be0582f0357072e82b13c69be7d) | masked K1 all/local/strided/fixed modes | Pinned dense equations and fixed-mode layout/callback pass output and gradient checks; paired median overhead is -82.5% to -89.8% forward and -79.4% to -85.2% forward/backward ([profile](../../results/unified-mixer/sparse-transformer-k1.json)). The optimized TensorFlow 1 BlocksparseTransformer traversal remains unprofiled |

## Wave 3: Generalized state, routing and axis coverage

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-009: Log-linear attention | [fla / `fla/ops/log_linear_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/log_linear_attn) | K2 dyadic levels | Profile and qualify streaming decode, variable-length batches and full cache ABI |
| arch-010: PaTH attention | [fla / `fla/ops/path_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/path_attn) | K1+transform recurrence | BF16 B1/T128, 4 query/2 KV heads, K=V=32. Independent equation output/gradient max errors 9.77e-4/<4.77e-7; pinned adapter exact; paired median overhead +3.8%/+1.5% forward/forward-backward. Projections, w short convolution, optional q/k normalization, cache and decode remain open |
| arch-011: Wall attention | [fla / `fla/ops/wall_attn`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/wall_attn) | K1 per-channel decay | FP32 batch 1, sequence 64, 2 query/1 KV heads, K=32, V=16: independent equation max errors 4.77e-7 output and 1.17e-9 gradients; pinned adapter exact; 21-pair median overhead +4.1%/+0.5% forward/forward-backward. Optional sink/window and decode/cache remain open |
| arch-012: Parallax | [fla / `fla/ops/parallax`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/parallax) | K1 secondary-query correction | BF16 batch 1, sequence 128, 2 heads, K=V=32: explicit equation max errors 7.81e-3 output and 1.53e-5 gradients; pinned adapter exact; 21-pair median overhead +5.7%/+0.2% forward/forward-backward. Model-side r construction and full layer remain open |
| arch-013: DeltaFormer | [fla / `fla/ops/deltaformer`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/deltaformer) | K1 causal softmax after strict-causal triangular K2 value correction | BF16 B1/T64, H=2, K=V=32: independent equation max errors 9.77e-4 output and 4.77e-7 input gradients; pinned adapter exact; median plan overhead +1.7%/+0.5% forward/forward-backward. Q/K/V/beta frontend, complete layer and mode qualification remain open |
| arch-014: BitAttention | [fla / `fla/layers/bitattn.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/bitattn.py) | K1 after BitLinear projections and RoPE | The shared BF16 attention slice at B1/T64/H=4/K=V=32 matches pinned FlashAttention output and gradients exactly; median plan overhead +2.6%/-9.9% forward/forward-backward. BitLinear quantization/surrogate gradients, positional transform and full layer remain open |
| arch-027: GDN2 | [fla / `fla/ops/gdn2`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gdn2) | K2 gated delta transition | Separate key erase and value write gates; full-layer projections, gates, normalization and output projection |
| arch-029: Gated DeltaProduct | [fla / `fla/ops/gated_delta_product`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_delta_product) | K2 multi-update | Ordered rank-2 recurrence parity/profile measured; per-update projection/gate production and full-layer training/prefill/decode integration remain |
| arch-030: Momentum DeltaNet | [fla / `fla/ops/momentum_delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/momentum_delta_rule) | K2 augmented state | BF16 B1/T256, H=2, K=V=32. Equation output/state/input-gradient max errors 4.9e-4/1.82e-3/2.45e-4; pinned adapter exact; median overhead +0.1%/-0.4% forward/forward-backward. Default normalization modes, frontend, projections and full-layer integration remain open |
| arch-031: Generalized delta IPLR | [fla / `fla/ops/generalized_delta_rule/iplr`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/generalized_delta_rule/iplr) | K2 low-rank transition | FP32 recurrence equation and all input gradients pass at batch 1, sequence 256, 2 heads, K=V=32; pinned recurrent adapter parity is exact; median overhead +7.3% forward/+2.5% forward-backward. Factor generation and full layer remain external |
| arch-032: Generalized delta DPLR | [fla / `fla/ops/generalized_delta_rule/dplr`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/generalized_delta_rule/dplr) | K2 low-rank transition | BF16 chunk equation and all input gradients pass at batch 1, sequence 256, 2 heads, K=V=32; pinned chunk adapter parity is exact; median overhead +6.6% forward/+0.9% forward-backward. Factor generation and full layer remain external |
| arch-033: Gated Oja rule | [fla / `fla/ops/gated_oja_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/gated_oja_rule) | K2 value-channel-decayed Oja state | BF16 B1/T256, H=2, K=V=16: independent recurrence max errors 6.11e-5 output, 4.61e-4 final state, and 5.74e-7 gradients; pinned adapter exact; median overhead +3.1%/+1.9% forward/forward-backward. Projections, gate production, full-layer and decode integration remain open |
| arch-034: PGDN | [fla / `fla/ops/precond_gated_delta_rule`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/precond_gated_delta_rule) | K2 ATK-preconditioned gated delta | BF16 B1/T256, H=2, K=V=16: independent equation max errors 3.66e-4 output, 6.79e-4 matrix state and 1.38e-3 gradients; pinned adapter exact; median overhead +2.3%/+1.3% forward/forward-backward. ATK parameter generation, grouped value heads and full-layer integration remain open |
| arch-035: PKDA | [fla / `fla/ops/precond_kda`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/precond_kda) | K2 key-channel ATK-preconditioned delta | BF16 B1/T256, H=2, K=V=16: independent equation max errors 2.75e-4 output, 7.82e-4 matrix state, 1.56e-2 ATK state and 3.54e-3 gradients; pinned adapter exact; median overhead +3.4%/+1.1% forward/forward-backward. Gate generation, ATK parameters and full-layer integration remain open |
| arch-036: Rodimus | [fla / `fla/layers/rodimus.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/rodimus.py) | K2 BF16 key-channel GLA with V-first state and 1/√K read | BF16 B1/T64, H=1, K=64, V=128. Equation errors 1.22e-4 output, 4.57e-4 final state, and 9.54e-7 max input gradient; pinned adapter exact; median overhead +5.0%/+0.6% forward/forward-backward. Frontend, nonzero cache state, full layer and mode qualification remain open |
| arch-037: Comba | [fla / `fla/ops/comba`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/comba) | K2 dual-key gated delta | BF16 B1/T256, H=2, K=V=16: independent equation max errors 1.22e-4 output, 5.28e-4 final state and 6.52e-6 gradients; pinned adapter exact; median overhead +4.8%/+1.1% forward/forward-backward. Feature projection and full-layer integration remain open |
| arch-038: MesaNet | [fla / `fla/ops/mesa_net`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/mesa_net) | K2 dual covariance state plus regularized key-space solve | BF16 B1/T64/H2/K=V=16 output/state/all-input-gradient parity passes; paired plan overhead +3.8%/+1.2% forward/forward-backward. Nonzero initial-state gradients, streaming decode, Q/K normalization and lambda frontend, complete layer, and modes remain open |
| arch-039: Titans | [fla / `fla/ops/titans`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/titans) | K2 chunked associative memory with learned inner reconstruction update | FP32 B1/T64/H1/D=16, chunk 16: independent tokenwise equation errors are 3.05e-5 output, 1.43e-5 final state and at most 9.31e-4 gradient; pinned adapter exact. Median overhead -0.9%/+0.1% forward/forward-backward. This is FLA's eager PyTorch chunk operator, not a fused GPU kernel; outer attention, projections, hierarchy and full-model cache modes remain open |
| arch-042: RWKV-7 | [fla / `fla/ops/rwkv7`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/rwkv7) | K2 DPLR transition | BF16 chunk equation parity at batch 1, sequence 256, 2 heads, K=V=32: 7.81e-3 output, 2.79e-4 final state, and at most 7.63e-6 gradient error; pinned adapter exact. Median overhead +6.0%/+1.7%; source-factor frontend and complete layer remain external |
| arch-043: Mamba-1 | [mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | K2/SSM+convolution | Selective state update discretization local convolution and cache ABI |
| arch-044: Mamba-2 / SSD | [mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | K2/semiseparable+convolution | Full-layer projections, short convolution, dt bias/softplus, skip/output gate and cache/chunk-boundary ABI |
| arch-045: Mamba-3 | [mamba](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | K2 SISO rotary angle accumulator with trapezoidal four-state SSM recurrence | BF16 SISO at B1/T64/H2/K=V=16 passes output/four-state/all-input-gradient parity against pinned `mamba3_siso_combined`; equation errors are 7.82e-5 output, 1.09e-2 max state and 2.61e-3 max gradient; adapter exact; median plan overhead +2.7%/+0.3% forward/forward-backward. MIMO/TileLang, model frontend, full layer and cache remain open |
| arch-046: LogLinearMamba2 | [fla / `fla/layers/log_linear_mamba2.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/log_linear_mamba2.py) | K2 dyadic levels + Mamba-2 frontend | Integrate projection/dt/level transforms and qualify full layer, cache, prefill and decode |
| arch-050: Raven | [fla / `fla/layers/raven.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/raven.py) | GSA+route/frontend | Raven calls pinned GSA; equal-head kernel matches its shared profile. GQA faults on this A10G. Router top-k/scores, decay frontend and full layer remain open |
| arch-051: Mixture of Memories / MoM | [fla / `fla/layers/mom.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/mom.py) | K2 route-dispatched gated-delta memory core | BF16 two routed streams, T64/H2/K32/V16: equation max errors 9.77e-4 output, 4.97e-4 state and 5.07e-7 gradients; pinned adapter exact; plan overhead +0.8%/+1.4%. Router/top-k selection, packing/dispatch/merge and expert projections/convolution/full-layer composition remain open |
| arch-052: YOCO | [fla / `fla/layers/yoco.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/layers/yoco.py) | multi-layer/cache composition | GatedRetention uses the profiled Simple GLA kernel; rotary/gate frontend and output parity, shared-KV builder, self/cross-decoder composition and cache ABI remain |
| arch-053: Samba | [microsoft/Samba](https://github.com/microsoft/Samba/tree/617c7a0f8c71f1b7cb6180b86f9543d146f5c66f) | K1 on Samba_421M_nope attention branch | FP32 B1/T512/H12/D128 source CausalSelfAttention including QKV/output projections matches exactly; median overhead -5.8%/-2.0% ([profile](../../results/unified-mixer/samba-k1.json)). Full hybrid Block, Mamba state/cache, short convolution, rotary, normalization and MLP remain open |
| arch-054: Attention Residuals / AttnRes | [fla / `fla/ops/attnres`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/attnres) | depth-axis reduction | BF16 depth 8, B1/T256/width 128. Generic K1 equation max errors are 3.91e-3 output and 3.82e-6 gradients; pinned adapter output and gradients are exact. Median plan overhead +8.9%/+3.4% forward/forward-backward; layer scheduling and cross-layer gradient accumulation remain external |
| arch-057: TokenFormer / Pattention | [pattention](https://github.com/Haiyang-W/TokenFormer/tree/4d56c73f407635e62f6df16b97dc897b4477129e) | parameter-axis contraction | BF16 softmax mode, B1/T128, 256 parameter tokens, query/value width 32. Max output error 4.88e-4 and gradient error 1.53e-5. Paired median plan time is 0.8% faster forward and 22.3% faster forward/backward than the pinned source equation. GELU/L2 modes, parameter initialization and routing remain open |
| arch-058: SwiGLU MLP | [XMA](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | outside K1/K2/K3 | XMA's pointwise activation and linear projections belong in MLP/GEMM coverage, not sequence-mixer qualification |
| arch-059: Top-k MoE | [XMA](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | outside K1/K2/K3 | Token routing and expert GEMMs belong in MoE/routing coverage, not sequence-mixer qualification |
| arch-064: POLAR attention | atma `kernel/polar_triton.py` @ `28bb3de8afbe7c0b00115e0fbff36afc9ad49c11` | K1 direction plus magnitude | FP32 B1/T64/H2/D=16 equation and Triton output/magnitude/all-gradient parity pass exactly; median plan overhead +4.2%/+0.6%. Full layer projections, GQA, canonical convolution and output/count stages remain external |
| arch-065: Foveal sparse attention | atma `kernel/polar_triton.py` @ `28bb3de8afbe7c0b00115e0fbff36afc9ad49c11` | K1 local window plus selected remote pages | FP32 B1/T64/H2/D=16, page/local window 16, supplied routes: equation and Triton output/magnitude/all-gradient parity pass exactly; median plan overhead +5.8%/+0.4%. Geometric routing, projections and full layer/cache remain external |
| arch-067: Differential Attention | [microsoft/unilm / `Diff-Transformer/multihead_diffattn.py`](https://github.com/microsoft/unilm/tree/50224e387211f15ac6a3b2685730b9a0c850f145/Diff-Transformer) | two K1 reductions plus V1 coefficient | Preprojected FP32 B1/T64/H2/D=16 V1 core: output error 4.47e-8, maximum gradient error 3.64e-11, paired median plan latency -59.5%/-51.1%. RoPE, lambda generation, RMSNorm, projection and cache remain external |
| arch-068: TDA | [snap-research/TDA / `triton_threshold_attention.py`](https://github.com/snap-research/TDA/blob/cd8ddc9d5b43a1dcf86f9cfda302edb5cc108da2/triton_threshold_attention.py) | two thresholded causal K1 reductions | FP32 B1/T64/H2/D=32 source adapter exact; equation max errors 9.40e-5 output and 4.94e-8 gradients; median plan overhead +2.6%/+1.8%. Full projections, beta frontend and variable-length masks remain |
| arch-069: TPA | [tensorgi/TPA](https://github.com/tensorgi/TPA/tree/c276c80d5ad807881dedb4707d8d3c20b4e97ec6) | factorized projection+K1 audit | Full T6 CausalSelfAttention matches exactly at FP32 B1/T64/H2/D32; median overhead +2.0%/-0.9%. Decode/cache and full decoder integration remain open |
| arch-070: Tucker attention | [ScSteffen/Tucker-Attention](https://github.com/ScSteffen/Tucker-Attention/tree/c3e3d3cec991f4303b824c7fb7cbb95e3748d5c7) | factorized query+K1 | BF16 B4/T2048/H4/rank16 exact output; maximum gradient error 1.16e-10; median overhead +6.0%/+0.8%. Full frontend/projection and cache expansion remain open |

## Wave 4: Nonlinear updates and newly resolved targets

| Architecture ID / name | Comparator | Proposed lowering | Required work |
|---|---|---|---|
| arch-055: TTT | [fla / `fla/ops/ttt`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/ops/ttt) | K2 TTT-Linear matrix/bias update | BF16 B1/T64/H2/D=16 chunk-16 core: independent naive output/state/gradient errors 1.55e-2/<8e-7/1.56e-2 pass configured tolerances; pinned adapter exact; median overhead +3.1%/-0.2%. MLP variant, full layer, higher-order gradients and cache integration remain open |
| arch-056: FwPKM | [SakanaAI/FwPKM](https://github.com/SakanaAI/fast-weight-product-key-memory/tree/b1c8e234b523d70245fa197eed4b80a985c413a8) | K1 fused softmax/value read over selected product-key logits | Pinned `retrieve_values` output and input gradients pass for dot-product and IDW routing; default IDW profile at FP32 B4/T256/H2/D64/V64/topk8 has +9.0% forward and -0.6% forward/backward median overhead ([profile](../../results/unified-mixer/fwpkm-k1.json)). Online fast-weight writes, addressing entropy updates, optimizer and chunk/cache semantics remain open |
| arch-060: RNN | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | K2 tanh nonlinear state | FP32 B1/T64/H=1/D=16: equation and Triton output/state/all-gradient checks pass at 3e-6; median plan overhead +3.7%/+0.4% forward/forward-backward. Multi-head replication, packed sequences, clipping and full-layer integration remain open |
| arch-061: GRU | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | K2 reset/update nonlinear state | FP32 B1/T64/H=1/D=16: equation and Triton output/state/all-gradient checks pass at 3e-6; median plan overhead +3.2%/+1.5% forward/forward-backward. Multi-head replication, packed sequences, clipping and full-layer integration remain open |
| arch-062: M2RNN | [xma](https://github.com/open-lm-engine/accelerated-model-architectures/tree/384ed0a7bd82ced1f40609603dd541cac5416844) | K2 nonlinear matrix-memory state | FP32 B1/T64/H=1/K=V=16: equation and Triton output/state/all-gradient checks pass at 3e-6; median plan overhead +3.9%/+0.9% forward/forward-backward. Multi-head replication, packed sequences, clipping and full-layer integration remain open |
| arch-063: BDH | [bdh](https://github.com/pathwaycom/bdh/tree/2b0d7a45b058d4309c84a10e0768d541fe18bdc2) | K2 strict-past rotary linear attention | FP32 B1/T64/H2/QK32/V16: independent recurrence max errors 2.98e-8 output and 2.91e-11 query/value gradients; pinned adapter exact; paired median plan overhead +4.5%/+1.9%. Full BDH projections, normalization, gated MLP, dropout and residual graph remain open |
| arch-066: CAT / Compress-and-Attend | [fla / `fla/models/cat/modeling_cat.py`](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2/fla/models/cat/modeling_cat.py) | compression+K1 composition | BF16 B1/T22/H2/D=16 source BlockMask: exact FlexAttention output, max input-gradient error 1.49e-8, equation output error 9.77e-4; median relative latency -2.6%/-12.0% forward/forward-backward. Compression, adaptive/separator tokens, rotary frontend, projections and full decoder layer remain |
| arch-073: KATA | [ayghri/KATA](https://github.com/ayghri/KATA/tree/f93fe75750be6400a0068749794985d70666926d) | K1 normalized-positive grouped scores | BF16 B4/T1024/H8/D64/M4 output and gradients exact; independent equation output error 9.77e-4; median overhead +5.3%/+1.4%. Model projections, packed variable-length path and cache remain open |
| arch-074: HLA | [yifanzhang-pro/HLA](https://github.com/yifanzhang-pro/HLA/tree/484fef2bb40d4ed58f7656e545cf5ef64c40c962) | K2 masked second-order streaming recurrence | Pinned paper Equation (3.3) parity passes; FP32 B1/T1024/H8/D=16 Triton output/gradient errors are 5.9e-6/3.4e-9, median plan speedup is 30.8%/46.5% ([profile](../../results/unified-mixer/hla-k2.json)). Higher/asymmetric/decayed orders, chunk scan scheduling and cache remain open |
| arch-075: Conformer | [ESPnet](https://github.com/espnet/espnet/tree/2950325ea62c8052f448aaf11affdabe169ec8ab) | K1 bidirectional self-attention | Pinned `MultiHeadedAttention` SDPA path with projections at FP32 B2/T1024/H4/D32 matches output and all gradients exactly; median overhead +5.8%/+2.7%. Relative position, convolution, FFN and full encoder remain open |
| arch-076: H3 | [HazyResearch/H3](https://github.com/HazyResearch/H3/tree/5c4d06b5795405170387c80998b58d76179a8a1a) | K2 two-stage causal FFT convolution | Pinned FP32 `H3.forward` at B1/T128/D64/state16/head_dim1 passes output and all input/parameter gradients; median overhead -4.8%/-2.7% ([profile](../../results/unified-mixer/h3-k2.json)). Head_dim>1, inference cache/step and fused FFT path remain open |
| arch-077: Hyena | [HazyResearch/safari](https://github.com/HazyResearch/safari/tree/02220c69d247e5473616cd053a443ad99fd2559b) | K2 order-2 implicit-filter causal FFT convolution | Pinned FP32 `HyenaOperator.forward` B1/T128/D32/filter-width 16 matches output and all gradients exactly; median overhead -4.6%/-3.3% ([profile](../../results/unified-mixer/hyena-k2.json)). Higher orders, low-precision fused FFT and streaming/cache remain open |
| arch-078: Hopfield | [ml-jku/hopfield-layers](https://github.com/ml-jku/hopfield-layers/tree/f56f929c95b77a070ae675ea4f56b6d54d36e730) | K1 unscaled softmax association | One source update at scaling 1.0, FP32 B2/T256/H4/D32: max output error 7.57e-10, max gradient error 4.10e-12, median overhead -67.0%/-45.0%. Iterative retrieval, learned scaling and normalization remain open |
| arch-079: MAML | [cbfinn/maml](https://github.com/cbfinn/maml/tree/a7f45f1bcd7457fe97b227a21e89b8a82cc5fa49) | outside K1/K2/K3 | Meta-learning inner adaptation and outer meta-gradients are a training algorithm, not a mixer kernel; track separately from mixer qualification |
| arch-080: Reptile | [OpenAI supervised-reptile](https://github.com/openai/supervised-reptile/tree/8f2b71c67a31c1a605ced0cecb76db876b607a7a) | outside K1/K2/K3 | Reptile's task training and parameter interpolation are a meta-learning algorithm, not a mixer kernel; track separately from mixer qualification |

## Unified kernel prototype status

The compiler entry point selects K1 softmax reduction, K2 recurrent state, or
K3 sparse delta state from one typed semantic descriptor. Its executable
reference path covers attention with explicit masks, matrix and diagonal state
recurrences, and ordered sparse-slot updates. Optional bindings use PyTorch SDPA,
the pinned FLA gated-delta adapter, and the existing native Triton sparse-state
anchor under their narrower capability contracts.

The 76 tagged rows are kernel slices rather than supported full architectures.
For example, masked K1 execution accepts routes from the caller and still
computes dense scores; Mamba recipes cover their state equation after projection
and convolution; sparse delta accepts already generated routes. A pinned
identity with no compatible source runtime remains an explicit comparison blocker.
Training, prefill, decode, numerical, full-layer, model and latency status stay
unqualified until the parity campaign establishes them.

## What counts as coverage

Report frontend expression, external-adapter execution, native execution and
performance parity separately. Each target needs kernel, full-layer and model
results where those boundaries exist. A kernel does not implicitly cover a
router, convolution, positional transform, normalization, cache or inner optimizer.

Qualify training, prefill and decode separately. Unsupported upstream modes must
be recorded with evidence as `upstream_unavailable`, never counted as passes.
Catalog items that are MLP, MoE or meta-learning algorithms use `not_applicable`
for mixer-kernel parity and point to the separate compiler domain.
Wave 1 is the first release gate; subsequent waves are full-version construction
work. Every row exits with qualification or a published blocker and concrete
implementation task. Blocked rows remain in the backlog and outside advertised
support. New upstream variants add comparisons rather than overwrite old baselines.

Use the [parity plan](../validation/parity.md) and
[generality axes](../compiler/generality-axes.md) to build each row.
