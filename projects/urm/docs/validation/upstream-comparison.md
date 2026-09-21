# Upstream comparison register: coverage, parity, and dispatch overhead

Consolidated rollup of the validated per-architecture comparisons against
pinned upstream sources. Each row is a kernel-slice comparison recorded in
the named artifact under `results/unified-mixer/`; it is not full-layer or
end-to-end qualification. Numbers are reproduced from the committed
artifacts, measured on the validated A10G / torch 2.8.0 / triton 3.4.0 line
against the pinned upstream revision recorded in each artifact.

**76/76 compared architectures pass parity** against their
pinned upstream callable. Upstream sources compared: atma, bdh, conformer, differential, fla, flash, fwpkm, h3, hla_higher_order, hopfield, kata, longformer, mamba, pattention, safari, samba, sdm, sparse_transformer, tda, tpa, tucker, xma.

- **Parity** is the artifact's output/gradient/state correctness verdict
  against the exact upstream callable. A row shows `pass` only when every
  case named in the register passes within the frozen tolerances; a failed
  case shows `fail`, and missing or incomplete evidence shows `incomplete`.
- **Overhead** is the paired median of per-pair `(compiled - direct) / direct`
  dispatch fractions: negative is faster than the upstream call, positive is
  slower. Forward and forward+backward are reported separately; the
  least-favorable case is shown for multi-case artifacts.

**These are dispatch-overhead numbers, and dispatch coverage is not native
coverage.** For most rows the compiled plan invokes the pinned upstream
kernel through URM's library adapter, so the overhead measures only the
compiler's dispatch cost on top of that shared upstream kernel (typically
a few percent). It does **not** mean URM computes the operation with its
own kernel - a generator that only dispatches would be a thin wrapper.
The honest measure of the unified generator's reach is native generation:
see `docs/validation/native-coverage.md` for which recipes URM computes
natively, and `docs/planning/production-matrix.md` plus
`results/qualification/` for native-replacement performance qualification.

## K1

| Architecture | Upstream | Parity | Forward overhead | Fwd+Bwd overhead | Scope |
|---|---|---|---|---|---|
| BitAttention | flash | pass | +2.6% | -9.9% | Shared BitAttention K1 kernel only: BF16 B1/T64, Hq=Hkv=4 and K=V=32. Direct pinned FlashAttention output and gradien… |
| Conformer | conformer | pass | +5.8% | +2.7% | Pinned ESPnet Conformer self-attention with full Q/K/V and output projections; profile excludes relative-position/pos… |
| DSA | fla | pass | -58.5% | -60.1% | BF16 DSA selected-attention slice with supplied token indices and shared precomputed boolean mask. The compiler uses … |
| DeltaFormer | fla | pass | +1.7% | +0.5% | Pinned FLA DeltaFormer operator at BF16 B1/T64/H2/K=V=32. The independent strict-causal triangular correction plus ca… |
| Forgetting Transformer / FoX | fla | pass | +5.0% | +0.0% | BF16 FoX attention with per-token log-decay gates. The reference equation uses the pairwise cumulative-gate bias and … |
| Foveal sparse attention | atma | pass | +5.8% | +0.4% | ATMA Foveal local-window plus caller-supplied selected remote-page Polar reduction, FP32 B1/T64/H2/D=16, page/local w… |
| FwPKM | fwpkm | pass | +9.0% | -0.6% | Pinned retrieve_values call compared with exact product-key IDW routing, selected-value gather and URM fused K1 softm… |
| Hopfield | hopfield | pass | -67.0% | -45.0% | Pinned HopfieldCore single-update attention including source Q/K/V and output projections; iterative retrieval and as… |
| KATA | kata | pass | +5.3% | +1.4% | causal KATA normalized-positive attention core vs pinned Triton source; all source and compiler work inside the repor… |
| Longformer | longformer | pass | +3.1% | -1.8% | bidirectional local-window softmax attention using Longformer's sliding-chunks K1 operator; full-source call/profile … |
| MLA | flash | pass | +7.7% | -9.4% | MLA causal attention after latent-KV expansion and RoPE concatenation; FlashAttention pads V to qk_head_dim and MLA c… |
| MoBA | fla | pass | +1.1% | +0.6% | BF16 supplied-route MoBA attention vs pinned FLA parallel_moba; nested FlashAttention causal/noncausal D=32 kernel so… |
| NSA | fla | pass | -30.0% | -41.8% | BF16 selected-only NSA attention against fla.ops.nsa.parallel.parallel_nsa with exact precomputed block routes and co… |
| POLAR attention | atma | pass | +4.2% | +0.6% | ATMA Polar causal direction and bounded-magnitude reduction with learned null sink, FP32 B1/T64/H2/D=16. The independ… |
| PaTH attention | fla | pass | +3.8% | +1.5% | PaTH triangular transformed causal attention with explicit float32 w/beta/g; layer projections, short convolution and… |
| Sparse Transformer | sparse_transformer | pass | -83.5% | -81.3% | Pinned TensorFlow source attention_impl all/local/strided equations and exact fixed-mode block layout plus callback e… |
| Transformer GQA | flash | pass | +1.2% | -10.6% | causal K1 BF16 kernel slice vs direct FlashAttention; D=32; full-layer positional/projection stages excluded |
| Transformer MHA | flash | pass | +2.6% | -9.9% | causal K1 BF16 kernel slice vs direct FlashAttention; D=32; full-layer positional/projection stages excluded |
| Transformer MQA | flash | pass | +4.8% | -11.1% | causal K1 BF16 kernel slice vs direct FlashAttention; D=32; full-layer positional/projection stages excluded |
| Wall attention | fla | pass | +4.1% | +0.5% | FP32 Wall per-channel decay attention against pinned parallel_wall_attn at batch 1, sequence 64, 2 query heads/1 KV h… |

## K2

| Architecture | Upstream | Parity | Forward overhead | Fwd+Bwd overhead | Scope |
|---|---|---|---|---|---|
| ABC | fla | pass | +3.9% | +1.6% | BF16 two-stage ABC against pinned chunk_abc: independent recurrence errors 0.000488 output, 7.41e-05 max final state … |
| BDH | bdh | pass | +4.5% | +1.9% | BDH Attention.forward at FP32 B1/T64/H2/QK32/V16; compiler uses the same pinned source callable, so profile records p… |
| Based | fla | pass | +6.7% | +2.4% | based_attention_core causal kernel slice only; compiler library adapter invokes the pinned FLA source callable, with … |
| Comba | fla | pass | +4.8% | +1.1% | Head-decayed dual-key COMBA recurrence against pinned chunk_comba. The independent equation has 1.22e-4 maximum outpu… |
| DeltaNet | fla | pass | +4.1% | +1.5% | compiled K2 kernel slice; architecture gate/projection stages remain external |
| GDN2 | fla | pass | +4.4% | +3.0% | GDN-2 recurrence only; plan invokes the pinned FLA chunk operator, while full-layer Q/K/V projections, erase/write/de… |
| GLA | fla | pass | +5.5% | +1.6% | compiled K2 kernel slice; architecture gate/projection stages remain external |
| GRU | xma | pass | +3.2% | +1.5% | FP32 B1/T64/H=1/D=16: independent tokenwise equation maximum errors 0.00e+00 output, 0.00e+00 final state and 2.33e-1… |
| GSA | fla | pass | +4.6% | +2.9% | BF16 two-stage GSA against pinned chunk_gsa: independent recurrence errors 0.00391 output, 0.00153 max final state an… |
| Gated DeltaNet | fla | pass | +2.2% | +1.2% | compiled K2 kernel slice; architecture gate/projection stages remain external |
| Gated DeltaProduct | fla | pass | +4.4% | -0.4% | BF16 ordered rank-2 gated delta kernel slice vs the exact pinned FLA chunk operator; profile measures plan/validation… |
| Gated Oja rule | fla | pass | +3.1% | +1.9% | Value-channel-decayed Oja recurrence against pinned chunk_gated_oja_rule. The independent equation has 6.11e-5 max ou… |
| Generalized delta DPLR | fla | pass | +5.1% | -0.9% | BF16 generalized-delta DPLR core with external diagonal/low-rank transition factors and additive KV write. The librar… |
| Generalized delta IPLR | fla | pass | +7.3% | +2.5% | FP32 generalized-delta IPLR core with external transition factors and additive KV write. The library adapter calls th… |
| H3 | h3 | pass | -4.8% | -2.7% | Pinned H3.forward with use_fast_fftconv=False and head_dim=1, including SSKernel parameter-derived filter generation,… |
| HGRN | fla | pass | +1.1% | +0.1% | vector-state recurrent kernel; the compiler library anchor invokes the pinned FLA recurrent HGRN source operator |
| HGRN2 | fla | pass | +3.9% | -0.6% | GLA recurrent operator; full HGRN2 projections, activation gates and value-first state layout are excluded |
| HLA | hla_higher_order | pass | -30.8% | -46.5% | Exact masked second-order HLA from pinned paper Equation (3.3), checked against an independent dense transcription an… |
| Hyena | safari | pass | -4.6% | -3.3% | Pinned standalone HyenaOperator.forward(order=2), including the source implicit-filter MLP and modulation, short dept… |
| KDA | fla | pass | +2.1% | -1.4% | Precomputed-gate KDA recurrence. The compiler library adapter calls the pinned FLA chunk kernel, so timing measures p… |
| LightNet | fla | pass | +5.6% | +2.0% | K2 recurrence kernel slice with LightNet-transformed inputs; the complete LightNet layer frontend and gated normaliza… |
| Lightning Attention | fla | pass | +8.8% | +2.0% | K2 recurrent kernel slice compared with the exact FLA chunk_simple_gla call used by Lightning Attention; the artifact… |
| Linear attention | fla | pass | +0.6% | -2.2% | compiled K2 kernel slice; architecture gate/projection stages remain external |
| LogLinearMamba2 | fla | pass | +1.8% | -0.5% | Dyadic LogLinear operator core with precomputed log-decay and level-scale tensors. The compiler library adapter calls… |
| M2RNN | xma | pass | +3.9% | +0.9% | FP32 B1/T64/H=1/D=16: independent tokenwise equation maximum errors 7.45e-09 output, 0.00e+00 final state and 3.64e-1… |
| Mamba-1 | mamba | pass | +4.2% | +6.3% | selective scan only; the compiler library anchor calls the pinned source operator, while convolution, projections, ac… |
| Mamba-2 / SSD | mamba | pass | +4.4% | -0.8% | SSD scan only; the library adapter invokes the pinned Mamba-2 source operator, while projections, short convolution, … |
| Mamba-3 | mamba | pass | +2.7% | +0.3% | Pinned Mamba-3 SISO Triton callable at BF16 B1/T64/H2/K=V=16. Independent four-state trapezoidal recurrence errors ar… |
| MesaNet | fla | pass | +3.8% | +1.2% | Pinned FLA chunk_mesa_net at BF16 B1/T64/H2/K=V=16 with its default 30-step conjugate-gradient solver and zero initia… |
| Mixture of Memories / MoM | fla | pass | +0.8% | +1.4% | Per-routed-memory BF16 MoM recurrence core using FLA chunk_gated_delta_rule with its source Q/K L2 normalization and … |
| Momentum DeltaNet | fla | pass | +0.1% | -0.4% | Two-state fast-weight/momentum chunk recurrence with q/k/p normalization and p-times-alpha disabled; model frontend, … |
| PGDN | fla | pass | +2.3% | +1.3% | PGDN ATK-preconditioned gated delta recurrence against pinned chunk_precond_gated_delta_rule with default x=1.5, eps=… |
| PKDA | fla | pass | +3.4% | +1.1% | PKDA key-channel-gated preconditioned delta recurrence against pinned chunk_precond_kda with explicit log-space gates… |
| RNN | xma | pass | +3.7% | +0.4% | FP32 B1/T64/H=1/D=16: independent tokenwise equation maximum errors 0.00e+00 output, 0.00e+00 final state and 0.00e+0… |
| RWKV-4 | fla | pass | +2.1% | +2.1% | FP32 RWKV-4 WKV recurrent core with stable per-channel alpha/beta/eps state. The library adapter calls the pinned FLA… |
| RWKV-6 | fla | pass | +5.8% | +0.9% | RWKV-6 recurrent kernel with static bonus and key-channel log decay; input gradients use output loss because the pinn… |
| RWKV-7 | fla | pass | +6.0% | +1.7% | BF16 RWKV-7 chunk core with source-derived DPLR transition and unit-scale r read. The library adapter calls pinned FL… |
| Raven | fla | pass | +4.6% | +2.9% | Raven directly dispatches to GSA after feature, router, slot-weight and decay generation. The GSA kernel slice matche… |
| ReBased | fla | pass | +6.2% | +0.7% | rebased_attention_core causal kernel slice only; compiler library adapter invokes the pinned FLA source callable, wit… |
| RetNet / retention | fla | pass | +5.7% | +1.5% | K2 recurrent kernel slice compared with the exact FLA fused_chunk_simple_gla call used by Retention; the artifact als… |
| Rodimus | fla | pass | +5.0% | +0.6% | Pinned FLA chunk_gla at BF16 B1/T64/H1/K64/V128 using source default 1/sqrt(K) scaling and V-first returned state. In… |
| Simple GLA | fla | pass | +7.5% | +3.3% | compiled K2 kernel slice; architecture gate/projection stages remain external |
| YOCO | fla | pass | +7.5% | +3.3% | YOCO GatedRetention directly dispatches to FLA chunk_simple_gla after rotary Q/K and per-head gate generation. The sh… |

## K3

| Architecture | Upstream | Parity | Forward overhead | Fwd+Bwd overhead | Scope |
|---|---|---|---|---|---|
| Sparse Delta Memory | sdm | pass | -16.6% | -41.4% | compiled native K3 state slice; direct pinned SDM gated_write_read; score/projection stages excluded |

## other

| Architecture | Upstream | Parity | Forward overhead | Fwd+Bwd overhead | Scope |
|---|---|---|---|---|---|
| Attention Residuals / AttnRes | fla | pass | +8.9% | +3.4% | AttnRes residual-depth aggregation including query/RMS parameters and all residual source gradients; generic K1 RMS-n… |
| CAT / Compress-and-Attend | fla | pass | -2.6% | -12.0% | CAT source FlexAttention BlockMask vs K1 SDPA plan under the pinned interleaved causal mask. Output parity is exact, … |
| Differential Attention | differential | pass | -59.5% | -51.1% | On the pinned V1 K1 slice with preprojected branch inputs, identity RoPE and post-normalization/output projection dis… |
| Log-linear attention | fla | pass | +1.8% | -0.5% | Dyadic LogLinear operator core with precomputed log-decay and level-scale tensors. The compiler library adapter calls… |
| Parallax | fla | pass | +5.7% | +0.2% | BF16 Parallax q/r/k/v equation at batch 1, sequence 128 and 2 heads. The independent secondary-query correction has 7… |
| Samba | samba | pass | -5.8% | -2.0% | Pinned Samba_421M_nope CausalSelfAttention attention layer index 1 with source QKV/output projections and causal atte… |
| TDA | tda | pass | +2.6% | +1.8% | The exact pinned TDA Triton adapter has zero output and input-gradient error. The independent thresholded-equation re… |
| TPA | tpa | pass | +2.0% | -0.9% | softmax attention after caller-supplied projections/transforms; full-source call/profile scope and all external stage… |
| TTT | fla | pass | +3.1% | -0.2% | Pinned FLA TTT-Linear chunk recurrence with BF16 Q/K/V/update parameters and FP32 nonzero matrix/bias initial states.… |
| Titans | fla | pass | -0.9% | +0.1% | Exact pinned FLA chunk_titans_linear_ref(use_chunk=True) at FP32 B1/T64/H1/D=16, chunk 16. The independent tokenwise … |
| TokenFormer / Pattention | pattention | pass | -0.8% | -22.3% | TokenFormer Pattention softmax normalization mode over caller-supplied parameter tokens, with the source token-count … |
| Tucker attention | tucker | pass | +6.0% | +0.8% | softmax attention after caller-supplied projections/transforms; full-source call/profile scope and all external stage… |
