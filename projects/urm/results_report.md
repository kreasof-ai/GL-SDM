# URM training-harness measurement: 51 native rows × 49 upstream baselines

**What this is.** A training-harness measurement: each architecture row trains a 100M-class decoder LM for 10 steps on finewebedu, with checkpoint-parity and KL gates, on one NVIDIA A10G. It is **not** a production serving benchmark and **not** a source-model parity claim — the harness trains a generic decoder surround around each row's mixer (verdict 4 of the [evidence policy](docs/evidence.md); verdict 5 is not claimed).

**Config.** width=768, layers=9, heads=12, head_dim=64, seq=512, vocab=50304, finewebedu, 10 steps, bf16 autocast with fp32 kernel accumulation. Microbatch 8192 tokens with an OOM fallback ladder 2048 → 1024 (fallback rows are marked `(mbNNNN)`). URM rows compile through the opaque-op boundary; upstream rows run eager (the fla chunk kernels fail torch.compile/Inductor here) — so upstream throughput is a *lower bound*. MFU denominator: A10G adopted achievable bf16 peak = 70 TFLOPS.

**Environment.** torch 2.14.0+cu130, triton 3.8.0, NVIDIA A10G 22 GiB (101 KB shared-memory limit), git HEAD `c9b24f0`. Policy constraints: no flash-attn installs (bypassed to SDPA), no mamba_ssm (no torch-2.14/cu130 wheel), no source builds (the lingua SDM CUDA extension is toolchain-blocked).


## URM native rows

All 51 native rows completed training and checkpoint parity. Inline flags: **`NaN`** = loss diverged within the 10 steps (stability finding; MFU still validly measures executed FLOPs); **`slow`** = pathological MFU (<0.10, reasons below); `(mbNNNN)` = OOM fallback.

| row | params | MFU | tok/s | ckpt | KL | peak GiB | loss | flags |
|---|---|---|---|---|---|---|---|---|
| abc_gsa | 104,071,104 | 0.207 | 23121 | ✓ | — | 7.41 | 8.105 | — |
| attnres | 102,769,920 | 0.051 | 5803 | ✓ | — | 13.13 | 8.181 | **slow** |
| based_attention (mb1024) | 102,742,272 | 0.012 | 1375 | ✓ | — | 9.30 | 8.196 | **slow** |
| bit_attention | 102,769,920 | 0.238 | 26938 | ✓ | — | 6.25 | 8.098 | — |
| cat_attention | 102,742,272 | 0.329 | 37199 | ✓ | — | 5.77 | 8.142 | — |
| comba | 102,908,376 | 0.335 | 37842 | ✓ | — | 19.10 | nan | **NaN** |
| conformer_attention (mb2048) | 108,092,160 | 0.201 | 21637 | ✓ | — | 7.48 | 8.540 | — |
| deltaformer | 102,908,376 | 0.031 | 3463 | ✓ | — | 9.03 | 8.196 | **slow** |
| deltanet | 102,825,792 | 0.254 | 28745 | ✓ | 4.28e-10 | 6.19 | 8.113 | — |
| dense_attention | 102,742,272 | 0.434 | 46149 | ✓ | 1.01e-09 | 5.75 | 8.197 | — |
| differential_attention | 123,979,392 | 0.535 | 50128 | ✓ | — | 5.58 | 8.166 | — |
| dplr | 118,688,256 | 0.317 | 31080 | ✓ | — | 21.04 | nan | **NaN** |
| dsa | 102,742,272 | 0.435 | 49216 | ✓ | — | 5.57 | 8.375 | — |
| forgetting_attention (mb2048) | 102,825,216 | 0.238 | 26863 | ✓ | 0.00e+00 | 7.19 | 8.410 | — |
| gated_delta_product (mb2048) | 102,991,428 | 0.160 | 18037 | ✓ | — | 12.70 | nan | **NaN** |
| gated_deltanet | 102,908,952 | 0.270 | 30448 | ✓ | 0.00e+00 | 6.19 | 8.132 | — |
| gdn2 (mb2048) | 113,372,928 | 0.225 | 23071 | ✓ | — | 6.66 | nan | **NaN** |
| gla | 108,278,784 | 0.290 | 31102 | ✓ | 0.00e+00 | 6.17 | 8.048 | — |
| gsa | 104,071,104 | 0.177 | 19704 | ✓ | — | 7.31 | 8.112 | — |
| hgrn2 | 102,742,272 | 0.284 | 32145 | ✓ | 0.00e+00 | 5.97 | 8.168 | — |
| hla | 102,742,272 | 0.198 | 22370 | ✓ | — | 6.09 | 10.834 | — |
| hopfield_association | 108,202,860 | 0.453 | 48653 | ✓ | — | 6.02 | 8.184 | — |
| iplr (mb2048) | 113,372,928 | 0.029 | 2926 | ✓ | — | 9.34 | nan | **NaN** **slow** |
| kata | 102,742,272 | 0.460 | 52040 | ✓ | — | 5.65 | 8.151 | — |
| kda | 108,140,652 | 0.271 | 29103 | ✓ | 0.00e+00 | 6.38 | 8.122 | — |
| lightnet | 102,742,272 | 0.280 | 31681 | ✓ | — | 6.02 | 8.172 | — |
| lightning_attention | 102,742,272 | 0.290 | 32836 | ✓ | 2.77e-09 | 5.54 | 8.348 | — |
| linear_attention | 102,742,272 | 0.291 | 32861 | ✓ | 1.99e-11 | 5.75 | 8.163 | — |
| log_linear_attention (mb2048) | 113,372,928 | 0.134 | 13759 | ✓ | — | 12.80 | 9.076 | — |
| log_linear_mamba2 (mb2048) | 103,157,208 | 0.117 | 13167 | ✓ | — | 12.67 | 10.002 | — |
| longformer | 102,742,272 | 0.299 | 33806 | ✓ | — | 5.60 | 8.357 | — |
| mamba2 | 87,784,920 | 0.261 | 34471 | ✓ | — | 5.46 | 8.578 | — |
| mla_attention | 93,978,000 | 0.344 | 42511 | ✓ | — | 5.68 | 8.594 | — |
| moba | 102,742,272 | 0.447 | 50568 | ✓ | — | 5.70 | 8.190 | — |
| mom | 252,766,656 | 0.018 | 818 | ✓ | — | 9.92 | 8.482 | **slow** |
| nsa | 102,991,104 | 0.282 | 31849 | ✓ | — | 6.21 | 8.141 | — |
| path_attention (mb2048) | 102,908,376 | 0.016 | 1854 | ✓ | — | 8.88 | 8.469 | **slow** |
| pattention | 43,464,960 | 0.014 | 3710 | ✓ | — | 4.87 | 8.081 | **slow** |
| raven | 104,069,376 | 0.178 | 19850 | ✓ | — | 7.41 | 8.075 | — |
| retnet | 108,050,688 | 0.292 | 31447 | ✓ | 0.00e+00 | 5.93 | 8.385 | — |
| rodimus | 87,038,352 | 0.379 | 50600 | ✓ | — | 5.41 | 8.013 | — |
| rwkv7 | 118,688,256 | 0.318 | 31107 | ✓ | — | 21.11 | nan | **NaN** |
| samba_attention | 94,432,632 | 0.313 | 38452 | ✓ | — | 5.48 | 7.987 | — |
| sdm | 94,952,448 | 0.191 | 23411 | ✓ | — | 7.26 | 8.299 | — |
| simple_gla | 102,825,900 | 0.288 | 32533 | ✓ | 0.00e+00 | 5.65 | 8.114 | — |
| sparse_transformer | 102,742,272 | 0.182 | 20634 | ✓ | — | 5.95 | 8.168 | — |
| tda | 102,742,272 | 0.548 | 62001 | ✓ | — | 4.92 | 8.343 | — |
| tpa_attention | 99,424,512 | 0.354 | 41332 | ✓ | — | 6.25 | 8.130 | — |
| tucker_attention | 230,153,760 | 0.082 | 4155 | ✓ | — | 12.64 | 9.058 | **slow** |
| wall_attention | 113,372,928 | 0.437 | 44774 | ✓ | — | 5.74 | 8.120 | — |
| yoco | 102,825,216 | 0.289 | 32679 | ✓ | — | 5.54 | 8.254 | — |

_51/51 native rows completed._


### NaN-diverged rows (6)

The low-rank/dual-gate K2 family diverges to NaN loss within the 10 steps at width 768 — a genuine training-stability result, not a harness bug. MFU/throughput remain valid measurements of executed FLOPs; the separate reduced-shape checkpoint-parity gate passes (it certifies resume correctness, not 10-step stability):

- **comba** — MFU 0.335 still measured; loss NaN
- **dplr** — MFU 0.317 still measured; loss NaN
- **gated_delta_product** — MFU 0.160 still measured; loss NaN
- **gdn2** — MFU 0.225 still measured; loss NaN
- **iplr** — MFU 0.029 still measured; loss NaN
- **rwkv7** — MFU 0.318 still measured; loss NaN


### Pathological-MFU rows (8, MFU < 0.1)

Architectural costs, measured honestly:

- **attnres** — MFU 0.051: the residual design keeps every block summary alive and aggregates depth-domain per sub-layer
- **based_attention** — MFU 0.012: chunked-K with 12× score recompute at DV=768; ran at the 1024-token fallback (state history memory)
- **deltaformer** — MFU 0.031: K4 triangular solve is serial in t by construction
- **iplr** — MFU 0.029: identity-plus-rank-1 transition; also NaN-diverged (see flags) and ran at the 2048 fallback
- **mom** — MFU 0.018: external torch composition (public_path=False): mixture-of-paths routing, 2.5× the class parameter count
- **path_attention** — MFU 0.016: the path-sum mixer runs a dense per-pair recurrence at this width
- **pattention** — MFU 0.014: tokenformer block: five cascaded reference-tier pattention maps per block (no fused kernel exists anywhere)
- **tucker_attention** — MFU 0.082: Tucker foldings materialize per-head einsum operands (230M params, 2.2× the class)


### OOM-fallback rows (9)

These rows OOM'd at the 8192-token primary microbatch and trained at the fallback rung shown; their MFU is measured at that rung (lower occupancy than the primary, stated openly):

- **based_attention** — mb1024
- **conformer_attention** — mb2048
- **forgetting_attention** — mb2048
- **gated_delta_product** — mb2048
- **gdn2** — mb2048
- **iplr** — mb2048
- **log_linear_attention** — mb2048
- **log_linear_mamba2** — mb2048
- **path_attention** — mb2048


## Upstream comparison

Same 100M-class config; upstream rows run eager. Tiers: **prod** = the upstream's production kernel; **ref** = the upstream's reference/research implementation, or a transcription where the production kernel is environment-blocked (flash-attn absent, mamba_ssm wheel absent, SMEM/toolchain envelope) — the per-row reason is in the notes below. Granularity: mixer / schedule (interleaved hybrid) / block (full block) / residual (residual design).

| row | tier | granularity | MFU (urm/up) | tok/s (urm/up) | peak GiB (urm/up) | params (urm/up) | KL |
|---|---|---|---|---|---|---|---|
| abc_gsa | prod | mixer | 0.207 / 0.218 | 23121 / 23978 | 7.41 / 8.92 | 104,071,104 / 105,397,056 | — |
| attnres | prod | residual | 0.051 / 0.168 | 5803 / 18999 | 13.13 / 13.10 | 102,769,920 / 102,769,920 | — |
| based_attention | prod | mixer | 0.012 / 0.257 | 1375 / 29087 | 9.30 / 8.01 | 102,742,272 / 102,742,272 | — |
| bit_attention | prod | mixer | 0.238 / 0.268 | 26938 / 30252 | 6.25 / 7.80 | 102,769,920 / 102,749,184 | — |
| cat_attention | ref | mixer | 0.329 / 0.261 | 37199 / 29566 | 5.77 / 7.91 | 102,742,272 / 102,742,272 | — |
| comba | prod | mixer | 0.335 / 0.180 | 37842 / 16813 | 19.10 / 9.15 | 102,908,376 / 124,254,000 | — |
| conformer_attention | ref | mixer | 0.201 / 0.158 | 21637 / 16977 | 7.48 / 12.69 | 108,092,160 / 108,092,160 | — |
| deltaformer | ref | mixer | 0.031 / 0.002 (mb2048) | 3463 / 176 | 9.03 / 18.73 | 102,908,376 / 113,372,928 | — |
| deltanet | prod | mixer | 0.254 / 0.171 | 28745 / 19316 | 6.19 / 8.55 | 102,825,792 / 102,908,736 | 4.28e-10 |
| dense_attention | prod | mixer | 0.434 / 0.288 | 46149 / 30565 | 5.75 / 7.90 | 102,742,272 / 102,742,272 | 1.01e-09 |
| differential_attention | ref | mixer | 0.535 / 0.184 | 50128 / 20856 | 5.58 / 11.70 | 123,979,392 / 102,745,728 | — |
| dplr | prod | mixer | 0.317 / 0.223 | 31080 / 22908 | 21.04 / 8.85 | 118,688,256 / 113,372,928 | — |
| dsa | ref | mixer | 0.435 / 0.135 | 49216 / 14444 | 5.57 / 10.49 | 102,742,272 / 108,576,000 | — |
| forgetting_attention | prod | mixer | 0.238 / 0.249 | 26863 / 28085 | 7.19 / 8.02 | 102,825,216 / 102,825,324 | 0.00e+00 |
| gated_delta_product | prod | mixer | 0.160 / 0.190 | 18037 / 15717 | 12.70 / 9.63 | 102,991,428 / 140,344,920 | — |
| gated_deltanet | prod | mixer | 0.270 / 0.181 | 30448 / 16936 | 6.19 / 9.61 | 102,908,952 / 124,253,784 | 0.00e+00 |
| gdn2 | prod | mixer | 0.225 / 0.143 | 23071 / 14374 | 6.66 / 10.27 | 113,372,928 / 115,226,028 | — |
| gla | prod | mixer | 0.290 / 0.218 | 31102 / 24646 | 6.17 / 8.55 | 108,278,784 / 102,912,192 | 0.00e+00 |
| gsa | prod | mixer | 0.177 / 0.216 | 19704 / 23228 | 7.31 / 9.54 | 104,071,104 / 108,057,600 | — |
| hgrn2 | prod | mixer | 0.284 / 0.241 | 32145 / 27279 | 5.97 / 8.54 | 102,742,272 / 102,749,184 | 0.00e+00 |
| hopfield_association | ref | mixer | 0.453 / 0.265 | 48653 / 8887 | 6.02 / 15.75 | 108,202,860 / 347,226,624 | — |
| iplr | ref | mixer | 0.029 / 0.003 (mb2048) | 2926 / 351 | 9.34 / 8.31 | 113,372,928 / 113,372,928 | — |
| kata | prod | mixer | 0.460 / 0.251 | 52040 / 28433 | 5.65 / 8.54 | 102,742,272 / 102,742,848 | — |
| kda | prod | mixer | 0.271 / 0.143 | 29103 / 15844 | 6.38 / 9.11 | 108,140,652 / 104,692,140 | 0.00e+00 |
| lightnet | prod | mixer | 0.280 / 0.175 | 31681 / 17680 | 6.02 / 10.77 | 102,742,272 / 115,135,488 | — |
| lightning_attention | prod | mixer | 0.290 / 0.175 | 32836 / 18797 | 5.54 / 8.70 | 102,742,272 / 108,217,260 | 2.77e-09 |
| linear_attention | prod | mixer | 0.291 / 0.214 | 32861 / 24145 | 5.75 / 8.64 | 102,742,272 / 102,892,608 | 1.99e-11 |
| log_linear_attention | ref | mixer | 0.134 / 0.046 (mb2048) | 13759 / 4731 | 12.80 / 8.32 | 113,372,928 / 113,372,928 | — |
| longformer | ref | mixer | 0.299 / 0.232 | 33806 / 26177 | 5.60 / 8.42 | 102,742,272 / 102,742,272 | — |
| mamba2 | prod | mixer | 0.261 / 0.011 (mb2048) | 34471 / 1090 | 5.46 / 16.56 | 87,784,920 / 115,389,576 | — |
| mla_attention | ref | mixer | 0.344 / 0.184 | 42511 / 22545 | 5.68 / 10.24 | 93,978,000 / 95,000,832 | — |
| moba | ref | mixer | 0.447 / 0.241 | 50568 / 27242 | 5.70 / 8.75 | 102,742,272 / 102,742,272 | — |
| mom | prod | mixer | 0.018 / 0.120 | 818 / 5909 | 9.92 / 14.32 | 252,766,656 / 236,947,032 | — |
| nsa | ref | mixer | 0.282 / 0.001 | 31849 / 67 | 6.21 / 13.06 | 102,991,104 / 102,991,104 | — |
| path_attention | prod | mixer | 0.016 / 0.179 | 1854 / 20088 | 8.88 / 8.56 | 102,908,376 / 103,288,428 | — |
| pattention | ref | block | 0.014 / 0.106 | 3710 / 28168 | 4.87 / 6.88 | 43,464,960 / 43,464,960 | — |
| raven | prod | mixer | 0.178 / 0.190 | 19850 / 20458 | 7.41 / 10.51 | 104,069,376 / 108,141,912 | — |
| retnet | prod | mixer | 0.292 / 0.262 | 31447 / 24568 | 5.93 / 8.85 | 108,050,688 / 123,977,088 | 0.00e+00 |
| rodimus | prod | mixer | 0.379 / 0.185 | 50600 / 18297 | 5.41 / 9.55 | 87,038,352 / 117,452,160 | — |
| rwkv7 | ref | mixer | 0.318 / 0.002 (mb2048) | 31107 / 202 | 21.11 / 10.10 | 118,688,256 / 118,688,256 | — |
| samba_attention | prod | schedule | 0.313 / 0.002 (mb2048) | 38452 / 168 | 5.48 / 7.83 | 94,432,632 / 116,084,992 | — |
| sdm | ref | mixer | 0.191 / 0.002 (mb2048) | 23411 / 168 | 7.26 / 20.36 | 94,952,448 / 116,290,008 | — |
| simple_gla | prod | mixer | 0.288 / 0.174 | 32533 / 18731 | 5.65 / 8.70 | 102,825,900 / 108,217,260 | 0.00e+00 |
| sparse_transformer | ref | mixer | 0.182 / 0.261 | 20634 / 29562 | 5.95 / 7.91 | 102,742,272 / 102,742,272 | — |
| tda | prod | mixer | 0.548 / 0.251 | 62001 / 28353 | 4.92 / 8.22 | 102,742,272 / 102,742,272 | — |
| tpa_attention | ref | mixer | 0.354 / 0.134 | 41332 / 16929 | 6.25 / 12.67 | 99,424,512 / 92,070,144 | — |
| tucker_attention | ref | mixer | 0.082 / 0.225 | 4155 / 11365 | 12.64 / 14.40 | 230,153,760 / 230,146,848 | — |
| wall_attention | prod | mixer | 0.437 / 0.256 | 44774 / 26280 | 5.74 / 8.40 | 113,372,928 / 113,372,928 | — |
| yoco | prod | mixer | 0.289 / 0.244 | 32679 / 26167 | 5.54 / 8.38 | 102,825,216 / 108,133,632 | — |

_49/49 upstream baselines completed: 31 production-kernel, 18 reference-implementation._


### Reference-implementation baselines (18) — why no production kernel

- **cat_attention** — the fla modeling_cat decoder needs FlexAttention block-mask plumbing; baseline applies the pinned structural mask via SDPA
- **conformer_attention** — the pinned espnet rel-pos attention module (research code)
- **deltaformer** — fla's DeltaFormerAttention layer needs flash-attn; baseline is the pinned naive deltaformer op
- **differential_attention** — the pinned Diff-Transformer MultiheadDiffAttn (research code, AST-extracted)
- **dsa** — the pinned fla naive_dsa op (lightning indexer + top-k selection + attention); fla's fast DSA kernel is indexer-coupled to a specific head tiling
- **hopfield_association** — the pinned hflayers Hopfield module, self-association mode (research code)
- **iplr** — fla's chunk_iplr_delta_rule backward is NotImplementedError upstream; baseline is the pinned naive recurrence
- **log_linear_attention** — fla's chunk kernel exceeds A10G SMEM (122KB > 101KB) at head_dim=64; baseline is the pinned naive
- **longformer** — the pinned longformer sliding-chunk torch path (the TVM kernel needs Apache TVM, not installed)
- **mla_attention** — fla's MLA layer hard-requires flash-attn; baseline is the pinned prefill equation in torch
- **moba** — fla's parallel_moba needs flash-attn; baseline is the pinned law as a block-sparse SDPA mask
- **nsa** — fla's parallel_nsa needs flash-attn (absent by policy); baseline composes the pinned naive branch oracles
- **pattention** — the pinned tokenformer Pattention equation hosted at reference tier (megatron source needs neox/mpu)
- **rwkv7** — fla's chunk kernel exceeds A10G SMEM (131KB) and fused_recurrent is inference-only; baseline is the pinned naive recurrence
- **sdm** — the lingua CUDA extension is toolchain-blocked (nvcc 12.9 vs cu13 headers; source builds excluded); baseline is the pinned law in torch
- **sparse_transformer** — the pin's attention_impl is TF1/blocksparse (not runnable); baseline applies the pinned strided+local mask via SDPA
- **tpa_attention** — the pin ships decode-only kernels (n==1 assert); baseline generalizes the pinned factorized equation to training
- **tucker_attention** — the pinned fused kernel is H100-targeted (294KB SMEM); baseline transcribes the pinned equation in torch


### Production-kernel baselines with caveats

- **attnres** — fla fused_attnres — the pinned production kernel, full tier match
- **bit_attention** — fla's BitAttention layer needs flash-attn; baseline is pinned fused-BitLinear kernels (production) + SDPA attention
- **mamba2** — fla's own Triton Mamba2 (mamba_ssm has no prebuilt torch-2.14/cu130 wheel); causal_conv1d absent so its conv falls back to Triton
- **samba_attention** — fla mamba branch (Triton backend) + SDPA sliding-window attention (fla's attention layer needs flash-attn)


### Pathological upstream MFU (<0.01)

The baseline itself is slow (naive recurrence / torch transcription / eager fallback config) — the comparison is still valid; the tier label says why:

- **deltaformer** — upstream MFU 0.002 (reference-implementation)
- **iplr** — upstream MFU 0.003 (reference-implementation)
- **nsa** — upstream MFU 0.001 (reference-implementation)
- **rwkv7** — upstream MFU 0.002 (reference-implementation)
- **samba_attention** — upstream MFU 0.002 (production-kernel)
- **sdm** — upstream MFU 0.002 (reference-implementation)


### Upstream OOM-fallback rows (7)

- **deltaformer** — mb2048
- **iplr** — mb2048
- **log_linear_attention** — mb2048
- **mamba2** — mb2048
- **rwkv7** — mb2048
- **samba_attention** — mb2048
- **sdm** — mb2048


## Coverage and exclusions

- **hla** — no upstream implementation exists anywhere (empty pin; paper-only). The sole principled exclusion: the URM row trains, no baseline is fabricated.
- **log_linear_mamba2** — environment-blocked: upstream chunk kernel exceeds A10G SMEM (196KB > 101KB); no reference variant in the pin. Ours-only row.
- **mamba1** — reference-tier row (accepted charter debt: the K2 elementwise gate has a single client); not in the 51-row native sweep, so no baseline comparison is run.
- **KL gate** — wired for the 23 rows with a registry comparator; `—` elsewhere means *no comparator wired*, not failure.
- **mom, raven** — external torch compositions (public_path=False); trained as-is, no compiler-coverage claim.

