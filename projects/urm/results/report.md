# URM training measurements — corrected campaign

**Coverage.** 51/51 URM rows have finite training trajectories and passing checkpoint gates; 35 have eligible production-kernel measurements.

Each row trains a decoder surround on finewebedu for 10 measured steps after two full optimizer warmup steps. This measures the training harness; it does not claim source-model or serving parity. See the [evidence policy](../docs/evidence.md).

**Protocol.** width=768, layers=9, heads=12, head_dim=64, sequence=512, vocab=50304. Effective batch and microbatch are both 8192 tokens. Both arms compile the surround; Python plan dispatch and unsupported upstream kernels remain eager boundaries. Optimizer roles and clipping are identical in both arms. bf16 autocast with fp32 kernel accumulation; timing is synchronized before and after measurement. MFU is an approximate parameter/state FLOP estimate divided by the adopted 70 TFLOPS A10G peak, not a hardware utilization counter.

Memory-heavy rows use the same explicit activation-checkpointing policy in both arms. Based uses its upstream default of 16 query/key features and 64 value channels. OOM retries are disabled in this campaign; failures are recorded without substituting a smaller batch. Non-finite losses or gradient norms fail the run.

TDA and Differential Attention training use existing public native calls with external differentiable merges to retain projection and mixing-weight gradients. Two core autograd-wrapper fixes retain only flags/shapes rather than bias/mask or gate tensors, preventing graph retention after backward; kernel math is unchanged.

SDM uses native public product-key routes with an external compiled PyTorch state schedule (chunk size 128), including decay and its gradients. Its shared-frontend baseline calls the unmodified pinned Meta CUDA/Triton kernels (chunk size 64), built with an isolated matching CUDA toolkit. The core URM backend is unchanged by this SDM optimization. See [SDM measurements and historical MFU accounting](../docs/sdm-optimization.md). Accepted non-SDM measurements retain their original source fingerprints; each paired row still requires identical fingerprints in both arms.

**Provenance.** torch 2.14.0+cu130, NVIDIA A10G; measurement version 2; source fingerprint(s) `345b161d5a98, e4b6d1444e12`. Each JSON contains its actual config, full source hash, loss trajectory, memory trajectory, and attempted microbatches. FLA and Mamba production adapters verify their pinned sources.

**Memory audit.** The largest within-run step-end allocation range across verified URM rows is 0.000 GiB over the measured steps. This checks intermediate steps as well as the first-to-last drift.

The largest step-end allocation range among eligible upstream runs is 12.0 KiB.

## URM rows

`slow` marks approximate MFU below 0.10. `ckpt` means activation checkpointing is enabled; checkpoint correctness is reported separately. Memory drift is the last minus first step-end allocation.

| row | params | MFU | tok/s | checkpoint gate | peak GiB | loss | memory drift GiB | flags |
|---|---:|---:|---:|:---:|---:|---:|---:|---|
| abc_gsa | 104,071,104 | 0.186 | 20793 | ✓ | 7.17 | 16.853 | 0.000 | — |
| attnres | 102,769,920 | 0.171 | 19352 | ✓ | 12.57 | 17.010 | 0.000 | — |
| based_attention | 94,779,648 | 0.119 | 14562 | ✓ | 6.99 | 16.266 | 0.000 | — |
| bit_attention | 102,769,920 | 0.243 | 27480 | ✓ | 6.02 | 15.642 | 0.000 | — |
| cat_attention | 102,742,272 | 0.267 | 30207 | ✓ | 5.50 | 13.739 | 0.000 | — |
| comba | 102,908,376 | 0.258 | 29137 | ✓ | 19.03 | 9.051 | 0.000 | — |
| conformer_attention | 108,092,160 | 0.172 | 18498 | ✓ | 3.59 | 16.551 | 0.000 | ckpt |
| deltaformer | 102,908,376 | 0.097 | 10925 | ✓ | 11.08 | 15.094 | 0.000 | slow |
| deltanet | 102,825,792 | 0.213 | 24025 | ✓ | 5.95 | 9.011 | 0.000 | — |
| dense_attention | 102,742,272 | 0.318 | 33777 | ✓ | 5.50 | 16.590 | 0.000 | — |
| differential_attention | 123,979,392 | 0.211 | 19752 | ✓ | 6.96 | 16.945 | 0.000 | — |
| dplr | 118,688,256 | 0.238 | 23278 | ✓ | 19.82 | 8.723 | 0.000 | — |
| dsa | 102,742,272 | 0.312 | 35331 | ✓ | 5.31 | 10.151 | 0.000 | — |
| forgetting_attention | 102,825,216 | 0.188 | 21230 | ✓ | 3.55 | 9.426 | 0.000 | ckpt |
| gated_delta_product | 102,991,428 | 0.114 | 12863 | ✓ | 6.29 | 8.854 | 0.000 | ckpt |
| gated_deltanet | 102,908,952 | 0.222 | 25086 | ✓ | 5.95 | 9.039 | 0.000 | — |
| gdn2 | 113,372,928 | 0.175 | 17894 | ✓ | 3.63 | 8.834 | 0.000 | ckpt |
| gla | 108,278,784 | 0.231 | 24824 | ✓ | 5.88 | 8.720 | 0.000 | — |
| gsa | 104,071,104 | 0.154 | 17205 | ✓ | 7.06 | 8.812 | 0.000 | — |
| hgrn2 | 102,742,272 | 0.231 | 26160 | ✓ | 5.72 | 9.036 | 0.000 | — |
| hla | 102,742,272 | 0.175 | 19732 | ✓ | 5.85 | 12.049 | 0.000 | — |
| hopfield_association | 108,202,860 | 0.319 | 34299 | ✓ | 5.77 | 9.726 | 0.000 | — |
| iplr | 113,372,928 | 0.240 | 24551 | ✓ | 19.56 | 9.617 | 0.000 | — |
| kata | 102,742,272 | 0.320 | 36168 | ✓ | 5.39 | 8.859 | 0.000 | — |
| kda | 108,140,652 | 0.220 | 23685 | ✓ | 6.10 | 9.136 | 0.000 | — |
| lightnet | 102,742,272 | 0.229 | 25866 | ✓ | 5.72 | 9.572 | 0.000 | — |
| lightning_attention | 102,742,272 | 0.235 | 26583 | ✓ | 5.30 | 11.140 | 0.000 | — |
| linear_attention | 102,742,272 | 0.235 | 26599 | ✓ | 5.51 | 16.045 | 0.000 | — |
| log_linear_attention | 113,372,928 | 0.069 | 7078 | ✓ | 6.35 | 8.953 | 0.000 | slow, ckpt |
| log_linear_mamba2 | 103,157,208 | 0.065 | 7267 | ✓ | 6.22 | 8.955 | 0.000 | slow, ckpt |
| longformer | 102,742,272 | 0.271 | 30665 | ✓ | 5.34 | 10.667 | 0.000 | — |
| mamba2 | 87,784,920 | 0.226 | 29921 | ✓ | 5.28 | 9.066 | 0.000 | — |
| mla_attention | 93,978,000 | 0.270 | 33316 | ✓ | 5.47 | 9.420 | 0.000 | — |
| moba | 102,742,272 | 0.314 | 35516 | ✓ | 5.44 | 9.285 | 0.000 | — |
| mom | 252,766,656 | 0.144 | 6640 | ✓ | 13.49 | 9.019 | 0.000 | — |
| nsa | 102,991,104 | 0.230 | 25966 | ✓ | 5.96 | 16.715 | 0.000 | — |
| path_attention | 102,908,376 | 0.059 | 6686 | ✓ | 4.25 | 15.087 | 0.000 | slow, ckpt |
| pattention | 43,464,960 | 0.174 | 46141 | ✓ | 5.67 | 14.906 | 0.000 | — |
| raven | 104,069,376 | 0.142 | 15869 | ✓ | 7.17 | 12.663 | 0.000 | — |
| retnet | 108,050,688 | 0.233 | 25092 | ✓ | 5.67 | 8.779 | 0.000 | — |
| rodimus | 87,038,352 | 0.302 | 40222 | ✓ | 5.23 | 9.226 | 0.000 | — |
| rwkv7 | 118,688,256 | 0.238 | 23274 | ✓ | 19.82 | 8.723 | 0.000 | — |
| samba_attention | 94,432,632 | 0.254 | 31269 | ✓ | 5.27 | 8.897 | 0.000 | — |
| sdm | 94,952,448 | 0.197 | 24215 | ✓ | 11.17 | 9.173 | 0.000 | external state schedule |
| simple_gla | 102,825,900 | 0.233 | 26359 | ✓ | 5.41 | 9.108 | 0.000 | — |
| sparse_transformer | 102,742,272 | 0.140 | 15814 | ✓ | 5.71 | 12.976 | 0.000 | — |
| tda | 102,742,272 | 0.093 | 10514 | ✓ | 5.28 | 10.095 | 0.000 | slow |
| tpa_attention | 99,424,512 | 0.277 | 32318 | ✓ | 6.02 | 9.076 | 0.000 | — |
| tucker_attention | 230,153,760 | 0.123 | 6246 | ✓ | 4.68 | 22.029 | 0.000 | ckpt |
| wall_attention | 113,372,928 | 0.311 | 31865 | ✓ | 5.46 | 16.217 | 0.000 | — |
| yoco | 102,825,216 | 0.234 | 26477 | ✓ | 5.31 | 8.776 | 0.000 | — |

### Remaining low-MFU rows (5)

These measurements remain visible. The listed work explains the execution path, and does not establish that the implementation is optimal.

- **deltaformer**: strict-causal correction and block triangular solve (MFU 0.097).
- **log_linear_attention**: four saved state banks and activation recomputation (MFU 0.069).
- **log_linear_mamba2**: four saved state banks and activation recomputation (MFU 0.065).
- **path_attention**: Householder score correction and block triangular solve (MFU 0.059).
- **tda**: threshold-ReLU-square forward/backward with restored Q/K/V gradients (MFU 0.093).

## Production-kernel measurements

Only verified runs with matching shapes, effective batch, microbatch, precision, compilation policy, checkpointing policy, environment, and source fingerprint enter this table. There is no reference-kernel replacement after an upstream failure. `shared` uses a common external frontend; `family` is an architecture-family baseline whose projections or other mixer-side layers can differ. Parameter counts are shown explicitly; family measurements are not isolated kernel speedup claims.

| row | scope | MFU (URM / upstream) | tok/s (URM / upstream) | peak GiB (URM / upstream) | params (URM / upstream) |
|---|---|---:|---:|---:|---:|
| abc_gsa | family | 0.186 / 0.281 | 20793 / 30959 | 7.17 / 6.61 | 104,071,104 / 105,397,056 |
| attnres | shared | 0.171 / 0.186 | 19352 / 21059 | 12.57 / 9.65 | 102,769,920 / 102,769,920 |
| based_attention | family | 0.119 / 0.321 | 14562 / 39381 | 6.99 / 5.47 | 94,779,648 / 94,779,648 |
| bit_attention | family | 0.243 / 0.279 | 27480 / 31513 | 6.02 / 5.39 | 102,769,920 / 102,749,184 |
| comba | shared | 0.258 / 0.314 | 29137 / 35488 | 19.03 / 5.62 | 102,908,376 / 102,908,376 |
| deltanet | family | 0.213 / 0.292 | 24025 / 32999 | 5.95 / 6.24 | 102,825,792 / 102,908,736 |
| dense_attention | shared | 0.318 / 0.363 | 33777 / 38597 | 5.50 / 5.39 | 102,742,272 / 102,742,272 |
| dplr | shared | 0.238 / 0.273 | 23278 / 26779 | 19.82 / 6.18 | 118,688,256 / 118,688,256 |
| forgetting_attention | family | 0.188 / 0.267 | 21230 / 30148 | 3.55 / 3.56 | 102,825,216 / 102,825,324 |
| gated_delta_product | shared | 0.114 / 0.206 | 12863 / 23282 | 6.29 / 3.55 | 102,991,428 / 102,991,428 |
| gated_deltanet | family | 0.222 / 0.275 | 25086 / 25742 | 5.95 / 7.30 | 102,908,952 / 124,253,784 |
| gdn2 | shared | 0.175 / 0.197 | 17894 / 20168 | 3.63 / 3.64 | 113,372,928 / 113,372,928 |
| gla | family | 0.231 / 0.302 | 24824 / 34124 | 5.88 / 6.25 | 108,278,784 / 102,912,192 |
| gsa | family | 0.154 / 0.254 | 17205 / 27354 | 7.06 / 7.23 | 104,071,104 / 108,057,600 |
| hgrn2 | family | 0.231 / 0.285 | 26160 / 32258 | 5.72 / 6.23 | 102,742,272 / 102,749,184 |
| kata | family | 0.320 / 0.300 | 36168 / 33904 | 5.39 / 6.24 | 102,742,272 / 102,742,848 |
| kda | family | 0.220 / 0.261 | 23685 / 28974 | 6.10 / 6.81 | 108,140,652 / 104,692,140 |
| lightnet | family | 0.229 / 0.205 | 25866 / 20659 | 5.72 / 8.47 | 102,742,272 / 115,135,488 |
| lightning_attention | family | 0.235 / 0.298 | 26583 / 32009 | 5.30 / 6.40 | 102,742,272 / 108,217,260 |
| linear_attention | family | 0.235 / 0.302 | 26599 / 34141 | 5.51 / 6.34 | 102,742,272 / 102,892,608 |
| log_linear_attention | shared | 0.069 / 0.211 | 7078 / 21656 | 6.35 / 3.63 | 113,372,928 / 113,372,928 |
| log_linear_mamba2 | shared | 0.065 / 0.210 | 7267 / 23660 | 6.22 / 3.56 | 103,157,208 / 103,157,208 |
| mamba2 | shared | 0.226 / 0.324 | 29921 / 42850 | 5.28 / 4.83 | 87,784,920 / 87,784,920 |
| mom | family | 0.144 / 0.171 | 6640 / 8413 | 13.49 / 12.01 | 252,766,656 / 236,947,032 |
| path_attention | family | 0.059 / 0.220 | 6686 / 24755 | 4.25 / 3.56 | 102,908,376 / 103,288,428 |
| raven | shared | 0.142 / 0.304 | 15869 / 33925 | 7.17 / 5.76 | 104,069,376 / 104,069,376 |
| retnet | family | 0.233 / 0.303 | 25092 / 28396 | 5.67 / 6.55 | 108,050,688 / 123,977,088 |
| rodimus | family | 0.302 / 0.269 | 40222 / 26624 | 5.23 / 7.25 | 87,038,352 / 117,452,160 |
| rwkv7 | shared | 0.238 / 0.275 | 23274 / 26898 | 19.82 / 6.08 | 118,688,256 / 118,688,256 |
| samba_attention | shared | 0.254 / 0.323 | 31269 / 39766 | 5.27 / 5.12 | 94,432,632 / 94,432,632 |
| sdm | shared | 0.197 / 0.169 | 24215 / 20747 | 11.17 / 7.80 | 94,952,448 / 94,952,448 |
| simple_gla | family | 0.233 / 0.298 | 26359 / 32034 | 5.41 / 6.40 | 102,825,900 / 108,217,260 |
| tda | shared | 0.093 / 0.129 | 10514 / 14632 | 5.28 / 5.60 | 102,742,272 / 102,742,272 |
| wall_attention | family | 0.311 / 0.321 | 31865 / 32926 | 5.46 / 5.67 | 113,372,928 / 113,372,928 |
| yoco | family | 0.234 / 0.305 | 26477 / 32805 | 5.31 / 6.08 | 102,825,216 / 108,133,632 |

### Production adapters

- Mamba-2 uses the unmodified pinned SSD Triton package without importing its optional CUDA extension; RWKV-7 uses the pinned chunk kernel with `chunk_size=16`.
- Comba, GDN2, DeltaProduct, DPLR, RWKV-7, Mamba-2 and both log-linear rows share the URM external frontend. The upstream call replaces only the mixer kernel.
- Samba uses the same Mamba-2/RoPE schedule with pinned SSD and production SDPA. Raven shares the eight-slot/top-k-two deterministic frontend and uses pinned chunk GSA. Equal duplication of all slots meets its 16-slot backward minimum while preserving outputs and gradients.
- TDA supplies contiguous head batches and gradients, applies the native query scaling, and matches the registered differential merge (identical paths with lambda=0.5). Supported Triton launch options select IEEE fp32 dots and one pipeline stage.
- Log-linear attention uses independent heads as single-group batches and an A10G one-stage pipeline. Operands and level scales are bf16, and the last scale repeats for the capped bank. The upstream checkout remains unmodified.

- SDM uses the verified original Meta sparse-IP/gather CUDA extensions and Triton WY kernels. Partition-local identity padding, autocast isolation, and a saved terminal snapshot adapt the production API without changing its source or substituting a reference kernel.

## Upstreams excluded from production comparison

Unavailable kernels, research implementations, failed training, and mismatched measurements are listed without paired throughput. Reference implementations can be requested as separate diagnostics using `train.upstream --include-reference`; they remain excluded from the table above.

- **cat_attention**: the fla modeling_cat decoder needs FlexAttention block-mask plumbing; baseline applies the pinned structural mask via SDPA.
- **conformer_attention**: the pinned espnet rel-pos attention module (research code).
- **deltaformer**: fla's DeltaFormerAttention layer needs flash-attn; baseline is the pinned naive deltaformer op.
- **differential_attention**: the pinned Diff-Transformer MultiheadDiffAttn (research code, AST-extracted).
- **dsa**: the pinned fla naive_dsa op (lightning indexer + top-k selection + attention); fla's fast DSA kernel is indexer-coupled to a specific head tiling.
- **hopfield_association**: the pinned hflayers Hopfield module, self-association mode (research code).
- **iplr**: fla's chunk_iplr_delta_rule backward is NotImplementedError upstream; baseline is the pinned naive recurrence.
- **longformer**: the pinned longformer sliding-chunk torch path (the TVM kernel needs Apache TVM, not installed).
- **mla_attention**: fla's MLA layer hard-requires flash-attn; baseline is the pinned prefill equation in torch.
- **moba**: fla's parallel_moba needs flash-attn; baseline is the pinned law as a block-sparse SDPA mask.
- **nsa**: fla's parallel_nsa needs flash-attn (absent by policy); baseline composes the pinned naive branch oracles.
- **pattention**: the pinned tokenformer Pattention equation hosted at reference tier (megatron source needs neox/mpu).
- **sparse_transformer**: the pin's attention_impl is TF1/blocksparse (not runnable); baseline applies the pinned strided+local mask via SDPA.
- **tpa_attention**: the pin ships decode-only kernels (n==1 assert); baseline generalizes the pinned factorized equation to training.
- **tucker_attention**: the pinned fused kernel is H100-targeted (294KB SMEM); baseline transcribes the pinned equation in torch.
- **hla**: no upstream measurement available.

## Validation limits

Checkpoint gates use reduced eager models and certify resume behavior, not full-scale accuracy. Finite full-scale loss and gradient norms are checked separately at every update. KL is retained in the per-row JSON wherever a comparator is wired; absent KL is not a parity claim. These ten-step runs do not establish long-run convergence.
