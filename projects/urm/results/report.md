# URM training benchmark

**51/51 URM configurations trained with finite losses and passed the checkpoint resume check.** 35 also have a usable upstream production-kernel comparison.

Each run trains a nine-layer decoder on FineWebEdu for ten measured updates after two warmup updates. Batch and microbatch are both 8192 tokens. Results measure decoder training on an A10G GPU.

MFU is shown as a percentage: estimated model work (`6 × parameters × tokens`, plus mixer/state work) divided by elapsed time and 70 TFLOPS. It is an estimate, not a GPU hardware counter.

## URM results

Peak memory includes temporary activations. Memory change is the last minus first allocation after a training step. The checkpoint column records whether saving, reloading and resuming passed.

| Model | Parameters | MFU (%) | Tokens/s | Checkpoint | Peak (GiB) | Final loss | Memory change (GiB) | Notes |
|---|---:|---:|---:|:---:|---:|---:|---:|---|
| abc_gsa | 104,071,104 | 18.6 | 20793 | ✓ | 7.17 | 16.853 | 0.000 | — |
| attnres | 102,769,920 | 17.1 | 19352 | ✓ | 12.57 | 17.010 | 0.000 | — |
| based_attention | 94,779,648 | 11.9 | 14562 | ✓ | 6.99 | 16.266 | 0.000 | — |
| bit_attention | 102,769,920 | 24.3 | 27480 | ✓ | 6.02 | 15.642 | 0.000 | — |
| cat_attention | 102,742,272 | 26.7 | 30207 | ✓ | 5.50 | 13.739 | 0.000 | — |
| comba | 102,908,376 | 25.8 | 29137 | ✓ | 19.03 | 9.051 | 0.000 | — |
| conformer_attention | 108,092,160 | 17.2 | 18498 | ✓ | 3.59 | 16.551 | 0.000 | activation checkpointing |
| deltaformer | 102,908,376 | 9.7 | 10925 | ✓ | 11.08 | 15.094 | 0.000 | MFU below 10% |
| deltanet | 102,825,792 | 21.3 | 24025 | ✓ | 5.95 | 9.011 | 0.000 | — |
| dense_attention | 102,742,272 | 31.8 | 33777 | ✓ | 5.50 | 16.590 | 0.000 | — |
| differential_attention | 123,979,392 | 21.1 | 19752 | ✓ | 6.96 | 16.945 | 0.000 | — |
| dplr | 118,688,256 | 23.8 | 23278 | ✓ | 19.82 | 8.723 | 0.000 | — |
| dsa | 102,742,272 | 31.2 | 35331 | ✓ | 5.31 | 10.151 | 0.000 | — |
| forgetting_attention | 102,825,216 | 18.8 | 21230 | ✓ | 3.55 | 9.426 | 0.000 | activation checkpointing |
| gated_delta_product | 102,991,428 | 11.4 | 12863 | ✓ | 6.29 | 8.854 | 0.000 | activation checkpointing |
| gated_deltanet | 102,908,952 | 22.2 | 25086 | ✓ | 5.95 | 9.039 | 0.000 | — |
| gdn2 | 113,372,928 | 17.5 | 17894 | ✓ | 3.63 | 8.834 | 0.000 | activation checkpointing |
| gla | 108,278,784 | 23.1 | 24824 | ✓ | 5.88 | 8.720 | 0.000 | — |
| gsa | 104,071,104 | 15.4 | 17205 | ✓ | 7.06 | 8.812 | 0.000 | — |
| hgrn2 | 102,742,272 | 23.1 | 26160 | ✓ | 5.72 | 9.036 | 0.000 | — |
| hla | 102,742,272 | 17.5 | 19732 | ✓ | 5.85 | 12.049 | 0.000 | — |
| hopfield_association | 108,202,860 | 31.9 | 34299 | ✓ | 5.77 | 9.726 | 0.000 | — |
| iplr | 113,372,928 | 24.0 | 24551 | ✓ | 19.56 | 9.617 | 0.000 | — |
| kata | 102,742,272 | 32.0 | 36168 | ✓ | 5.39 | 8.859 | 0.000 | — |
| kda | 108,140,652 | 22.0 | 23685 | ✓ | 6.10 | 9.136 | 0.000 | — |
| lightnet | 102,742,272 | 22.9 | 25866 | ✓ | 5.72 | 9.572 | 0.000 | — |
| lightning_attention | 102,742,272 | 23.5 | 26583 | ✓ | 5.30 | 11.140 | 0.000 | — |
| linear_attention | 102,742,272 | 23.5 | 26599 | ✓ | 5.51 | 16.045 | 0.000 | — |
| log_linear_attention | 113,372,928 | 6.9 | 7078 | ✓ | 6.35 | 8.953 | 0.000 | MFU below 10%, activation checkpointing |
| log_linear_mamba2 | 103,157,208 | 6.5 | 7267 | ✓ | 6.22 | 8.955 | 0.000 | MFU below 10%, activation checkpointing |
| longformer | 102,742,272 | 27.1 | 30665 | ✓ | 5.34 | 10.667 | 0.000 | — |
| mamba2 | 87,784,920 | 22.6 | 29921 | ✓ | 5.28 | 9.066 | 0.000 | — |
| mla_attention | 93,978,000 | 27.0 | 33316 | ✓ | 5.47 | 9.420 | 0.000 | — |
| moba | 102,742,272 | 31.4 | 35516 | ✓ | 5.44 | 9.285 | 0.000 | — |
| mom | 252,766,656 | 14.4 | 6640 | ✓ | 13.49 | 9.019 | 0.000 | — |
| nsa | 102,991,104 | 23.0 | 25966 | ✓ | 5.96 | 16.715 | 0.000 | — |
| path_attention | 102,908,376 | 5.9 | 6686 | ✓ | 4.25 | 15.087 | 0.000 | MFU below 10%, activation checkpointing |
| pattention | 43,464,960 | 17.4 | 46141 | ✓ | 5.67 | 14.906 | 0.000 | — |
| raven | 104,069,376 | 14.2 | 15869 | ✓ | 7.17 | 12.663 | 0.000 | — |
| retnet | 108,050,688 | 23.3 | 25092 | ✓ | 5.67 | 8.779 | 0.000 | — |
| rodimus | 87,038,352 | 30.2 | 40222 | ✓ | 5.23 | 9.226 | 0.000 | — |
| rwkv7 | 118,688,256 | 23.8 | 23274 | ✓ | 19.82 | 8.723 | 0.000 | — |
| samba_attention | 94,432,632 | 25.4 | 31269 | ✓ | 5.27 | 8.897 | 0.000 | — |
| sdm | 94,952,448 | 19.4 | 23768 | ✓ | 11.16 | 9.173 | 0.000 | native chunks |
| simple_gla | 102,825,900 | 23.3 | 26359 | ✓ | 5.41 | 9.108 | 0.000 | — |
| sparse_transformer | 102,742,272 | 14.0 | 15814 | ✓ | 5.71 | 12.976 | 0.000 | — |
| tda | 102,742,272 | 9.3 | 10514 | ✓ | 5.28 | 10.095 | 0.000 | MFU below 10% |
| tpa_attention | 99,424,512 | 27.7 | 32318 | ✓ | 6.02 | 9.076 | 0.000 | — |
| tucker_attention | 230,153,760 | 12.3 | 6246 | ✓ | 4.68 | 22.029 | 0.000 | activation checkpointing |
| wall_attention | 113,372,928 | 31.1 | 31865 | ✓ | 5.46 | 16.217 | 0.000 | — |
| yoco | 102,825,216 | 23.4 | 26477 | ✓ | 5.31 | 8.776 | 0.000 | — |

### Rows below 10% MFU (5)

These rows need further profiling. The operations below describe their implementations; they have not been confirmed as the bottlenecks.

- **deltaformer**: strict-causal correction and block triangular solve (MFU 9.7%).
- **log_linear_attention**: four saved state banks and activation recomputation (MFU 6.9%).
- **log_linear_mamba2**: four saved state banks and activation recomputation (MFU 6.5%).
- **path_attention**: Householder score correction and block triangular solve (MFU 5.9%).
- **tda**: threshold-ReLU-square forward/backward with restored Q/K/V gradients (MFU 9.3%).

## Production-kernel measurements

Every pair uses the same decoder dimensions, data, batch sizes, precision, optimizer, compilation and activation-checkpointing settings. Both runs must pass the training checks and record the same hardware and benchmark source version. Failed upstream kernels are excluded; a reference implementation cannot replace them.

Each cell shows **URM / upstream**. The tables separate comparisons that replace only the mixer kernel from those that use a different mixer module.

### Same projections and routing; only the kernel changes (14)

Both runs use the same projections, gates and routing code. The upstream run replaces the URM mixer kernel with the upstream production kernel.

| Model | MFU (%) | Tokens/s | Peak (GiB) | Parameters |
|---|---:|---:|---:|---:|
| attnres | 17.1 / 18.6 | 19352 / 21059 | 12.57 / 9.65 | 102,769,920 / 102,769,920 |
| comba | 25.8 / 31.4 | 29137 / 35488 | 19.03 / 5.62 | 102,908,376 / 102,908,376 |
| dense_attention | 31.8 / 36.3 | 33777 / 38597 | 5.50 / 5.39 | 102,742,272 / 102,742,272 |
| dplr | 23.8 / 27.3 | 23278 / 26779 | 19.82 / 6.18 | 118,688,256 / 118,688,256 |
| gated_delta_product | 11.4 / 20.6 | 12863 / 23282 | 6.29 / 3.55 | 102,991,428 / 102,991,428 |
| gdn2 | 17.5 / 19.7 | 17894 / 20168 | 3.63 / 3.64 | 113,372,928 / 113,372,928 |
| log_linear_attention | 6.9 / 21.1 | 7078 / 21656 | 6.35 / 3.63 | 113,372,928 / 113,372,928 |
| log_linear_mamba2 | 6.5 / 21.0 | 7267 / 23660 | 6.22 / 3.56 | 103,157,208 / 103,157,208 |
| mamba2 | 22.6 / 32.4 | 29921 / 42850 | 5.28 / 4.83 | 87,784,920 / 87,784,920 |
| raven | 14.2 / 30.4 | 15869 / 33925 | 7.17 / 5.76 | 104,069,376 / 104,069,376 |
| rwkv7 | 23.8 / 27.5 | 23274 / 26898 | 19.82 / 6.08 | 118,688,256 / 118,688,256 |
| samba_attention | 25.4 / 32.3 | 31269 / 39766 | 5.27 / 5.12 | 94,432,632 / 94,432,632 |
| sdm | 19.4 / 16.9 | 23768 / 20726 | 11.16 / 7.80 | 94,952,448 / 94,952,448 |
| tda | 9.3 / 12.9 | 10514 / 14632 | 5.28 / 5.60 | 102,742,272 / 102,742,272 |

### Different mixer modules (21)

The upstream run uses a separate mixer implementation. Its projections, gates or other layers can differ, as can its parameter count. These numbers compare decoder implementations; they do not isolate kernel speed.

| Model | MFU (%) | Tokens/s | Peak (GiB) | Parameters |
|---|---:|---:|---:|---:|
| abc_gsa | 18.6 / 28.1 | 20793 / 30959 | 7.17 / 6.61 | 104,071,104 / 105,397,056 |
| based_attention | 11.9 / 32.1 | 14562 / 39381 | 6.99 / 5.47 | 94,779,648 / 94,779,648 |
| bit_attention | 24.3 / 27.9 | 27480 / 31513 | 6.02 / 5.39 | 102,769,920 / 102,749,184 |
| deltanet | 21.3 / 29.2 | 24025 / 32999 | 5.95 / 6.24 | 102,825,792 / 102,908,736 |
| forgetting_attention | 18.8 / 26.7 | 21230 / 30148 | 3.55 / 3.56 | 102,825,216 / 102,825,324 |
| gated_deltanet | 22.2 / 27.5 | 25086 / 25742 | 5.95 / 7.30 | 102,908,952 / 124,253,784 |
| gla | 23.1 / 30.2 | 24824 / 34124 | 5.88 / 6.25 | 108,278,784 / 102,912,192 |
| gsa | 15.4 / 25.4 | 17205 / 27354 | 7.06 / 7.23 | 104,071,104 / 108,057,600 |
| hgrn2 | 23.1 / 28.5 | 26160 / 32258 | 5.72 / 6.23 | 102,742,272 / 102,749,184 |
| kata | 32.0 / 30.0 | 36168 / 33904 | 5.39 / 6.24 | 102,742,272 / 102,742,848 |
| kda | 22.0 / 26.1 | 23685 / 28974 | 6.10 / 6.81 | 108,140,652 / 104,692,140 |
| lightnet | 22.9 / 20.5 | 25866 / 20659 | 5.72 / 8.47 | 102,742,272 / 115,135,488 |
| lightning_attention | 23.5 / 29.8 | 26583 / 32009 | 5.30 / 6.40 | 102,742,272 / 108,217,260 |
| linear_attention | 23.5 / 30.2 | 26599 / 34141 | 5.51 / 6.34 | 102,742,272 / 102,892,608 |
| mom | 14.4 / 17.1 | 6640 / 8413 | 13.49 / 12.01 | 252,766,656 / 236,947,032 |
| path_attention | 5.9 / 22.0 | 6686 / 24755 | 4.25 / 3.56 | 102,908,376 / 103,288,428 |
| retnet | 23.3 / 30.3 | 25092 / 28396 | 5.67 / 6.55 | 108,050,688 / 123,977,088 |
| rodimus | 30.2 / 26.9 | 40222 / 26624 | 5.23 / 7.25 | 87,038,352 / 117,452,160 |
| simple_gla | 23.3 / 29.8 | 26359 / 32034 | 5.41 / 6.40 | 102,825,900 / 108,217,260 |
| wall_attention | 31.1 / 32.1 | 31865 / 32926 | 5.46 / 5.67 | 113,372,928 / 113,372,928 |
| yoco | 23.4 / 30.5 | 26477 / 32805 | 5.31 / 6.08 | 102,825,216 / 108,133,632 |

### Production adapters

The upstream runs use pinned production kernels. Loading requirements and numerical adjustments are documented in [benchmark implementation details](../docs/benchmark.md#corrections).

SDM runs through native URM with chunks of up to 128 tokens. Its upstream comparison uses the original Meta CUDA/Triton kernels with 64-token chunks and the same projections and routes. See [SDM results and historical MFU accounting](../docs/sdm-optimization.md).

## Upstreams excluded from production comparison

These rows have no usable production-kernel comparison. The reason is listed for each row. Reference or research implementations can be run separately with `train.upstream --include-reference`.

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

## Measurement details

- Decoder: width 768, 9 layers, 12 heads, head dimension 64, sequence length 512, vocabulary 50,304.
- BF16 training with FP32 kernel accumulation. Both runs compile the model around Python plan dispatch and upstream kernels that execute outside the compiled graph.
- GPU timing is synchronized. Optimizer settings and gradient clipping match in both runs.
- Activation checkpointing is enabled for the rows marked in the table. It recomputes activations to reduce memory use and is separate from the checkpoint resume check.
- OOM retries are disabled. Failed runs are recorded at the requested batch size.

Torch 2.14.0+cu130; GPU: NVIDIA A10G; measurement version 2. Benchmark source hashes: `345b161d5a98, 487a6ddbd732`. SDM was remeasured after native integration; the other rows retain their accepted measurements. Each URM/upstream pair has matching source hashes.

The largest change in step-end allocation within any URM run is 0.0 KiB. This includes intermediate steps, not just the first and last.

The largest step-end allocation change among compared upstream runs is 12.0 KiB.

Full configs, losses, memory traces and source hashes are recorded in the per-model JSONs under [sweep/](sweep/) and [upstream/](upstream/). See [how to reproduce the benchmark](../docs/benchmark.md#running).

## What the checks establish

Checkpoint resume is tested on smaller models without compilation. Full-size runs separately check losses and gradient norms at every update. Output-distribution comparisons (KL) are recorded in JSON where available. Ten training updates do not establish long-run convergence or equivalence to complete upstream models and serving workloads. See the [evidence policy](../docs/evidence.md).
