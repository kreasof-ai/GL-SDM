# Catalog upstream validation: coverage, upstream parity, upstream performance

Status: evidence record, regenerated from the live compiler by
`benchmarks/catalog_upstream_validation.py`. For each of the 62
representation-covered recipes (recipes that lower into a canonical core and
match their independent equation), this table records three measurements:

- **coverage**: the recipe lowers into a canonical core
  (`urm.oracles.composition.execute_canonical`).
- **upstream parity**: the LIBRARY plan (pinned upstream FLA / SDM / SDPA
  adapter) on CUDA vs the REFERENCE plan (independent equation) on identical
  operands; pass when max abs err < 0.02 or max rel err < 0.01.
  Upstream adapters run in float32/bfloat16 with chunked kernels, so the
  criterion is looser than the FP64 representation-coverage gate.
- **upstream performance**: when both NATIVE and LIBRARY compile, paired
  interleaved synchronized wall timing on CUDA; overhead is the paired median
  of `(native - library) / library` (negative = URM-native is faster).

**42 of 62 covered recipes pass upstream parity; 23 have native-vs-upstream performance measured; 1 have no upstream adapter; 36 have no native kernel.**

## (a) Native kernel + upstream parity pass + performance measured

| Recipe | Family | upstream anchor | dtype | abs err | overhead (native vs upstream) | native ms | upstream ms |
|---|---|---|---|---|---|---|---|
| `gated_delta_net` | RECURRENCE | `fla_gated_delta_rule_adapter` | bfloat16 | 7.10e-02 | -69.5% | 0.235 | 0.773 |
| `delta_net` | RECURRENCE | `fla_chunk_delta_rule_adapter` | bfloat16 | 1.23e-01 | -62.0% | 0.322 | 0.843 |
| `lightnet_gla_core` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.10e-02 | -58.4% | 0.196 | 0.460 |
| `hgrn2_ssm_core` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.10e-02 | -57.8% | 0.197 | 0.461 |
| `gla` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.10e-02 | -56.9% | 0.197 | 0.459 |
| `mom_selected_memory_core` | RECURRENCE | `fla_gated_delta_rule_adapter` | bfloat16 | 4.31e-03 | -55.6% | 0.363 | 0.801 |
| `rodimus_gla_core` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.37e-03 | -49.9% | 0.228 | 0.468 |
| `simple_gla` | RECURRENCE | `fla_chunk_simple_gla_adapter` | float32 | 1.18e-02 | -45.8% | 0.193 | 0.348 |
| `nsa_selected_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +12.2% | 0.282 | 0.255 |
| `dsa_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +18.1% | 0.365 | 0.316 |
| `sparse_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +21.5% | 0.296 | 0.245 |
| `cat_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +25.5% | 0.394 | 0.317 |
| `differential_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 1.79e-07 | +98.3% | 0.398 | 0.208 |
| `mha` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +170.7% | 0.269 | 0.098 |
| `mla_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +178.3% | 0.266 | 0.098 |
| `mqa` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +183.5% | 0.265 | 0.098 |
| `pattention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 1.79e-07 | +184.4% | 0.266 | 0.095 |
| `samba_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +191.0% | 0.264 | 0.095 |
| `conformer_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.09e-07 | +193.1% | 0.308 | 0.108 |
| `tpa_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +207.4% | 0.271 | 0.094 |
| `hopfield_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 1.79e-07 | +213.7% | 0.275 | 0.096 |
| `gqa` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +216.6% | 0.271 | 0.089 |
| `foveal_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +223.1% | 0.361 | 0.119 |

## (b) Upstream parity only (no native-vs-upstream timing)

| Recipe | Family | upstream anchor | dtype | abs err | rel err | parity | note |
|---|---|---|---|---|---|---|---|
| `based_attention_core` | RECURRENCE | `fla_fused_chunk_based_adapter` | float32 | 2.38e-06 | 1.52e-06 | pass | no native kernel |
| `comba_core` | RECURRENCE | `fla_chunk_comba_adapter` | bfloat16 | 4.22e+00 | 4.94e-01 | fail | no native kernel |
| `gated_delta_product_core` | RECURRENCE | `fla_chunk_gated_delta_product_adapter` | bfloat16 | 1.18e-01 | 4.80e-03 | pass | no native kernel |
| `gated_oja_core` | RECURRENCE | `fla_chunk_gated_oja_adapter` | bfloat16 | 1.32e-02 | 3.57e-03 | pass | no native kernel |
| `gdn2_core` | RECURRENCE | `fla_chunk_gdn2_adapter` | float32 | 5.92e-03 | 1.09e-03 | pass | no native kernel |
| `generalized_delta_dplr_core` | RECURRENCE | `fla_chunk_dplr_adapter` | bfloat16 | 3.61e-02 | 6.84e-03 | pass | no native kernel |
| `generalized_delta_iplr_core` | RECURRENCE | `fla_fused_recurrent_iplr_adapter` | float32 | 5.96e-07 | 8.13e-08 | pass | no native kernel |
| `h3_ssm_fft_core` | RECURRENCE | `h3_ssm_fft_convolution_adapter` | float32 | 9.54e-07 | 5.09e-07 | pass | no native kernel |
| `hla_second_order_core` | RECURRENCE | `hla_second_order_triton_adapter` | float32 | 5.72e-06 | 1.18e-07 | pass | no native kernel |
| `hyena_fftconv_core` | RECURRENCE | `hyena_fft_convolution_adapter` | float32 | 4.77e-07 | 6.33e-08 | pass | no native kernel |
| `kda_core` | RECURRENCE | `fla_chunk_kda_adapter` | float32 | 1.12e-02 | 2.41e-03 | pass | no native kernel |
| `lightning_attention_core` | RECURRENCE | `fla_chunk_simple_gla_adapter` | float32 | 1.19e-02 | 1.04e-03 | pass | no native kernel |
| `linear_attention` | RECURRENCE | `fla_chunk_linear_attention_adapter` | bfloat16 | 1.09e-02 | 7.77e-03 | pass | no native kernel |
| `mesa_net_core` | RECURRENCE | `fla_chunk_mesa_net_adapter` | bfloat16 | 6.64e-03 | 6.28e-03 | pass | no native kernel |
| `rebased_attention_core` | RECURRENCE | `fla_parallel_rebased_adapter` | float32 | 2.23e-04 | 1.59e-04 | pass | no native kernel |
| `retention_core` | RECURRENCE | `fla_chunk_simple_gla_adapter` | float32 | 1.19e-02 | 1.04e-03 | pass | no native kernel |
| `rwkv4_memory_core` | RECURRENCE | `fla_fused_recurrent_rwkv4_adapter` | float32 | 1.19e-07 | 8.40e-08 | pass | no native kernel |
| `rwkv6_memory_core` | RECURRENCE | `fla_fused_recurrent_rwkv6_adapter` | float32 | 2.38e-07 | 7.88e-08 | pass | no native kernel |
| `rwkv7_transition_core` | RECURRENCE | `fla_chunk_rwkv7_adapter` | bfloat16 | 3.74e-02 | 3.54e-03 | pass | no native kernel |
| `wall_attention_core` | SOFTMAX | `fla_parallel_wall_attention_adapter` | float32 | 1.19e-07 | 9.21e-08 | pass | no native kernel |

## (c) No upstream adapter

| Recipe | Family | reason |
|---|---|---|
| `sparse_delta_memory` | SPARSE_DELTA | ValueError: K3 uses the URM-native or reference sparse-state anchor |

## (d) No native kernel

| Recipe | Family | upstream parity | note |
|---|---|---|---|
| `abc_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `based_attention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `comba_core` | RECURRENCE | fail | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `deltaformer_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `gated_delta_product_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `gated_oja_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `gdn2_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `generalized_delta_dplr_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `generalized_delta_iplr_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `gru_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `gsa_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `h3_ssm_fft_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `hla_second_order_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `hyena_fftconv_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `kata_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `kda_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `lightning_attention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `linear_attention` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `longformer_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `m2rnn_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `mamba2_ssm_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `mamba3_siso_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `mesa_net_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `momentum_delta_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `parallax_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `rebased_attention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `retention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `rnn_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `rwkv4_memory_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `rwkv6_memory_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `rwkv7_transition_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `tda_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `titans_linear_memory_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `ttt_linear_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `tucker_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |
| `wall_attention_core` | SOFTMAX | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential compos… |

## Upstream parity failures and errors

| Recipe | Family | status | abs err | rel err | note |
|---|---|---|---|---|---|
| `comba_core` | RECURRENCE | fail | 4.22e+00 | 4.94e-01 |  |
| `abc_core` | RECURRENCE | upstream error | - | - | ValueError: ABC requires matching query/key heads and key dimensions |
| `deltaformer_attention_core` | SOFTMAX | upstream error | - | - | ImportError: Please install Flash Attention via `pip install flash-attn --no-build-isolation` first |
| `gru_core` | RECURRENCE | upstream error | - | - | ModuleNotFoundError: No module named 'xma' |
| `gsa_core` | RECURRENCE | upstream error | - | - | RuntimeError: PassManager::run failed |
| `hgrn_ssm_core` | RECURRENCE | upstream error | - | - | RuntimeError: FLA HGRN anchor requires its pinned source checkout |
| `kata_attention_core` | SOFTMAX | upstream error | - | - | ModuleNotFoundError: No module named 'kata' |
| `longformer_attention_core` | SOFTMAX | upstream error | - | - | ModuleNotFoundError: pinned Longformer source requires its checkout root on PYTHONPATH |
| `m2rnn_core` | RECURRENCE | upstream error | - | - | ModuleNotFoundError: No module named 'xma' |
| `mamba1_ssm_core` | RECURRENCE | upstream error | - | - | ModuleNotFoundError: No module named 'mamba_ssm' |
| `mamba2_ssm_core` | RECURRENCE | upstream error | - | - | ModuleNotFoundError: No module named 'mamba_ssm' |
| `mamba3_siso_core` | RECURRENCE | upstream error | - | - | ModuleNotFoundError: No module named 'mamba_ssm' |
| `momentum_delta_core` | RECURRENCE | upstream error | - | - | ValueError: Momentum DeltaNet operands must share device and dtype |
| `parallax_attention_core` | SOFTMAX | upstream error | - | - | CompilationError: at 91:13: col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS) m_kv = (col_indices[:, None] < T) & m_k[None, :… |
| `rnn_core` | RECURRENCE | upstream error | - | - | ModuleNotFoundError: No module named 'xma' |
| `tda_attention_core` | SOFTMAX | upstream error | - | - | ModuleNotFoundError: No module named 'triton_threshold_attention' |
| `titans_linear_memory_core` | RECURRENCE | upstream error | - | - | RuntimeError: Titans requires the exact recorded FLA source revision |
| `ttt_linear_core` | RECURRENCE | upstream error | - | - | RuntimeError: TTT-Linear requires the exact recorded FLA source revision |
| `tucker_attention_core` | SOFTMAX | upstream error | - | - | ModuleNotFoundError: No module named 'src' |
