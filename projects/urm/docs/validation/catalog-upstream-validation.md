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

**39 of 62 covered recipes pass upstream parity; 23 have native-vs-upstream performance measured; 13 have no upstream adapter; 36 have no native kernel.**

## (a) Native kernel + upstream parity pass + performance measured

| Recipe | Family | upstream anchor | dtype | abs err | overhead (native vs upstream) | native ms | upstream ms |
|---|---|---|---|---|---|---|---|
| `gated_delta_net` | RECURRENCE | `fla_gated_delta_rule_adapter` | bfloat16 | 7.26e-02 | -68.7% | 0.193 | 0.611 |
| `delta_net` | RECURRENCE | `fla_chunk_delta_rule_adapter` | bfloat16 | 1.23e-01 | -62.9% | 0.205 | 0.563 |
| `hgrn2_ssm_core` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.10e-02 | -58.2% | 0.232 | 0.556 |
| `gla` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.10e-02 | -57.9% | 0.206 | 0.492 |
| `lightnet_gla_core` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.10e-02 | -57.1% | 0.202 | 0.473 |
| `mom_selected_memory_core` | RECURRENCE | `fla_gated_delta_rule_adapter` | bfloat16 | 3.61e-03 | -54.0% | 0.382 | 0.843 |
| `rodimus_gla_core` | RECURRENCE | `fla_chunk_gla_adapter` | float32 | 1.37e-03 | -51.3% | 0.228 | 0.462 |
| `simple_gla` | RECURRENCE | `fla_chunk_simple_gla_adapter` | float32 | 1.18e-02 | -45.2% | 0.205 | 0.379 |
| `nsa_selected_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +11.1% | 0.319 | 0.288 |
| `cat_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +19.4% | 0.292 | 0.254 |
| `dsa_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +21.8% | 0.341 | 0.286 |
| `sparse_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +23.3% | 0.351 | 0.289 |
| `differential_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 1.79e-07 | +100.6% | 0.349 | 0.175 |
| `gqa` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +191.8% | 0.273 | 0.093 |
| `mla_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +196.5% | 0.269 | 0.093 |
| `pattention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 1.79e-07 | +198.3% | 0.277 | 0.097 |
| `foveal_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +200.3% | 0.317 | 0.107 |
| `mqa` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +218.6% | 0.289 | 0.093 |
| `conformer_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.09e-07 | +219.2% | 0.285 | 0.096 |
| `tpa_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +219.3% | 0.330 | 0.106 |
| `mha` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +221.0% | 0.287 | 0.096 |
| `samba_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 2.38e-07 | +221.5% | 0.335 | 0.105 |
| `hopfield_attention_core` | SOFTMAX | `torch.nn.functional.scaled_dot_product_attention` | float32 | 1.79e-07 | +222.8% | 0.283 | 0.093 |

## (b) Upstream parity only (no native-vs-upstream timing)

| Recipe | Family | upstream anchor | dtype | abs err | rel err | parity | note |
|---|---|---|---|---|---|---|---|
| `based_attention_core` | RECURRENCE | `fla_fused_chunk_based_adapter` | float32 | 2.38e-06 | 1.52e-06 | pass | no native kernel |
| `gated_delta_product_core` | RECURRENCE | `fla_chunk_gated_delta_product_adapter` | bfloat16 | 1.18e-01 | 4.80e-03 | pass | no native kernel |
| `gdn2_core` | RECURRENCE | `fla_chunk_gdn2_adapter` | float32 | 5.92e-03 | 1.09e-03 | pass | no native kernel |
| `generalized_delta_dplr_core` | RECURRENCE | `fla_chunk_dplr_adapter` | bfloat16 | 3.61e-02 | 6.84e-03 | pass | no native kernel |
| `generalized_delta_iplr_core` | RECURRENCE | `fla_fused_recurrent_iplr_adapter` | float32 | 5.96e-07 | 8.13e-08 | pass | no native kernel |
| `h3_ssm_fft_core` | RECURRENCE | `h3_ssm_fft_convolution_adapter` | float32 | 9.54e-07 | 5.09e-07 | pass | no native kernel |
| `hla_second_order_core` | RECURRENCE | `hla_second_order_triton_adapter` | float32 | 5.72e-06 | 1.18e-07 | pass | no native kernel |
| `hyena_fftconv_core` | RECURRENCE | `hyena_fft_convolution_adapter` | float32 | 4.77e-07 | 6.33e-08 | pass | no native kernel |
| `kda_core` | RECURRENCE | `fla_chunk_kda_adapter` | float32 | 1.12e-02 | 2.41e-03 | pass | no native kernel |
| `lightning_attention_core` | RECURRENCE | `fla_chunk_simple_gla_adapter` | float32 | 1.19e-02 | 1.04e-03 | pass | no native kernel |
| `rebased_attention_core` | RECURRENCE | `fla_parallel_rebased_adapter` | float32 | 2.23e-04 | 1.59e-04 | pass | no native kernel |
| `retention_core` | RECURRENCE | `fla_chunk_simple_gla_adapter` | float32 | 1.19e-02 | 1.04e-03 | pass | no native kernel |
| `rwkv4_memory_core` | RECURRENCE | `fla_fused_recurrent_rwkv4_adapter` | float32 | 1.19e-07 | 8.40e-08 | pass | no native kernel |
| `rwkv6_memory_core` | RECURRENCE | `fla_fused_recurrent_rwkv6_adapter` | float32 | 2.38e-07 | 7.88e-08 | pass | no native kernel |
| `rwkv7_transition_core` | RECURRENCE | `fla_chunk_rwkv7_adapter` | bfloat16 | 3.46e-02 | 3.28e-03 | pass | no native kernel |
| `wall_attention_core` | SOFTMAX | `fla_parallel_wall_attention_adapter` | float32 | 1.19e-07 | 9.21e-08 | pass | no native kernel |

## (c) No upstream adapter

| Recipe | Family | reason |
|---|---|---|
| `deltaformer_attention_core` | SOFTMAX | ImportError: Please install Flash Attention via `pip install flash-attn --no-build-isolation` first |
| `gru_core` | RECURRENCE | ModuleNotFoundError: No module named 'xma' |
| `kata_attention_core` | SOFTMAX | ModuleNotFoundError: No module named 'kata' |
| `longformer_attention_core` | SOFTMAX | ModuleNotFoundError: pinned Longformer source requires its checkout root on PYTHONPATH |
| `m2rnn_core` | RECURRENCE | ModuleNotFoundError: No module named 'xma' |
| `mamba1_ssm_core` | RECURRENCE | ModuleNotFoundError: No module named 'mamba_ssm' |
| `mamba2_ssm_core` | RECURRENCE | ModuleNotFoundError: No module named 'mamba_ssm' |
| `mamba3_siso_core` | RECURRENCE | ModuleNotFoundError: No module named 'mamba_ssm' |
| `momentum_delta_core` | RECURRENCE | ModuleNotFoundError: No module named 'fla.ops.momentum_delta_rule' |
| `rnn_core` | RECURRENCE | ModuleNotFoundError: No module named 'xma' |
| `sparse_delta_memory` | SPARSE_DELTA | ValueError: K3 uses the URM-native or reference sparse-state anchor |
| `tda_attention_core` | SOFTMAX | ModuleNotFoundError: No module named 'triton_threshold_attention' |
| `tucker_attention_core` | SOFTMAX | ModuleNotFoundError: No module named 'src' |

## (d) No native kernel

| Recipe | Family | upstream parity | note |
|---|---|---|---|
| `abc_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `based_attention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `comba_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `deltaformer_attention_core` | SOFTMAX | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `gated_delta_product_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `gated_oja_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `gdn2_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `generalized_delta_dplr_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `generalized_delta_iplr_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `gru_core` | RECURRENCE | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `gsa_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `h3_ssm_fft_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `hla_second_order_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `hyena_fftconv_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `kata_attention_core` | SOFTMAX | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `kda_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `lightning_attention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `linear_attention` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `longformer_attention_core` | SOFTMAX | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `m2rnn_core` | RECURRENCE | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `mamba2_ssm_core` | RECURRENCE | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `mamba3_siso_core` | RECURRENCE | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `mesa_net_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `momentum_delta_core` | RECURRENCE | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `parallax_attention_core` | SOFTMAX | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `rebased_attention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `retention_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `rnn_core` | RECURRENCE | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `rwkv4_memory_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `rwkv6_memory_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `rwkv7_transition_core` | RECURRENCE | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `tda_attention_core` | SOFTMAX | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `titans_linear_memory_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `ttt_linear_core` | RECURRENCE | upstream error | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `tucker_attention_core` | SOFTMAX | no upstream adapter | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |
| `wall_attention_core` | SOFTMAX | pass | ValueError: URM-native anchors support K1 normalized softmax (and its differential composition), K3 sparse delta, K2 diagonal SSM semantics, or the plain K2 matrix-state re |

## Upstream parity failures and errors

| Recipe | Family | status | abs err | rel err | note |
|---|---|---|---|---|---|
| `abc_core` | RECURRENCE | upstream error | - | - | ValueError: ABC requires matching query/key heads and key dimensions |
| `comba_core` | RECURRENCE | upstream error | - | - | ValueError: COMBA log_decay and beta must be float32 |
| `gated_oja_core` | RECURRENCE | upstream error | - | - | ValueError: gated Oja gv and beta must be float32 |
| `gsa_core` | RECURRENCE | upstream error | - | - | RuntimeError: PassManager::run failed |
| `hgrn_ssm_core` | RECURRENCE | upstream error | - | - | RuntimeError: FLA HGRN anchor requires its pinned source checkout |
| `linear_attention` | RECURRENCE | upstream error | - | - | ValueError: the pinned FLA linear-attention chunk backward requires even key and value dimensions for float16/bfloat16 tensors |
| `mesa_net_core` | RECURRENCE | upstream error | - | - | ValueError: MesaNet log_decay, beta and lamb must be float32 |
| `parallax_attention_core` | SOFTMAX | upstream error | - | - | CompilationError: at 91:13:
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bo |
| `titans_linear_memory_core` | RECURRENCE | upstream error | - | - | RuntimeError: Titans requires the exact recorded FLA source revision |
| `ttt_linear_core` | RECURRENCE | upstream error | - | - | RuntimeError: TTT-Linear requires the exact recorded FLA source revision |
