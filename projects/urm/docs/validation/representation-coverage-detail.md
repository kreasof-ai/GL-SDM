# Representation coverage: every architecture lowers into a canonical core

Status: evidence record, regenerated from the live compiler. This is the
durable contract behind the unified generator: each named architecture recipe
either lowers into the canonical NumPy execution path for its core (K1/K2/K3)
and is verified against its independent architecture equation in float64, or it
is recorded as declined with the precise reason. Once a recipe lowers and
matches, any later optimization of the canonical K1/K2/K3 kernel lifts it
automatically - there is no per-architecture re-derivation.

**23 of 74 named recipes lower into a canonical core and match their independent equation.** 0 lower but are not yet verified; 51 decline (under-specified or exotic).

## Verified (lower + match independent equation)

| Recipe | Family | output err | state err |
|---|---|---|---|
| `mha` | SOFTMAX | 1.33e-07 | - |
| `mqa` | SOFTMAX | 1.33e-07 | - |
| `gqa` | SOFTMAX | 1.33e-07 | - |
| `sparse_attention_core` | SOFTMAX | 1.33e-07 | - |
| `cat_attention_core` | SOFTMAX | 1.33e-07 | - |
| `nsa_selected_attention_core` | SOFTMAX | 1.33e-07 | - |
| `dsa_attention_core` | SOFTMAX | 1.33e-07 | - |
| `mla_attention_core` | SOFTMAX | 1.33e-07 | - |
| `foveal_attention_core` | SOFTMAX | 1.33e-07 | - |
| `pattention_core` | SOFTMAX | 1.42e-07 | - |
| `tpa_attention_core` | SOFTMAX | 1.33e-07 | - |
| `conformer_attention_core` | SOFTMAX | 1.45e-07 | - |
| `hopfield_attention_core` | SOFTMAX | 1.42e-07 | - |
| `samba_attention_core` | SOFTMAX | 1.33e-07 | - |
| `lightnet_gla_core` | RECURRENCE | 6.96e-07 | 1.89e-07 |
| `simple_gla` | RECURRENCE | 3.83e-07 | 3.03e-07 |
| `gla` | RECURRENCE | 6.96e-07 | 1.89e-07 |
| `rodimus_gla_core` | RECURRENCE | 8.70e-08 | 1.89e-07 |
| `mom_selected_memory_core` | RECURRENCE | 3.13e-07 | 8.22e-08 |
| `hgrn2_ssm_core` | RECURRENCE | 6.96e-07 | 1.89e-07 |
| `delta_net` | RECURRENCE | 4.09e-06 | 1.68e-06 |
| `gated_delta_net` | RECURRENCE | 1.49e-06 | 3.23e-07 |
| `sparse_delta_memory` | SPARSE_DELTA | 6.10e-08 | 1.59e-07 |

## Lowers but not yet verified

| Recipe | Family | note |
|---|---|---|

## Declined (under-specified or exotic)

| Recipe | Family | reason |
|---|---|---|
| `polar_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; polar_attention is a distinct equation |
| `foveal_sparse_polar_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; sparse_polar_attention is a distinct equation |
| `fox` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; forgetting_gated_softmax_attention is a distinct equation |
| `wall_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; gated_attention is a distinct equation |
| `differential_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; difference_of_softmax_attention is a distinct equation |
| `tda_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; thresholded_softmax_attention is a distinct equation |
| `moba_selected_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; block_routed_softmax_attention is a distinct equation |
| `bdh_attention_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `path_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; path_transform_attention is a distinct equation |
| `deltaformer_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; delta_transform_attention is a distinct equation |
| `parallax_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; position_indexed_attention is a distinct equation |
| `attnres_depth_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; depth_weighted_attention is a distinct equation |
| `tucker_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; projected_softmax_attention is a distinct equation |
| `longformer_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; local_window_softmax_attention is a distinct equation |
| `kata_attention_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; positive_feature_attention is a distinct equation |
| `fwpkm_memory_read_core` | SOFTMAX | K1 canonical path covers the normalized softmax reduction only; selected_memory_read is a distinct equation |
| `h3_ssm_fft_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `hyena_fftconv_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `hla_second_order_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `linear_attention` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `based_attention_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rebased_attention_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `retention_core` | RECURRENCE | exotic composition flags: ['static_head_decay'] |
| `lightning_attention_core` | RECURRENCE | exotic composition flags: ['static_head_decay'] |
| `hgrn_ssm_core` | RECURRENCE | the canonical matrix-state path covers the matrix layout only |
| `atma_gated_delta_decode_core` | RECURRENCE | only functional state is canonical |
| `gdn2_core` | RECURRENCE | exotic composition flags: ['gdn2_ssm'] |
| `gated_oja_core` | RECURRENCE | exotic composition flags: ['gated_oja'] |
| `comba_core` | RECURRENCE | exotic composition flags: ['comba_rule'] |
| `pgdn_core` | RECURRENCE | exotic composition flags: ['preconditioned_gated_delta'] |
| `pkda_core` | RECURRENCE | exotic composition flags: ['preconditioned_kda'] |
| `abc_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `gsa_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `kda_core` | RECURRENCE | exotic composition flags: ['kda_delta'] |
| `gated_delta_product_core` | RECURRENCE | exotic composition flags: ['gated_delta_product'] |
| `generalized_delta_iplr_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `generalized_delta_dplr_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rwkv4_memory_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rwkv6_memory_core` | RECURRENCE | exotic composition flags: ['rwkv6_memory'] |
| `momentum_delta_core` | RECURRENCE | exotic composition flags: ['momentum_delta'] |
| `mesa_net_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rwkv7_transition_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `mamba1_ssm_core` | RECURRENCE | the canonical matrix-state path covers the matrix layout only |
| `mamba2_ssm_core` | RECURRENCE | exotic composition flags: ['mamba2_ssm'] |
| `mamba3_siso_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `titans_linear_memory_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `ttt_linear_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rnn_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `gru_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `m2rnn_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `log_linear_attention_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
