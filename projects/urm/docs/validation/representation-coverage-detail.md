# Representation coverage: every architecture lowers into a canonical core

Status: evidence record, regenerated from the live compiler. This is the
durable contract behind the unified generator: each named architecture recipe
either lowers into the canonical NumPy execution path for its core (K1/K2/K3)
and is verified against its independent architecture equation in float64, or it
is recorded as declined with the precise reason. Once a recipe lowers and
matches, any later optimization of the canonical K1/K2/K3 kernel lifts it
automatically - there is no per-architecture re-derivation.

**37 of 74 named recipes lower into a canonical core and match their independent equation.** 0 lower but are not yet verified; 37 decline (under-specified or exotic).

## Verified (lower + match independent equation)

| Recipe | Family | output err | state err |
|---|---|---|---|
| `mha` | SOFTMAX | 1.33e-07 | - |
| `mqa` | SOFTMAX | 1.33e-07 | - |
| `gqa` | SOFTMAX | 1.33e-07 | - |
| `sparse_attention_core` | SOFTMAX | 1.33e-07 | - |
| `cat_attention_core` | SOFTMAX | 1.33e-07 | - |
| `differential_attention_core` | SOFTMAX | 1.89e-07 | - |
| `nsa_selected_attention_core` | SOFTMAX | 1.33e-07 | - |
| `dsa_attention_core` | SOFTMAX | 1.33e-07 | - |
| `mla_attention_core` | SOFTMAX | 1.33e-07 | - |
| `parallax_attention_core` | SOFTMAX | 3.55e-07 | - |
| `foveal_attention_core` | SOFTMAX | 1.33e-07 | - |
| `pattention_core` | SOFTMAX | 1.42e-07 | - |
| `tpa_attention_core` | SOFTMAX | 1.33e-07 | - |
| `tucker_attention_core` | SOFTMAX | 1.98e-07 | - |
| `longformer_attention_core` | SOFTMAX | 9.18e-08 | - |
| `kata_attention_core` | SOFTMAX | 1.02e-07 | - |
| `conformer_attention_core` | SOFTMAX | 1.45e-07 | - |
| `hopfield_attention_core` | SOFTMAX | 1.42e-07 | - |
| `samba_attention_core` | SOFTMAX | 1.33e-07 | - |
| `linear_attention` | RECURRENCE | 1.70e-07 | 6.64e-07 |
| `based_attention_core` | RECURRENCE | 3.11e-07 | 6.97e-07 |
| `rebased_attention_core` | RECURRENCE | 4.08e-07 | 4.82e-07 |
| `retention_core` | RECURRENCE | 8.56e-07 | 1.88e-07 |
| `lightning_attention_core` | RECURRENCE | 8.56e-07 | 1.88e-07 |
| `lightnet_gla_core` | RECURRENCE | 6.96e-07 | 1.89e-07 |
| `simple_gla` | RECURRENCE | 3.83e-07 | 3.03e-07 |
| `gla` | RECURRENCE | 6.96e-07 | 1.89e-07 |
| `rodimus_gla_core` | RECURRENCE | 8.70e-08 | 1.89e-07 |
| `mom_selected_memory_core` | RECURRENCE | 3.13e-07 | 8.22e-08 |
| `hgrn_ssm_core` | RECURRENCE | 2.06e-07 | 1.95e-07 |
| `hgrn2_ssm_core` | RECURRENCE | 6.96e-07 | 1.89e-07 |
| `delta_net` | RECURRENCE | 4.09e-06 | 1.68e-06 |
| `gated_delta_net` | RECURRENCE | 1.49e-06 | 3.23e-07 |
| `gdn2_core` | RECURRENCE | 2.58e-06 | 1.87e-06 |
| `kda_core` | RECURRENCE | 8.26e-07 | 5.53e-07 |
| `mamba1_ssm_core` | RECURRENCE | 9.02e-08 | 7.27e-08 |
| `sparse_delta_memory` | SPARSE_DELTA | 6.10e-08 | 1.59e-07 |

## Lowers but not yet verified

| Recipe | Family | note |
|---|---|---|

## Declined (under-specified or exotic)

| Recipe | Family | reason |
|---|---|---|
| `polar_attention_core` | SOFTMAX | K1 canonical path does not yet cover polar_attention |
| `foveal_sparse_polar_attention_core` | SOFTMAX | K1 canonical path does not yet cover sparse_polar_attention |
| `fox` | SOFTMAX | K1 canonical path does not yet cover forgetting_gated_softmax_attention |
| `wall_attention_core` | SOFTMAX | K1 canonical path does not yet cover gated_attention |
| `tda_attention_core` | SOFTMAX | K1 canonical path does not yet cover thresholded_softmax_attention |
| `moba_selected_attention_core` | SOFTMAX | K1 canonical path does not yet cover block_routed_softmax_attention |
| `bdh_attention_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `path_attention_core` | SOFTMAX | K1 canonical path does not yet cover path_transform_attention |
| `deltaformer_attention_core` | SOFTMAX | K1 canonical path does not yet cover delta_transform_attention |
| `attnres_depth_core` | SOFTMAX | K1 canonical path does not yet cover depth_weighted_attention |
| `fwpkm_memory_read_core` | SOFTMAX | K1 canonical path does not yet cover selected_memory_read |
| `h3_ssm_fft_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `hyena_fftconv_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `hla_second_order_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `atma_gated_delta_decode_core` | RECURRENCE | only functional state is canonical |
| `gated_oja_core` | RECURRENCE | exotic composition flags: ['gated_oja'] |
| `comba_core` | RECURRENCE | exotic composition flags: ['comba_rule'] |
| `pgdn_core` | RECURRENCE | exotic composition flags: ['preconditioned_gated_delta'] |
| `pkda_core` | RECURRENCE | exotic composition flags: ['preconditioned_kda'] |
| `abc_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `gsa_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `gated_delta_product_core` | RECURRENCE | exotic composition flags: ['gated_delta_product'] |
| `generalized_delta_iplr_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `generalized_delta_dplr_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rwkv4_memory_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rwkv6_memory_core` | RECURRENCE | exotic composition flags: ['rwkv6_memory'] |
| `momentum_delta_core` | RECURRENCE | exotic composition flags: ['momentum_delta'] |
| `mesa_net_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rwkv7_transition_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `mamba2_ssm_core` | RECURRENCE | exotic composition flags: ['mamba2_ssm'] |
| `mamba3_siso_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `titans_linear_memory_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `ttt_linear_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `rnn_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `gru_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `m2rnn_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
| `log_linear_attention_core` | RECURRENCE | additive/no-decay is under-specified in the IR (collision group) |
