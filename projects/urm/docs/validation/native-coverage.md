# Native generation coverage

This is the honest measure of URM's unified-generator reach: the recipes
URM computes with its **own** generated kernels, not by dispatching to an
upstream library. The [upstream comparison table](upstream-comparison.md)
measures dispatch overhead against pinned upstream sources; it does not
establish that URM computes these operations natively. A unified generator
that merely dispatches would be a thin wrapper - native coverage is what
distinguishes a generator from a wrapper.

**26 of 74 named recipes compile to a native kernel (35%).** The remaining 48 decline to
the reference or an upstream adapter.

## Natively generated (URM computes these)

Each row is one reusable native generator template and the recipes it covers
- one semantic kernel covering trivial and exotic variants, which is the
unified-generator value proposition.

| Native generator | Recipes covered | Count |
|---|---|---|
| K1 online softmax (`urm_native_k1_online_softmax_v1`) | `cat_attention_core`, `conformer_attention_core`, `differential_attention_core`, `dsa_attention_core`, `foveal_attention_core`, `gqa`, `hopfield_attention_core`, `mha`, `mla_attention_core`, `mqa`, `nsa_selected_attention_core`, `pattention_core`, `samba_attention_core`, `sparse_attention_core`, `tpa_attention_core` | 15 |
| urm_native_matrix_state_recurrence_v1 (`urm_native_matrix_state_recurrence_v1`) | `delta_net`, `gated_delta_net`, `gla`, `hgrn2_ssm_core`, `lightnet_gla_core`, `mom_selected_memory_core`, `rodimus_gla_core`, `simple_gla` | 8 |
| K2 diagonal recurrence (`urm_native_diagonal_recurrence_v1`) | `hgrn_ssm_core`, `mamba1_ssm_core` | 2 |
| K3 sparse state (`urm_native_sparse_state_mixer_v0`) | `sparse_delta_memory` | 1 |

## Dispatch/reference only (no native kernel yet)

These recipes compile only through the reference oracle or an upstream
library adapter. They are the native-generation backlog, grouped here as a
flat list; the production matrix orders the mandatory subset.

`abc_core`, `atma_gated_delta_decode_core`, `attnres_depth_core`, `based_attention_core`, `bdh_attention_core`, `comba_core`, `deltaformer_attention_core`, `foveal_sparse_polar_attention_core`, `fox`, `fwpkm_memory_read_core`, `gated_delta_product_core`, `gated_oja_core`, `gdn2_core`, `generalized_delta_dplr_core`, `generalized_delta_iplr_core`, `gru_core`, `gsa_core`, `h3_ssm_fft_core`, `hla_second_order_core`, `hyena_fftconv_core`, `kata_attention_core`, `kda_core`, `lightning_attention_core`, `linear_attention`, `log_linear_attention_core`, `longformer_attention_core`, `m2rnn_core`, `mamba2_ssm_core`, `mamba3_siso_core`, `mesa_net_core`, `moba_selected_attention_core`, `momentum_delta_core`, `parallax_attention_core`, `path_attention_core`, `pgdn_core`, `pkda_core`, `polar_attention_core`, `rebased_attention_core`, `retention_core`, `rnn_core`, `rwkv4_memory_core`, `rwkv6_memory_core`, `rwkv7_transition_core`, `tda_attention_core`, `titans_linear_memory_core`, `ttt_linear_core`, `tucker_attention_core`, `wall_attention_core`.

## Reading this table

- A recipe is **native** only when the compiler emits a URM kernel that
  computes the operation; parity for native kernels is established against
  the reference oracle and the pinned upstream comparator.
- A recipe is **dispatch/reference only** when the compiler cannot yet
  generate a native kernel for its full semantics and must run the
  reference or call upstream. Expanding native coverage means lowering more
  of these to native generation - not adding dispatch adapters.
