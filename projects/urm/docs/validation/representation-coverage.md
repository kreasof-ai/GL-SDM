# K1/K2/K3 representation coverage: evidence

Status: evidence record. This document separates **representational coverage**
(the K1/K2/K3 IR can express the kernel) from **native generation** (the
compiler emits a native kernel for it), per the acceptance requirement to keep
claims tied to evidence. The distinction matters: a workload can be fully
represented and reference-correct against upstream while still lacking a native
generated kernel.

## Method

For each mandatory-matrix workload class we probe four stages and record the
selected anchor or the explicit decline:

- **Expressible**: a `UnifiedMixerSpec` exists with the family's semantic fields.
- **Reference**: the compiler's reference oracle executes the equation.
- **Library (upstream)**: the compiler binds the pinned upstream comparator.
- **Native**: the compiler emits a URM-native kernel anchor.

## Evidence

| Workload | Family | Expressible | Reference | Library (upstream) | Native |
|---|---|---|---|---|---|
| Dense MHA/GQA/MQA | K1 | yes | `urm.unified.k1.softmax_reference.v1` | `torch.nn.functional.scaled_dot_product_attention` | `urm_native_k1_online_softmax_v1` |
| Diagonal recurrence (HGRN / Mamba-1) | K2 | yes | `urm.unified.k2.state_reference.v1` | `fla_fused_recurrent_hgrn_adapter` | `urm_native_diagonal_recurrence_v1` |
| Matrix-state gated-delta (Gated DeltaNet) | K2 | yes | `urm.unified.k2.state_reference.v1` | `fla_gated_delta_rule_adapter` | `urm_native_matrix_state_recurrence_v1` |
| Sparse Delta Memory | K3 | yes | `urm.unified.k3.sparse_delta_reference.v1` | declines (native/reference only) | `urm_native_sparse_state_mixer_v0` |

Reproduce with the probe in the section below; the anchors above are the live
compiler output, not prose.

## What this proves

- **K1, K2, and K3 each represent their mandatory-matrix kernels.** Every row is
  expressible and reference-correct. The K2 matrix-state gated-delta equation is
  the recurrence `Z_t = G_t M_{t-1}`, `h_t = k_t^T Z_t`,
  `delta_t = beta_t (v_t - h_t)`, `M_t = Z_t + k_t delta_t^T`,
  `y_t = scale * q_t^T M_t` (the `c=1` delta-corrected case in
  `urm/ir/recurrence.py`), and the URM K2 reference matches the pinned FLA
  `chunk_gated_delta_rule` to bf16 tolerance once the read-scale convention is
  aligned (URM reference defaults to scale 1.0; FLA to `K^-0.5`).

- **The K2 matrix-state native-generation gap is closed for the plain class.**
  A general native matrix-state kernel
  (`urm/backends/triton/recurrence/matrix_state.py`, anchor
  `urm_native_matrix_state_recurrence_v1`) covers the plain delta/additive
  recurrence across decay granularities (head/key-channel) and read timings,
  selected from semantic fields. It matches the reference oracle to fp32
  tolerance across the full axis grid
  (`test_native_matrix_state_recurrence_matches_reference`) and matches the
  pinned FLA `fused_recurrent_gated_delta_rule` exactly (output to 1.2e-7,
  final state exact). It natively covers seven recipes: `gated_delta_net`,
  `delta_net`, `gla`, `hgrn2_ssm_core`, `lightnet_gla_core`, `rodimus_gla_core`,
  `simple_gla`.

- **The remaining matrix-state gap is the additive/no-decay collision group.**
  The IR does not yet distinguish a plain additive recurrence from the exotic
  additive equations that share its semantic fields: `gru_core`, `rnn_core`,
  `m2rnn_core`, `h3_ssm_fft_core`, `hyena_fftconv_core`, `hla_second_order_core`,
  `mamba3_siso_core`, `mesa_net_core`, `titans_linear_memory_core`,
  `ttt_linear_core`, and `bdh_attention_core` all present as
  `additive / no-decay / identity-feature / no-normalizer`, yet compute GRU
  tanh/gate nonlinearities, FFT long convolutions, second-order corrections, and
  similar. Dispatching them on the spec alone would silently compute the wrong
  equation, so the native generator declines the whole additive/no-decay group
  (pinned by
  `test_native_matrix_state_declines_underdetermined_recipes`). Closing this
  requires the name-dependence fix: carrying those equations explicitly in the
  IR so different equations have distinguishable semantic representations
  (acceptance-contract section 4).

- **K3 has no upstream library adapter** by design (the pinned SDM checkout is
  invoked through the external adapter boundary, not the in-process library
  path); its native and reference anchors both exist.

## Reproduce

```bash
cd projects/urm
PYTHONPATH=src python - <<'PY'
from urm.frontend.mixer_recipes import (
    softmax_attention_spec, diagonal_ssm_spec, delta_rule_spec, sparse_delta_spec,
)
from urm.ir.mixer import DecayGranularity
from urm.compiler.unified_mixer import compile_mixer, MixerBackend

for label, spec in [
    ("K1 MHA", softmax_attention_spec()),
    ("K2 diagonal", diagonal_ssm_spec("hgrn_ssm_core", hgrn=True)),
    ("K2 gated-delta", delta_rule_spec("gated_delta", decay=DecayGranularity.HEAD)),
    ("K3 sparse", sparse_delta_spec()),
]:
    for backend in (MixerBackend.REFERENCE, MixerBackend.LIBRARY, MixerBackend.NATIVE):
        dtypes = ("float32", "bfloat16") if backend is MixerBackend.LIBRARY else ("float32",)
        for dt in dtypes:
            try:
                plan = compile_mixer(spec, backend=backend, intent="training", dtype=dt)
                print(label, backend.value, "->", plan.anchor)
                break
            except Exception as exc:
                print(label, backend.value, "-> DECLINE", type(exc).__name__)
PY
```
