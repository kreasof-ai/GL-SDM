# Frozen contract: FLA gated delta-rule (v1)

Pin used by every committed result in this iteration:

| Item | Value |
| --- | --- |
| Upstream | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) |
| Version | **0.5.2** (PyPI sdist/wheel `flash-linear-attention==0.5.2`, transitive helper `fla-core==0.5.2`; corresponds to GitHub tag `v0.5.2`) |
| Installation source | PyPI into the validated environment (`validated-environment.json` records torch/triton/CUDA) |
| License | MIT. URM calls the package externally as an installed dependency; no FLA source is copied into this repository. |
| Operations frozen | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (prefill) and `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` (decode) |

The API was inspected in the installed distribution (not assumed): signatures,
docstrings, and the shipped reference `fla.ops.gated_delta_rule.naive_recurrent_gated_delta_rule`
were checked against observed kernel behavior before freezing this contract.

## Recurrence (per batch, per value-head)

State ``S ∈ R^{K x V}``. Inputs per position ``t``: queries ``q_t ∈ R^K``,
keys ``k_t ∈ R^K``, values ``v_t ∈ R^V``, scalar log-decay ``g_t``, scalar
write strength ``β_t ∈ (0, 1)``.

```text
u_t  = S_{t-1}^T k_t              # retrieve
dv_t = β_t * (v_t - u_t)          # beta-scaled write error
S_t  = exp(g_t) * S_{t-1} + k_t dv_t^T   # decay, then rank-1 delta update
o_t  = scale * S_t^T q_t          # output reads the POST-update state
```

Ordering facts verified against the upstream reference (exact to ~2e-7 with
`fused_recurrent`, ~2.5e-3 with the chunked form in fp32):

- the decay `exp(g_t)` is applied **before** retrieval and update;
- the output at step ``t`` reads the state **after** step ``t``'s update;
- ``g`` is consumed in **log space** (`exp(cumulative sum)` inside chunks);
- `scale` multiplies ``q`` only in the output projection (default `1/sqrt(K)`).

## Typed boundary (`GatedDeltaRuleSpec`)

| Field | Meaning | Constraint |
| --- | --- | --- |
| `batch` | independent sequences | `>= 1` |
| `sequence` | tokens per sequence | `>= 1` (decode is `sequence == 1`) |
| `heads` | query/key heads `H` | `>= 1` |
| `value_heads` | value/state heads `HV` (GVA) | `HV >= H` and `HV % H == 0`; `g`/`β` have shape `[B,T,HV]` |
| `key_dim` | `K`, state row dim | `>= 1`; `scale` default `1/sqrt(K)` |
| `value_dim` | `V`, state column dim | `>= 1` |
| `dtype` | q/k/v storage dtype | bf16 or fp16 for the CUDA kernels compared here |
| `mode` | `"prefill"` (chunk) or `"decode"` (fused recurrent) | see below |
| `has_initial_state` / `output_final_state` | explicit state handling | state tensors `[N, HV, K, V]`, `N == B` for equal lengths |

Boundary layout is **[B, T, H, K]** for q/k, **[B, T, HV, V]** for v, output
**[B, T, HV, V]**, state **[B, HV, K, V]** (row-major K-major layout, fp32).

## Frozen semantic choices

- `use_qk_l2norm_in_kernel=False`: q/k normalization is **caller-side** in this
  contract. Production GDN models apply L2 normalization before the operator;
  tests that need it normalize explicitly. The adapter rejects any other value
  rather than silently comparing different normalization semantics.
- `use_gate_in_kernel=False`: `g` arrives already in log space
  (e.g. `logsigmoid`). The fused Mamba-style `-exp(A_log)*softplus(...)` gate
  activation is out of scope for v1 and rejected.
- `use_beta_sigmoid_in_kernel=False`, `allow_neg_eigval=False`: `β` is passed
  post-sigmoid in `(0, 1)`. The `(0, 2)` negative-eigenvalue variant is a
  different semantic family and is not forced into this adapter.
- `state_v_first=False`: state layout stays K-by-V.
- Variable-length packing (`cu_seqlens`) and context parallelism are upstream
  capabilities but out of scope for v1; the adapter does not expose them.
- Chunk size is left at the upstream default (BT=64) for prefill; chunked and
  recurrent forms agree to floating-point reassociation tolerance, not bitwise.

## Path support matrix

| Path | Forward | Backward | Notes |
| --- | --- | --- | --- |
| `chunk_gated_delta_rule` (prefill) | yes | yes: dq, dk, dv, dg, dbeta, d(initial_state) | chunk-parallel; deterministic forward (bitwise repeat verified on A10G) |
| `fused_recurrent_gated_delta_rule` (decode) | yes | **not implemented upstream** (raises `NotImplementedError`) | used token-by-token for inference decode; backward benchmarks must target the chunk path |

## Precision, dtypes, devices

- Accumulation is fp32 inside both upstream kernels; I/O dtypes compared here:
  bf16 (primary), fp16 (secondary). fp32 I/O worked with the fused path in
  probes but the published Triton kernels target half precision, so fp32 is
  recorded as unsupported-for-benchmark rather than benchmarked.
- Devices: CUDA (Triton kernels). CPU execution is possible via upstream's
  naive/torch fallbacks but is not part of the GPU comparison.
- Tolerances live in the adapter/tests: bf16 output vs fp32 oracle uses
  atol 2e-2 / rtol 2e-2 for outputs and 6e-2 for long-sequence states;
  fp16 tightens to 1.5e-2. Gradient checks compare against an fp32 eager
  recurrence with dtype-scaled tolerances.

## What URM adds

The adapter performs spec validation, capability selection (prefill vs decode),
and dispatch to the pinned upstream function. It receives the same
preallocated tensors as a direct upstream call and executes the same kernel,
so measured differences are URM integration overhead only. This iteration does
**not** include a native URM gated-delta Triton kernel.
