# Linear and delta recurrence family

Construction contract; family 2 of the [coverage matrix](../planning/coverage.md).
The contract is an equation, not a named model or external-library signature.

## Operation

For each independent partition, state `M` has shape `[K,Dv]`; q/k have shape
`[K]`, v has shape `[Dv]`. Initial state and final state are explicit operands.
Define a nonnegative diagonal transition `G_t`, scalar write strength `beta_t`,
and a static correction choice `c` in `{0,1}`:

```text
Z_t = G_t M_(t-1)
h_t = k_t^T Z_t
delta_t = beta_t * (v_t - c*h_t)
M_t = Z_t + k_t delta_t^T
y_t = scale * q_t^T M_t
```

`c=0` is an additive linear update; `c=1` is a delta-corrected update.
Decay precedes retrieval, and reads occur after update. Other orderings require
explicit operations or verified transformations. Scalar decay is the first
implementation target; per-key-channel decay is a subsequent capability.
Query/key normalization and feature maps are separate frontend operations.
Normalized linear attention additionally needs denominator state and a specified
division/epsilon rule; it is not silently represented by the raw numerator alone.

## Chunk formulation

Within a chunk, let `E(t,j)=G_t ... G_(j+1)` and `E(t,t)=I`.
Using incoming state `M_0`, define

```text
V0_t = k_t^T E(t,0) M_0
Y0_t = scale * q_t^T E(t,0) M_0
A[t,j] = k_t^T E(t,j) k_j          for j<t, otherwise zero
Omega[t,j] = scale*q_t^T E(t,j) k_j for j<=t, otherwise zero
B = diag(beta)
(I + c*B*A) Delta = B*(V-c*V0)
Y = Y0 + Omega*Delta
M_C = E(C,0)*M_0 + sum_j E(C,j)*k_j*delta_j^T
```

With `c=0` the solve disappears. With `c=1` use a unit-lower-triangular solve.
Carry boundary state between chunks. Do not clip inverse decay factors or assume
that coefficient precomputation removes boundary dependencies. Stable decay
factorization and conditioning must be validated for the chosen implementation.

This is the dense counterpart of the [sparse-delta derivation](sparse-delta.md).
For delta correction, its recurrence VJP applies with `w=k`, `q=scale*q`, and
decay on all state rows. For additive updates the retrieval correction and its
gradients disappear. Both paths require gradients through transition schedules,
initial state and final-state losses; discrete mode choices are not differentiable.

## Lowering and parity route

Implement a float64 recurrence and chunk oracle first, then reuse the existing
gated-delta adapter only on its proven compatible envelope. Implement scalar-decay
chunked prefill/training and a recurrent decode kernel; add per-channel transitions
and normalized variants as separately tested capabilities.

Freeze comparator equations against the pinned implementation/reference, not its
name. The archived gated-delta document contains an inconsistency between its
written recurrence and its decay-order prose; resolve that against executable
code before claiming this contract matches it. Do not inherit that ambiguity.

Compare all operand VJPs, readouts and state continuation under identical precision
policies. Tune chunk size and state layout under a declared workspace budget.
Measure the full state stage and complete model workloads using the
[parity protocol](../validation/parity.md).
