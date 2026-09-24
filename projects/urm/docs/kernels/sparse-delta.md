# Sparse routed delta update: kernel formulation

Canonical operation: `sparse_delta`. This contract is independent of architecture
and routing frontend. SDM is one comparison workload, not its semantic definition.
Other mixers can use this lowering when their typed operations satisfy the same
equations; different update rules require a separate derivation.

Status: real-arithmetic derivation with float64 NumPy differential and adjoint
checks. This is not GPU certification, a registered compiler rewrite, or a claim
of bitwise equivalence under BF16 state rounding.

## Semantic contract

Work independently per partition/head. Let memory `M` have shape `[S,D]`.
For token `t`, dense vectors `w_t,q_t` of shape `[S]` encode sparse write/read
weights; `v_t` has shape `[D]`. Write indices are unique within a token; duplicate
write routes need a separately specified collision rule. `selected[t,s]` is the
explicit write-selection mask. A selected slot decays even when its weight is zero.

For scalar `g_t <= 0`, define diagonal `G_t[s,s] = exp(g_t * selected[t,s])`.
The decay applies once to each selected slot, before retrieval and update:

```text
Z_t = G_t M_(t-1)
h_t = w_t^T Z_t
delta_t = beta_t (v_t - h_t)
M_t = Z_t + w_t delta_t^T
y_t = q_t^T M_t                       # after-update read
```

Coefficients and route choices must be available independently of the evolving
memory. This contract does not cover state-dependent routing or a nonlinear state
update. Before-update reads require a separately derived read operator; they must
not silently select the after-update implementation.

## Chunk-local triangular system

Index a chunk from `1` to `C`, with incoming state `M_0`. Define
`L_t[s] = sum_(i=1..t) g_i selected[i,s]`, `L_0=0`, and
`E(t,j)=diag(exp(L_t-L_j))` for `0 <= j <= t`. Then

```text
V0_t = w_t^T E(t,0) M_0
Y0_t = q_t^T E(t,0) M_0
A[t,j] = w_t^T E(t,j) w_j       for j < t; otherwise zero
Omega[t,j] = q_t^T E(t,j) w_j   for j <= t; otherwise zero
B = diag(beta)
H = I + B A
H Delta = B (V - V0)
Y = Y0 + Omega Delta
M_C = E(C,0) M_0 + sum_j E(C,j) w_j delta_j^T
```

`A`, `Omega` and `H` have shape `[C,C]`, not `[S,S]`. `H` is unit lower triangular,
so it is nonsingular in exact arithmetic. Its conditioning still depends on the
coefficients. Solve for `Delta`; do not require explicit matrix inversion.

Coefficient construction can be batched across chunks because it does not depend
on incoming memory. Initial-state projections and boundary propagation do depend
on incoming memory. A correct first implementation carries `M_C` to the next chunk
in order. Independent chunks with the original `M_0` are incorrect. Parallel chunk
composition needs an additional proved representation and cost analysis.

For no decay, `A=tril(W W^T,-1)` and `Omega=tril(Q W^T)`. This is the particularly
simple GEMM case. Selected-slot decay requires interval factors; it cannot be
removed to recover those GEMMs. Algebraically separable factors involving
`exp(L_t)` and `exp(-L_j)` can overflow or underflow. Clipping only the inverse
factor changes even diagonal interactions. The oracle computes interval
differences directly. A production GEMM implementation needs a proved stable
scaling/tiling scheme; no efficient universal scheme is certified here.

## Backward contract

For the solve, with upstream derivative `dDelta`, compute
`Lambda = solve(H^T, dDelta)`, `dRHS = Lambda`, and
`dH = -Lambda Delta^T`, respecting the fixed triangular structure. Backpropagate
through `B`, `A`, initial projections, read coefficients and the final-state fold.
Final-state cotangents must flow backward across chunk boundaries.

An independent token-level VJP provides an oracle. Let `F` be the cotangent of
`M_t` from subsequent state use, plus `q_t dy_t^T` from its reading:

```text
dq_t = M_t dy_t
ddelta_t = w_t^T F
dv_t = beta_t ddelta_t
dbeta_t = dot(ddelta_t, v_t - h_t)
dw_t = F delta_t - Z_t dv_t
dZ_t = F - w_t dv_t^T
dg_t = sum_(selected slots s,d) dZ_t[s,d] Z_t[s,d]
dM_(t-1) = G_t dZ_t
```

The write-weight derivative is restricted to fixed selected routes. Selection
indices are discrete and have no ordinary derivative. These formulas cover both
reading losses and final-state losses; returning a constant zero decay gradient
violates this contract.

## Precision and acceptance

The real-arithmetic rewrite changes operation ordering. The old per-token BF16
state commit and a chunk-boundary commit are different numerical contracts.
Declare storage, accumulation and commit policy separately; changing chunk size
must not silently change an advertised semantic guarantee.

`urm.backends.reference.numpy.k3` contains independent recurrence, chunked solve and
analytical reverse recurrence implementations. Tests compare multiple chunk sizes,
partial chunks, repeated slots, strong decay, selected zero-weight slots, and
finite differences for memory, write/read weights, values, beta and decay.

These checks establish the float64 formulation on tested cases. Before registering
a GPU implementation, repeat output/state and VJP comparisons on that exact path,
including production-shaped collision stress and dtype-specific envelopes. Keep
the historical dual-form prototypes experimental until those gates pass.
