# K2: compact fixed-address ordered state

**Written canonical equation.** Per independent partition, let `M_t ∈ R[K,Dv]`, `q_t,k_t ∈ R[K]`, `v_t ∈ R[Dv]`, nonnegative diagonal `G_t`, scalar `β_t`, and `c ∈ {0,1}`:

```text
Z_t = G_t M_(t-1)
h_t = k_tᵀ Z_t
δ_t = β_t (v_t - c h_t)
M_t = Z_t + k_t δ_tᵀ
y_t = scale * q_tᵀ M_t
```

`c=0` is additive; `c=1` is delta correction. Decay precedes retrieval and the read is **after** update. Initial state, final state and final-state cotangent are explicit. Gate scope (scalar, head, channel), accumulation/cast points and output scale are semantic fields. A normalized linear variant needs a separate denominator state and specified epsilon division. Read-before-write, distinct erase/write factors, low-rank transition, nonlinear update, optimizer step, stabilized max-shift state and hierarchical state are **different descriptors** until proved; no architecture name selects them.

For a chunk, let `E(t,j)=G_t…G_(j+1)`, `V0_t=k_tᵀE(t,0)M_0`, `Y0_t=scale*q_tᵀE(t,0)M_0`, `A[t,j]=k_tᵀE(t,j)k_j` for `j<t` (zero otherwise), `Ω[t,j]=scale*q_tᵀE(t,j)k_j` for `j≤t`, and `B=diag(β)`. The solve and outputs are:

```text
H = I + c B A
H Δ = B (V - c V0)
Y = Y0 + Ω Δ
M_C = E(C,0) M_0 + Σ_j E(C,j) k_j Δ_jᵀ
```

`H` is unit lower triangular; its conditioning still matters. The incoming state propagates across chunk boundaries. For the solve VJP, `λ=solve(Hᵀ,dΔ)`, `dRHS=λ`, and `dH=-λΔᵀ` on the legal lower-triangular entries, followed by derivatives through factors and boundary state. A token-level reverse recurrence is the independent oracle. This algebra allows a candidate parallel schedule; rank growth, conditioning, state traffic, casts and backward decide whether it is efficient or numerically acceptable. State-dependent nonlinear coefficients do not receive an affine scan by label.

An independent token recurrence and VJP must cover every operand, transition, initial state and final-state loss. NumPy, Torch and Triton use the same typed request/result ABI with separately stated capability envelopes. **Current gap:** K2 has lower-level reference/native pieces but no executable public graph binder; selecting a K2 anchor is not K2 coverage. See the [roadmap](../planning/roadmap.md).
