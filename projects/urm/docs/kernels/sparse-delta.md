# K3: indexed mutable state

**Written sparse-delta equation.** For logical slots `s`, mutable state `M_t[s,:]`, sparse write weights `w_t[s]`, read weights `q_t[s]`, value `v_t`, scalar `β_t`, decay `g_t≤0`, and selected-slot mask `a_t[s]`:

```text
G_t[s] = exp(g_t * a_t[s])
Z_t[s,:] = G_t[s] * M_(t-1)[s,:]
h_t = Σ_s w_t[s] Z_t[s,:]
δ_t = β_t (v_t - h_t)
M_t[s,:] = Z_t[s,:] + w_t[s] δ_t
y_t = Σ_s q_t[s] M_t[s,:]        # after-update read
```

Selected slots decay even if their write weight is zero. Within-token duplicate write indices are rejected in this contract; cross-token collisions occur in token order. Route selection, tie/capacity policy, index dtype, provenance, read timing, commit/version effects, initial/final state, precision and gradients are declared separately. A before-update read or another collision/update rule cannot borrow this result. Routes are logical until placement maps them to physical pages or communication.

For a chunk, interval decay `E(t,j)` yields `V0_t=w_tᵀE(t,0)M_0`, `Y0_t=q_tᵀE(t,0)M_0`, `A[t,j]=w_tᵀE(t,j)w_j` for `j<t`, and `Ω[t,j]=q_tᵀE(t,j)w_j` for `j≤t`. With `B=diag(β)`, solve `HΔ=B(V-V0)` for `H=I+BA`, then `Y=Y0+ΩΔ` and `M_C=E(C,0)M_0+Σ_j E(C,j)w_jΔ_jᵀ`. `H` is unit lower triangular; the boundary state remains ordered.

The independent token VJP uses `F=dM_t+q_t dy_tᵀ`, `dq_t=M_t dy_t`, `dδ_t=w_tᵀF`, `dv_t=β_t dδ_t`, `dβ_t=dot(dδ_t,v_t-h_t)`, `dw_t=Fδ_t-Z_t dv_t`, `dZ_t=F-w_t dv_tᵀ`, `dg_t=Σ_{selected s,d}dZ_t[s,d]Z_t[s,d]`, and `dM_(t-1)=G_t dZ_t`. The chunk solve VJP uses `λ=solve(Hᵀ,dΔ)` and `dH=-λΔᵀ` on legal entries. Fixed discrete route indices have no ordinary derivative; score/weight gradients are separate. Floating-point state-commit frequency and interval factor stability are part of the numerical envelope; a real-arithmetic identity is not BF16 bitwise equality.

The existing route-to-state graph fragment is narrow. Its route provenance bridge, overlap behavior, reference read timing and source-model frontend must be qualified separately. Product-key route generation is a pure typed operation; it does not make K3 specific to SDM. A different indexed fast-weight update needs a distinct law and independent clients before entering core. See the [composition ledger](../planning/architecture-composition.md) and [evidence rules](../validation/evidence.md).
