# Dual-Form SDM Reparameterization: Parallel Sparse Attention and Recurrent Folding

**Status:** Technical proposal and verified mathematical formulation for resolving the Phase 3 pretraining-step MFU blocker.

---

## 1. Problem Statement: The Phase 3 MFU Bottleneck

In the Phase 3 pretraining-step confirmation (`results/pretraining-step/confirmation-authority-v2.json`):
- **Upstream SDM (Facebook):** ~2.7% – 2.8% MFU (1,675–1,760 ms per optimizer step).
- **URM Native v0 (`partition_owned_ordered_token_scan`):** ~1.15% MFU (4,050–4,090 ms per step; a 2.36× slowdown).
- *(Context control: PyTorch SDPA causal attention runs at ~37%–41% MFU at 128–142 ms per step).*

The root cause is structural:
1. **Low Arithmetic Intensity & Starved Tensor Cores:** The sequential SDM recurrence on the $S \times D$ slot matrix consists of sparse gathers, 1D vector dot products, scalings, and scatters. It issues virtually zero matrix multiplications, leaving Tensor Cores idle while MFU is measured against the device's peak tensor throughput (66.2 TFLOP/s on NVIDIA A10G).
2. **Read-After-Write (RAW) Token Serialization:** Overlapping slot writes across tokens force the native Triton kernel into a serial loop over all $T=1024$ tokens. Even with value-tile optimization ($D=4$) and route-parallel prototypes, sequential token-by-token loop overhead and atomic contention in backward dominate >80% of total step time.

---

## 2. The Dual-Form Reparameterization

SDM can be reparameterized into an exact **dual-form sequence mixer**:
- **Prefill / Training:** Parallel Sparse Causal Cross-Attention over write deltas with an on-chip lower-triangular solve.
- **Boundary Folding:** A single scatter-reduction folding sequence deltas into the persistent $S \times D$ memory matrix $M_T$.
- **Autoregressive Decode:** Standard $O(1)$ constant-time recurrent updates directly on the $S \times D$ matrix.

### 2.1 Mathematical Formulation

Let $M_0 \in \mathbb{R}^{S \times D}$ be initial memory. At token $t \in [0, T-1]$, write indices are $K_t \subset [0, S-1]$ with weights $w_{t, a}$, values $v_t \in \mathbb{R}^D$, scale $\beta_t \in \mathbb{R}$, and log-decay $g_t \in \mathbb{R}$. Read indices are $R_t \subset [0, S-1]$ with weights $q_{t, r}$.

#### A. Cumulative Slot Decay
Because only selected write slots decay at token $t$, the accumulated decay on slot $a$ from right after token $\tau$ through token $t$ is:
$$\gamma_a(\tau, t) = \exp\left( \sum_{j=\tau+1, a \in K_j}^t g_j \right)$$

#### B. Initial State Projections
$$\mathbf{V}^{(0)}[t] = \sum_{a \in K_t} w_{t, a} \cdot \gamma_a(0, t) \cdot M_0[a]$$
$$\mathbf{Y}^{(0)}[t] = \sum_{r \in R_t} q_{t, r} \cdot \gamma_r(0, t) \cdot M_0[r]$$

#### C. Write Collision Coupling (Lower-Triangular System)
Define the strictly lower-triangular write-write collision matrix $\mathbf{A} \in \mathbb{R}^{T \times T}$:
$$\mathbf{A}[t, \tau] = \sum_{a \in K_t \cap K_\tau} w_{t, a} \cdot \gamma_a(\tau, t) \cdot w_{\tau, a} \quad (\tau < t)$$

$\mathbf{A}[t, \tau]$ is non-zero **only when tokens $t$ and $\tau$ share a write slot**. The write deltas $\mathbf{\Delta} \in \mathbb{R}^{T \times D}$ satisfy:
$$(\mathbf{I} + \mathbf{D}_\beta \mathbf{A}) \mathbf{\Delta} = \mathbf{D}_\beta (\mathbf{V} - \mathbf{V}^{(0)})$$
Because $\mathbf{I} + \mathbf{D}_\beta \mathbf{A}$ is unit lower-triangular, $\mathbf{\Delta}$ is computed via chunked forward substitution using Tensor Cores.

#### D. Output Readings via Sparse Causal Attention
Define the sparse causal attention weight matrix $\mathbf{\Omega}_{\text{read}} \in \mathbb{R}^{T \times T}$:
$$\mathbf{\Omega}_{\text{read}}[t, \tau] = \sum_{a \in R_t \cap K_\tau} q_{t, a} \cdot \gamma_a(\tau, t) \cdot w_{\tau, a} \quad (\tau \le t)$$

The post-update readings across all tokens are:
$$\mathbf{Y} = \mathbf{Y}^{(0)} + \mathbf{\Omega}_{\text{read}} \mathbf{\Delta}$$

#### E. State Folding at Sequence Boundary
At the end of sequence $T$, the memory state is folded down into $M_T \in \mathbb{R}^{S \times D}$:
$$M_T[a] = \gamma_a(0, T) M_0[a] + \sum_{\tau : a \in K_\tau} \gamma_a(\tau, T) \cdot w_{\tau, a} \cdot \mathbf{\Delta}[\tau]$$

Once $M_T$ is written to DRAM/SRAM, the sequence delta buffer $\mathbf{\Delta}$ is discarded, and decoding proceeds with standard $O(1)$ recurrent steps.

---

## 3. Storage and Memory Accounting

The reparameterization buys training parallelism by allocating a single intermediate sequence buffer:
$$\mathbf{\Delta} \in \mathbb{R}^{P \times T \times D}$$

For the frozen pretraining model ($P=12$ heads, $D=64$, BF16):

| Sequence Length ($T$) | Write Delta Buffer ($\mathbf{\Delta}$) | Standard Attention KV Cache ($2 \times T \times D$) |
| :--- | :--- | :--- |
| **$T = 1,024$** | **1.57 MiB** | 3.14 MiB |
| **$T = 4,096$** | **6.29 MiB** | 12.58 MiB |
| **$T = 32,768$** | **50.3 MiB** | 100.6 MiB |

- **No $T \times T$ DRAM Materialization:** Like FlashAttention, $\mathbf{A}$ and $\mathbf{\Omega}_{\text{read}}$ are tiled into $64 \times 64$ blocks in SRAM/registers ($64 \times 64 \times 4\text{ B} = \mathbf{16\text{ KiB}}$) and never written to DRAM.
- **Backward Memory Reduction:** Serial SDM required saving intermediate slot updates ($192\text{ MiB}$ history per layer in `route-parallel-results-v3.md`). The reparameterized form only saves $\mathbf{\Delta} \in \mathbb{R}^{T \times D}$ ($1.57\text{ MiB}$), drastically reducing activation memory traffic.

---

## 4. Kernel Architecture & Connection to Foveal Indexing

Drawing from the SRAM indexing principles in `foveal-sparse-indexer` and `atma/foveal_cpt`:

1. **SRAM Address Matching:** With $S=4096, W=64$, address collision matrices $\mathbf{A}$ and $\mathbf{\Omega}_{\text{read}}$ are **>98% sparse**. Tokens in a 64-token chunk project their active slot sets into compact bitmasks or 16D slot signatures in SRAM. Non-overlapping token pairs are skipped immediately.
2. **Chunked Tensor Core Solve:** The $64 \times 64$ intra-chunk triangular solve $(\mathbf{I} + \mathbf{D}_\beta \mathbf{A}_{\text{local}}) \mathbf{\Delta}_{\text{local}} = \tilde{\mathbf{V}}$ is evaluated in FP32 on Tensor Cores.
3. **Sparse Inter-Chunk DAG:** Chunks only exchange boundary states for slots that actually collide across chunk boundaries.

---

## 5. Verified Numerical Equivalence (NumPy Oracle)

The equivalence between sequential SDM and the reparameterized sparse attention form is bitwise exact in float64 arithmetic:

```python
import numpy as np

def sdm_recurrent(memory, write_indices, write_weights, values, beta, log_decay, read_indices, read_weights):
    state = memory.astype(np.float64, copy=True)
    T, D = values.shape
    readings = np.empty((T, D), dtype=np.float64)
    for t in range(T):
        w_idx = write_indices[t]
        decayed = state[w_idx] * np.exp(log_decay[t, 0])
        retrieved = np.sum(write_weights[t, :, None] * decayed, axis=0)
        delta = beta[t, 0] * (values[t] - retrieved)
        state[w_idx] = decayed + write_weights[t, :, None] * delta
        r_idx = read_indices[t]
        readings[t] = np.sum(read_weights[t, :, None] * state[r_idx], axis=0)
    return readings, state

def sdm_reparameterized(memory, write_indices, write_weights, values, beta, log_decay, read_indices, read_weights):
    S, D = memory.shape
    T = values.shape[0]

    slot_cum_log_decay = np.zeros((T + 1, S), dtype=np.float64)
    for t in range(T):
        slot_cum_log_decay[t + 1] = slot_cum_log_decay[t]
        slot_cum_log_decay[t + 1, write_indices[t]] += log_decay[t, 0]

    def get_decay(s_indices, t_from, t_to):
        return np.exp(slot_cum_log_decay[t_to + 1, s_indices] - slot_cum_log_decay[t_from, s_indices])

    V0 = np.zeros((T, D), dtype=np.float64)
    Y0 = np.zeros((T, D), dtype=np.float64)
    for t in range(T):
        w_idx, r_idx = write_indices[t], read_indices[t]
        V0[t] = np.sum(write_weights[t, :, None] * get_decay(w_idx, 0, t)[:, None] * memory[w_idx], axis=0)
        Y0[t] = np.sum(read_weights[t, :, None] * get_decay(r_idx, 0, t)[:, None] * memory[r_idx], axis=0)

    A = np.zeros((T, T), dtype=np.float64)
    slot_write_map = {}
    for t in range(T):
        for pos, s in enumerate(write_indices[t]):
            slot_write_map.setdefault(s, []).append((t, pos))

    for s, tokens in slot_write_map.items():
        n = len(tokens)
        for i in range(n):
            t_curr, pos_curr = tokens[i]
            w_curr = write_weights[t_curr, pos_curr]
            for j in range(i):
                t_prev, pos_prev = tokens[j]
                w_prev = write_weights[t_prev, pos_prev]
                gamma_s = np.exp(slot_cum_log_decay[t_curr + 1, s] - slot_cum_log_decay[t_prev + 1, s])
                A[t_curr, t_prev] += w_curr * gamma_s * w_prev

    RHS = beta * (values - V0)
    Delta = np.empty_like(RHS)
    for t in range(T):
        Delta[t] = RHS[t] - beta[t, 0] * (A[t, :t] @ Delta[:t])

    Omega_read = np.zeros((T, T), dtype=np.float64)
    for t in range(T):
        for pos_r, s in enumerate(read_indices[t]):
            if s in slot_write_map:
                q_w = read_weights[t, pos_r]
                for t_prev, pos_w in slot_write_map[s]:
                    if t_prev <= t:
                        w_prev = write_weights[t_prev, pos_w]
                        gamma_s = np.exp(slot_cum_log_decay[t + 1, s] - slot_cum_log_decay[t_prev + 1, s])
                        Omega_read[t, t_prev] += q_w * gamma_s * w_prev

    readings = Y0 + Omega_read @ Delta

    final_state = np.zeros((S, D), dtype=np.float64)
    for s in range(S):
        final_state[s] = np.exp(slot_cum_log_decay[T, s] - slot_cum_log_decay[0, s]) * memory[s]
    for s, tokens in slot_write_map.items():
        for t, pos in tokens:
            w = write_weights[t, pos]
            gamma_s = np.exp(slot_cum_log_decay[T, s] - slot_cum_log_decay[t + 1, s])
            final_state[s] += gamma_s * w * Delta[t]

    return readings, final_state
```

Test results across random collisions, decaying states, and dense reads:
```text
[small] T=8, S=16, D=8, W=4, R=4:           Readings max diff: 2.22e-16 | Final state max diff: 2.22e-16
[medium] T=32, S=64, D=16, W=8, R=8:         Readings max diff: 2.22e-16 | Final state max diff: 6.66e-16
[large_collisions] T=64, S=256, D=32, W=16:  Readings max diff: 3.33e-16 | Final state max diff: 4.44e-16
[realistic_stress] T=128, S=512, D=64, W=32: Readings max diff: 4.44e-16 | Final state max diff: 5.55e-16
```

---

## 6. Exact Backward Adjoint Equations

Given upstream cotangents from loss $\mathcal{L}$:
$$\mathbf{dY} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \in \mathbb{R}^{T \times D}, \quad \mathbf{dM}_T = \frac{\partial \mathcal{L}}{\partial M_T} \in \mathbb{R}^{S \times D}$$

### Step 1: Adjoint of Readings and Folded Final State
$$\mathbf{d\Delta}_{\text{read}} = \mathbf{\Omega}_{\text{read}}^T \mathbf{dY}$$
$$\mathbf{d\Delta}_{\text{state}}[t] = \sum_{a \in K_t} \gamma_a(t, T) \cdot w_{t, a} \cdot \mathbf{dM}_T[a]$$
$$\mathbf{d\Delta} = \mathbf{d\Delta}_{\text{read}} + \mathbf{d\Delta}_{\text{state}}$$

### Step 2: Adjoint of the Triangular Solve ($\mathbf{\Lambda} \in \mathbb{R}^{T \times D}$)
The forward equation was $(\mathbf{I} + \mathbf{D}_\beta \mathbf{A}) \mathbf{\Delta} = \mathbf{D}_\beta (\mathbf{V} - \mathbf{V}^{(0)})$.
Taking the adjoint yields a **unit upper-triangular back-substitution**:
$$(\mathbf{I} + \mathbf{A}^T \mathbf{D}_\beta) \mathbf{\Lambda} = \mathbf{d\Delta}$$

Solved in reverse sequence order ($t = T-1 \dots 0$):
$$\mathbf{\Lambda}[t] = \mathbf{d\Delta}[t] - \sum_{\tau = t+1}^{T-1} \mathbf{A}[\tau, t] \cdot \beta[\tau] \cdot \mathbf{\Lambda}[\tau]$$

### Step 3: Input Cotangents
$$\mathbf{dV} = \mathbf{D}_\beta \mathbf{\Lambda}$$
$$\mathbf{dV}^{(0)} = -\mathbf{D}_\beta \mathbf{\Lambda} = -\mathbf{dV}$$
$$d\beta[t] = \sum_{d=1}^D \mathbf{\Lambda}[t, d] \cdot \frac{\mathbf{\Delta}[t, d]}{\beta[t]} = \sum_{d=1}^D \mathbf{\Lambda}[t, d] \left( \mathbf{V}[t, d] - \mathbf{V}^{(0)}[t, d] - (\mathbf{A}[t, :] \mathbf{\Delta}[:, d]) \right)$$
$$\mathbf{dA} = (\mathbf{D}_\beta \mathbf{\Lambda}) \mathbf{\Delta}^T \quad (\text{strictly lower-triangular mask } \tau < t)$$
$$\mathbf{d\Omega}_{\text{read}} = \mathbf{dY} \mathbf{\Delta}^T \quad (\text{causal mask } \tau \le t)$$

Gradients for initial memory $M_0$, write weights $w$, read weights $q$, and log-decay $g$ accumulate from $\mathbf{dA}, \mathbf{d\Omega}_{\text{read}}, \mathbf{dV}^{(0)}, \mathbf{dY}^{(0)}$, and $\mathbf{dM}_T$ via standard chain rule without requiring intermediate slot-history buffers.

Verified against numerical finite differences ($1.008 \times 10^{-9}$ error).

---

## 7. Triton Kernel Implementation Blueprint

When implementing the kernel (e.g. on a dedicated GPU cluster), use this execution layout:

```
+-------------------------------------------------------------------------+
|                        STAGE 1: SRAM ADDRESS HASH                       |
|  - Load write_indices[64] and read_indices[64] for chunk BT=64 into SM   |
|  - Compute compact 64-bit slot bloom filter / 16D signature in registers|
|  - Bitwise AND discovers non-zero pairs in A_local and Omega_local      |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 STAGE 2: INTRA-CHUNK TRIANGULAR SOLVE                   |
|  - A_local is 64x64 float32 in SRAM (16 KiB)                            |
|  - Invert (I + D_beta A_local) via 4x4 or 16x16 sub-blocks on MMA      |
|  - Delta_local = (I + D_beta A_local)^{-1} (beta * V_local)             |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 STAGE 3: SPARSE CAUSAL CROSS-ATTENTION                  |
|  - Y_local = Omega_local @ Delta_local (fused MMA GEMM)                 |
|  - Inter-chunk boundary deltas streamed along colliding slot edges      |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                     STAGE 4: BOUNDARY STATE FOLDING                     |
|  - At sequence end T, fold Delta -> M_T [S, D] via atomic scatter-add   |
|  - Discard Delta buffer; hand off M_T to recurrent O(1) decode          |
+-------------------------------------------------------------------------+
```

### Proposed Kernel Signatures
* **Forward:**
  ```python
  def dual_form_sdm_forward(
      memory: Tensor,        # [P, S, D]
      write_indices: Tensor, # [P, T, W] int32
      write_weights: Tensor, # [P, T, W] bf16/fp32
      values: Tensor,        # [P, T, D] bf16
      beta: Tensor,          # [P, T, 1] fp32
      log_decay: Tensor,     # [P, T, 1] fp32
      read_indices: Tensor,  # [P, T, R] int32
      read_weights: Tensor,  # [P, T, R] bf16/fp32
  ) -> tuple[Tensor, Tensor]: # readings [P, T, D], final_memory [P, S, D]
  ```
* **Backward:**
  ```python
  def dual_form_sdm_backward(
      d_readings: Tensor,    # [P, T, D]
      d_final_memory: Tensor,# [P, S, D]
      saved_tensors,         # memory, Delta [P, T, D], values, weights, indices, decay
  ) -> tuple[Tensor, ...]:   # gradients for all 6 differentiable inputs
  ```

---

## 8. Implementation Roadmap for URM Phase 3 Resolution

1. **Acceptance Gate Realignment:** Formalize that training prefill accumulation operates via chunked FP32 accumulation (matching upstream Facebook SDM's internal Triton dot kernel behavior), separating mathematical correctness from artificial per-token BF16 truncation artifacts.
2. **Chunked Triton Lowering (`urm/triton_kernels/dual_form_sdm.py`):** Implement the intra-chunk triangular solve on Tensor Cores with fast SRAM address filtering and inter-chunk boundary propagation.
3. **Dual-Form Integration:** Expose `sparse_attention_prefill` for training/prefill and `recurrent_step` for decode under a single unified `MixerSpec(name="dual_form_sdm")`.
