# Frozen original Sparse Delta Memory contract (v1)

This document freezes URM's Phase 3 **SDM baseline integration**. It is not a
native URM kernel and does not claim completion of a page-local GL-SDM kernel.

## Upstream identity and installation

- Repository: [`facebookresearch/sparse-delta-memory`](https://github.com/facebookresearch/sparse-delta-memory)
- Pinned commit: `183e7df809131b80ad4393741029d0f20fc3640b`
  (the repository's initial commit, dated 2026-07-06)
- License: CC-BY-NC 4.0. The checkout also states that it derives from
  BSD-3-Clause Meta Lingua. URM neither copies nor modifies upstream source.
- Installation: the upstream has no `pyproject.toml` or `setup.py`. Clone it,
  check out the exact commit, create its environment with
  `bash setup/create_env.sh`, and put the checkout root on `PYTHONPATH`.
- Frozen validated runtime: Python 3.12.13, PyTorch 2.8.0, Triton 3.4.0,
  CUDA 12.9, `einops` 0.8.2, `ninja` 1.13.2, a runtime CUDA toolkit with
  `nvcc`, CUDA driver/CUDART/CRT/NVVM development files, system `gcc/g++`, and
  an NVIDIA SM80-or-newer GPU. The authors specify Python 3.11 in their Conda
  script and PyTorch >=2.8 / Triton >=3.4 / SM80+ in their README. The adapter
  pins the exact validated Torch/Triton pair and fails closed on drift.
- When CUDART is installed under a Conda prefix, use the upstream setup's
  activation rule: `LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"`.

The exact callable API is:

```text
SparseDeltaMemory._get_product_keys_for_head(x, num_keys, half_key)
SparseDeltaMemory.read(memory, q_val, q_idx)
SparseDeltaMemory.gated_write_read(
    memory, k_idx, k_val, v, beta, g, q_idx, q_val,
    grad_final_memory=None,
)
SDMLayerState(memory, seq_len=0); update_(new_memory, seq_len=1)
```

The first method is private upstream, but it is the only callable that exposes
the authors' exact product-key address trace. Its use is therefore explicit in
the pin and compatibility contract; a revision changing it is rejected.

## Typed boundary

`SDMAddressTrace` contains four contiguous CUDA tensors:

| Field | Layout | Dtype | Meaning |
| --- | --- | --- | --- |
| `write_indices` | `[P,T,W]` | `int64` | Global addresses, including `P` partition offsets |
| `write_weights` | `[P,T,W]` | state dtype | Softmax over selected product-key scores |
| `read_indices` | `[P,T,R]` | `int64` | Global post-update read addresses |
| `read_weights` | `[P,T,R]` | state dtype | Softmax over selected product-key scores |

Addresses are strictly increasing and unique within every token. Different
tokens may select the same address. Each parallel lane owns a contiguous
`slots_per_partition` range, and traces may not cross that range. Empty `P`,
`T`, `W`, or `R` is unsupported; width one is the minimal route.

Address generation splits a score vector of width `2*sqrt(slots)` in half,
takes top candidates from each half, takes top sums from their Cartesian
product, maps a pair `(i,j)` to `i*sqrt(slots)+j`, sorts the final addresses,
then applies Softmax to the carried scores. Product-key ties are rejected from
the equivalence envelope because upstream `torch.topk` does not specify a
portable tie order.

`SDMState` contains a contiguous CUDA memory tensor
`[P*slots_per_partition,D]` and a host sequence length. `float32` and
`bfloat16` state/value/weight/gate tensors are supported. `D` may be
non-power-of-two where upstream kernels support it; slots must be a perfect
square and divisible by eight. The adapter is deliberately not the existing
routed-reduction signature because state mutation and token order are
observable.

## Mutation and collision semantics

For each parallel lane and token, in sequence order:

```text
M_decay[a] = exp(g_t) * M[a]                       for selected write slots a
retrieved  = sum_w k_weight[t,w] * M_decay[k_idx[t,w]]
delta      = beta_t * (value_t - retrieved)
M[a]       = M_decay[a] + k_weight[t,w] * delta
reading_t  = sum_r q_weight[t,r] * M[q_idx[t,r]]   # observes this token's write
```

Cross-token address collisions therefore use ordered recurrent semantics, not
a transactional sum/mean/last-write merge. Within-token duplicate addresses
are outside the product-key generator's reachable set and are rejected by the
adapter; they are not benchmarked as a supported atomic policy. Collision-heavy
tests repeat valid unique address sets across tokens and compare both output
traces and final memory, so matching only final outputs cannot pass.

The upstream call mutates `memory` during execution. Only after it returns does
the adapter advance `SDMState.sequence_length`, matching upstream
`SDMLayerState.update_`. A read-only call neither mutates memory nor changes the
length. Repeated decode calls reuse the same tensor object and preserve state.

## Supported paths

- Sparse read-only: upstream `SparseDeltaMemory.read` / `fast_embedding_bag`.
- Inference `T=1`: upstream fused decode write then post-update read.
- Inference `2 <= T <= 64`: upstream token-step path.
- Inference `T>64`: upstream WY/chunk prefill path.
- Training prefill with `T>=16`, a tensor-core-safe configured chunk size of at
  least 16, `snapshot_quant="none"`, and upstream autograd. Although the Python
  wrapper admits shorter sequences, its Triton dot kernel rejects them on the
  pinned Torch/Triton/A10G stack, so the adapter fails closed before dispatch.
- Scalar per-token decay `g[P,T,1]`, delta-rule updates, Softmax read/write
  weights, local single-device memory, and persistent cache state.

## Explicitly rejected

- Missing checkout, a checkout not at the pinned commit, a dirty checkout,
  incompatible Torch/Triton, CUDA absence, or pre-SM80 hardware.
- CPU execution, fp16/fp64 state, non-contiguous/mixed-device tensors,
  product-key ties as a portable guarantee, unsorted or duplicate within-token
  addresses, empty routes, cross-partition addresses, and non-square slot
  counts.
- Key-weighted decay, snapshot quantization, memory normalization, alternate
  read/write activations, arbitrary precomputed routing semantics, context or
  tensor parallelism, sharded/offloaded state, transactional GL-SDM commits,
  and deterministic accumulation requests.
- Native URM SDM or page-local lowering claims. The compiler anchor is named
  `facebook_sparse_delta_memory_183e7df_external_adapter`.

## Comparison and measurement

The four levels are NumPy product-key/read/ordered-update oracles, transparent
PyTorch formulations, the pinned original methods called directly, and those
same bound method objects called through `UrmSparseDeltaMemoryAdapter`.
Address indices must match exactly; outputs and full memory state use
dtype-specific tolerances in tests.

`benchmarks/sparse_delta_memory.py` covers read-only smoke, batched prefill,
cached decode, write/update, collision-heavy routing, training prefill, and a
capacity-oriented A10G case. It excludes preparation, cloning, reset, address
generation, cache initialization, and pre-start synchronization from each
steady-state region. Intrinsic upstream output/workspace allocation remains in
the call because the callable API does not accept caller-provided buffers.
Raw alternating AB/BA wall/CUDA-event samples, median, p95, throughput,
separate direct/adapter allocator peaks, analytical traffic estimates, route
distributions, call identity, cold process/import/address/read/update timings,
and complete upstream/runtime provenance are retained in
`results/sparse-delta-memory/benchmark.json`. Decode samples preserve their
preallocated states across invocations; unsupported cases and read-only write
traffic are recorded as `not_applicable`, never as zero.
