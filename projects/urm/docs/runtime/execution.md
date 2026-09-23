# URM compiler and runtime boundaries

Status: implementation specification. The [compiler charter](../compiler/compiler-charter.md)
is normative.

## Pipeline and ownership

1. `frontend/spec.py` owns model-facing declarations; `compiler/semantic.py` owns
   typed semantic operations over logical domains and explicit state effects.
2. `compiler/rewrite.py` owns registered transformations, their preconditions,
   numerical envelope, backward obligations and saved-state policy.
3. `compiler/planner.py` owns legality, candidate selection and executable plans.
   Existing constraint, solver and schedule modules remain compiler facilities;
   bounded schedule selection is distinct from architecture-discovery experiments.
4. `compiler/execution.py` selects legal anchors; `runtime/` binds the serialized
   plan to an executable backend. The serialized plan must drive execution. Do not
   add a competing dispatcher. The compiler holds no GPU execution bodies.
5. `runtime/registry.py` retains the existing low-level backend protocol. Backend
   implementations and external adapters own physical layouts and library APIs.
6. Provider compilation, binary loading and hardware caching belong behind the
   execution boundary. A future Tensor adapter must not redefine mixer semantics.

`urm.ir` is the canonical IR module. The former `urm.backend` and `urm.reference`
wildcard compatibility shims were removed; import `BackendRegistry` from
`urm.runtime` and the NumPy oracle (`execute`, `merge_writes`) from `urm.oracles`.
The existing compiler classes and binder APIs remain in place.

## Initial lowering families

| Family | Required semantics | Initial implementation route |
|---|---|---|
| [Softmax attention](../kernels/softmax-attention.md) | Mask, normalization, head mapping, score scale, training/inference mode | Existing attention adapter plus independently checked native lowerings |
| [Linear/delta](../kernels/linear-delta.md) | Update rule, decay granularity, state shape, read timing, boundary state | Existing gated-delta adapter; additional rules require individual derivations |
| [Sparse delta](../kernels/sparse-delta.md) | Explicit selected slots, collision/order policy, decay, delta correction, read timing and commit precision | Existing certified native anchor; repaired chunked lowering as a separate candidate |

These are families, not exactly three binaries. Prefill, decode, backward, memory
layouts and hardware capabilities may require separate implementations. An
unsupported combination must return a structured decline. No backend may omit
decay, gradients, writes or masks to fit a family signature.

## Rewrite and training legality

The sparse-slot chunk transform is specified in [the formulation](../kernels/sparse-delta.md).
It is a floating-point reassociation, not bitwise equivalence to per-token BF16
commits. No compiler registration is added until a concrete backend passes its
declared numerical envelope, including final-state and decay gradients.

Training resolution requires a validated VJP for every differentiable operand,
including the initial and final persistent state. A forward-only opaque kernel
cannot acquire a backward by naming an autograd fallback. Such a fallback must
explicitly recompute a differentiable equivalent operation, preserve its numerical
contract and RNG state where applicable, and pass VJP tests.

Core IR accepts typed update descriptors, not arbitrary Python callbacks. An
outer framework can compose compiled operations into optimization loops without
claiming that a nonlinear inner optimizer is a supported delta-rule rewrite.

## Runtime and provider contract

Selection checks semantics before performance. A resolved artifact must identify:

- semantic spec and rewrite/version fingerprints;
- training, prefill, decode or forward-only intent;
- shape, strides/layout, operand and accumulation dtypes, state commit policy;
- read timing, masks, selection/collision rules and gradient coverage;
- provider, architecture/device capabilities, compiler and library versions;
- workspace needs, saved tensors, backward/recomputation implementation;
- selected schedule and compiled-source identity.

All properties affecting legality or generated code belong in the cache identity
or must be revalidated on invocation. Capabilities are requirements, not promises
that a provider makes any algorithm efficient. Compile failures and cache misses
may select another *semantically compatible* implementation or decline explicitly.

## Acceptance and integration

Each family needs one frontend-to-executable vertical slice with output/state
agreement, required gradients, serialization/dispatch agreement and supported
inference state continuation. Reuse existing verified anchors first.

Measure complete training steps and prefill/decode separately. Distinguish useful
model FLOPs from extra implementation work; do not reuse a single dense FLOP count
across chunk sizes or algorithms. Record measured latency, peak memory, raw samples
and provenance. Numerical and performance gates apply to the same backend settings.

Tensor integration, broader architecture coverage and additional hardware providers
are subsequent milestones. No MFU percentage, universal portability claim, or
kernel-development duration is an acceptance fact before it is measured.
