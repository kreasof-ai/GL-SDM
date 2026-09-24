# Runtime and provider contract

The runtime receives a **complete immutable plan** from the compiler. It validates role-indexed operands and state sessions, then invokes the selected provider. It never selects an architecture, changes an equation, guesses a schedule, loads a source comparator, or silently changes provider tier.

## One request and result ABI

For K1/K2/K3 and their reference/native implementations, a semantic request contains: family and closed equation descriptor; role names independent of user tensor spelling; logical axis/shape constraints; dtype, accumulation and cast policy; mode (`training`, `prefill`, `decode`); output and gradient arity; route/head map; state layout and initial/final ownership; read/write/collision/commit effects. Provider capability matching includes all of those fields, plus device/layout and exact executable entry point. A provider either accepts the whole request or returns a structured decline before plan emission.

The result contains output tensors, explicitly named final states/caches, and requested gradients or a differentiable execution handle. Reference NumPy and Torch providers implement the **same semantic signatures** as Triton; they need not mirror physical tile variants. NumPy supplies independent high-precision equations, Torch a transparent differentiable reference, and Triton a qualified native schedule. Library adapters are a separate tier with provenance and cost.

## Serialized plan and binding

Each region step records a stable region ID, descriptor hash, provider ID and tier, role-to-edge bindings, effect ordering, placement, concrete launch config, mode, numerical envelope and backward/decode path. The plan also records unfused ordinary steps or a proved fused region, route provenance, state lifetime and materialization boundaries. The binder rejects missing, extra, reordered or tampered steps and incompatible runtime shapes, routes or state sessions. It does not look up tensors globally by literal names such as `query` and `key`.

The compiler owns capability and schedule decisions. Runtime owns tensor-value validation, state alias/lifetime checks, provider invocation and cache-session continuation. Backend modules contain device kernels and minimal launch thunks; profiler policy and search remain outside them. A generated route retains its certificate through graph edges; an externally supplied route is validated or uses an explicitly declared trust contract. `certify_trusted` cannot stand in for value/provenance validation.

## State and request isolation

K1 KV caches, K2 fixed-address state bundles and K3 indexed states have distinct typed ABIs. Prefill followed by repeated decode must match reference continuation at the same positions and precision policy. Training final-state cotangents propagate across chunks. The serving application owns request scheduling, eviction, cancellation and batching; it cannot make persistent state global by architecture name. Library/reference fallback is visible in the plan and measurement, never counted as native.

The [roadmap](../planning/roadmap.md) lists the implementation order. The [evidence rules](../validation/evidence.md) define when a mode can be reported as supported.
