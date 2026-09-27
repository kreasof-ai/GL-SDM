# URM integration and the large-bank bottleneck

The current 16-layer GL-SDM stack compiles product-key routing and snapshot
reads, including backward, through frozen URM revision
`604bfdf5d2c827266a32ef142ca996cc712d70f0`. Local attention and dense projections
use PyTorch. GL-SDM owns the transaction clock, proposal buffering and ordered
commit. The new stack introduces no Triton kernel.

## What the adapter does

The 134,217,728-entry shared bank requires product-key factor extent 512.
URM's declared support stops at 256. The project's explicitly enabled compiler
support override permits only factor 512, eight routes, FP32 scores and INT32
indices. It preserves dependency/hardware checks and restores the original
probe after compilation. The actual native kernel plan receives the original
larger shape. Forward/backward comparisons at the larger shape pass, but this
is experimental shape support, not an upstream guarantee.

During training the adapter pads logical value width 64 to physical width 128
to select an existing faster URM read-backward schedule. All global layers reuse
one padded snapshot per chunk. Inference has no backward and reads width 64
directly. URM source and `shared/requirements-urm.txt` are unchanged.

## Measured cost

The [profile](sixteen_layers/gl_sdm_memory_profile.json) covers one batch of one,
length 512, forward/backward only. It excludes clipping and optimizer updates.
Its 380.03 ms cumulative GPU kernel time is an instrumented diagnostic, not
an update-throughput measurement.

| Operation | Share of measured GPU kernel time |
| --- | ---: |
| FP32 additions | 56.19% |
| FP32 fills | 21.85% |
| Copies | 5.00% |

Each of the four global layers reads once per chunk. Write prediction also
reads in the first three chunks, for 28 native read backwards in total.
URM's current read backward returns a dense `zeros_like(memory)` adjoint:
`[1, 8, 262144, 128]` is 1 GiB in FP32. Autograd then adds these full gradients.
Shape attribution records 20 full-width-128 `add_` calls taking 134.40 ms and
28 width-128 `zeros_like` calls taking 60.91 ms. Nested `zero_`/`fill_` timings
must not be added to their parent timings. Logical-width-64 gradients, padding
backward and functional commits add further state traffic.

This explains why reducing dense parameter FLOPs does not automatically improve
speed. The 128-token transaction clock and much larger bank expose costs that
were small in the earlier tied-weight controls. The profile implicates the
current composition and dense gradient representation; it does not prove that
GL-SDM needs an architecture-specific URM kernel.

## Next optimizations to investigate

1. **Use existing URM operations on compact rows.** Keep native product-key
   routing, gather the union of touched rows, and evaluate whether URM's
   provided-route read can operate on that compact state. Accumulate shared-bank
   gradients once, rather than materializing a full adjoint for every read.
   This is initially project-owned integration work. It needs full-size
   output/gradient and continuation checks before replacing the current path.
2. **Use transaction overlays.** Keep the immutable learned base and compact
   committed deltas instead of copying a full bank at each version. The reusable
   GL-SDM memory contract should expose identical snapshots to CSDM. Stable
   collision ordering and gradients through prior commits must be preserved.
3. **Consider generic URM improvements only after that audit.** A width-64
   read schedule that avoids adapter padding, or reusable selected-row adjoint
   aggregation, could benefit multiple architectures. Either requires an
   explicitly scoped URM task, a new pin and dependent-workload validation.

BF16 reference parity is a separate unresolved requirement. The full-size
FP32 model passes strict logit, gradient and continuation checks; that result
must not be used to dismiss the BF16 failure. The 40% MFU target remains unmet.
