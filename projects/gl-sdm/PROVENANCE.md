# Sources

The experiment structure is adapted from
[ATMA at 28bb3de](https://github.com/kreasof-ai/atma/tree/28bb3de8afbe7c0b00115e0fbff36afc9ad49c11):

| Here | ATMA source |
| --- | --- |
| `model.py` | `external_baselines/model.py`, `model/layers.py`: model/block contracts, normalization, gated squared-ReLU MLP, soft-capped head |
| `mixers.py:Transformer` | `train/model.py`: RoPE softmax projections, QK normalization and output gate |
| `train.py`, `checkpoint.py` | `raven_baseline/train.py`: token batches, microbatch accumulation, AdamW/Muon options, validation curves, checkpoint files and structured log blocks |
| `regularization.py`, `muon.py` | `train/reg.py`, `train/optimizer.py`, copied with their equations intact |
| `data.py` | `train/data.py`: nanoGPT shard header and next-token shift |
| `evaluate.py` | `eval.py`, `scaled_ablation/evaluate.py`: chunked head, clean/junk prefix CE, digit-code induction needle and absent-needle control |
| `inference.py`, `benchmark.py`, `verify.py` | ATMA's prefill/decode, warmup timing and reference verification approach, implemented against the common model contract |

The Transformer uses full attention in **every block**, ordinary `nn.Linear` and
PyTorch SDPA. It has no local convolution blocks, Titans memory, custom linear
kernels or URM calls. RoPE matches ATMA's half-truncated 1024 frequency schedule.

SDM uses Meta's complete `SparseDeltaMemory` layer at
[183e7df](https://github.com/facebookresearch/sparse-delta-memory/tree/183e7df809131b80ad4393741029d0f20fc3640b).
Its launch adapter pads each independent head/request for arbitrary sequence
lengths and disables autocast around the upstream FP32 WY solve. This preserves
upstream learned initial memory and gradients; it does not replace the kernel.
Training uses `GatedSparseMemoryWriteRead`; prefill uses the upstream inference
WY implementation; decode uses `fused_decode_step`. Both CUDA extensions must
load successfully before constructing a model. Snapshot quantization is off.

GDN2 uses FLA's complete `GatedDeltaNet2` layer, including its short convolutions,
gates and output normalization, at
[864a87f](https://github.com/fla-org/flash-linear-attention/tree/864a87f6ce5be8828bef81eb22baafd41937cdf2).
FLA selects chunk training/prefill and fused recurrent short inference.

Reference checks explicitly substitute independent PyTorch attention/recurrence
equations. They share projections, routing and normalization with production to
check kernel integration. They are never enabled by a production runner.

Deliberate harness corrections: non-finite runs abort; evaluation never skips
OOM samples; CPU and absolute shard paths work; checkpoints include optimizer,
RNG and consumed-batch state; GPU timings synchronize and exclude warmup;
MFU is explicit 6ND with the sparse learned memory bank excluded from active N.
No Transformer attention FLOP proxy is assigned to recurrent baselines. A10G
uses 70 dense BF16 TFLOP/s. All parameter counts are also recorded. Equal-width
configs are starting points, **not parameter-matched quality claims**.

## GL-SDM

`global_model.py` and `memory.py` implement this repository's proposed model;
they do not wrap a baseline as GL-SDM. They reuse the ATMA model/head/block
interfaces and MLP form, with a tied reasoner, product-key routes, one global
learned FP32 bank, token-level frozen snapshots and weighted delta commits.
The adaptive variant uses ACT's cumulative halt probabilities, remainder mass,
weighted latent output and ponder cost. Fixed depth, final-only writes and
write-every-step controls are configurable within GL-SDM.

The PyTorch memory implementation uses stable address sorting, per-address
segmented reduction and unique-address index updates. Its explicit oracle uses
dense one-hot reads and writes. Snapshot identity and version are checked;
memory and routing parameters are never changed during a forward transaction.
Benchmark logs report actual reasoning depth and retain unique-parameter 6ND.
Repeated tied-weight applications are not counted in that MFU estimate.

The chunk path in `chunk.py` composes routing and snapshot reads from the public
frozen [URM revision 604bfdf](https://github.com/kreasof-ai/urm/tree/604bfdf5d2c827266a32ef142ca996cc712d70f0),
including both backwards. It adopts URM's highest-address ties. Physical zero
padding of width-64 values selects its existing width-128 schedule; logical
values and memory capacity remain unchanged. The actual plan and both widths
are recorded. No URM source, baseline kernel or dependency pin was copied or
changed. Chunk boundaries and bounded local causal SDPA follow the proposal's
chunk-commit clock. The token-clock control retains its earlier implementation.

Dense reasoner/proposal/head-loss arithmetic uses ordinary PyTorch compilation
with eager BF16 rounding casts preserved. Fixed writes batch across reasoning
steps because their snapshot is immutable. `training_graph.py` captures only
fixed-shape forward/backward; data copies, finite checks, clipping and optimizer
updates are timed by the same runner. Warmup/capture do not update parameters or
consume training data. The project-owned `kernels.py` continues to lower ordered
sparse commits, for which the frozen URM package has no executable lowering.

The token path still uses the project's stable smaller-address router,
selected-row read/proposal backwards and ordered commits. Chunk-model reference
checks use independent PyTorch route/gather equations, segmented commits and
dense attention in FP32; BF16 shares production SDPA to isolate memory kernels.
Its output/gradient limits stay unchanged. Split continuation records a separate
1% relative L2 bound, with an independent strict FP32 check: the same BF16
route drift reproduces in the pure PyTorch memory control. Separate small
fixtures use dense one-hot memory equations.
Large-bank whole-model checks expose failures in the unchanged output/gradient
gates, including a long FP32 check; these are preserved in the diagnostic
artifacts rather than reported as passing. Production-bank read operands and
their gradients independently pass stricter comparisons.
Changing the write clock or reasoning depth is reported explicitly. Quality
comparisons and learned adaptive depth require converged training.
