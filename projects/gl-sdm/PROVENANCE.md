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
