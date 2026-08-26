# Baseline catalog

Baseline choices were checked against their official documentation on
26 August 2026. Versions and exact commits must be recorded in every benchmark
result rather than treated as permanent defaults.

## Coverage matrix

| Family | Semantic oracle | Framework baseline | Optimized baseline | URM target | Phase |
| --- | --- | --- | --- | --- | --- |
| Dense causal attention | NumPy dense softmax-reduce | PyTorch SDPA math backend | PyTorch SDPA flash backend and FlashAttention | Dense sequence lowering | 1 |
| Block/local/sparse attention | NumPy masked softmax-reduce | PyTorch FlexAttention | FlexAttention compiled block-sparse backend | Block-sparse sequence lowering | 2 |
| Linear attention | Explicit recurrent/chunk reference | PyTorch eager recurrence | Flash Linear Attention (pinned 0.5.2 gated delta-rule comparator landed; see docs/fla-gated-delta-rule.md) | Recurrent lowering | 2 |
| Selective SSM | Explicit scan reference | PyTorch eager scan | Mamba selective scan | Recurrent lowering | 2 |
| Top-k MoE | NumPy stable top-k gate | PyTorch per-expert eager | MegaBlocks; SonicMoE on supported hardware | Expert grouped-GEMM lowering | 1-2 |
| Advanced MoE routing | Per-family routing/dispatch oracle | PyTorch explicit dispatch | Original model implementation plus grouped-GEMM backend | Typed score/select/balance lowering | 2 |
| Parameter-token mixer | NumPy dense/top-k reduce | PyTorch SDPA formulation | Attention-compatible fused path | Parameter-block lowering | 2 |
| Sparse Delta Memory | NumPy top-k gather-reduce | PyTorch `topk` + gather | Original SDM Triton/CUDA kernels | HBM memory-page lowering | 1-2 |
| Transactional GL-SDM write | NumPy sort/merge/commit | PyTorch sort + segment reduction | Original SDM sparse update plus URM transaction wrapper | Buffered transactional lowering | 1-3 |
| Learned token sparse attention | Dense attention plus selected-index trace | PyTorch gather + SDPA | DeepSeek FlashMLA / DeepGEMM | Token-index lowering | 2 |
| Learned block sparse attention | Dense attention plus selected-block trace | PyTorch block gather + SDPA | MiniMax MSA kernels | GQA-group block lowering | 2 |

## Upstream references

- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
  is the framework-level dense attention API. Its backend selector lets the
  harness force math and flash implementations separately.
- [PyTorch FlexAttention](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html)
  provides score modification and `BlockMask`-based sparse routing; it is the
  first structured sparse-attention comparator.
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) is the direct
  optimized exact-attention comparator when supported by the test GPU.
- The [Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
  is the initial kernel-development reference, not the performance ceiling.
- The authors' [Sparse Delta Memory repository](https://github.com/facebookresearch/sparse-delta-memory)
  is the authoritative SDM implementation and kernel baseline. Its
  `lingua/sparse_delta_memory/` package contains the model layer, product-key
  routing, sparse inner product, fused decay/scatter, inference cache, and
  Triton/CUDA kernels.
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
  supplies baselines across linear-attention and related recurrent mixer
  families. It is also part of the gated delta-rule lineage cited by the SDM
  authors, but it is **not** the source of the SDM kernels.
- [Mamba](https://github.com/state-spaces/mamba) supplies the selective-scan
  comparator and keeps ordered recurrence separate from gather/reduce semantics.
- [MegaBlocks](https://github.com/databricks/megablocks) is the established
  dropless/grouped-GEMM MoE comparator.
- [SonicMoE](https://github.com/Dao-AILab/sonic-moe) is an additional modern MoE
  comparator for supported Hopper and Blackwell systems.
- [DeepSeekMoE](https://github.com/deepseek-ai/DeepSeek-MoE) is the reference
  for fine-grained routed experts, shared-expert isolation, and expert/device
  auxiliary balancing.
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) adds Sigmoid routing,
  routing-only expert bias updates, dropless execution, and node-limited
  dispatch to the DeepSeekMoE structure.
- [Routing-Free MoE](https://github.com/liuyilun2000/RoutingFreeMoE/tree/release)
  is the reference for expert-local threshold activation and adaptive density
  balancing.
- [Kimi K3](https://arxiv.org/abs/2607.24653) is the reference for Stable
  LatentMoE and Quantile Balancing. The linked report is not the original
  auxiliary-loss-free method; it replaces the fixed-step bias update with a
  quantile update at its much larger expert count.
- [DeepSeek FlashMLA](https://github.com/deepseek-ai/FlashMLA) is the optimized
  baseline for DeepSeek token-level sparse attention, together with the
  indexer kernels released through the DeepSeek-V3.2 repositories.
- [MiniMax MSA](https://github.com/MiniMax-AI/MSA) is the optimized baseline for
  per-GQA-group block selection and sparse main attention.
- [Gated DeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) provides the
  reference for independent channel-wise erase and write gates. Mamba-family
  kernels come from the official Mamba repository, and KDA/GDN-family kernels
  come from FLA or their authors' repositories as applicable.

## SDM integration rule

The first SDM adapter should call a pinned revision of the original repository
and treat its outputs and address traces as the optimized baseline. Only after
semantic equivalence is established should URM isolate or rewrite its routing,
gather, and update kernels.

The upstream implementation currently requires PyTorch 2.8 or newer, Triton 3.4
or newer, a CUDA toolkit and host compiler at runtime, and an SM80-or-newer GPU.
It is licensed CC-BY-NC 4.0, so copying or redistributing kernel source requires
a separate license review; an external adapter avoids silently mixing that code
into URM's eventual license.

## Baseline rules

1. Compare against both a transparent framework implementation and the best
   applicable upstream kernel.
2. Pin package version, repository commit, compiler version, driver, GPU, dtype,
   and determinism flags in result metadata.
3. Do not compare different semantics: causal alignment, GQA broadcasting,
   token dropping, gate normalization, collision handling, and mutation timing
   must match.
4. Do not count compilation or autotuning in steady-state latency, but report it
   separately.
5. Keep unsupported hardware results as `not_applicable`, never as zero or a
   failed performance result.

## Scope control

"Each sequence mixer" means one maintained adapter per distinct semantic and
kernel family, not every published architecture. New adapters enter the catalog
only when they add a new routing, reduction, mutation, residency, or scheduling
case.
