"""Model-agnostic decoder LM for the URM training harness.

A generic ``URMDecoderLM`` (token+position embedding → N decoder blocks → norm →
lm_head) whose per-block mixer is pluggable via :class:`MixerSpec`. The mixer runs
through the public URM path (a K1/K2/K3 layer module) — this module owns only the
ordinary surround (embeddings, norms, MLP, lm_head), never a mixer equation.

The model returns ``(logits, loss)``; loss is cross-entropy over the next-token
targets. ``forward_distribution`` returns the softmax distribution for the KL gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class MixerSpec:
    """How to build one block's mixer through the public URM path.

    ``builder`` maps ``(model_dim, num_heads, head_dim, intent)`` to a mixer module
    taking ``hidden [B,T,C]`` and returning ``[B,T,C]``. ``upstream`` names the pinned
    reference oracle for the KL gate (``None`` when no reference kernel exists, e.g.
    HLA); ``has_reference_kernel`` / ``has_decode_kernel`` record the upstream's
    capability envelope honestly.
    """

    name: str
    builder: object  # Callable[[int, int, int, str, str], nn.Module]  (dim, heads, head_dim, intent, target)
    upstream: str | None
    has_reference_kernel: bool
    has_decode_kernel: bool
    stateful: bool = False  # carries persistent memory across microbatches (SDM/K3)
    public_path: bool = True  # routes through a URM plan (False = external torch composition)
    tier: str = "reference"  # "native" where the envelope serves it, else "reference"
    # Architecture granularity (the HF-modeling axis — a row is faithful at its own
    # level, per the pinned upstream's modeling format):
    #   "mixer"    — builder(dim, heads, head_dim, intent, target) -> [B,T,C]->[B,T,C]
    #                module placed as one DecoderBlock's mixer (the default).
    #   "schedule" — builder(layer_idx, dim, heads, head_dim, intent, target) -> mixer;
    #                the row owns the per-layer mixer CHOICE (interleaved hybrids, e.g.
    #                Samba's mamba/attention alternation).
    #   "block"    — builder(dim, heads, head_dim, intent, target) -> a full decoder
    #                block module (x -> x, owning norms/residual/MLP), e.g. Tokenformer
    #                (parameter tokens replace the QKV projections AND the MLP).
    #   "residual" — builder(dim, layers, intent, target) -> a ResidualDesign owning the
    #                cross-block residual aggregation (AttnRes: depth-domain softmax over
    #                block summaries, not a sequence mixer); base blocks are dense
    #                attention, the row under test is the residual law itself.
    granularity: str = "mixer"


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = mlp_ratio * dim
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, mixer: nn.Module, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mixer = mixer
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class URMDecoderLM(nn.Module):
    """A pluggable-mixer decoder LM. All mixer work routes through the public path."""

    def __init__(self, *, vocab_size: int, sequence_length: int, layers: int,
                 width: int, num_heads: int, head_dim: int, mixer: MixerSpec,
                 mlp_ratio: int = 4, intent: str = "training", target: str = "reference",
                 batch_size: int | None = None):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, sequence_length=sequence_length,
                           layers=layers, width=width, num_heads=num_heads,
                           head_dim=head_dim, mixer=mixer.name, target=target)
        self.mixer_spec = mixer
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(sequence_length, width)

        def _build(layer_idx: int | None = None) -> nn.Module:
            if mixer.stateful:
                # Persistent-state mixers size their memory banks per (batch, head) at
                # construction; the harness supplies the microbatch batch size.
                if batch_size is None:
                    raise ValueError(f"stateful mixer {mixer.name!r} requires batch_size")
                return mixer.builder(width, num_heads, head_dim, intent, target, batch_size)
            if mixer.granularity == "schedule":
                return mixer.builder(layer_idx, width, num_heads, head_dim, intent, target)
            return mixer.builder(width, num_heads, head_dim, intent, target)

        self.granularity = mixer.granularity
        if self.granularity == "block":
            # The row supplies the full decoder block (it owns norms/residual/MLP).
            self.blocks = nn.ModuleList(_build() for _ in range(layers))
        elif self.granularity == "residual":
            # Standard dense-attention blocks; the row under test is the residual design
            # threaded across the block stack (see _hidden).
            from architectures.head_map_attention import HeadMapAttention
            self.blocks = nn.ModuleList(
                DecoderBlock(width, HeadMapAttention(
                    width, query_heads=num_heads, kv_heads=num_heads, head_dim=head_dim,
                    causal=True, head_map="equal", target=target, intent=intent,
                ), mlp_ratio)
                for _ in range(layers)
            )
            self.residual_design = mixer.builder(width, layers, intent, target)
        else:
            self.blocks = nn.ModuleList(
                DecoderBlock(width, _build(i), mlp_ratio)
                for i in range(layers)
            )
        self.norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight  # tied embeddings
        self.apply(self._initialize)
        if self.granularity == "residual":
            # The AttnRes query projections stay zero-init (paper §5) — re-apply after
            # the blanket normal_ init above.
            self.residual_design.rezero_()

    @staticmethod
    def _initialize(module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None]
        if self.granularity == "residual":
            return self._hidden_residual(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def _hidden_residual(self, x: torch.Tensor) -> torch.Tensor:
        """AttnRes residual design (fla SambaModel pattern, against fla/ops/attnres).

        A running ``prefix_sum`` rides on the hidden state; each sub-layer's input is
        the depth-domain AttnRes aggregation over the block summaries with the prenorm
        folded in (``output_rms_weight``). At block boundaries the prefix joins the
        summary list. No plain ``+`` residuals anywhere — the residual law IS the row
        under test.
        """
        design = self.residual_design
        summaries: list[torch.Tensor] = []
        prefix: torch.Tensor | None = x
        for i, block in enumerate(self.blocks):
            # -- attention sub-layer ------------------------------------------------
            if prefix is not None and not summaries:
                # First sub-layer ever: L=1 aggregation is the identity (p=1, mix=v0);
                # apply the prenorm directly (the Megatron-LM first-layer bypass).
                summaries = [prefix]
                h, prefix = block.norm1(prefix), None
            else:
                residuals = summaries if prefix is None else [*summaries, prefix]
                if prefix is not None:  # block boundary: the prefix becomes a summary
                    summaries, prefix = residuals, None
                h = design.aggregate(2 * i, residuals, block.norm1.weight)
            attn_out = block.mixer(h)
            prefix = attn_out if prefix is None else prefix + attn_out
            # -- MLP sub-layer ------------------------------------------------------
            residuals = summaries if prefix is None else [*summaries, prefix]
            h = design.aggregate(2 * i + 1, residuals, block.norm2.weight)
            mlp_out = block.mlp(h)
            prefix = mlp_out if prefix is None else prefix + mlp_out
        x = prefix if prefix is not None else summaries[-1]
        return self.norm(x)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.lm_head(self._hidden(tokens))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss

    def reset_state(self) -> None:
        """Zero persistent mixer state (stateful mixers only; no-op otherwise)."""
        for block in self.blocks:
            reset = getattr(block.mixer, "reset_state", None)
            if callable(reset):
                reset()

    def detach_state(self) -> None:
        """Persist the last forward's final state, detached from the autograd graph.

        Without the detach, every microbatch extends one autograd graph through the
        memory bank — unbounded graph growth and wrong gradients (the classic
        stateful-recurrence problem K1/K2 mixers don't have).
        """
        for block in self.blocks:
            detach = getattr(block.mixer, "detach_state", None)
            if callable(detach):
                detach()

    def forward_distribution(self, tokens: torch.Tensor) -> torch.Tensor:
        """Softmax next-token distribution (the KL-gate surface)."""
        logits = self.lm_head(self._hidden(tokens)).float()
        return F.softmax(logits, dim=-1)


__all__ = ["MixerSpec", "URMDecoderLM", "DecoderBlock", "MLP", "RMSNorm"]
