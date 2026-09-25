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
                 mlp_ratio: int = 4, intent: str = "training", target: str = "reference"):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, sequence_length=sequence_length,
                           layers=layers, width=width, num_heads=num_heads,
                           head_dim=head_dim, mixer=mixer.name, target=target)
        self.mixer_spec = mixer
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(sequence_length, width)
        self.blocks = nn.ModuleList(
            DecoderBlock(width, mixer.builder(width, num_heads, head_dim, intent, target), mlp_ratio)
            for _ in range(layers)
        )
        self.norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.token.weight  # tied embeddings
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.lm_head(self._hidden(tokens))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)
            )
        return logits, loss

    def forward_distribution(self, tokens: torch.Tensor) -> torch.Tensor:
        """Softmax next-token distribution (the KL-gate surface)."""
        logits = self.lm_head(self._hidden(tokens)).float()
        return F.softmax(logits, dim=-1)


__all__ = ["MixerSpec", "URMDecoderLM", "DecoderBlock", "MLP", "RMSNorm"]
