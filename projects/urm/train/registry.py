"""The architecture→upstream map for the URM training harness.

Each registered mixer runs through the public URM path and records its pinned upstream
oracle and capability envelope honestly:

- ``upstream``: the pinned source the KL/output gate compares against, or ``None``
  where no reference kernel exists (HLA ships only a paper — no reference kernel).
- ``has_reference_kernel``: an independent equation to compare against exists.
- ``has_decode_kernel``: the upstream ships a single-token decode/inference path
  (TDA's threshold attention, for example, ships a training kernel but no separate
  decode kernel — recorded so the inference harness does not over-claim).
- ``stateful``: the mixer carries persistent state across steps (SDM); the generic
  loop resets it per sequence.

Only mixers that route through the public URM path are registered here; an
architecture whose mixer is not yet trainable through the public path stays out (the
honest scope), recorded in the per-architecture recipe, not faked here.
"""

from __future__ import annotations

from train.model import MixerSpec


def _build_dense_attention(model_dim, num_heads, head_dim, intent):
    """K1: dense causal softmax attention (MHA) through the public K1 path."""
    from architectures.head_map_attention import HeadMapAttention

    return HeadMapAttention(
        model_dim, query_heads=num_heads, kv_heads=num_heads, head_dim=head_dim,
        causal=True, head_map="equal", target="reference", intent=intent,
    )


def _build_gla(model_dim, num_heads, head_dim, intent):
    """K2: GLA channel-gated linear-delta state through the public K2 path."""
    from architectures.gla import GLALayer

    return GLALayer(
        model_dim, num_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim,
        target="reference", intent=intent,
    )


def _build_deltanet(model_dim, num_heads, head_dim, intent):
    """K2: DeltaNet delta-rule state through the public K2 path."""
    from architectures.deltanet import DeltaNetLayer

    return DeltaNetLayer(
        model_dim, num_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim,
        target="reference", intent=intent,
    )


def _build_fox(model_dim, num_heads, head_dim, intent):
    """K1: FoX forgetting attention (cumulative-gate score bias) through public K1."""
    return _FoXAdapter(model_dim, num_heads, head_dim, intent)


class _FoXAdapter(__import__("torch").nn.Module):
    """FoX's layer takes q/k/v/g directly; adapt the [B,T,C] boundary to projections."""

    def __init__(self, model_dim, num_heads, head_dim, intent):
        super().__init__()
        import torch
        from architectures.forgetting_attention import ForgettingAttentionLayer

        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.f_proj = torch.nn.Linear(model_dim, num_heads, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self._mixer = ForgettingAttentionLayer(num_heads, head_dim, intent=intent)

    def forward(self, hidden):
        import torch

        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        f = torch.nn.functional.logsigmoid(self.f_proj(hidden))  # [B,T,H]
        out = self._mixer(q, k, v, f)                            # [B,T,H,D]
        return self.o_proj(out.reshape(B, T, H * D))


# The registered trainable mixers (public-path only). Upstream oracles are the pinned
# comparator callables; ``None`` means no reference kernel exists.
MIXER_REGISTRY: dict[str, MixerSpec] = {
    "dense_attention": MixerSpec(
        name="dense_attention",
        builder=_build_dense_attention,
        upstream="torch.nn.functional.scaled_dot_product_attention",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "gla": MixerSpec(
        name="gla",
        builder=_build_gla,
        upstream="fla.ops.gla.naive.naive_recurrent_gla",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "deltanet": MixerSpec(
        name="deltanet",
        builder=_build_deltanet,
        upstream="fla.ops.delta_rule.naive.delta_rule_recurrence",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "forgetting_attention": MixerSpec(
        name="forgetting_attention",
        builder=_build_fox,
        upstream="fla.ops.forgetting_attn.naive.naive_forgetting_attn",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
}


def get_mixer(name: str) -> MixerSpec:
    if name not in MIXER_REGISTRY:
        legal = ", ".join(sorted(MIXER_REGISTRY))
        raise KeyError(f"unknown mixer {name!r}; registered: {legal}")
    return MIXER_REGISTRY[name]


__all__ = ["MIXER_REGISTRY", "get_mixer"]
