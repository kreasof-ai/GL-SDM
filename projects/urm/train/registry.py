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


def _build_dense_attention(model_dim, num_heads, head_dim, intent, target="reference"):
    """K1: dense causal softmax attention (MHA) through the public K1 path."""
    from architectures.head_map_attention import HeadMapAttention

    return HeadMapAttention(
        model_dim, query_heads=num_heads, kv_heads=num_heads, head_dim=head_dim,
        causal=True, head_map="equal", target=target, intent=intent,
    )


def _build_gla(model_dim, num_heads, head_dim, intent, target="reference"):
    """K2: GLA channel-gated linear-delta state through the public K2 path."""
    from architectures.gla import GLALayer

    return GLALayer(
        model_dim, num_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim,
        target=target, intent=intent,
    )


def _build_deltanet(model_dim, num_heads, head_dim, intent, target="reference"):
    """K2: DeltaNet delta-rule state through the public K2 path."""
    from architectures.deltanet import DeltaNetLayer

    return DeltaNetLayer(
        model_dim, num_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim,
        target=target, intent=intent,
    )


def _build_fox(model_dim, num_heads, head_dim, intent, target="reference"):
    """K1: FoX forgetting attention (cumulative-gate score bias) through public K1."""
    return _FoXAdapter(model_dim, num_heads, head_dim, intent)


def _build_k2_family(layer_path):
    """Builder factory for the K2LinearStateLayer subclasses (shared constructor shape)."""
    def build(model_dim, num_heads, head_dim, intent, target="reference"):
        module_name, class_name = layer_path.rsplit(".", 1)
        module = __import__(f"architectures.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)(
            model_dim, num_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim,
            target=target, intent=intent,
        )
    return build


class _HLAAdapter(__import__("torch").nn.Module):
    """HLA's pinned Algorithm 1 has no projections and takes q/k/v [B,H,T,*]; adapt the
    [B,T,C] boundary with external q/k/v/o projections (the same pattern as FoX)."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target):
        super().__init__()
        import torch
        from architectures.hla import HLALayer

        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self._mixer = HLALayer(head_dim=head_dim, target=target, intent=intent)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        out = self._mixer(q, k, v)                      # [B,H,T,D]
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def _build_hla(model_dim, num_heads, head_dim, intent, target="reference"):
    """K2×2: HLA higher-order linear attention (two native K2 stages share one state).

    No upstream reference kernel exists (paper only) — the training gate is the kernel
    parity report (native vs reference tier on the two-K2 law).
    """
    return _HLAAdapter(model_dim, num_heads, head_dim, intent, target)


def _build_sdm(model_dim, num_heads, head_dim, intent, target="native", batch_size=None):
    """K3: SDM sparse delta memory through the public K3 path (native Triton).

    The memory bank is persistent external state carried across microbatches within a
    step (the harness resets it per step, detaches per microbatch — the
    benchmarks/pretraining_step.py lifecycle). ``width = heads * value_dim``;
    slots_per_partition must be a perfect square (product-key geometry).
    """
    from architectures.sdm_memory import SparseDeltaMemoryLayer

    if batch_size is None:
        raise ValueError("SDM requires the microbatch batch size at construction")
    return SparseDeltaMemoryLayer(
        width=model_dim, heads=num_heads, value_dim=head_dim,
        slots_per_partition=256, reads=8, writes=8,
        batch_size=batch_size, target=target, intent=intent,
    )


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
    "sdm": MixerSpec(
        name="sdm",
        builder=_build_sdm,
        # The pinned upstream (lingua sparse_delta_memory @ 183e7df8) exists and the
        # U3.D law was verified against it in the arch-047 work, but its router's tie
        # policy is backend-dependent torch.topk, so route identity is explicitly not
        # claimed and the KL gate cannot compare against it — recorded as
        # has_reference_kernel=False (the flag's contract: upstream is None iff no
        # comparable kernel). The training gate for SDM is the kernel parity report
        # (native vs reference tier vs the numpy oracle).
        upstream=None,
        has_reference_kernel=False,
        has_decode_kernel=True,
        stateful=True,
    ),
    # --- The native-K2 envelope family (all bind urm_native_diagonal_recurrence_v1) ---
    "gated_deltanet": MixerSpec(
        name="gated_deltanet",
        builder=_build_k2_family("gated_deltanet.GatedDeltaNetLayer"),
        upstream="fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "hgrn2": MixerSpec(
        name="hgrn2",
        builder=_build_k2_family("hgrn2.HGRN2Layer"),
        upstream="fla.ops.gla.naive.naive_recurrent_gla",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "kda": MixerSpec(
        name="kda",
        builder=_build_k2_family("kda.KDALayer"),
        upstream="fla.ops.kda.naive.naive_recurrent_kda",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "linear_attention": MixerSpec(
        name="linear_attention",
        builder=_build_k2_family("linear_attention.LinearAttentionLayer"),
        upstream="fla.ops.linear_attn.naive.naive_recurrent_linear_attn",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "retnet": MixerSpec(
        name="retnet",
        builder=_build_k2_family("retnet.RetNetLayer"),
        upstream="fla.ops.retention.naive.naive_retention",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "simple_gla": MixerSpec(
        name="simple_gla",
        builder=_build_k2_family("simple_gla.SimpleGLALayer"),
        upstream="fla.ops.simple_gla.naive.naive_recurrent_simple_gla",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "lightning_attention": MixerSpec(
        name="lightning_attention",
        builder=_build_k2_family("simple_gla.LightningAttentionLayer"),
        upstream="fla.ops.simple_gla.naive.naive_recurrent_simple_gla",
        has_reference_kernel=True,
        has_decode_kernel=True,
    ),
    "hla": MixerSpec(
        name="hla",
        builder=_build_hla,
        upstream=None,  # paper only — no reference kernel exists
        has_reference_kernel=False,
        has_decode_kernel=False,
    ),
}


def get_mixer(name: str) -> MixerSpec:
    if name not in MIXER_REGISTRY:
        legal = ", ".join(sorted(MIXER_REGISTRY))
        raise KeyError(f"unknown mixer {name!r}; registered: {legal}")
    return MIXER_REGISTRY[name]


__all__ = ["MIXER_REGISTRY", "get_mixer"]
