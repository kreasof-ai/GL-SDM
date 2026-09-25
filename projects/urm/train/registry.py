"""The architecture→upstream map for the URM training harness — the full catalog.

Every trainable module in ``architectures/`` is registered, with its tier recorded
honestly (``tier``):

- **native** — the mixer binds a native anchor through the public URM path.
- **reference** — the mixer routes through the public plan but the native tier declines
  its descriptor (dual-gate, multi-rank, left-transition, feature maps, normalized,
  dyadic-banked, elementwise, TriangularSolve) or is unqualified for training (native
  K1 backward); the verified reference tier runs.
- **external compositions** — plain-torch modules that don't route through a URM plan
  (``public_path=False``): trained as-is so the report never claims compiler coverage
  it didn't exercise.

``upstream`` names the pinned oracle the KL gate compares against (``None`` where no
comparable kernel exists); ``has_reference_kernel`` / ``has_decode_kernel`` record the
upstream's envelope honestly; ``stateful`` marks persistent memory (SDM). Exact
trainable-parameter counts are reported per config (they vary — the mixers' projection
structures differ). attnres is intentionally excluded: it is a depth-residual
combination op, not a per-block sequence mixer, and registering it would fake the
architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from train.adapters import QKVAdapter
from train.model import MixerSpec


# ---- Sequence mixers ([B,T,C] → [B,T,C]) ------------------------------------------------

def _k2(layer_path):
    """K2LinearStateLayer subclass: (hidden, num_heads, head_k_dim, head_v_dim)."""
    def build(model_dim, num_heads, head_dim, intent, target="reference"):
        module_name, class_name = layer_path.rsplit(".", 1)
        module = __import__(f"architectures.{module_name}", fromlist=[class_name])
        return getattr(module, class_name)(
            model_dim, num_heads=num_heads, head_k_dim=head_dim, head_v_dim=head_dim,
            target=target, intent=intent,
        )
    return build


def _build_dense_attention(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.head_map_attention import HeadMapAttention
    return HeadMapAttention(
        model_dim, query_heads=num_heads, kv_heads=num_heads, head_dim=head_dim,
        causal=True, head_map="equal", target=target, intent=intent,
    )


def _build_fox(model_dim, num_heads, head_dim, intent, target="reference"):
    return _FoXAdapter(model_dim, num_heads, head_dim, intent)


class _FoXAdapter(torch.nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, intent):
        super().__init__()
        from architectures.forgetting_attention import ForgettingAttentionLayer
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.f_proj = torch.nn.Linear(model_dim, num_heads, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self._mixer = ForgettingAttentionLayer(num_heads, head_dim, intent=intent)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        f = F.logsigmoid(self.f_proj(hidden))
        out = self._mixer(q, k, v, f)
        return self.o_proj(out.reshape(B, T, H * D))


class _HLAAdapter(torch.nn.Module):
    """HLA takes q/k/v [B,H,T,*] with no projections (pinned Algorithm 1); adapt."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target):
        super().__init__()
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
        out = self._mixer(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def _build_hla(model_dim, num_heads, head_dim, intent, target="reference"):
    return _HLAAdapter(model_dim, num_heads, head_dim, intent, target)


def _build_sdm(model_dim, num_heads, head_dim, intent, target="native", batch_size=None):
    from architectures.sdm_memory import SparseDeltaMemoryLayer
    if batch_size is None:
        raise ValueError("SDM requires the microbatch batch size at construction")
    return SparseDeltaMemoryLayer(
        width=model_dim, heads=num_heads, value_dim=head_dim,
        slots_per_partition=256, reads=8, writes=8,
        batch_size=batch_size, target=target, intent=intent,
    )


def _build_mamba2_k2(model_dim, num_heads, head_dim, intent, target="native"):
    from architectures.mamba import Mamba2K2Layer
    return Mamba2K2Layer(model_dim, num_heads, head_dim, 64, target=target, intent=intent)


def _build_mamba1_k2(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.mamba import Mamba1K2Layer
    return Mamba1K2Layer(model_dim, 16, target=target, intent=intent)


def _build_mamba1_external(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.mamba import Mamba1Layer
    return Mamba1Layer(model_dim, 16)


def _build_mom(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.mom import MoMLayer
    return MoMLayer(model_dim, num_heads, head_dim, head_dim, 8, 2, target=target, intent=intent)


def _build_raven(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.raven import RavenLayer
    return RavenLayer(model_dim, num_heads, head_dim, head_dim, 8, 2, target=target, intent=intent)


def _build_rodimus(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.rodimus import RodimusLayer
    return RodimusLayer(model_dim, 8, target=target, intent=intent)


def _build_mla(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.mla_attention import MLALayer
    return MLALayer(model_dim, num_heads, 16, 32, head_dim, 48, target=target, intent=intent)


def _build_samba(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.samba_attention import SambaAttentionLayer
    return SambaAttentionLayer(model_dim, num_heads, head_dim, target=target, intent=intent)


def _build_tpa(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.tpa_attention import TPAAttentionLayer
    return TPAAttentionLayer(model_dim, num_heads, head_dim, 8, 8, target=target, intent=intent)


def _build_tucker(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.tucker_attention import TuckerAttentionLayer
    return TuckerAttentionLayer(model_dim, num_heads, target=target, intent=intent)


def _build_bit_attention(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.bit_attention import BitAttentionLayer
    return BitAttentionLayer(model_dim, num_heads, target=target, intent=intent)


def _build_cat_attention(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.cat_attention import CATDecoderAttention
    return CATDecoderAttention(model_dim, num_heads, 16, target=target, intent=intent)


def _build_differential(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.differential_attention import DifferentialAttentionLayer
    return DifferentialAttentionLayer(model_dim, num_heads, head_dim, 1, target=target, intent=intent)


def _build_lightnet(model_dim, num_heads, head_dim, intent, target="native"):
    from architectures.lightnet import LightNetLayer
    return LightNetLayer(model_dim, num_heads, head_dim, head_dim, target=target, intent=intent)


def _build_h3(model_dim, num_heads, head_dim, intent, target="reference"):
    return _H3Adapter(model_dim, num_heads, head_dim)


def _build_hyena(model_dim, num_heads, head_dim, intent, target="reference"):
    from architectures.hyena_operator import HyenaOperatorLayer
    return HyenaOperatorLayer(model_dim, 512)


class _H3Adapter(torch.nn.Module):
    """H3's forward takes the S4D/shift SSM kernels as external operands (the pinned
    design); the adapter learns them as parameters (the trainable form)."""

    def __init__(self, model_dim, num_heads, head_dim, seq_len=512):
        super().__init__()
        from architectures.h3_mixer import H3MixerLayer
        self._mixer = H3MixerLayer(model_dim, head_dim)
        h = model_dim // head_dim
        self.ssm_kernel = torch.nn.Parameter(torch.randn(h, 2 * seq_len) * 0.02)
        self.ssm_k_kernel = torch.nn.Parameter(torch.randn(model_dim, 2 * seq_len) * 0.02)

    def forward(self, hidden):
        return self._mixer(hidden, self.ssm_kernel, self.ssm_k_kernel)


class _LogLinearMamba2Adapter(torch.nn.Module):
    """LogLinearMamba2Layer returns the banked mixer's [B,T,H,D] unflattened; the adapter
    adds the output projection the composition omits."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target):
        super().__init__()
        from architectures.log_linear_mamba2 import LogLinearMamba2Layer
        self.num_heads, self.head_dim = num_heads, head_dim
        self._layer = LogLinearMamba2Layer(model_dim, num_heads, head_dim, 4,
                                           target=target, intent=intent)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        out = self._layer(hidden)
        return self.o_proj(out.reshape(B, T, self.num_heads * self.head_dim))


def _build_log_linear_mamba2(model_dim, num_heads, head_dim, intent, target="reference"):
    return _LogLinearMamba2Adapter(model_dim, num_heads, head_dim, intent, target)


# ---- Operand mixers (QKVAdapter + input-derived operands) ---------------------------------

def _op(layer_path, *, layout="bhtd", extra=None, gate_out_dim=None, **ctor):
    """Wrap an operand mixer in the QKVAdapter. ``gate_out_dim`` sizes the input-derived
    gate projection: "heads" → num_heads, "heads_dim" → num_heads*head_dim."""
    def build(model_dim, num_heads, head_dim, intent, target="reference"):
        module_name, class_name = layer_path.rsplit(".", 1)
        module = __import__(f"architectures.{module_name}", fromlist=[class_name])
        cls = getattr(module, class_name)

        def factory(H, D):
            import inspect
            sig = inspect.signature(cls.__init__)
            kwargs = dict(ctor)
            for pname, val in (("num_heads", H), ("head_dim", D), ("head_k_dim", D),
                               ("head_v_dim", D)):
                if pname in sig.parameters:
                    kwargs.setdefault(pname, val)
            if "target" in sig.parameters:
                kwargs["target"] = target
            if "intent" in sig.parameters:
                kwargs["intent"] = intent
            return cls(**kwargs)

        god = None
        if gate_out_dim == "heads":
            god = num_heads
        elif gate_out_dim == "heads_dim":
            god = num_heads * head_dim
        return QKVAdapter(model_dim, num_heads, head_dim, factory, layout=layout,
                          extra=extra, gate_out_dim=god)
    return build


def _beta(adapter, hidden, q, k, v):
    return {"beta": torch.sigmoid(adapter.gate_proj(hidden)).transpose(1, 2)}


def _comba_ops(adapter, hidden, q, k, v):
    g = F.logsigmoid(adapter.gate_proj(hidden)).transpose(1, 2)      # [B,H,T]
    beta = torch.sigmoid(adapter.gate_proj2(hidden)).transpose(1, 2)
    return {"p": q, "beta": beta, "g": g}


def _iplr_ops(adapter, hidden, q, k, v):
    # alpha / low_rank_beta per-channel [B,H,T,K], input-derived, contractive at init.
    B, H, T, K = k.shape
    gate = torch.tanh(adapter.gate_proj(hidden)).view(B, T, H, K).transpose(1, 2) * 0.1
    return {"alpha": gate, "beta": gate * 0.5}


def _gdn2_ops(adapter, hidden, q, k, v):
    B, H, T, K = k.shape
    V = v.shape[-1]
    gate = adapter.gate_proj(hidden).view(B, T, H, K).transpose(1, 2)
    wr = torch.sigmoid(adapter.gate_proj2(hidden)).view(B, T, H, V).transpose(1, 2)
    return {"g": -F.softplus(gate) * 0.3,
            "erase_gate": torch.sigmoid(gate), "write_gate": wr}


def _deltaformer_ops(adapter, hidden, q, k, v):
    return {"beta": torch.sigmoid(adapter.gate_proj(hidden)).transpose(1, 2)}


def _path_ops(adapter, hidden, q, k, v):
    B, T = hidden.shape[0], hidden.shape[1]
    H = adapter.num_heads
    return {
        "w": k,  # Householder vectors share the key width
        "beta": torch.sigmoid(adapter.gate_proj(hidden)).view(B, T, H),
        "g": torch.zeros(B, T, H, device=hidden.device),
        "scale": float(adapter.head_dim) ** -0.5,
    }


def _wall_ops(adapter, hidden, q, k, v):
    # per-channel log gate [B,T,H,D], pre-cumsum, input-derived (bthd layout)
    B, T, H, D = q.shape
    return {"g": F.logsigmoid(adapter.gate_proj(hidden)).view(B, T, H, D)}


def _abc_ops(adapter, hidden, q, k, v):
    # s: slot logits [B,H,T,M], input-derived
    B, T = hidden.shape[0], hidden.shape[1]
    H = adapter.num_heads
    M = getattr(adapter._mixer, "n_slots", 8)
    return {"s": adapter.gate_proj(hidden).view(B, T, H, M).transpose(1, 2)}


def _gsa_ops(adapter, hidden, q, k, v):
    """GSA: slot logits s + per-slot log-decay g, both [B,H,T,M], input-derived."""
    ops = _abc_ops(adapter, hidden, q, k, v)
    B, T = hidden.shape[0], hidden.shape[1]
    M = ops["s"].shape[-1]
    ops["g"] = F.logsigmoid(adapter.gate_proj2(hidden)).view(B, T, adapter.num_heads, M).transpose(1, 2)
    return ops


def _log_linear_ops(adapter, hidden, q, k, v):
    # g per-head [B,T,H]; level_scales [B,T,H,L]; both input-derived
    B, T = hidden.shape[0], hidden.shape[1]
    H = adapter.num_heads
    L = getattr(adapter._mixer, "num_levels", 4)
    g = F.logsigmoid(adapter.gate_proj(hidden))[..., :H]
    scales = adapter.gate_proj2(hidden)[..., : H * L].view(B, T, H, L)
    return {"g": g, "level_scales": scales}


def _nsa_ops(adapter, hidden, q, k, v):
    B, T = hidden.shape[0], hidden.shape[1]
    H = adapter.num_heads
    BS = getattr(adapter._mixer, "block_size", 8)
    idx = torch.stack(
        [torch.tensor([0, max(0, t // BS)]) for t in range(T)]
    ).view(1, T, 1, 2).expand(B, T, H, 2).contiguous().to(hidden.device)
    return {"block_indices": idx}


def _dsa_ops(adapter, hidden, q, k, v):
    B, T = hidden.shape[0], hidden.shape[1]
    H = adapter.num_heads
    S = 8
    base = torch.arange(T, device=hidden.device).view(1, T, 1, 1)
    off = torch.arange(S, device=hidden.device).view(1, 1, 1, S)
    idx = (base - off).clamp(min=-1).expand(B, T, H, S).contiguous()
    return {"indices": idx}


def _longformer_ops(adapter, hidden, q, k, v):
    T = hidden.shape[1]
    gather = adapter._mixer.build_gather_indices(T, global_positions=[0]).to(hidden.device)
    return {"gather_indices": gather}


def _sparse_transformer_ops(adapter, hidden, q, k, v):
    T = hidden.shape[1]
    gather = adapter._mixer.build_gather_indices(T).to(hidden.device)
    return {"gather_indices": gather}


def _kata_ops(adapter, hidden, q, k, v):
    return {}


def _moba_ops(adapter, hidden, q, k, v):
    return {}


def _tda_ops(adapter, hidden, q, k, v):
    return {"query2": q, "key2": k, "lam": 0.5, "cosine": False}


def _build_abc(layer_path, gate=False):
    """ABC/GSA: slot-attention layers with (num_heads, head_k_dim, head_v_dim, n_slots)."""
    def build(model_dim, num_heads, head_dim, intent, target="reference"):
        module_name, class_name = layer_path.rsplit(".", 1)
        module = __import__(f"architectures.{module_name}", fromlist=[class_name])
        cls = getattr(module, class_name)

        def factory(H, D):
            return cls(num_heads=H, head_k_dim=D, head_v_dim=D, n_slots=8,
                       target=target, intent=intent)
        extra = _gsa_ops if gate else _abc_ops
        return QKVAdapter(model_dim, num_heads, head_dim, factory, extra=extra,
                          gate_out_dim=num_heads * 8)
    return build


class _GDPAdapter(torch.nn.Module):
    """Gated delta product: R rank-1 updates per token — k/v/beta run at T*R while q/g
    run at T. The adapter projects and repeats k/v R-fold (the pinned test's layout),
    with the gates derived from the input (deterministic)."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target, R=2):
        super().__init__()
        from architectures.gated_delta_product import GatedDeltaProductLayer
        self.num_heads, self.head_dim, self.R = num_heads, head_dim, R
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.g_proj = torch.nn.Linear(model_dim, num_heads, bias=True)
        self.b_proj = torch.nn.Linear(model_dim, num_heads * R, bias=True)
        self._mixer = GatedDeltaProductLayer(num_heads, head_dim, head_dim, R,
                                             target=target, intent=intent)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D, R = self.num_heads, self.head_dim, self.R
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = k.repeat_interleave(R, dim=2)
        v = v.repeat_interleave(R, dim=2)
        g = F.logsigmoid(self.g_proj(hidden)).transpose(1, 2)
        beta = torch.sigmoid(self.b_proj(hidden)).view(B, T, H, R).permute(0, 2, 1, 3).reshape(B, H, T * R)
        out = self._mixer(q, k, v, g, beta, float(D) ** -0.5)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def _build_gdp(model_dim, num_heads, head_dim, intent, target="reference"):
    return _GDPAdapter(model_dim, num_heads, head_dim, intent, target)


class _PattentionAdapter(torch.nn.Module):
    """Pattention attends the query sequence over a learned parameter-token bank."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target):
        super().__init__()
        from architectures.pattention import PattentionLayer
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self._mixer = PattentionLayer(head_dim, head_dim, param_token_num=8,
                                      target=target, intent=intent)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D).reshape(B, T * H, D)
        out = self._mixer(q)
        return self.o_proj(out.reshape(B, T, H * D))


def _build_pattention(model_dim, num_heads, head_dim, intent, target="reference"):
    return _PattentionAdapter(model_dim, num_heads, head_dim, intent, target)


class _HopfieldAdapter(torch.nn.Module):
    """Hopfield association: state patterns query a learned stored-pattern bank."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target):
        super().__init__()
        from architectures.hopfield_association import HopfieldAssociationLayer
        self.num_heads, self.head_dim = num_heads, head_dim
        self.state_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.stored = torch.nn.Parameter(torch.randn(16, num_heads * head_dim) * 0.02)
        self._mixer = HopfieldAssociationLayer(num_heads * head_dim, num_heads,
                                               target=target, intent=intent)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        state = self.state_proj(hidden)
        stored = self.stored.unsqueeze(0).expand(B, -1, -1)
        out = self._mixer(state, stored)
        return self.o_proj(out if isinstance(out, torch.Tensor) else out[0])


def _build_hopfield(model_dim, num_heads, head_dim, intent, target="reference"):
    return _HopfieldAdapter(model_dim, num_heads, head_dim, intent, target)


class _ConformerAdapter(torch.nn.Module):
    """Conformer rel-pos attention: q/k/v [B,T,E] + pos_enc [B,2T-1,E] (synthesized)."""

    def __init__(self, model_dim, num_heads, head_dim, intent, target):
        super().__init__()
        from architectures.conformer_attention import ConformerRelPosAttention
        self._mixer = ConformerRelPosAttention(model_dim, num_heads, target=target, intent=intent)

    def forward(self, hidden):
        B, T, C = hidden.shape
        pos_enc = torch.zeros(B, 2 * T - 1, C, device=hidden.device)
        return self._mixer(hidden, hidden, hidden, pos_enc)


def _build_conformer(model_dim, num_heads, head_dim, intent, target="reference"):
    return _ConformerAdapter(model_dim, num_heads, head_dim, intent, target)


# =====================================================================================
# The registry. Tier per mixer: "native" where the envelope serves it, "reference" else.
# =====================================================================================

MIXER_REGISTRY: dict[str, MixerSpec] = {
    # --- K2 native envelope (native + upstream KL gates) ---
    "gla": MixerSpec("gla", _k2("gla.GLALayer"),
                     "fla.ops.gla.naive.naive_recurrent_gla", True, True, tier="native"),
    "deltanet": MixerSpec("deltanet", _k2("deltanet.DeltaNetLayer"),
                          "fla.ops.delta_rule.naive.delta_rule_recurrence", True, True, tier="native"),
    "gated_deltanet": MixerSpec("gated_deltanet", _k2("gated_deltanet.GatedDeltaNetLayer"),
                                "fla.ops.gated_delta_rule.naive.naive_recurrent_gated_delta_rule", True, True, tier="native"),
    "hgrn2": MixerSpec("hgrn2", _k2("hgrn2.HGRN2Layer"),
                       "fla.ops.gla.naive.naive_recurrent_gla", True, True, tier="native"),
    "kda": MixerSpec("kda", _k2("kda.KDALayer"),
                     "fla.ops.kda.naive.naive_recurrent_kda", True, True, tier="native"),
    "linear_attention": MixerSpec("linear_attention", _k2("linear_attention.LinearAttentionLayer"),
                                  "fla.ops.linear_attn.naive.naive_recurrent_linear_attn", True, True, tier="native"),
    "retnet": MixerSpec("retnet", _k2("retnet.RetNetLayer"),
                        "fla.ops.retention.naive.naive_retention", True, True, tier="native"),
    "simple_gla": MixerSpec("simple_gla", _k2("simple_gla.SimpleGLALayer"),
                            "fla.ops.simple_gla.naive.naive_recurrent_simple_gla", True, True, tier="native"),
    "lightning_attention": MixerSpec("lightning_attention", _k2("simple_gla.LightningAttentionLayer"),
                                     "fla.ops.simple_gla.naive.naive_recurrent_simple_gla", True, True, tier="native"),
    # --- two-K2 graph (no upstream kernel: paper only) ---
    "hla": MixerSpec("hla", _build_hla, None, False, False, tier="native"),
    # --- K3 native (persistent state; pinned router tie policy not claimed) ---
    "sdm": MixerSpec("sdm", _build_sdm, None, False, True, stateful=True, tier="native"),
    # --- public-path sequence mixers, native ---
    "mamba2": MixerSpec("mamba2", _build_mamba2_k2, "mamba_ssm", True, True, tier="native"),
    "lightnet": MixerSpec("lightnet", _build_lightnet, None, False, False, tier="native"),
    "rodimus": MixerSpec("rodimus", _build_rodimus, None, False, False, tier="native"),
    "samba_attention": MixerSpec("samba_attention", _build_samba, None, False, False, tier="native"),
    "tpa_attention": MixerSpec("tpa_attention", _build_tpa, None, False, False, tier="native"),
    "yoco": MixerSpec("yoco", _k2("yoco.YOCOSelfDecoder"), None, False, False, tier="native"),
    "bit_attention": MixerSpec("bit_attention", _build_bit_attention, None, False, False, tier="native"),
    "cat_attention": MixerSpec("cat_attention", _build_cat_attention, None, False, False, tier="native"),
    # --- public-path, reference tier (native declines the descriptor) ---
    "deltaformer": MixerSpec("deltaformer", _op("deltaformer.DeltaFormerLayer", extra=_deltaformer_ops, gate_out_dim="heads"),
                             None, False, False),
    "path_attention": MixerSpec("path_attention", _op("path_attention.PaTHAttentionLayer", layout="bthd", extra=_path_ops, gate_out_dim="heads"),
                                None, False, False),
    "comba": MixerSpec("comba", _op("comba.CombaLayer", extra=_comba_ops, gate_out_dim="heads"),
                       "fla.ops.comba", True, True),
    "iplr": MixerSpec("iplr", _op("iplr.IPLRLayer", extra=_iplr_ops, gate_out_dim="heads_dim"),
                      None, False, False),
    "gated_delta_product": MixerSpec("gated_delta_product", _build_gdp,
                                     "fla.ops.gated_delta_product", True, True),
    "gdn2": MixerSpec("gdn2", _op("gdn2.GDN2Layer", extra=_gdn2_ops, gate_out_dim="heads_dim"),
                      "fla.ops.gdn2.naive.naive_recurrent_gdn2", True, True),
    "wall_attention": MixerSpec("wall_attention", _op("wall_attention.WallAttentionLayer", layout="bthd", extra=_wall_ops, gate_out_dim="heads_dim"),
                                None, False, False),
    "kata": MixerSpec("kata", _op("kata.KATALayer", extra=_kata_ops, num_groups=4),
                      None, False, False),
    "moba": MixerSpec("moba", _op("moba.MoBALayer", extra=_moba_ops, chunk_size=8, topk=2),
                      None, False, False),
    "nsa": MixerSpec("nsa", _op("nsa.NSASelectedLayer", layout="bthd", extra=_nsa_ops, block_size=8),
                     None, False, False),
    "dsa": MixerSpec("dsa", _op("dsa.DSALayer", layout="bthd", extra=_dsa_ops),
                     None, False, False),
    "longformer": MixerSpec("longformer", _op("longformer.LongformerLayer", layout="bthd", extra=_longformer_ops, window=8),
                            None, False, False),
    "sparse_transformer": MixerSpec("sparse_transformer", _op("sparse_transformer.SparseTransformerLayer", layout="bthd", extra=_sparse_transformer_ops, stride=4, local_ctx=4),
                                    None, False, False),
    "abc_gsa": MixerSpec("abc_gsa", _build_abc("abc_gsa.ABCLayer"),
                         "fla.ops.abc", True, True),
    "gsa": MixerSpec("gsa", _build_abc("abc_gsa.GSALayer", gate=True),
                     "fla.ops.gsa", True, True),
    "log_linear_attention": MixerSpec("log_linear_attention", _op("log_linear_attention.BankedLogLinearMixer", layout="bthd", extra=_log_linear_ops, num_levels=4, gate_out_dim="heads_dim"),
                                      None, False, False),
    "hopfield_association": MixerSpec("hopfield_association", _build_hopfield, None, False, False),
    "pattention": MixerSpec("pattention", _build_pattention, None, False, False),
    "conformer_attention": MixerSpec("conformer_attention", _build_conformer, None, False, False),
    "mamba1": MixerSpec("mamba1", _build_mamba1_k2, "mamba_ssm", True, True),
    "log_linear_mamba2": MixerSpec("log_linear_mamba2", _build_log_linear_mamba2, None, False, False),
    "mla_attention": MixerSpec("mla_attention", _build_mla, None, False, False),
    "tucker_attention": MixerSpec("tucker_attention", _build_tucker, None, False, False),
    "differential_attention": MixerSpec("differential_attention", _build_differential, None, False, False),
    # --- K1 reference tier (native K1 training backward unqualified — roadmap debt) ---
    "dense_attention": MixerSpec("dense_attention", _build_dense_attention,
                                 "torch.nn.functional.scaled_dot_product_attention", True, True),
    "forgetting_attention": MixerSpec("forgetting_attention", _build_fox,
                                      "fla.ops.forgetting_attn.naive.naive_forgetting_attn", True, True),
    # --- external plain-torch compositions (public_path=False) ---
    "mom": MixerSpec("mom", _build_mom, None, False, False, public_path=False),
    "raven": MixerSpec("raven", _build_raven, None, False, False, public_path=False),
    "h3_mixer": MixerSpec("h3_mixer", _build_h3, None, False, False, public_path=False),
    "hyena_operator": MixerSpec("hyena_operator", _build_hyena, None, False, False, public_path=False),
    "mamba1_external": MixerSpec("mamba1_external", _build_mamba1_external, None, False, False, public_path=False),
}


def get_mixer(name: str) -> MixerSpec:
    if name not in MIXER_REGISTRY:
        legal = ", ".join(sorted(MIXER_REGISTRY))
        raise KeyError(f"unknown mixer {name!r}; registered: {legal}")
    return MIXER_REGISTRY[name]


__all__ = ["MIXER_REGISTRY", "get_mixer"]
