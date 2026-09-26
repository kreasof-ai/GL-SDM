"""Upstream-kernel baselines for the benchmark sweep (ATMA-pattern comparison arm).

Trains full decoder LMs whose mixer is the pinned upstream's FAST kernel (fla chunk
mode, torch SDPA) — not the slow naive recurrent ops the KL gate uses as parity
oracles — on the same 100M-class config, so MFU / throughput / peak memory compare
against the upstream's production kernel. The surround (embeddings, norms, MLP,
lm_head, optimizers, harness loop) is identical to the URM rows; only the mixer
module differs.

Honest scope:
- Upstream rows run EAGER (compile_model=False): the fla chunk kernels fail
  torch.compile/Inductor in this environment (measured), while the URM native rows
  compile through the opaque custom-op boundary. The eager penalty on the upstream
  side is the unfused surround only — the fla mixer kernels are already fused Triton.
- mamba2's upstream (mamba_ssm) needs a compiled CUDA extension not built here, so it
  has no upstream baseline (recorded, not fabricated).
- The flops numerator reuses the URM row's mixer name so the MFU accounting is
  identical between arms.

Usage:
    PYTHONPATH=src:. python -m train.upstream --out-dir results_upstream
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "/tmp/urm-comparator-pins/fla")

from train.harness import TrainConfig, train
from train.model import MixerSpec
from train.data import data_generator, get_data


class _FlaWrap(torch.nn.Module):
    """Wrap an fla chunk-mode layer as a [B,T,C] -> [B,T,C] mixer (unwrap the tuple)."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden):
        out = self.layer(hidden)
        return out[0] if isinstance(out, tuple) else out


class _TDAUpstream(torch.nn.Module):
    """The pinned TDA Triton kernel (threshold ReLU² attention) as a [B,T,C] mixer.

    Unlike the research-code pins, tda's pin ships a fused FlashAttention-style kernel
    (fwd+bwd), so it is a legitimate fast upstream baseline, not just a parity oracle.
    """

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        sys.path.insert(0, "/tmp/urm-comparator-pins/tda")
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        from triton_threshold_attention import threshold_rela_triton
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        out = threshold_rela_triton(q, k, v, beta=1.0, relu_power=2.0)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


class _FlaOpWrap(torch.nn.Module):
    """Wrap an fla chunk-mode OP (not a layer) as a [B,T,C] mixer: own the QKV/O
    projections and the operand derivation, call the op per the pinned signature."""

    def __init__(self, model_dim, num_heads, head_dim, op_path, derive):
        super().__init__()
        module_name, fn_name = op_path.rsplit(".", 1)
        self._op = getattr(__import__(module_name, fromlist=[fn_name]), fn_name)
        self._derive = derive  # (module, hidden, q, k, v) -> extra kwargs
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.gate_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=True)
        self.gate_proj2 = torch.nn.Linear(model_dim, num_heads * head_dim, bias=True)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        extra = self._derive(self, hidden, q, k, v)
        out = self._op(q, k, v, **extra)
        if isinstance(out, tuple):
            out = out[0]
        return self.o_proj(out.reshape(B, T, H * D))


def _wall_upstream(model_dim, num_heads, head_dim, intent, target="reference"):
    """fla parallel_wall_attn: the pinned per-channel cumulative-decay law — the same
    law as our wall_attention row (verified against fla.ops.wall_attn.naive)."""
    def derive(mod, hidden, q, k, v):
        # g: per-channel log decay [B,T,H,D], pre-cumsum, contractive at init.
        B, T = hidden.shape[0], hidden.shape[1]
        g = F.logsigmoid(mod.gate_proj(hidden)).view(B, T, mod.num_heads, mod.head_dim) / 8
        return {"g": g}
    return _FlaOpWrap(model_dim, num_heads, head_dim,
                      "fla.ops.wall_attn.parallel.parallel_wall_attn", derive)


def _iplr_upstream(model_dim, num_heads, head_dim, intent, target="reference"):
    """fla iplr — the chunk kernel's backward is NotImplementedError upstream, so this
    baseline is the pinned naive recurrence (pure torch, autograd-friendly): a
    REFERENCE-IMPLEMENTATION tier baseline, labeled as such in the report."""
    class _IPLRNaive(_FlaOpWrap):
        def __init__(self, model_dim, num_heads, head_dim):
            super().__init__(model_dim, num_heads, head_dim,
                             "fla.ops.generalized_delta_rule.iplr.naive.iplr_recurrence",
                             lambda *a: {})

        def forward(self, hidden):
            B, T, _ = hidden.shape
            H, D = self.num_heads, self.head_dim
            q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
            k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
            v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
            a = torch.tanh(self.gate_proj(hidden)).view(B, T, H, D).transpose(1, 2) * 0.1
            b = torch.tanh(self.gate_proj2(hidden)).view(B, T, H, D).transpose(1, 2) * 0.1
            out = self._op(q, k, v, a, b, output_final_state=False)
            if isinstance(out, tuple):
                out = out[0]
            return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))

    return _IPLRNaive(model_dim, num_heads, head_dim)


def _dplr_upstream(model_dim, num_heads, head_dim, intent, target="reference"):
    """fla chunk_dplr_delta_rule: the decay∘low-rank composition — our dplr row's
    pinned upstream op (registry upstream = fla.ops.generalized_delta_rule.dplr)."""
    def derive(mod, hidden, q, k, v):
        B, T, H, K = k.shape
        gk = F.logsigmoid(mod.gate_proj(hidden)).view(B, T, H, K)
        a = torch.tanh(mod.gate_proj2(hidden)).view(B, T, H, K) * 0.1
        b = torch.tanh(q).view(B, T, H, K) * 0.1  # beta from the query stream
        return {"a": a, "b": b, "gk": gk}
    return _FlaOpWrap(model_dim, num_heads, head_dim,
                      "fla.ops.generalized_delta_rule.dplr.chunk.chunk_dplr_delta_rule", derive)


def _log_linear_upstream(model_dim, num_heads, head_dim, intent, target="reference"):
    """fla chunk_log_linear_attn: the dyadic-banked log-linear law — our
    log_linear_attention row's pinned family (verified vs fla .../log_linear_attn/naive).
    The kernel is single-group (k/v/g/level_scales [B,T,1,*], q multi-head shares)."""
    class _LogLinearUpstream(_FlaOpWrap):
        def forward(self, hidden):
            B, T, _ = hidden.shape
            H, D = self.num_heads, self.head_dim
            L = 4  # dyadic levels, matching the URM row's num_levels
            q = self.q_proj(hidden).view(B, T, H, D)
            k = self.k_proj(hidden).view(B, T, H, D)
            v = self.v_proj(hidden).view(B, T, H, D)
            g = F.logsigmoid(self.gate_proj(hidden)).view(B, T, H, D).mean(-1)   # [B,T,H]
            # per-head level scales [B,T,H,levels]; the pinned naive indexes
            # ceil(log2(T))+1 dyadic levels — pad the projection's L=4 with zeros
            # (zero scale = level contributes nothing beyond its mask).
            import math
            levels = int(math.ceil(math.log2(T))) + 1
            scales = hidden.new_zeros(B, T, H, levels)
            scales[..., :L] = self.gate_proj2(hidden).view(B, T, H, D)[..., :L]
            # The chunk kernel exceeds the A10G SMEM limit at head_dim=64 (122KB >
            # 101KB — hardware envelope, recorded); the baseline runs the pinned naive
            # (pure-torch, autograd-friendly) — REFERENCE-IMPLEMENTATION tier.
            from fla.ops.log_linear_attn.naive import naive_log_linear_attn
            out = naive_log_linear_attn(q, k, v, g, scales)
            return self.o_proj(out.reshape(B, T, H * D))

    return _LogLinearUpstream(model_dim, num_heads, head_dim,
                              "fla.ops.log_linear_attn.naive.naive_log_linear_attn",
                              lambda *a: {})


class _SDPA(torch.nn.Module):
    """torch SDPA causal attention as a [B,T,C] mixer (the K1 family's practical upstream)."""

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


# ---- Granularity-matched upstreams for the remodeled rows -------------------------------

class _SDPASlidingWindow(torch.nn.Module):
    """Sliding-window causal attention via SDPA with an explicit window mask — the
    samba attention branch's upstream (fla's layer requires flash-attn; bypassed to
    SDPA per the no-flash-attn constraint)."""

    def __init__(self, model_dim, num_heads, head_dim, window):
        super().__init__()
        self.num_heads, self.head_dim, self.window = num_heads, head_dim, window
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D, W = self.num_heads, self.head_dim, self.window
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        i = torch.arange(T, device=hidden.device).view(T, 1)
        j = torch.arange(T, device=hidden.device).view(1, T)
        allow = (j <= i) & (j > i - W)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


def _samba_upstream_layer(layer_idx, model_dim, num_heads, head_dim, intent,
                          target="reference"):
    """fla Samba schedule: mamba on even layers, sliding-window attention on odd.
    The mamba branch is fla.layers.mamba.Mamba (Triton backend — causal_conv1d absent,
    its own fallback); the attention branch is SDPA+window (fla's layer requires
    flash-attn, bypassed per the no-FA constraint)."""
    if layer_idx % 2 == 0:
        from fla.layers.mamba import Mamba
        return _FlaWrap(Mamba(hidden_size=model_dim, layer_idx=layer_idx))
    return _SDPASlidingWindow(model_dim, num_heads, head_dim, window=512)


class _NSANaiveUpstream(torch.nn.Module):
    """fla's FAST NSA (parallel_nsa) requires flash-attn for its sliding-window
    sub-branch — bypassed per the no-FA constraint, so the production kernel is
    unavailable here. This baseline composes the pinned naive NSA oracles
    (compression + selection + torch sliding window + gated merge): a
    REFERENCE-IMPLEMENTATION tier baseline, labeled as such."""

    def __init__(self, model_dim, num_heads, head_dim, block_size=8, topk=2, window=16):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.block_size, self.topk, self.window = block_size, topk, window
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.gate_proj = torch.nn.Linear(model_dim, 3 * num_heads, bias=False)

    def forward(self, hidden):
        from fla.ops.nsa.naive import naive_nsa_compression, naive_nsa_selection
        from fla.ops.utils.pooling import mean_pooling
        B, T, _ = hidden.shape
        H, D, BS, W = self.num_heads, self.head_dim, self.block_size, self.window
        device = hidden.device
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        gates = torch.sigmoid(self.gate_proj(hidden)).view(B, T, H, 3)
        scale = D ** -0.5

        k_cmp, v_cmp = mean_pooling(k, BS), mean_pooling(v, BS)
        o_cmp, _ = naive_nsa_compression(q, k_cmp, v_cmp, BS, scale)

        # Route: current + previous block (the pinned forced blocks), matching our row.
        t_idx = torch.arange(T, device=device)
        current = t_idx // BS
        previous = torch.where(current > 0, current - 1, torch.full_like(current, -1))
        block_indices = torch.stack([current, previous], dim=-1)
        block_indices = block_indices.view(1, T, 1, 2).expand(B, T, H, 2).contiguous()
        o_slc = naive_nsa_selection(q, k, v, block_indices, BS, scale)

        i = t_idx.view(T, 1)
        j = t_idx.view(1, T)
        allow = (j <= i) & (j > i - W)
        o_swa = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allow,
        ).transpose(1, 2)

        out = (gates[..., 0:1] * o_cmp + gates[..., 1:2] * o_slc + gates[..., 2:3] * o_swa)
        return self.o_proj(out.reshape(B, T, H * D))


def _nsa_upstream(model_dim, num_heads, head_dim, intent, target="reference"):
    return _NSANaiveUpstream(model_dim, num_heads, head_dim)


class _TokenformerUpstreamBlock(torch.nn.Module):
    """The pinned megatron tokenformer block, AST-extracted by the comparator package
    (benchmarks.comparators.pattention) — the exact upstream module, block granularity."""

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        # The pinned Pattention equation, re-hosted on our projector-free block shell
        # (the megatron class needs neox_args/mpu; the comparator's adapter extracts
        # the Pattention equation — we compose it per ParallelTokenformerLayer).
        from architectures.pattention import PattentionLayer
        self.norm1 = torch.nn.RMSNorm(model_dim)
        self.norm2 = torch.nn.RMSNorm(model_dim)
        pattn = lambda: PattentionLayer(model_dim, model_dim, param_token_num=64,  # noqa: E731
                                        target="reference", intent="training")
        self.query, self.key, self.value, self.proj = (pattn() for _ in range(4))
        self.mlp = PattentionLayer(model_dim, model_dim, param_token_num=64,
                                   target="reference", intent="training")
        for m in (self.query, self.key, self.value, self.proj, self.mlp):
            torch.nn.init.normal_(m.key_param_tokens, std=0.02)
            torch.nn.init.normal_(m.value_param_tokens, std=0.02)
        self.num_heads, self.head_dim = num_heads, head_dim

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        h = self.norm1(x)
        q = self.query(h).view(B, T, H, D).transpose(1, 2)
        k = self.key(h).view(B, T, H, D).transpose(1, 2)
        v = self.value(h).view(B, T, H, D).transpose(1, 2)
        ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(ctx.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.norm2(x))


class _AttnResUpstreamDesign(torch.nn.Module):
    """fla's fused_attnres residual aggregation, same call pattern as our AttnResDesign."""

    def __init__(self, model_dim, num_sublayers):
        super().__init__()
        from fla.ops.attnres import fused_attnres  # noqa: F401 — import check
        self.num_sublayers = num_sublayers
        self.query = torch.nn.Parameter(torch.zeros(num_sublayers, 1, model_dim))
        self.rms_weight = torch.nn.Parameter(torch.ones(num_sublayers, model_dim))

    def rezero_(self):
        torch.nn.init.zeros_(self.query)

    def aggregate(self, sublayer_idx, residuals, output_rms_weight):
        from fla.ops.attnres import fused_attnres
        return fused_attnres(
            query=self.query[sublayer_idx, 0],
            residuals=residuals,
            rms_weight=self.rms_weight[sublayer_idx],
            output_rms_weight=output_rms_weight,
            rms_eps=1e-6,
        )


def _fla_builder(cls_path, extra=None):
    """Build an fla fast-kernel layer, passing only the kwargs its constructor takes.

    The fla layer classes differ (``head_dim`` vs ``feature_dim``, ``mode`` present or
    not, low-rank dims, …), so the builder inspects the signature and filters the
    candidate kwargs — a constructor mismatch records an error row rather than a crash.
    """
    module_name, class_name = cls_path.rsplit(".", 1)

    def build(model_dim, num_heads, head_dim, intent, target="reference"):
        import inspect
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        params = inspect.signature(cls.__init__).parameters
        candidates = dict(
            hidden_size=model_dim, d_model=model_dim, num_heads=num_heads,
            head_dim=head_dim, feature_dim=head_dim, mode="chunk", layer_idx=0,
        )
        if extra:
            candidates.update(extra)
        kwargs = {k: v for k, v in candidates.items() if k in params}
        return _FlaWrap(cls(**kwargs))
    return build


# The fast upstream kernels, keyed by the URM row they baseline. The mixer name in the
# MixerSpec is the URM row's name so model_flops_per_step's numerator is identical.
# Coverage: every native-tier row whose registry upstream has a FAST kernel. Excluded:
# - mamba2 — mamba_ssm's fused kernel needs a compiled CUDA extension not built here.
# - dplr — its pinned upstream is an op (fla.ops.generalized_delta_rule.dplr), not a
#   standalone layer; the GatedDeltaNet layer already represents that kernel family.
UPSTREAM_BUILDERS = {
    "dense_attention": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _SDPA(model_dim, num_heads, head_dim)
    ),
    "gla": _fla_builder("fla.layers.gla.GatedLinearAttention"),
    "gated_deltanet": _fla_builder("fla.layers.gated_deltanet.GatedDeltaNet"),
    "deltanet": _fla_builder("fla.layers.delta_net.DeltaNet"),
    "linear_attention": _fla_builder("fla.layers.linear_attn.LinearAttention"),
    "retnet": _fla_builder("fla.layers.multiscale_retention.MultiScaleRetention"),
    "simple_gla": _fla_builder("fla.layers.simple_gla.SimpleGatedLinearAttention"),
    "hgrn2": _fla_builder("fla.layers.hgrn2.HGRN2Attention"),
    "kda": _fla_builder("fla.layers.kda.KimiDeltaAttention"),
    # Extended set: the remaining native rows with a fast upstream kernel.
    "comba": _fla_builder("fla.layers.comba.Comba"),
    "gdn2": _fla_builder("fla.layers.gdn2.GatedDeltaNet2"),
    "gsa": _fla_builder("fla.layers.gsa.GatedSlotAttention"),
    "abc_gsa": _fla_builder("fla.layers.abc.ABCAttention"),
    "gated_delta_product": _fla_builder("fla.layers.gated_deltaproduct.GatedDeltaProduct"),
    "rwkv7": _fla_builder("fla.layers.rwkv7.RWKV7Attention"),
    "based_attention": _fla_builder("fla.layers.based.BasedLinearAttention"),
    "forgetting_attention": _fla_builder("fla.layers.forgetting_attn.ForgettingAttention"),
    # lightning_attention's pinned comparator IS simple_gla's kernel (same class).
    "lightning_attention": _fla_builder("fla.layers.simple_gla.SimpleGatedLinearAttention"),
    # tda's pin ships a fused Triton kernel (fwd+bwd) — a genuine fast baseline, unlike
    # the research-code pins (tucker/tpa/samba/kata/…) which are plain-torch references.
    "tda": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _TDAUpstream(model_dim, num_heads, head_dim)
    ),
    # fla-family throughput baselines for rows whose registry upstream is None (the
    # project claims no pinned KL oracle for them, but fla ships a production kernel
    # for the same architecture family — a legitimate throughput reference, clearly
    # not the KL oracle). MFU numerators match the URM row (same mixer name → same
    # flops bucket), noting the harness's 3-bucket flops accounting is coarse.
    "lightnet": _fla_builder("fla.layers.lightnet.LightNetAttention"),
    "mom": _fla_builder("fla.layers.mom.MomAttention"),
    "mla_attention": _fla_builder("fla.layers.mla.MultiheadLatentAttention"),
    "moba": _fla_builder("fla.layers.moba.MoBA"),
    "deltaformer": _fla_builder("fla.layers.deltaformer.DeltaFormerAttention"),
    "log_linear_mamba2": _fla_builder("fla.layers.log_linear_mamba2.LogLinearMamba2"),
    "bit_attention": _fla_builder("fla.layers.bitattn.BitAttention"),
    # Same-family matches verified to run (fwd+bwd) here. wall_attention/nsa moved to
    # their own entries: fla's WallAttention IS our channel-decay law (window_size is
    # decode-cache chunking, not the law) — wired below via the parallel op; nsa is the
    # full 3-branch layer, wired with the remodeled full-NSA row.
    "path_attention": _fla_builder("fla.layers.path_attn.PaTHAttention"),
    "rodimus": _fla_builder("fla.layers.rodimus.RodimusAttention"),
    "raven": _fla_builder("fla.layers.raven.Raven"),
    # Our yoco row is the YOCO self-decoder half → fla's YOCOGatedRetention.
    "yoco": _fla_builder("fla.layers.yoco.YOCOGatedRetention"),
    # Op-level fast baselines (no fla layer class; the pinned chunk/parallel OP wrapped
    # with our own projections/operand derivation — same kernel family as the row's law):
    "wall_attention": lambda: _wall_upstream,       # parallel_wall_attn (same law — verified)
    "iplr": lambda: _iplr_upstream,                  # chunk_iplr_delta_rule
    "dplr": lambda: _dplr_upstream,                  # chunk_dplr_delta_rule (registry upstream)
    "log_linear_attention": lambda: _log_linear_upstream,  # chunk_log_linear_attn
    # Granularity-matched upstreams for the remodeled rows (same granularity both sides —
    # the responsible-comparison requirement):
    "samba_attention": lambda: _samba_upstream_layer,   # fla mamba/attn interleave (schedule)
    "nsa": lambda: _nsa_upstream,                        # fla full NativeSparseAttention
    # mamba rows: mamba_ssm's CUDA extension has no prebuilt wheel for torch 2.14+cu130
    # (kernels-community has torch211/cu128 max) and source builds are excluded — the
    # baselines are fla's own Triton mamba layers (no mamba_ssm dependency). Mamba2's
    # expand=2 inner width is 2*hidden_size, so num_heads*head_dim is set to match.
    "mamba2": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _FlaWrap(__import__("fla.layers.mamba2", fromlist=["Mamba2"]).Mamba2(
            hidden_size=model_dim, num_heads=2 * model_dim // head_dim,
            head_dim=head_dim, layer_idx=0))
    ),
    # mamba1 is the reference-tier row (charter debt, not in the 50-row native sweep);
    # an upstream-only row would have no comparison partner — recorded, not run.
    # Reference-implementation baselines (pinned research code; labeled tier):
    "differential_attention": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _DifferentialUpstream(model_dim, num_heads, head_dim)
    ),
    "hopfield_association": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _HopfieldUpstream(model_dim, num_heads, head_dim)
    ),
    "longformer": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _LongformerUpstream(model_dim, num_heads, head_dim)
    ),
    "tucker_attention": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _TuckerUpstream(model_dim, num_heads, head_dim)
    ),
    "conformer_attention": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _ConformerUpstream(model_dim, num_heads, head_dim)
    ),
    "dsa": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _DSAUpstream(model_dim, num_heads, head_dim)
    ),
    "sparse_transformer": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _SparseTransformerUpstream(model_dim, num_heads, head_dim)
    ),
    "tpa_attention": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _TPAUpstream(model_dim, num_heads, head_dim)
    ),
    # kata: the pinned KataAttention (mode="chunk" → fla chunk_linear_attn) with our
    # row's group count. Production kernel (fla-composed).
    "kata": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference": _FlaWrap(
            __import__("kata.layer", fromlist=["KataAttention"]).KataAttention(
                mode="chunk", hidden_size=model_dim, num_heads=num_heads,
                spd_num_groups=4, layer_idx=0))
    ),
    # cat: the CAT structural mask law via masked SDPA (the fla modeling_cat decoder
    # needs the FlexAttention block-mask + config plumbing; the mask IS the law).
    "cat_attention": lambda: (
        lambda model_dim, num_heads, head_dim, intent, target="reference":
        _CATUpstream(model_dim, num_heads, head_dim)
    ),
}

# sdm is stateful (persistent memory bank) — the upstream MixerSpec must carry the
# stateful contract so the harness supplies the microbatch batch size and applies
# the reset/detach lifecycle.
UPSTREAM_STATEFUL = {"sdm"}

UPSTREAM_STATEFUL_BUILDERS = {
    "sdm": lambda: _SDMUpstream,
}

import sys as _sys
if "/tmp/urm-comparator-pins/kata" not in _sys.path:
    _sys.path.insert(0, "/tmp/urm-comparator-pins/kata")


class _CATUpstream(torch.nn.Module):
    """CAT (Compress And Attend) structural-mask decoder attention via masked SDPA:
    queries attend causally within their local block AND to prior compressed tokens
    at kv % block_size == 0 — the pinned fla modeling_cat mask law.
    Reference-implementation (the compressor tower is a separate model stage, not
    part of the decoder-attention row)."""

    def __init__(self, model_dim, num_heads, head_dim, block_size=16):
        super().__init__()
        self.num_heads, self.head_dim, self.block_size = num_heads, head_dim, block_size
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        i = torch.arange(T, device=hidden.device).view(T, 1)
        j = torch.arange(T, device=hidden.device).view(1, T)
        within_block = (i // self.block_size) == (j // self.block_size)
        compressed = (j % self.block_size) == 0
        causal = j <= i
        allow = (within_block | compressed) & causal
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))

# ---- Reference-implementation baselines (research-code pins; labeled tier) ---------------
# These are the upstream's shipped implementations — plain-torch research code, not
# production kernels. They count toward coverage with the report labeling the baseline
# tier (reference-implementation vs production-kernel).

class _DifferentialUpstream(torch.nn.Module):
    """The pinned Diff-Transformer MultiheadDiffAttn (AST-extracted by the comparator
    package) with its rotary pair synthesized per forward. Reference-implementation."""

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        from benchmarks.comparators.differential import _pinned_diffattn_class
        cls = _pinned_diffattn_class()
        # Pinned convention: num_heads is HALF the baseline transformer's head count.
        self._module = cls(embed_dim=model_dim, depth=0, num_heads=max(1, num_heads // 2))
        self.head_dim_half = model_dim // max(1, num_heads // 2) // 2

    def _rotary(self, T, device, dtype):
        # Pinned apply_rotary: cos/sin are [seqlen_ro, rotary_dim/2] (2-D), matching
        # the diff head_dim, same dtype as the input (the rotary is a CUDA kernel).
        d = self.head_dim_half
        theta = 1.0 / (10000 ** (torch.arange(0, d, 2, device=device).float() / d))
        idx = torch.outer(torch.arange(T, device=device).float(), theta)
        return idx.cos().to(dtype), idx.sin().to(dtype)

    def forward(self, hidden):
        B, T, C = hidden.shape
        rel_pos = self._rotary(T, hidden.device, hidden.dtype)
        return self._module(hidden, rel_pos)


class _HopfieldUpstream(torch.nn.Module):
    """The pinned hflayers Hopfield module in its self-association mode (stored
    patterns = the sequence, the pinned default; a learned static bank is a different
    hflayers API mode with equal-count constraints). Sequence-first layout per the
    pinned core. Reference-implementation."""

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        import sys
        sys.path.insert(0, "/tmp/urm-comparator-pins/hopfield")
        from hflayers import Hopfield
        inner = num_heads * head_dim
        self.state_proj = torch.nn.Linear(model_dim, inner, bias=False)
        self.o_proj = torch.nn.Linear(inner, model_dim, bias=False)
        self._hopfield = Hopfield(input_size=inner, hidden_size=inner,
                                  output_size=inner, num_heads=num_heads)

    def forward(self, hidden):
        state = self.state_proj(hidden).transpose(0, 1)  # [T, B, E] per the pinned core
        out = self._hopfield((state, state, state))
        return self.o_proj(out.transpose(0, 1))


class _LongformerUpstream(torch.nn.Module):
    """The pinned Longformer sliding-chunk attention (comparator adapter) driven as a
    trainable mixer. Reference-implementation (the TVM kernel needs Apache TVM)."""

    def __init__(self, model_dim, num_heads, head_dim, window=8):
        super().__init__()
        import sys
        if "/tmp/urm-comparator-pins/longformer" not in sys.path:
            sys.path.insert(0, "/tmp/urm-comparator-pins/longformer")
        self.num_heads, self.head_dim, self.window = num_heads, head_dim, window
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        from benchmarks.comparators.longformer import longformer_attention_adapter
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        out, _ = longformer_attention_adapter(q, k, v, self.window)
        return self.o_proj(out.reshape(B, T, H * D))


class _TuckerUpstream(torch.nn.Module):
    """The pinned ViTTuckerAttention equation transcribed to torch. The pin's fused
    FlashAttentionTucker kernel is H100-targeted — it requires 294KB SMEM at the
    benchmark config, over the A10G's 101KB hardware limit (recorded, not fabricated)
    — so the baseline runs the pinned equation (einsum foldings + per-head
    softmax(Q̃·B_pre[h]·K̃ᵀ/√R)·Ṽ) in torch. REFERENCE-IMPLEMENTATION tier."""

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        H = num_heads
        self.n_embd, self.n_head = model_dim, H
        self.head_dim = model_dim // H
        # Full-rank defaults (no compression), per the pinned _default_ranks.
        self.Us_pre = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(model_dim, model_dim) * 0.02),
            torch.nn.Parameter(torch.randn(H, H) * 0.02),
            torch.nn.Parameter(torch.randn(model_dim, model_dim) * 0.02),
        ])
        self.Core_pre = torch.nn.Parameter(torch.randn(model_dim, H, model_dim) * 0.02)
        self.Us_post = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(model_dim, model_dim) * 0.02),   # V fold
            torch.nn.Parameter(torch.randn(H, H) * 0.02),                    # head fold
            torch.nn.Parameter(torch.randn(model_dim, model_dim) * 0.02),    # output fold
        ])
        self.Core_post = torch.nn.Parameter(torch.randn(model_dim, H, model_dim) * 0.02)

    def forward(self, x):
        B, N, _ = x.shape
        H = self.n_head
        # Pinned foldings (ViT/src/attn/tucker.py _tucker_foldings_attn_optimized).
        K_tilde = torch.einsum("BNd, dT->BNT", x, self.Us_pre[2]).contiguous()
        B_pre = torch.einsum("Hs,RsT-> HRT", self.Us_pre[1], self.Core_pre)
        Q_tilde = torch.einsum("BNd, dR->BNR", x, self.Us_pre[0]).contiguous()
        V_tilde = torch.einsum("BNd, dR->BNR", x, self.Us_post[0]).contiguous()
        R = Q_tilde.shape[-1]
        # Per-head softmax(Q̃·B_pre[h]·K̃ᵀ/√R)·Ṽ (the pinned kernel's equation);
        # the pin's kernel is non-causal (causal=False in the pinned constructor).
        q_eff = torch.einsum("bnr,hrt->bnht", Q_tilde, B_pre)
        attn = torch.einsum("bnht,bnt->bnth", q_eff, K_tilde) * (R ** -0.5)
        attn = torch.softmax(attn.float(), dim=-1).to(x.dtype)
        y = torch.einsum("bnth,bnr->bhnr", attn, V_tilde)  # [B,H,N,r_v]
        # Pinned output folding (_tucker_foldings_output): out = einsum over the
        # post factors and the post core.
        out = torch.einsum(
            "BhNr, hs, Dt, rst->BND",
            y, self.Us_post[1], self.Us_post[2], self.Core_post,
        ).contiguous()
        return out


class _ConformerUpstream(torch.nn.Module):
    """The pinned espnet RelPositionMultiHeadedAttention (AST-extracted by the
    comparator package), driven with a synthesized pos_enc. Reference-implementation."""

    def __init__(self, model_dim, num_heads, head_dim):
        super().__init__()
        from benchmarks.comparators.conformer import _pinned_rel_pos_attention_class
        cls = _pinned_rel_pos_attention_class()
        self._module = cls(num_heads=num_heads, embed_size=model_dim, dropout_rate=0.0)

    def forward(self, hidden):
        B, T, C = hidden.shape
        # The pinned forward: q/k/v [B,T,E], pos_enc [B,2T-1,E], mask [B,T2] True=mask.
        pos_enc = torch.zeros(B, 2 * T - 1, C, device=hidden.device, dtype=hidden.dtype)
        mask = torch.zeros(B, T, dtype=torch.bool, device=hidden.device)
        return self._module(hidden, hidden, hidden, pos_enc, mask)


class _DSAUpstream(torch.nn.Module):
    """DeepSeek Sparse Attention per the pinned fla dsa law (lightning indexer +
    selected attention); the fla fast kernel is indexer-coupled, so the baseline runs
    the pinned naive op for the selection with SDPA over the gathered tokens.
    Reference-implementation."""

    def __init__(self, model_dim, num_heads, head_dim, topk=8):
        super().__init__()
        self.num_heads, self.head_dim, self.topk = num_heads, head_dim, topk
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)
        # The lightning indexer: a single shared index head (MQA-style).
        self.idx_q = torch.nn.Linear(model_dim, head_dim, bias=False)
        self.idx_k = torch.nn.Linear(model_dim, head_dim, bias=False)
        self.idx_w = torch.nn.Linear(model_dim, head_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D, S = self.num_heads, self.head_dim, min(self.topk, T)
        q = self.q_proj(hidden).view(B, T, H, D)
        k = self.k_proj(hidden).view(B, T, H, D)
        v = self.v_proj(hidden).view(B, T, H, D)
        # Lightning indexer: I[t,s] = sum_j w[t,j] * relu(q_idx[t,j] * k_idx[s,j]).
        qi = self.idx_q(hidden)  # [B,T,D]
        ki = self.idx_k(hidden)  # [B,T,D]
        wi = self.idx_w(hidden)  # [B,T,D]
        scores = torch.einsum("btd,bsd->bts", qi * wi, ki).relu()
        causal = torch.ones(T, T, dtype=torch.bool, device=hidden.device).tril_()
        scores = scores.masked_fill(~causal, float("-inf"))
        idx = scores.topk(S, dim=-1).indices  # [B,T,S]
        # Selected attention over gathered tokens: k [B,T,H,D] → per-query gather over
        # the kv axis. k_hd [B,H,T,D] → [B,1,H,T,D] → [B,T,H,T,D], gather dim 3.
        k_hd = k.transpose(1, 2)  # [B,H,T,D]
        v_hd = v.transpose(1, 2)
        k_exp = k_hd.unsqueeze(1).expand(B, T, H, T, D)
        v_exp = v_hd.unsqueeze(1).expand(B, T, H, T, D)
        tok = idx.view(B, T, 1, S, 1).expand(B, T, H, S, D)
        k_g = torch.gather(k_exp, 3, tok)  # [B,T,H,S,D]
        v_g = torch.gather(v_exp, 3, tok)
        attn = torch.einsum("bthd,bthsd->bths", q, k_g) * (D ** -0.5)
        attn = torch.softmax(attn.float(), dim=-1).to(hidden.dtype)
        out = torch.einsum("bths,bthsd->bthd", attn, v_g)
        return self.o_proj(out.reshape(B, T, H * D))


class _SparseTransformerUpstream(torch.nn.Module):
    """The pinned OpenAI sparse transformer law (fixed strided+local mask) via masked
    SDPA. The pin's attention_impl is TF1/blocksparse (not runnable here — recorded);
    the mask law is the architecture. Reference-implementation."""

    def __init__(self, model_dim, num_heads, head_dim, stride=4, local_ctx=4):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.stride, self.local_ctx = stride, local_ctx
        self.q_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, num_heads * head_dim, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * head_dim, model_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)
        i = torch.arange(T, device=hidden.device).view(T, 1)
        j = torch.arange(T, device=hidden.device).view(1, T)
        # Strided + local fixed mask (the pinned Sparse Transformer pattern), causal.
        causal = j <= i
        strided = (i - j) % self.stride == 0
        local = (i - j) < self.local_ctx
        allow = causal & (strided | local)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D))


class _SDMUpstream(torch.nn.Module):
    """The pinned lingua SparseDeltaMemory (the K3 row's upstream). Stateful: the
    memory bank persists across microbatches via the harness's stateful contract
    (reset_state/detach_state). The pinned layer's own CUDA kernels run the
    sparse write/read. Production kernel (the lingua CUDA path)."""

    def __init__(self, model_dim, num_heads, head_dim, intent="training",
                 target="reference", batch_size=None):
        super().__init__()
        import sys
        sdm_root = "/tmp/urm-comparator-pins/sdm"
        if sdm_root not in sys.path:
            sys.path.insert(0, sdm_root)
        from lingua.sparse_delta_memory.layer import SparseDeltaMemory, SparseDeltaMemoryArgs
        slots = 256 * 256  # perfect square (product-key addressing), per the pin
        args = SparseDeltaMemoryArgs(
            dim=model_dim, num_writes=8, num_reads=8, slots_per_head=slots,
            num_heads=num_heads, backprop_on_memory=True, log_memory_access_stats=False,
        )
        self._layer = SparseDeltaMemory(args, layer_id=0)
        self._cache = None

    def reset_state(self):
        self._cache = None

    def detach_state(self):
        if self._cache is not None:
            self._cache = self._cache.detach() if hasattr(self._cache, "detach") else self._cache

    def forward(self, hidden):
        out, self._cache = self._layer(hidden, cache=self._cache)
        return out


class _TPAUpstream(torch.nn.Module):
    """TPA (Tensor Product Attention) — the pin ships decode-only kernels
    (flashtpa_decode_torch asserts n==1), so the training baseline generalizes the
    pinned factorized equation to n=T queries: score = (aq ∘ (bq·bk))·ak, softmax,
    then ·av, ·bv (the pinned einsum chain, causal-masked). Reference-implementation."""

    def __init__(self, model_dim, num_heads, head_dim, rank=8):
        super().__init__()
        self.num_heads, self.head_dim, self.rank = num_heads, head_dim, rank
        # The pinned factor projections: aq [H,R], bq [R,D]; ak [H], av [H];
        # bk [D], bv [E=D]. All from the input hidden.
        R, D, E = rank, head_dim, head_dim
        self.aq_proj = torch.nn.Linear(model_dim, num_heads * R, bias=False)
        self.bq_proj = torch.nn.Linear(model_dim, R * D, bias=False)
        self.ak_proj = torch.nn.Linear(model_dim, num_heads, bias=False)
        self.av_proj = torch.nn.Linear(model_dim, num_heads, bias=False)
        self.bk_proj = torch.nn.Linear(model_dim, D, bias=False)
        self.bv_proj = torch.nn.Linear(model_dim, E, bias=False)
        self.o_proj = torch.nn.Linear(num_heads * E, model_dim, bias=False)

    def forward(self, hidden):
        B, T, _ = hidden.shape
        H, R, D, E = self.num_heads, self.rank, self.head_dim, self.head_dim
        aq = self.aq_proj(hidden).view(B, T, H, R)
        bq = self.bq_proj(hidden).view(B, T, R, D)
        ak = self.ak_proj(hidden)  # [B,M,H]
        av = self.av_proj(hidden)  # [B,M,H]
        bk = self.bk_proj(hidden)  # [B,M,D]
        bv = self.bv_proj(hidden)  # [B,M,E]
        # The pinned factorized score path, generalized to n=T queries.
        score1 = torch.einsum("bnrd,bmd->bnmr", bq, bk)
        score2 = torch.einsum("bnhr,bnmr->bnmh", aq, score1)
        score3 = torch.einsum("bnmh,bmh->bhnm", score2, ak)
        causal = torch.ones(T, T, dtype=torch.bool, device=hidden.device).tril_()
        score3 = score3.masked_fill(~causal, float("-inf"))
        prob = F.softmax(score3 * (D ** -0.5) / R, dim=-1)
        o = torch.einsum("bhnm,bmh->bnmh", prob, av)
        o = torch.einsum("bnmh,bme->bnhe", o, bv)
        return self.o_proj(o.reshape(B, T, H * E))


# Rows whose upstream runs at a non-mixer granularity (the MixerSpec must carry it so the
# surround model builds the same structure both arms).
UPSTREAM_GRANULARITY = {
    "samba_attention": "schedule",
    "pattention": "block",
    "attnres": "residual",
}

# Block/residual upstream builders (separate signature from the mixer builders).
UPSTREAM_BLOCK_BUILDERS = {
    # The pinned tokenformer block (Pattention everywhere; SDPA for the causal mixing,
    # which in the pinned source is flash_attention — bypassed to SDPA per the
    # no-flash-attn constraint).
    "pattention": lambda: _TokenformerUpstreamBlock,
}

def _attnres_upstream(model_dim, layers, intent, target="reference"):
    return _AttnResUpstreamDesign(model_dim, 2 * layers)

UPSTREAM_RESIDUAL_BUILDERS = {
    "attnres": lambda: _attnres_upstream,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="URM upstream-kernel baseline sweep")
    p.add_argument("--out-dir", default="results_upstream")
    p.add_argument("--rows", default=None,
                   help="comma-separated subset (default: all with a fast upstream)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=9)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--microbatch-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=int, default=1500)
    p.add_argument("--subprocess", action="store_true",
                   help="run each row in its own process (CUDA memory isolation)")
    return p.parse_args()


def _run_one(row: str, args: argparse.Namespace) -> dict:
    """Train one upstream row in-process (called directly or via --subprocess child)."""
    granularity = UPSTREAM_GRANULARITY.get(row, "mixer")
    stateful = row in UPSTREAM_STATEFUL
    if granularity == "block":
        builder = UPSTREAM_BLOCK_BUILDERS[row]()
    elif granularity == "residual":
        builder = UPSTREAM_RESIDUAL_BUILDERS[row]()
    elif stateful:
        builder = UPSTREAM_STATEFUL_BUILDERS[row]()
    else:
        builder = UPSTREAM_BUILDERS[row]()
    spec = MixerSpec(
        name=row, builder=builder, upstream=None,
        has_reference_kernel=False, has_decode_kernel=False,
        tier="reference", public_path=False, granularity=granularity,
        stateful=stateful,
    )
    cfg = TrainConfig(
        mixer=row, vocab_size=50304, sequence_length=args.sequence_length,
        layers=args.layers, width=args.width, num_heads=args.num_heads,
        head_dim=args.head_dim, batch_tokens=args.microbatch_tokens,
        microbatch_tokens=args.microbatch_tokens, steps=args.steps, seed=0,
        compile_model=False,  # fla chunk kernels fail torch.compile here (measured)
    )
    data = data_generator("finewebedu10B/finewebedu_train_*.bin",
                          cfg.microbatch_tokens, cfg.sequence_length)
    return train(cfg, spec, data, device="cuda").to_dict()


def main() -> None:
    args = _parse_args()
    all_rows = sorted(set(UPSTREAM_BUILDERS) | set(UPSTREAM_BLOCK_BUILDERS)
                      | set(UPSTREAM_RESIDUAL_BUILDERS) | set(UPSTREAM_STATEFUL_BUILDERS))
    rows = ([r.strip() for r in args.rows.split(",") if r.strip()]
            if args.rows else all_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    get_data("finewebedu_train_000001.bin")
    print(f"[upstream] {len(rows)} rows -> {out_dir}", flush=True)

    # Single-row in-process mode (used by the subprocess driver).
    if os.environ.get("URM_UPSTREAM_CHILD") == "1":
        rec = _run_one(rows[0], args)
        (out_dir / f"{rows[0]}.json").write_text(json.dumps(rec, indent=2))
        return

    summary = []
    for i, row in enumerate(rows, 1):
        out_file = out_dir / f"{row}.json"
        log_file = out_dir / f"{row}.log"
        if out_file.exists():
            rec = json.loads(out_file.read_text())
            print(f"[upstream] {i}/{len(rows)} {row}: cached "
                  f"(mfu={rec.get('mfu', 0):.3f})", flush=True)
            summary.append(rec)
            continue
        t0 = time.time()
        if args.subprocess:
            cmd = [sys.executable, "-m", "train.upstream", "--rows", row,
                   "--out-dir", str(out_dir), "--steps", str(args.steps),
                   "--layers", str(args.layers), "--width", str(args.width),
                   "--num-heads", str(args.num_heads), "--head-dim", str(args.head_dim),
                   "--sequence-length", str(args.sequence_length),
                   "--microbatch-tokens", str(args.microbatch_tokens)]
            env = dict(os.environ, PYTHONPATH="src:.", URM_UPSTREAM_CHILD="1")
            try:
                with log_file.open("w") as log:
                    proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          env=env, timeout=args.timeout)
                rec = json.loads(out_file.read_text()) if proc.returncode == 0 \
                    else {"mixer": row, "error": f"exit {proc.returncode} (see {log_file})"}
            except subprocess.TimeoutExpired:
                rec = {"mixer": row, "error": f"timeout after {args.timeout}s"}
        else:
            try:
                rec = _run_one(row, args)
            except Exception as e:  # noqa: BLE001 — record and continue
                rec = {"mixer": row, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            out_file.write_text(json.dumps(rec, indent=2))
        dt = time.time() - t0
        status = (f"mfu={rec['mfu']:.3f} tok/s={rec['throughput_tokens_s']:.0f} "
                  f"mem={rec['peak_memory_gib']:.1f}GiB" if "mfu" in rec
                  else f"ERROR: {rec['error'][:80]}")
        print(f"[upstream] {i}/{len(rows)} {row}: {status} ({dt:.0f}s)", flush=True)
        summary.append(rec)

    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for r in summary if "mfu" in r)
    print(f"[upstream] done: {n_ok}/{len(summary)} rows ok", flush=True)


if __name__ == "__main__":
    main()
