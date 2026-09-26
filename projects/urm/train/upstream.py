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
    # Same-family matches verified to run (fwd+bwd) here. Excluded despite an fla class:
    # wall_attention (fla's WallAttention is sliding-window — a DIFFERENT law from our
    # channel-decay Wall) and nsa (fla's NativeSparseAttention is the full 3-branch NSA;
    # our nsa row is the selected branch only — not apples-to-apples).
    "path_attention": _fla_builder("fla.layers.path_attn.PaTHAttention"),
    "rodimus": _fla_builder("fla.layers.rodimus.RodimusAttention"),
    "raven": _fla_builder("fla.layers.raven.Raven"),
    # Our yoco row is the YOCO self-decoder half → fla's YOCOGatedRetention.
    "yoco": _fla_builder("fla.layers.yoco.YOCOGatedRetention"),
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
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rows = ([r.strip() for r in args.rows.split(",") if r.strip()]
            if args.rows else sorted(UPSTREAM_BUILDERS))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    get_data("finewebedu_train_000001.bin")
    print(f"[upstream] {len(rows)} rows -> {out_dir}", flush=True)

    summary = []
    for i, row in enumerate(rows, 1):
        out_file = out_dir / f"{row}.json"
        if out_file.exists():
            rec = json.loads(out_file.read_text())
            print(f"[upstream] {i}/{len(rows)} {row}: cached "
                  f"(mfu={rec.get('mfu', 0):.3f})", flush=True)
            summary.append(rec)
            continue
        # The mixer name is the URM row's (identical MFU numerator); upstream=None and
        # has_reference_kernel=False so the KL gate records N/A (a self-comparison
        # against the upstream oracle would be meaningless noise here).
        spec = MixerSpec(
            name=row, builder=UPSTREAM_BUILDERS[row](), upstream=None,
            has_reference_kernel=False, has_decode_kernel=False,
            tier="reference", public_path=False,
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
        t0 = time.time()
        try:
            result = train(cfg, spec, data, device="cuda")
            rec = result.to_dict()
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
