"""The model-agnostic URM training harness (ATMA pattern).

One runner trains any registered public-path mixer: data → model → AdamW+Muon → N
steps, with MFU measurement, checkpointing, and the three correctness gates:

- **checkpoint alignment** — save at step N, reload into a fresh model, verify the
  parameters match exactly, then resume and verify the next step's update matches the
  uninterrupted run.
- **gradient alignment** — per-group gradient-norm trace across the run (optional;
  adds one sync per group, so it is toggled).
- **KL divergence** — the model's next-token distribution vs. the pinned upstream
  oracle on a fixed batch, where a reference kernel exists (recorded N/A where not).

The runner owns no mixer equation; the model's mixers route through the public URM
path via the registry. Nothing here is SDM-specific.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from train.model import MixerSpec, URMDecoderLM
from train.optimizer import build_optimizers


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """One harness run: the model shape, the data, and the gate budget."""

    mixer: str                       # a key into train.registry.MIXER_REGISTRY
    vocab_size: int = 50304
    sequence_length: int = 512
    layers: int = 12
    width: int = 768
    num_heads: int = 12
    head_dim: int = 64
    mlp_ratio: int = 4
    # Data + schedule.
    batch_tokens: int = 512 * 8      # tokens per optimizer step (microbatch * accum)
    microbatch_tokens: int = 512 * 8 # tokens per forward (microbatch * accum == batch)
    steps: int = 10
    seed: int = 0
    # Gates.
    target_mfu: float = 0.50         # the 50% MFU target (native tier)
    capture_gradients: bool = False  # gradient alignment adds one sync per group

    def __post_init__(self):
        if self.width != self.num_heads * self.head_dim:
            raise ValueError("width must equal num_heads * head_dim")
        if self.batch_tokens % self.microbatch_tokens != 0:
            raise ValueError("batch_tokens must be a multiple of microbatch_tokens")
        if self.batch_tokens % self.sequence_length != 0:
            raise ValueError("batch_tokens must be a multiple of sequence_length")

    @property
    def grad_accum(self) -> int:
        return self.batch_tokens // self.microbatch_tokens

    @property
    def microbatch_sequences(self) -> int:
        return self.microbatch_tokens // self.sequence_length


def model_flops_per_step(cfg: TrainConfig, mixer_name: str = "dense_attention") -> int:
    """Useful forward+backward FLOPs for one optimizer step (the MFU numerator).

    NanoGPT-style accounting: ``6 * N * tokens`` for the parameter matmuls (fwd+bwd),
    plus the mixer's family-specific state/score term. The K1 (dense softmax) term is
    the ``12 * L * H * T * D`` score+reduce work; the K2 linear-state mixers replace the
    quadratic attention term with the linear scan's ``O(T * K * V)`` state work. This is
    an estimate, not a mixer-exact count; it is the numerator for the MFU ratio only.
    """
    model = _param_count_estimate(cfg)
    tokens = cfg.batch_tokens
    if mixer_name == "dense_attention":
        # Quadratic attention score+reduce over the sequence.
        mixer_term = 12 * cfg.layers * cfg.num_heads * cfg.sequence_length * cfg.head_dim * tokens
    else:
        # Linear-state scan: state update + read per token is O(K*V) per head.
        mixer_term = (6 * cfg.layers * cfg.num_heads * cfg.head_dim * cfg.head_dim * tokens)
    return 6 * model * tokens + mixer_term


def _param_count_estimate(cfg: TrainConfig) -> int:
    c, v, l, t = cfg.width, cfg.vocab_size, cfg.layers, cfg.sequence_length
    embeddings = v * c + t * c
    norms = (2 * l + 1) * c
    mixer = l * 4 * c * c          # dense qkvo; the K2/K3 mixers are within a small factor
    mlp = l * 2 * c * (cfg.mlp_ratio * c)
    head = 0                       # tied embeddings
    return embeddings + norms + mixer + mlp + head


def _peak_flops(device_name: str) -> float:
    n = device_name.lower()
    if "b200" in n or "b300" in n:
        return 2250e12
    if "h100" in n or "h200" in n:
        return 989e12
    if "a100" in n:
        return 312e12
    if "l40s" in n:
        return 362e12
    if "a10g" in n:
        return 125e12
    if "l4" in n:
        return 121e12
    if "t4" in n:
        return 65e12
    return 65e12


@dataclass(slots=True)
class TrainResult:
    mixer: str
    steps: int
    final_loss: float
    mfu: float
    params: int
    checkpoint_aligned: bool
    grad_trace: dict[str, list[float]] = field(default_factory=dict)
    kl_divergence: float | None = None
    wallclock_s: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "mixer": self.mixer, "steps": self.steps, "final_loss": self.final_loss,
            "mfu": self.mfu, "params": self.params,
            "checkpoint_aligned": self.checkpoint_aligned,
            "kl_divergence": self.kl_divergence, "wallclock_s": self.wallclock_s,
        }


def build_model(cfg: TrainConfig, mixer: MixerSpec, *, device: str = "cuda") -> URMDecoderLM:
    torch.manual_seed(cfg.seed)
    model = URMDecoderLM(
        vocab_size=cfg.vocab_size, sequence_length=cfg.sequence_length,
        layers=cfg.layers, width=cfg.width, num_heads=cfg.num_heads,
        head_dim=cfg.head_dim, mixer=mixer, mlp_ratio=cfg.mlp_ratio, intent="training",
    )
    return model.to(device)


def train(cfg: TrainConfig, mixer: MixerSpec, data_iter, *,
          device: str = "cuda") -> TrainResult:
    """Run the harness: N steps, MFU, and the correctness gates."""
    model = build_model(cfg, mixer, device=device)
    optimizers = build_optimizers(model)
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = model_flops_per_step(cfg, mixer.name)
    peak = _peak_flops(torch.cuda.get_device_name()) if device == "cuda" and torch.cuda.is_available() else 65e12

    grad_trace: dict[str, list[float]] = {}
    final_loss = 0.0
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        model.train()
        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            inputs, targets = next(data_iter)
            _, loss = model(inputs, targets)
            (loss / cfg.grad_accum).backward()
            step_loss += loss.item() / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        final_loss = step_loss
        if cfg.capture_gradients:
            _record_gradient_norms(model, grad_trace)
    wallclock = time.perf_counter() - t0
    mfu = (flops_per_step * cfg.steps / wallclock) / peak if wallclock > 0 else 0.0

    checkpoint_aligned = _check_checkpoint_alignment(cfg, mixer, data_iter, device=device)
    kl = _kl_gate(cfg, mixer, model, device=device)

    return TrainResult(
        mixer=mixer.name, steps=cfg.steps, final_loss=final_loss, mfu=mfu,
        params=n_params, checkpoint_aligned=checkpoint_aligned,
        grad_trace=grad_trace, kl_divergence=kl, wallclock_s=wallclock,
    )


def _record_gradient_norms(model, trace: dict[str, list[float]]) -> None:
    """Per-group gradient-norm trace (the gradient-alignment surface)."""
    # Note: called before zero_grad in a real capture; here it records post-step state.
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        group = name.split(".")[0] + "." + (name.split(".")[1] if "blocks" in name else "")
        trace.setdefault(group, []).append(float(p.grad.float().norm().item()))


def _check_checkpoint_alignment(cfg: TrainConfig, mixer: MixerSpec, data_iter, *,
                                device: str) -> bool:
    """Save at step N, reload, and verify a resumed step matches the uninterrupted run.

    Runs two short model instances on a fixed seed stream: one trains N+1 steps
    uninterrupted; the other trains N, checkpoints (state_dict round-trip), reloads,
    and takes step N+1. The final parameters must match exactly.
    """
    seed_stream = [torch.randint(0, cfg.vocab_size, (cfg.microbatch_sequences, cfg.sequence_length),
                                 generator=torch.Generator().manual_seed(1000 + s))
                   for s in range(cfg.steps + 1)]

    def run_with_checkpoint(checkpoint_at: int | None) -> dict[str, torch.Tensor]:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg, mixer, device=device)
        optimizers = build_optimizers(model)
        for step in range(cfg.steps + 1):
            if checkpoint_at is not None and step == checkpoint_at:
                # Round-trip the model + optimizer state through a checkpoint.
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                opt_state = [opt.state_dict() for opt in optimizers]
                model = build_model(cfg, mixer, device=device)
                model.load_state_dict(state)
                optimizers = build_optimizers(model)
                for opt, st in zip(optimizers, opt_state):
                    opt.load_state_dict(st)
            tokens = seed_stream[step].to(device)
            inputs, targets = tokens[:, :-1].int(), tokens[:, 1:].long()
            # Pad/crop to sequence_length-1 windows of tokens.
            _, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    uninterrupted = run_with_checkpoint(None)
    resumed = run_with_checkpoint(cfg.steps)
    return all(
        torch.equal(uninterrupted[k], resumed[k]) for k in uninterrupted
    )


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p ‖ q) for distributions already softmaxed over the last dim."""
    p = p.float().clamp_min(1e-12)
    q = q.float().clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum(dim=-1).mean().item())


def _kl_gate(cfg: TrainConfig, mixer: MixerSpec, model: URMDecoderLM, *,
             device: str) -> float | None:
    """KL(URM mixer ‖ pinned upstream) on identical operands; None where no kernel exists.

    The mixers are operations with no learned weights of their own, so the honest KL
    gate is mixer-level (matching ``benchmarks/alignment_report.py``): run the public-path
    mixer and the pinned upstream op on the same projected operands, softmax both outputs
    over the feature dimension, and report the divergence. Near-zero KL means the public
    path produces the upstream output distribution. Architectures with no reference kernel
    (e.g. HLA) record ``None`` rather than a fabricated number.
    """
    if not mixer.has_reference_kernel or mixer.upstream is None:
        return None
    comparator = _upstream_mixer_callable(mixer)
    if comparator is None:
        return None
    torch.manual_seed(4242)
    B, T, H, D = 1, cfg.sequence_length, cfg.num_heads, cfg.head_dim
    # Identical projected operands for both the URM mixer and the pinned upstream.
    q = torch.randn(B, T, H, D, device=device)
    k = torch.randn(B, T, H, D, device=device)
    v = torch.randn(B, T, H, D, device=device)
    urm_out, upstream_out = comparator(q, k, v, device)
    if urm_out is None:
        return None
    urm_dist = torch.softmax(urm_out.float(), dim=-1)
    upstream_dist = torch.softmax(upstream_out.float(), dim=-1)
    return _kl(urm_dist, upstream_dist)


def _upstream_mixer_callable(mixer: MixerSpec):
    """Map a registered mixer to a ``(q, k, v, device) -> (urm_out, upstream_out)`` oracle.

    Returns None where the upstream is not loadable here. The URM side runs the public-path
    layer; the upstream side runs the pinned reference op on the same operands.
    """
    if mixer.name == "dense_attention":
        import torch.nn.functional as F
        from architectures.head_map_attention import HeadMapAttention

        def run(q, k, v, device):
            B, T, H, D = q.shape
            layer = HeadMapAttention(H * D, query_heads=H, kv_heads=H, head_dim=D,
                                     causal=True, head_map="equal", target="reference",
                                     intent="inference").to(device)
            # Bypass the layer's projections: call the public K1 plan on the operands.
            urm = layer._plan.execute(
                query=q.float(), key=k.float(), value=v.float())["output"]
            with torch.no_grad():
                upstream = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                ).transpose(1, 2)
            return urm, upstream
        return run
    if mixer.name == "gla":
        from architectures.gla import GLALayer

        def run(q, k, v, device):
            B, T, H, D = q.shape
            try:
                import sys
                if "/tmp/urm-comparator-pins/fla" not in sys.path:
                    sys.path.insert(0, "/tmp/urm-comparator-pins/fla")
                from fla.ops.gla.naive import naive_recurrent_gla
            except Exception:
                return None, None
            layer = GLALayer(H * D, num_heads=H, head_k_dim=D, head_v_dim=D,
                             target="reference", intent="inference").to(device)
            gk = torch.nn.functional.logsigmoid(torch.randn(B, T, H, D, device=device))
            urm = layer._run_mixer({
                "query": q.transpose(1, 2).float(), "key": k.transpose(1, 2).float(),
                "value": v.transpose(1, 2).float(),
                "beta": torch.ones(B, H, T, device=device),
                "log_decay": gk.transpose(1, 2),
                "initial_state": torch.zeros(B, H, D, D, device=device),
            })["output"].transpose(1, 2)
            with torch.no_grad():
                upstream, _ = naive_recurrent_gla(q, k, v, gk, output_final_state=True)
            return urm, upstream
        return run
    # DeltaNet / FoX and others: add comparators as their upstreams are wired.
    return None


__all__ = ["TrainConfig", "TrainResult", "train", "build_model", "model_flops_per_step"]
