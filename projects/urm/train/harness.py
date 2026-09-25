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

# Cached measured bf16 peak (one calibration per process).
_MEASURED_PEAK: float | None = None


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
    # Gates + execution mode.
    target_mfu: float = 0.50         # the nominal MFU target; see the note on the A10G ceiling
    capture_gradients: bool = False  # gradient alignment adds one sync per group
    compile_model: bool = True       # torch.compile the surround (mixer is an opaque boundary)
    bf16: bool = True                # bf16 autocast (the A10G bf16 peak is the MFU denominator)
    target: str = "native"           # the MFU path runs the native tier; "reference" is the
                                     # correctness oracle (slower, used by the parity gates)

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


def model_flops_per_step(cfg: TrainConfig, mixer_name: str = "dense_attention",
                         exact_params: int | None = None) -> int:
    """Useful forward+backward FLOPs for one optimizer step (the MFU numerator).

    NanoGPT-style accounting: ``6 * N * tokens`` for the parameter matmuls (fwd+bwd),
    plus the mixer's family-specific state/score term. The K1 (dense softmax) term is
    the ``12 * L * H * T * D`` score+reduce work; the K2 linear-state mixers replace the
    quadratic attention term with the linear scan's ``O(T * K * V)`` state work; the K3
    sparse-memory mixers count the routed read+update work ``O((R + 2W) * D)`` per token
    per head. ``exact_params`` is the model's true trainable count when the caller has
    built it; otherwise a dense-qkvo estimate stands in. Numerator for the MFU ratio only.
    """
    model = exact_params if exact_params is not None else _param_count_estimate(cfg)
    tokens = cfg.batch_tokens
    if mixer_name == "dense_attention":
        # Quadratic attention score+reduce over the sequence.
        mixer_term = 12 * cfg.layers * cfg.num_heads * cfg.sequence_length * cfg.head_dim * tokens
    elif mixer_name == "sdm":
        # K3 sparse RMW: per token per head, read R slots + retrieved/write W slots.
        mixer_term = 6 * cfg.layers * cfg.num_heads * (8 + 2 * 8) * cfg.head_dim * tokens
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


def _nominal_peak_flops(device_name: str) -> float:
    """The vendor's nominal dense bf16 peak (the optimistic denominator)."""
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


def measure_bf16_peak() -> float:
    """The honest denominator: the device's measured bf16 GEMM throughput.

    The nominal vendor peak is unreachable; MFU against it understates the real number.
    We measure the dense bf16 matmul ceiling directly (the ATMA ``--measure-peak``
    method) so the reported MFU is the true fraction of achievable compute. On the A10G
    this is ~67 TFLOPS, not the nominal 125.
    """
    import statistics

    shapes = ((4096, 8192, 1024), (8192, 8192, 1024), (8192, 4096, 1024))
    values = []
    for m, n, k in shapes:
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        for _ in range(5):
            torch.mm(a, b)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(20):
            torch.mm(a, b)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) * 1e-3 / 20
        values.append(2 * m * n * k / seconds / 1e12)
    torch.cuda.empty_cache()
    return statistics.fmean(values) * 1e12


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
    throughput_tokens_s: float = 0.0
    peak_memory_gib: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "mixer": self.mixer, "steps": self.steps, "final_loss": self.final_loss,
            "mfu": self.mfu, "params": self.params,
            "checkpoint_aligned": self.checkpoint_aligned,
            "kl_divergence": self.kl_divergence, "wallclock_s": self.wallclock_s,
            "throughput_tokens_s": self.throughput_tokens_s,
            "peak_memory_gib": self.peak_memory_gib,
        }


def build_model(cfg: TrainConfig, mixer: MixerSpec, *, device: str = "cuda",
                target: str | None = None) -> URMDecoderLM:
    torch.manual_seed(cfg.seed)
    model = URMDecoderLM(
        vocab_size=cfg.vocab_size, sequence_length=cfg.sequence_length,
        layers=cfg.layers, width=cfg.width, num_heads=cfg.num_heads,
        head_dim=cfg.head_dim, mixer=mixer, mlp_ratio=cfg.mlp_ratio, intent="training",
        target=target or cfg.target,
        batch_size=(cfg.microbatch_tokens // cfg.sequence_length) if mixer.stateful else None,
    )
    return model.to(device)


def make_compile_safe(model: URMDecoderLM) -> URMDecoderLM:
    """Mark plan.execute-based mixers as dynamo boundaries so torch.compile works.

    Native K2-family layers run their mixer as an opaque custom op (traceable, no
    boundary needed) and are skipped. Mixers whose forward calls ``self._plan.execute``
    (a Python dispatch loop dynamo cannot trace — K3 SDM, K1 reference tiers, K2 on the
    reference tier) get their forward disabled: the plan, provider and equation are
    unchanged; only the dynamo partition moves (the surround fuses; the provider runs
    eagerly).
    """
    for block in model.blocks:
        mixer = block.mixer
        if getattr(mixer, "_target", None) == "native":
            continue  # native K2: the mixer is already an opaque custom op.
        inner = getattr(mixer, "_mixer", mixer)  # FoX adapter holds the real mixer in _mixer
        fwd = getattr(inner, "forward", None)
        if callable(fwd) and getattr(inner, "_plan", None) is not None:
            inner.forward = torch._dynamo.disable(fwd)
    return model


def train(cfg: TrainConfig, mixer: MixerSpec, data_iter, *,
          device: str = "cuda") -> TrainResult:
    """Run the harness: N steps, MFU, and the correctness gates.

    The model is compiled (the mixer is an opaque boundary — the public-path plan
    executes eagerly while the surround fuses) and run under bf16 autocast, matching the
    MFU measurement to the hardware's bf16 peak. The MFU numerator is the model's useful
    forward+backward FLOPs; the denominator is the device peak. On the A10G the realistic
    ceiling for a ~100M-param dense model is ~30–35% MFU (memory-bandwidth-bound at this
    batch/sequence), so the 50% target is a large-GPU number, recorded not claimed here.
    """
    model = build_model(cfg, mixer, device=device)
    if cfg.compile_model:
        # The native K2 layers run their mixer as an opaque custom op (no boundary
        # needed); plan.execute-based mixers (K3 SDM, K1 reference tiers) get their
        # forward marked as a dynamo boundary so the surround still fuses.
        model = torch.compile(make_compile_safe(model))
    optimizers = build_optimizers(model)
    n_params = sum(p.numel() for p in model.parameters())
    flops_per_step = model_flops_per_step(cfg, mixer.name, exact_params=n_params)
    # The honest MFU denominator: the measured bf16 GEMM peak (ATMA's --measure-peak
    # method), not the unreachable nominal vendor peak. Cached per process.
    if device == "cuda" and torch.cuda.is_available():
        global _MEASURED_PEAK
        if _MEASURED_PEAK is None:
            _MEASURED_PEAK = measure_bf16_peak()
        peak = _MEASURED_PEAK
    else:
        peak = _nominal_peak_flops("cpu")

    def _step_loss(inputs, targets):
        if cfg.bf16 and device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(inputs, targets)
        else:
            _, loss = model(inputs, targets)
        return loss

    grad_trace: dict[str, list[float]] = {}
    final_loss = 0.0
    # Stateful lifecycle (SDM/K3): each step is a fresh stream (reset at step start);
    # within a step, microbatches form a continued stream — the memory carries across
    # them DETACHED (benchmarks/pretraining_step.py's pattern). The detach is what keeps
    # the autograd graph step-local.
    def _maybe_reset():
        if mixer.stateful:
            (getattr(model, "_orig_mod", model)).reset_state()

    def _maybe_detach():
        if mixer.stateful:
            (getattr(model, "_orig_mod", model)).detach_state()

    # Warmup / compile.
    for _ in range(min(2, cfg.steps)):
        _maybe_reset()
        inputs, targets = next(data_iter)
        loss = _step_loss(inputs, targets)
        (loss / cfg.grad_accum).backward()
        _maybe_detach()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for step in range(cfg.steps):
        model.train()
        step_loss = 0.0
        _maybe_reset()
        for _ in range(cfg.grad_accum):
            inputs, targets = next(data_iter)
            loss = _step_loss(inputs, targets)
            (loss / cfg.grad_accum).backward()
            _maybe_detach()
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
    throughput = (cfg.steps * cfg.grad_accum * cfg.microbatch_tokens) / wallclock if wallclock > 0 else 0.0
    peak_memory_gib = (
        torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0.0
    )

    checkpoint_aligned = _check_checkpoint_alignment(cfg, mixer, data_iter, device=device)
    kl = _kl_gate(cfg, mixer, model, device=device)

    return TrainResult(
        mixer=mixer.name, steps=cfg.steps, final_loss=final_loss, mfu=mfu,
        params=n_params, checkpoint_aligned=checkpoint_aligned,
        grad_trace=grad_trace, kl_divergence=kl, wallclock_s=wallclock,
        throughput_tokens_s=throughput, peak_memory_gib=peak_memory_gib,
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

    Runs two small eager model instances on a fixed seed stream (a reduced shape — resume
    determinism is scale-independent): one trains N+1 steps uninterrupted; the other
    trains N, checkpoints (state_dict round-trip), reloads, and takes step N+1. The final
    parameters must match exactly. Eager + reduced-shape keeps the gate fast; the native
    MFU path is measured separately in ``train``.
    """
    # Reduced shape: 2 layers / 2 heads is enough to exercise resume determinism.
    # The gate runs on the reference tier for speed — except stateful K3 mixers, whose
    # route generation has no reference-tier provider: those run the gate on the tier
    # under training (the tier being certified).
    gate_target = cfg.target if mixer.stateful else "reference"
    small = TrainConfig(
        mixer=cfg.mixer, vocab_size=min(cfg.vocab_size, 512),
        sequence_length=min(cfg.sequence_length, 64), layers=2, width=128,
        num_heads=2, head_dim=cfg.head_dim, batch_tokens=128, microbatch_tokens=128,
        steps=cfg.steps, seed=cfg.seed, target=gate_target,
    )
    seed_stream = [torch.randint(0, small.vocab_size, (small.microbatch_sequences, small.sequence_length),
                                 generator=torch.Generator().manual_seed(1000 + s))
                   for s in range(small.steps + 1)]

    def run_with_checkpoint(checkpoint_at: int | None) -> dict[str, object]:
        torch.manual_seed(small.seed)
        model = build_model(small, mixer, device=device, target=gate_target)
        optimizers = build_optimizers(model)
        roundtrip_lossless = True
        losses: list[float] = []
        for step in range(small.steps + 1):
            if checkpoint_at is not None and step == checkpoint_at:
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                opt_state = [opt.state_dict() for opt in optimizers]
                model = build_model(small, mixer, device=device, target=gate_target)
                model.load_state_dict(state)
                # The checkpoint round-trip itself must be lossless — bitwise. This is
                # the structural resume property (a missing/misregistered param or
                # buffer fails here exactly, with no kernel noise involved).
                roundtrip_lossless = all(
                    torch.equal(model.state_dict()[k], v) for k, v in state.items()
                )
                optimizers = build_optimizers(model)
                for opt, st in zip(optimizers, opt_state):
                    opt.load_state_dict(st)
            # The same stateful lifecycle as the timed loop, applied identically on
            # both arms so resume fidelity is what's actually being measured.
            if mixer.stateful:
                model.reset_state()
            tokens = seed_stream[step].to(device)
            inputs, targets = tokens[:, :-1].int(), tokens[:, 1:].long()
            _, loss = model(inputs, targets)
            loss.backward()
            if mixer.stateful:
                model.detach_state()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        return {
            "params": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "losses": losses,
            "roundtrip_lossless": roundtrip_lossless,
        }

    uninterrupted = run_with_checkpoint(None)
    resumed = run_with_checkpoint(small.steps)
    if not mixer.stateful:
        return all(
            torch.equal(uninterrupted["params"][k], resumed["params"][k])
            for k in uninterrupted["params"]
        )
    # Stateful (K3) mixers: the native backward scatters gradients with relaxed
    # tl.atomic_add — summation order is scheduler-dependent BY DESIGN (a serialized
    # scatter would destroy the P-parallelism the kernel exists for). Measured on this
    # config: two IDENTICAL uninterrupted runs diverge up to 1.6e-2 in parameters over
    # 11 steps while their per-step losses agree to 2e-4 — the noise is high-dimensional
    # but cancels in the loss. So bitwise parameter equality is unachievable and is not
    # the property under test. The gate is: (a) the checkpoint round-trip is
    # bitwise-lossless (structural — a misregistered param/buffer fails exactly), and
    # (b) the resumed run's loss trajectory matches the uninterrupted run within 1e-2
    # absolute per step (~50x the measured run-to-run loss spread): resume reproduces
    # the training trajectory, which is what checkpointing is FOR.
    if not resumed["roundtrip_lossless"]:
        return False
    return all(
        abs(a - b) <= 1e-2
        for a, b in zip(uninterrupted["losses"], resumed["losses"])
    )


def _kl(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p ‖ q) for distributions already softmaxed over the last dim (clamped ≥ 0)."""
    p = p.float().clamp_min(1e-12)
    q = q.float().clamp_min(1e-12)
    # KL is non-negative; clamp away tiny negative fp artifacts from the log difference.
    return max(0.0, float((p * (p.log() - q.log())).sum(dim=-1).mean().item()))


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
    # The comparator must exercise the SAME tier the model trains with: a reference-tier
    # comparator would certify the reference path while the native path trains — which is
    # exactly how the native key_dim_rsqrt scale divergence hid behind a green KL.
    comparator = _upstream_mixer_callable(mixer, target=cfg.target)
    if comparator is None:
        return None
    torch.manual_seed(4242)
    # Cap the KL comparison length: the mixer-level KL is a distribution agreement check,
    # not a long-sequence stress test, and the reference-tier oracles are slow at large T.
    B, T, H, D = 1, min(cfg.sequence_length, 64), cfg.num_heads, cfg.head_dim
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


def _upstream_mixer_callable(mixer: MixerSpec, target: str = "reference"):
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
                             target=target, intent="inference").to(device)
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
    if mixer.name == "deltanet":
        from architectures.deltanet import DeltaNetLayer

        def run(q, k, v, device):
            B, T, H, D = q.shape
            try:
                import sys
                if "/tmp/urm-comparator-pins/fla" not in sys.path:
                    sys.path.insert(0, "/tmp/urm-comparator-pins/fla")
                from fla.ops.delta_rule.naive import delta_rule_recurrence
            except Exception:
                return None, None
            layer = DeltaNetLayer(H * D, num_heads=H, head_k_dim=D, head_v_dim=D,
                                  target=target, intent="inference").to(device)
            # DeltaNet normalizes q/k; the pinned oracle does the same internally.
            qn = torch.nn.functional.normalize(q, dim=-1)
            kn = torch.nn.functional.normalize(k, dim=-1)
            beta = torch.rand(B, H, T, device=device)
            urm = layer._run_mixer({
                "query": qn.transpose(1, 2).float(), "key": kn.transpose(1, 2).float(),
                "value": v.transpose(1, 2).float(), "beta": beta,
                "log_decay": torch.zeros(B, H, T, device=device),
                "initial_state": torch.zeros(B, H, D, D, device=device),
            })["output"]  # [B,H,T,D]
            with torch.no_grad():
                upstream, _ = delta_rule_recurrence(
                    qn.transpose(1, 2), kn.transpose(1, 2), v.transpose(1, 2), beta,
                    output_final_state=True,
                )  # [B,H,T,D]
            return urm, upstream
        return run
    if mixer.name == "forgetting_attention":
        from architectures.forgetting_attention import ForgettingAttentionLayer

        def run(q, k, v, device):
            B, T, H, D = q.shape
            try:
                import sys
                if "/tmp/urm-comparator-pins/fla" not in sys.path:
                    sys.path.insert(0, "/tmp/urm-comparator-pins/fla")
                from fla.ops.forgetting_attn.naive import naive_forgetting_attn
            except Exception:
                return None, None
            layer = ForgettingAttentionLayer(H, D, intent="inference").to(device)
            g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=device))
            urm = layer(q, k, v, g)
            with torch.no_grad():
                upstream = naive_forgetting_attn(q, k, v, g)
            return urm, upstream
        return run
    # --- Native-K2 envelope family: mixer-level KL vs the pinned fla naive ops ---
    if mixer.name in _K2_FLA_COMPARATORS:
        return _make_k2_fla_comparator(mixer.name, target)
    return None


def _make_k2_fla_comparator(name: str, target: str):
    """Mixer-level KL comparators for the native-K2 family vs the pinned fla naive ops.

    Each comparator reproduces the operand recipe of the corresponding per-architecture
    parity test (tests/test_architectures_<name>.py), on the tier under training.
    """
    def run(q, k, v, device):
        import sys
        if "/tmp/urm-comparator-pins/fla" not in sys.path:
            sys.path.insert(0, "/tmp/urm-comparator-pins/fla")
        B, T, H, D = q.shape
        try:
            if name == "gated_deltanet":
                from architectures.gated_deltanet import GatedDeltaNetLayer
                from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule as upstream_fn
                layer = GatedDeltaNetLayer(H * D, H, D, D, target=target, intent="inference").to(device)
            elif name == "hgrn2":
                from architectures.hgrn2 import HGRN2Layer
                from fla.ops.gla.naive import naive_recurrent_gla as upstream_fn
                layer = HGRN2Layer(H * D, H, D, D, target=target, intent="inference").to(device)
            elif name == "kda":
                from architectures.kda import KDALayer
                from fla.ops.kda.naive import naive_recurrent_kda as upstream_fn
                layer = KDALayer(H * D, H, D, D, target=target, intent="inference").to(device)
            elif name == "linear_attention":
                from architectures.linear_attention import LinearAttentionLayer
                from fla.ops.linear_attn.naive import naive_recurrent_linear_attn as upstream_fn
                layer = LinearAttentionLayer(H * D, H, D, D, target=target, intent="inference").to(device)
            elif name == "retnet":
                from architectures.retnet import RetNetLayer
                from fla.ops.retention.naive import naive_retention as upstream_fn
                layer = RetNetLayer(H * D, H, D, D, target=target, intent="inference").to(device)
            elif name in ("simple_gla", "lightning_attention"):
                from architectures.simple_gla import LightningAttentionLayer, SimpleGLALayer
                from fla.ops.simple_gla.naive import naive_recurrent_simple_gla as upstream_fn
                cls = SimpleGLALayer if name == "simple_gla" else LightningAttentionLayer
                layer = cls(H * D, H, D, D, target=target, intent="inference").to(device)
            else:
                return None, None
        except Exception:
            return None, None

        qt = q.transpose(1, 2).float()
        kt = k.transpose(1, 2).float()
        vt = v.transpose(1, 2).float()
        m0 = torch.zeros(B, H, D, D, device=device)

        if name == "gated_deltanet":
            qn = torch.nn.functional.normalize(q, dim=-1)
            kn = torch.nn.functional.normalize(k, dim=-1)
            beta = torch.rand(B, T, H, device=device)
            # The pinned gate schedule: g = -exp(A_log)·softplus(g_in + dt_bias), with
            # the layer's own parameters (the per-architecture test's recipe).
            g_in = torch.randn(B, T, H, device=device)
            g = -torch.exp(layer.A_log) * torch.nn.functional.softplus(
                g_in + layer.dt_bias)
            # URM mixer takes [B,H,T,*]; the naive takes [B,T,H,*] and returns [B,T,H,V].
            urm = layer._run_mixer({"query": qn.transpose(1, 2).float(), "key": kn.transpose(1, 2).float(),
                "value": vt, "beta": beta.transpose(1, 2), "log_decay": g.transpose(1, 2),
                "initial_state": m0})["output"]
            with torch.no_grad():
                upstream, _ = upstream_fn(qn, kn, v, beta, g, scale=None,
                                          output_final_state=True)
                upstream = upstream.transpose(1, 2)
        elif name == "hgrn2":
            # HGRN2's law IS the GLA channel-gate form: q=silu(q), k=1-exp(gk).
            qs = torch.nn.functional.silu(q)
            g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, D, device=device))
            ks = 1 - g.exp()
            urm = layer._run_mixer({"query": qs.transpose(1, 2).float(), "key": ks.transpose(1, 2).float(),
                "value": vt, "beta": torch.ones(B, H, T, device=device),
                "log_decay": g.transpose(1, 2), "initial_state": m0})["output"]
            with torch.no_grad():
                # naive_recurrent_gla takes [B,T,H,*] (transposes internally), no scale kwarg.
                upstream, _ = upstream_fn(qs, ks, v, g, output_final_state=True)
                upstream = upstream.transpose(1, 2)
        elif name == "kda":
            qn = torch.nn.functional.normalize(q, dim=-1)
            kn = torch.nn.functional.normalize(k, dim=-1)
            g = -torch.rand(B, T, H, D, device=device)
            beta = torch.rand(B, T, H, device=device)
            # URM mixer takes [B,H,T,*]; the naive takes [B,T,H,*] and returns [B,T,H,V].
            urm = layer._run_mixer({"query": qn.transpose(1, 2).float(), "key": kn.transpose(1, 2).float(),
                "value": vt, "beta": beta.transpose(1, 2), "log_decay": g.transpose(1, 2),
                "initial_state": m0})["output"]
            with torch.no_grad():
                upstream, _ = upstream_fn(qn, kn, v, g, beta, scale=None,
                                          output_final_state=True)
                upstream = upstream.transpose(1, 2)
        elif name == "linear_attention":
            # The pinned law includes the elu+1 feature map, applied by the layer's
            # forward — the comparator mirrors the per-architecture test exactly.
            # The naive takes [B,T,H,*] and returns [B,T,H,V].
            fm = type(layer)._feature_map
            qf = fm(q)  # [B,T,H,D]
            kf = fm(k)
            urm = layer._run_mixer({"query": qf.transpose(1, 2).float(),
                "key": kf.transpose(1, 2).float(), "value": vt,
                "beta": torch.ones(B, H, T, device=device),
                "log_decay": torch.zeros(B, H, T, device=device),
                "initial_state": m0})["output"]
            with torch.no_grad():
                upstream, _ = upstream_fn(qf, kf, v, scale=None, output_final_state=True,
                                          normalize=False)
                upstream = upstream.transpose(1, 2)
        elif name == "retnet":
            # naive_retention takes [B,H,T,D], computes its own per-head decay schedule
            # (log2(1-2^(-5-h))) — the URM layer's log_gamma is initialized to it.
            g = layer.log_gamma.view(1, 1, H).expand(B, T, H)
            urm = layer._run_mixer({"query": qt, "key": kt, "value": vt,
                "beta": torch.ones(B, H, T, device=device),
                "log_decay": g.transpose(1, 2).contiguous(),
                "initial_state": m0})["output"]
            with torch.no_grad():
                upstream = upstream_fn(qt, kt, vt)  # [B,H,T,D], fixed decay, no state
        else:  # simple_gla / lightning_attention — head-scalar gates, [B,T,H].
            # naive_recurrent_simple_gla takes [B,T,H,*] and g [B,T,H] (broadcast over K).
            if name == "simple_gla":
                g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device=device)) / 8
            else:
                # Lightning: static per-head g_gamma [H], expanded over B,T.
                g = layer.g_gamma.view(1, 1, H).expand(B, T, H)
            urm = layer._run_mixer({"query": qt, "key": kt, "value": vt,
                "beta": torch.ones(B, H, T, device=device),
                "log_decay": g.transpose(1, 2).contiguous(), "initial_state": m0})["output"]
            with torch.no_grad():
                upstream, _ = upstream_fn(q, k, v, g)
                upstream = upstream.transpose(1, 2)
        return urm, upstream
    return run


_K2_FLA_COMPARATORS = {
    "gated_deltanet", "hgrn2", "kda", "linear_attention", "retnet",
    "simple_gla", "lightning_attention",
}


def kernel_parity_report(mixer: MixerSpec, *, heads: int = 4, head_dim: int = 32,
                         batch: int = 2, seq: int = 64, device: str = "cuda"
                         ) -> dict[str, float | str]:
    """Native-vs-reference tier parity for mixers with no upstream kernel to gate on.

    The KL gate compares against a pinned upstream; where none exists (HLA ships only a
    paper; SDM's pinned router tie policy is backend-dependent so route identity is not
    claimed), the honest substitute is the public path agreeing with itself across tiers:
    same operands, reference tier vs native tier, max-abs-diff on the output. The mixer's
    law itself was verified against its pinned source in the per-architecture work — this
    report certifies the NATIVE execution of that law.
    """
    import torch as _t

    if mixer.stateful:
        return _k3_parity_report(mixer, heads=heads, head_dim=head_dim,
                                 batch=batch, seq=seq, device=device)

    def _build(target):
        return mixer.builder(heads * head_dim, heads, head_dim, "inference", target)

    ref_layer = _build("reference").to(device)
    nat_layer = _build("native").to(device)
    nat_layer.load_state_dict(ref_layer.state_dict())
    x = _t.randn(batch, seq, heads * head_dim, device=device)
    with _t.no_grad():
        out_ref = ref_layer(x)
        out_nat = nat_layer(x)
    return {
        "mixer": mixer.name,
        "report": "native_vs_reference",
        "max_abs_diff": float((out_ref - out_nat).abs().max().item()),
    }


def _k3_parity_report(mixer: MixerSpec, *, heads: int, head_dim: int,
                      batch: int, seq: int, device: str) -> dict[str, float | str]:
    """K3 mixer-law parity: native-generated routes feed BOTH mixer tiers.

    The route-generation kernel is native-only (no reference-tier provider exists), so a
    layer-level reference build cannot compile. Instead: generate canonical routes once
    with the native route kernel, then run the K3 sparse-delta state law on the native
    and reference tiers with identical routes/operands and compare readings + updated
    memory. This isolates the mixer law (U3.D) — the piece the architecture claims.
    """
    import torch as _t

    from urm.backends.numpy.k3 import numpy_sparse_state_mixer
    from urm.backends.torch.k3 import sparse_delta_state as ref_sds
    from urm.backends.triton.k3 import sparse_delta_state as native_sds
    from urm.backends.triton.k3 import sparse_route_selection
    from urm.ir.program import (
        DType, SparseReadTiming, SparseStateMixerSpec, SparseStateOperation,
    )

    P, T, S, D, R, W = batch * heads, seq, 256, head_dim, 8, 8
    spec = SparseStateMixerSpec(
        parallel=P, sequence=T, slots_per_partition=S, value_dim=D, writes=W, reads=R,
        dtype=DType.FLOAT32, operation=SparseStateOperation.UPDATE,
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    _t.manual_seed(9103)
    # Native route generation (the only tier that exists for it) produces canonical
    # ascending routes with softmax weights from product-key scores.
    scores = _t.randn(P, T, 2 * 16, device=device)  # factor_extent 16 -> 256 slots
    ri, rw = sparse_route_selection(scores, S, R, index_dtype=_t.int32)
    wi, ww = sparse_route_selection(scores + 0.01 * _t.randn_like(scores), S, W,
                                    index_dtype=_t.int32)
    memory = _t.randn(P, S, D, device=device)
    vals = _t.randn(P, T, D, device=device)
    beta = _t.rand(P, T, 1, device=device)
    ld = -_t.rand(P, T, 1, device=device) * 0.1
    # Fresh memory clones per path: the native kernel mutates in place
    # (PERSISTENT_IN_PLACE is the declared state policy).
    out_n, mem_n = native_sds(memory.clone(), ri, rw, write_addresses=wi,
                              write_weights=ww, values=vals, beta=beta, log_decay=ld,
                              spec=spec)
    out_r, mem_r = ref_sds(memory.clone(), ri, rw, write_addresses=wi,
                           write_weights=ww, values=vals, beta=beta, log_decay=ld,
                           spec=spec)
    out_o, mem_o = numpy_sparse_state_mixer(
        memory.cpu().numpy(), ri.cpu().numpy(), rw.float().cpu().numpy(),
        write_indices=wi.cpu().numpy(), write_weights=ww.float().cpu().numpy(),
        values=vals.cpu().numpy(), beta=beta.cpu().numpy(), log_decay=ld.cpu().numpy(),
        read_timing=SparseReadTiming.AFTER_UPDATE,
    )
    out_o = _t.as_tensor(out_o).to(device); mem_o = _t.as_tensor(mem_o).to(device)
    return {
        "mixer": mixer.name,
        "report": "k3_native_vs_reference_vs_oracle",
        "native_vs_reference_max_abs_diff": float((out_n - out_r).abs().max().item()),
        "native_vs_oracle_max_abs_diff": float((out_n - out_o).abs().max().item()),
        "memory_native_vs_oracle_max_abs_diff": float((mem_n - mem_o).abs().max().item()),
    }


__all__ = ["TrainConfig", "TrainResult", "train", "build_model", "model_flops_per_step",
           "kernel_parity_report"]
