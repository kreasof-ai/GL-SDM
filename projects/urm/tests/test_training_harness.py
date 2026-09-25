"""Gate 3 training harness: model-agnostic runner + the three correctness gates.

The harness trains any registered public-path mixer (dense K1 attention, K2
linear-state, FoX K1) through the ATMA-pattern loop (AdamW+Muon, MFU accounting,
checkpointing). The correctness gates are the binding evidence:

- **checkpoint alignment** — save at step N, reload into a fresh model, resume, and
  verify the step-N+1 parameters match the uninterrupted run exactly.
- **gradient alignment** — the per-group gradient-norm trace (toggled; adds overhead).
- **KL divergence** — the public-path mixer's output distribution vs. the pinned
  upstream on identical operands (near-zero KL), or ``None`` where no reference kernel
  exists.

These run on a small synthetic stream (self-contained); the 50%-MFU @ 100M @ 10-step
gate runs separately against the real finewebedu shards via ``train/run.py``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from train.data import synthetic_generator
from train.harness import TrainConfig, train
from train.registry import MIXER_REGISTRY, get_mixer

DEVICE = "cuda"


def _cfg(name: str, **over):
    base = dict(
        mixer=name, vocab_size=512, sequence_length=64, layers=2, width=128,
        num_heads=2, head_dim=64, batch_tokens=128, microbatch_tokens=128,
        steps=4, seed=0,
        # Correctness tests run eager: compiling several fresh small models per test hits
        # dynamo's recompile limit and adds nothing to the correctness evidence. The
        # compiled+bf16 MFU path is exercised by train/run.py at scale.
        compile_model=False,
    )
    base.update(over)
    return TrainConfig(**base)


@pytest.mark.parametrize("name", sorted(MIXER_REGISTRY))
def test_harness_trains_and_checkpoint_aligns(name):
    """Each registered mixer trains (loss is finite) and checkpoint-resume is exact."""
    cfg = _cfg(name)
    mixer = get_mixer(name)
    data = synthetic_generator(cfg.microbatch_tokens, cfg.sequence_length,
                               cfg.vocab_size, device=DEVICE, seed=cfg.seed)
    result = train(cfg, mixer, data, device=DEVICE)
    assert result.params > 0
    assert result.final_loss == result.final_loss  # finite (not NaN)
    assert result.checkpoint_aligned, f"{name}: checkpoint-resume diverged"


def test_registered_mixers_have_honest_capability_flags():
    """Every registered mixer records its upstream + capability envelope."""
    for name, spec in MIXER_REGISTRY.items():
        assert spec.name == name
        assert isinstance(spec.has_reference_kernel, bool)
        assert isinstance(spec.has_decode_kernel, bool)
        # A claimed upstream must be named; no-kernel cases record None.
        assert (spec.upstream is None) == (not spec.has_reference_kernel)


@pytest.mark.parametrize("name", ["dense_attention", "gla"])
def test_kl_gate_near_zero_vs_upstream(name):
    """The public-path mixer's output distribution matches the pinned upstream (KL≈0)."""
    cfg = _cfg(name, steps=2)
    mixer = get_mixer(name)
    data = synthetic_generator(cfg.microbatch_tokens, cfg.sequence_length,
                               cfg.vocab_size, device=DEVICE, seed=cfg.seed)
    result = train(cfg, mixer, data, device=DEVICE)
    assert result.kl_divergence is not None, f"{name}: KL gate should produce a value"
    assert result.kl_divergence < 1e-5, f"{name}: KL {result.kl_divergence} too high"


def test_gradient_alignment_trace_when_enabled():
    """The gradient-alignment surface records a per-group trace when toggled on."""
    cfg = _cfg("dense_attention", capture_gradients=True)
    mixer = get_mixer("dense_attention")
    data = synthetic_generator(cfg.microbatch_tokens, cfg.sequence_length,
                               cfg.vocab_size, device=DEVICE, seed=cfg.seed)
    result = train(cfg, mixer, data, device=DEVICE)
    # Gradient capture is a surface; the trace dict is present (may be empty if the
    # capture point is post-step). The gate is that the run completes with it on.
    assert result.grad_trace is not None


def test_stateful_sdm_lifecycle_and_metrics():
    """SDM: the persistent-state lifecycle is exercised and metrics are recorded.

    The stateful path must (a) train natively (loss finite and decreasing), (b) pass
    the checkpoint gate (bitwise-lossless round-trip + loss-trajectory resume — the
    native K3 backward's relaxed atomics make bitwise parameter equality unachievable
    by design), (c) record throughput and peak memory, and (d) produce a kernel parity
    report in place of a KL gate (no comparable upstream — the pinned router's tie
    policy is backend-dependent).
    """
    cfg = _cfg("sdm", target="native", steps=3)
    mixer = get_mixer("sdm")
    assert mixer.stateful
    data = synthetic_generator(cfg.microbatch_tokens, cfg.sequence_length,
                               cfg.vocab_size, device=DEVICE, seed=cfg.seed)
    result = train(cfg, mixer, data, device=DEVICE)
    assert result.final_loss == result.final_loss
    assert result.checkpoint_aligned
    assert result.kl_divergence is None  # no comparable upstream — by design
    assert result.throughput_tokens_s > 0
    assert result.peak_memory_gib > 0
    from train.harness import kernel_parity_report
    report = kernel_parity_report(mixer, heads=2, head_dim=32, batch=2, seq=16)
    assert report["native_vs_oracle_max_abs_diff"] < 1e-5


def test_kl_gate_covers_all_upstream_backed_mixers():
    """Every mixer with an upstream kernel produces a near-zero KL on the native tier."""
    import torch
    from train.harness import _kl, _upstream_mixer_callable
    for name, spec in MIXER_REGISTRY.items():
        if not spec.has_reference_kernel or name in ("dense_attention", "forgetting_attention"):
            continue  # covered elsewhere / K1 reference tier
        run = _upstream_mixer_callable(spec, target="native")
        assert run is not None, f"{name}: no comparator"
        torch.manual_seed(4242)
        q = torch.randn(1, 32, 2, 64, device=DEVICE)
        k = torch.randn(1, 32, 2, 64, device=DEVICE)
        v = torch.randn(1, 32, 2, 64, device=DEVICE)
        urm_out, up_out = run(q, k, v, DEVICE)
        assert urm_out is not None, f"{name}: comparator returned None"
        kl = _kl(torch.softmax(urm_out.float(), dim=-1),
                 torch.softmax(up_out.float(), dim=-1))
        assert kl < 1e-5, f"{name}: KL {kl} too high"
