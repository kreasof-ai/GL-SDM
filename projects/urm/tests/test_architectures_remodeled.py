"""Granularity-remodel gates: samba schedule, attnres residual design, tokenformer block.

These cover the HF-granularity remodeling itself (the per-layer mixer choice, the
residual-design bookkeeping, the all-Pattention block structure). The component
laws are covered by their own parity gates (mamba/K2, samba attention branch,
AttnResLayer vs fla naive_attnres, PattentionLayer vs tokenformer).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from train.model import URMDecoderLM
from train.registry import get_mixer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


def test_samba_schedule_alternates_mamba_and_attention():
    from architectures.mamba import Mamba2K2Layer
    from architectures.samba_attention import SambaAttentionLayer
    mixer = get_mixer("samba_attention")
    assert mixer.granularity == "schedule"
    even = mixer.builder(0, 128, 2, 64, "training", "reference")
    odd = mixer.builder(1, 128, 2, 64, "training", "reference")
    assert isinstance(even, Mamba2K2Layer), "even layers are the state-space branch"
    assert isinstance(odd, SambaAttentionLayer), "odd layers are the attention branch"


def test_samba_hybrid_model_trains():
    mixer = get_mixer("samba_attention")
    model = URMDecoderLM(vocab_size=256, sequence_length=32, layers=4, width=128,
                         num_heads=2, head_dim=64, mixer=mixer, target="reference",
                         batch_size=1).cuda()
    tokens = torch.randint(0, 256, (1, 32), device="cuda")
    _, loss = model(tokens, targets=tokens)
    loss.backward()
    assert torch.isfinite(loss)


def test_attnres_design_aggregation_matches_layer():
    """The design's per-sub-layer aggregation IS the verified AttnResLayer call."""
    from architectures.attnres import AttnResDesign, AttnResLayer
    torch.manual_seed(0)
    design = AttnResDesign(16, num_sublayers=4, max_sources=3)
    residuals = [torch.randn(5, 16), torch.randn(5, 16)]
    out_norm = torch.randn(16)
    with torch.no_grad():
        got = design.aggregate(2, residuals, out_norm)
        ref = AttnResLayer(2)(
            query=design.query[2, 0], residuals=residuals,
            rms_weight=design.rms_weight[2], output_rms_weight=out_norm,
        )
    assert (got - ref).abs().max().item() < 1e-6


def test_attnres_design_zero_init_survives_model_init():
    mixer = get_mixer("attnres")
    assert mixer.granularity == "residual"
    model = URMDecoderLM(vocab_size=256, sequence_length=32, layers=2, width=64,
                         num_heads=2, head_dim=32, mixer=mixer, target="reference",
                         batch_size=1)
    assert (model.residual_design.query == 0).all(), "AttnRes queries stay zero-init"


def test_attnres_model_trains_and_gradient_flows():
    mixer = get_mixer("attnres")
    model = URMDecoderLM(vocab_size=256, sequence_length=32, layers=2, width=64,
                         num_heads=2, head_dim=32, mixer=mixer, target="reference",
                         batch_size=1).cuda()
    tokens = torch.randint(0, 256, (1, 32), device="cuda")
    _, loss = model(tokens, targets=tokens)
    loss.backward()
    assert torch.isfinite(loss)
    # The residual design's parameters receive gradients through the aggregation.
    assert model.residual_design.query.grad is not None
    assert torch.isfinite(model.residual_design.query.grad).all()


def test_tokenformer_block_structure():
    """Every linear map in the block is a Pattention; the mixer is the typed K1."""
    from architectures.pattention import PattentionLayer, TokenformerBlock
    block = TokenformerBlock(128, 2, 64, ffn_slots=32, qkv_slots=16)
    for name in ("query", "key", "value", "proj", "mlp"):
        assert isinstance(getattr(block, name), PattentionLayer), name
    linears = [m for m in block.modules()
               if isinstance(m, torch.nn.Linear)]
    assert not linears, "tokenformer block has no dense Linear anywhere"


def test_tokenformer_block_matches_pinned_pattention_stages():
    """The block's forward equals the pinned stage sequence computed step by step."""
    from architectures.pattention import TokenformerBlock
    torch.manual_seed(0)
    block = TokenformerBlock(64, 2, 32, ffn_slots=16, qkv_slots=8,
                             target="reference").eval()
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        h = block.norm1(x)
        q = block.query(h)
        k = block.key(h)
        v = block.value(h)
        ctx = block._attn_plan.execute(
            query=q.view(1, 8, 2, 32), key=k.view(1, 8, 2, 32),
            value=v.view(1, 8, 2, 32))["output"].reshape(1, 8, 64)
        expected = x + block.proj(ctx)
        expected = expected + block.mlp(block.norm2(expected))
        got = block(x)
    assert (got - expected).abs().max().item() < 1e-5
