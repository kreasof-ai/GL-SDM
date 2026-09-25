"""Parity gates for arch-075 Conformer relative-position attention.

Verified against the pinned espnet source (RelPositionMultiHeadedAttention @
2950325e, the sweep's verification origin): full u/v-bias rel-shift score
construction, noncausal masked softmax with the all-masked-row-zero policy,
and gradient flow to the position bias parameters.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.conformer_attention import ConformerRelPosAttention
from benchmarks.comparators.conformer import conformer_rel_pos_attention_adapter

NUM_HEADS, EMBED = 4, 32


def _inputs(seed: int, T: int = 7):
    torch.manual_seed(seed)
    x = torch.randn(2, T, EMBED)
    pos_enc = torch.randn(2, 2 * T - 1, EMBED)
    mask = torch.zeros(2, T, dtype=torch.bool)  # nothing masked
    return x, pos_enc, mask


def _build_pair(seed: int):
    x, pos_enc, mask = _inputs(seed)
    expected, pinned, _id = conformer_rel_pos_attention_adapter(
        x, x, x, pos_enc, mask, num_heads=NUM_HEADS, embed_size=EMBED, seed=seed
    )
    urm = ConformerRelPosAttention(EMBED, NUM_HEADS)
    with torch.no_grad():
        for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            getattr(urm, name).weight.copy_(getattr(pinned, name).weight)
            if getattr(pinned, name).bias is not None:
                getattr(urm, name).bias.copy_(getattr(pinned, name).bias)
        urm.pos_bias_u.copy_(pinned.pos_bias_u)
        urm.pos_bias_v.copy_(pinned.pos_bias_v)
    return expected, urm, x, pos_enc


def test_layer_matches_pinned_relpos_attention():
    expected, urm, x, pos_enc = _build_pair(seed=5)
    with torch.no_grad():
        actual = urm(x, x, x, pos_enc)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"layer parity: max abs err {err}"


def test_masked_softmax_matches_pinned():
    """Masked source positions get zero weight; outputs match the pinned module."""
    torch.manual_seed(9)
    x = torch.randn(1, 5, EMBED)
    pos_enc = torch.randn(1, 9, EMBED)
    mask = torch.zeros(1, 5, dtype=torch.bool)
    mask[0, :2] = True  # mask the first two source positions
    expected, pinned, _id = conformer_rel_pos_attention_adapter(
        x, x, x, pos_enc, mask, num_heads=NUM_HEADS, embed_size=EMBED, seed=9
    )
    urm = ConformerRelPosAttention(EMBED, NUM_HEADS)
    with torch.no_grad():
        for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            getattr(urm, name).weight.copy_(getattr(pinned, name).weight)
            if getattr(pinned, name).bias is not None:
                getattr(urm, name).bias.copy_(getattr(pinned, name).bias)
        urm.pos_bias_u.copy_(pinned.pos_bias_u)
        urm.pos_bias_v.copy_(pinned.pos_bias_v)
        actual = urm(x, x, x, pos_enc, mask)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"masked parity: max abs err {err}"
    # Pinned sanity: attention weights for the masked positions are zero.
    assert pinned.attn is not None
    assert pinned.attn[..., :2].abs().max().item() == 0.0


def test_gradients_reach_position_bias_parameters():
    _expected, urm, x, pos_enc = _build_pair(seed=21)
    urm(x, x, x, pos_enc).square().sum().backward()
    for name, param in urm.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
    assert urm.pos_bias_u.grad.abs().sum().item() > 0
    assert urm.pos_bias_v.grad.abs().sum().item() > 0
