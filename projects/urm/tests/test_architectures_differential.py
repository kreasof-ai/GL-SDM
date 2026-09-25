"""Parity gates for arch-067 Differential Attention (Diff Transformer V1).

Verified against the pinned unilm Diff-Transformer source
(multihead_diffattn.py @ 50224e38, the sweep's verification origin): the
admitted base plan — two typed K1 calls (softmax over each half) + a typed
λ-weighted merge — matches the pinned module on identical parameters. The λ
frontend (λ_full = exp(λq1·λk1) − exp(λq2·λk2) + λ_init) and the subln /
(1−λ_init) post-stages are external. The V2 paired-head fusion is a later
proven rewrite, not claimed here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.differential_attention import DifferentialAttentionLayer, lambda_init_fn
from benchmarks.comparators.differential import differential_attention_adapter

EMBED, NUM_HEADS, DEPTH = 64, 4, 2
HEAD_DIM = EMBED // NUM_HEADS // 2  # pinned: embed_dim // num_heads // 2


def _rope_cache(T: int, device=None):
    # GPT-J interleaved rotary cache (pinned kernel.rotary convention).
    d = HEAD_DIM
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    t = torch.arange(T)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def _build_pair(seed: int, T: int = 6, device: str = "cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("pinned rotary kernel requires CUDA")
    torch.manual_seed(seed)
    config = dict(embed_dim=EMBED, num_heads=NUM_HEADS, num_kv_heads=None,
                  depth=DEPTH)
    x = torch.randn(2, T, EMBED, device=device)
    rel_pos = _rope_cache(T, device=device)
    expected, pinned, _id = differential_attention_adapter(x, rel_pos, seed=seed, device=device, **config)
    urm = DifferentialAttentionLayer(EMBED, NUM_HEADS, HEAD_DIM, DEPTH).to(device)
    with torch.no_grad():
        urm.q_proj.weight.copy_(pinned.q_proj.weight)
        urm.k_proj.weight.copy_(pinned.k_proj.weight)
        urm.v_proj.weight.copy_(pinned.v_proj.weight)
        urm.out_proj.weight.copy_(pinned.out_proj.weight)
        urm.lambda_q1.copy_(pinned.lambda_q1)
        urm.lambda_k1.copy_(pinned.lambda_k1)
        urm.lambda_q2.copy_(pinned.lambda_q2)
        urm.lambda_k2.copy_(pinned.lambda_k2)
        urm.subln.weight.copy_(pinned.subln.weight)
    return expected, pinned, urm, x, rel_pos


def test_lambda_init_matches_pinned():
    for depth in (0, 1, 2, 7):
        assert abs(lambda_init_fn(depth) - (0.8 - 0.6 * 2.718281828459045 ** (-0.3 * depth))) < 1e-12


def test_lambda_full_matches_pinned():
    _e, pinned, urm, _x, _r = _build_pair(seed=5, device='cuda')
    with torch.no_grad():
        lf_urm = urm._lambda_full()
        l1 = torch.exp(torch.sum(pinned.lambda_q1 * pinned.lambda_k1, dim=-1))
        l2 = torch.exp(torch.sum(pinned.lambda_q2 * pinned.lambda_k2, dim=-1))
        lf_pin = l1 - l2 + pinned.lambda_init
    assert (lf_urm - lf_pin).abs().max().item() < 1e-6


def test_layer_matches_pinned_diff_attention():
    expected, pinned, urm, x, rel_pos = _build_pair(seed=5, device='cuda')
    with torch.no_grad():
        # Both sides apply the pinned rotary (interleaved) and the causal mask;
        # the URM module's public forward takes rel_pos.
        actual = urm(x, rel_pos=rel_pos)
    err = (actual - expected).abs().max().item()
    assert err < 2e-3, f"layer parity vs pinned V1: max abs err {err}"


def test_gradients_flow_to_lambda_and_projections():
    _e, _p, urm, x, rel_pos = _build_pair(seed=11, device='cuda')
    out = urm(x, rel_pos=rel_pos)
    out.square().sum().backward()
    for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
        p = getattr(urm, name)
        assert p.grad is not None and p.grad.abs().sum().item() > 0, f"{name} has no gradient"
    assert urm.q_proj.weight.grad is not None
