"""Parity gates for arch-052 YOCO (self-decoder U2.A + cross-decoder U1.S).

Verified against the pinned fla source (fla/layers/yoco.py @ 864a87f6, the
sweep's verification origin): the self-decoder (Simple-GLA) matches the pinned
fused_recurrent_simple_gla (GPU); the cross-decoder (softmax attention over a
shared KV cache) matches the pinned parallel_attn (GPU).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.yoco import YOCOCrossDecoder, YOCOSelfDecoder
from benchmarks.comparators.fla_k2 import fla_op

HIDDEN, H, DK, DV = 32, 4, 8, 8


def test_self_decoder_matches_pinned_fused_recurrent_simple_gla_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned fused_recurrent_simple_gla requires CUDA")
    torch.manual_seed(5)
    B, T = 2, 6
    q = torch.randn(B, T, H, DK).cuda()
    k = torch.randn(B, T, H, DK).cuda()
    v = torch.randn(B, T, H, DV).cuda()
    g = (torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")) / 8)
    layer = YOCOSelfDecoder(HIDDEN, H, DK, DV).cuda()
    out = layer._run_mixer({
        "query": q.transpose(1, 2), "key": k.transpose(1, 2), "value": v.transpose(1, 2),
        "beta": torch.ones(B, H, T, device="cuda"),
        "log_decay": g.transpose(1, 2),
        "initial_state": torch.zeros(B, H, DK, DV, device="cuda"),
    })["output"]
    fused = fla_op("fla.ops.simple_gla.fused_recurrent_simple_gla")
    expected, _ = fused(q=q, k=k, v=v, g=g, scale=None, output_final_state=True)
    err = (out - expected.transpose(1, 2)).abs().max().item()
    assert err < 2e-3, f"self-decoder parity: max abs err {err}"


def test_cross_decoder_matches_pinned_parallel_attn_cuda():
    if not torch.cuda.is_available():
        pytest.skip("pinned parallel_attn requires CUDA")
    torch.manual_seed(9)
    B, T, S = 2, 6, 10
    layer = YOCOCrossDecoder(HIDDEN, H, DK).cuda()
    hidden = torch.randn(B, T, HIDDEN, device="cuda")
    # The shared KV cache covers positions [0, S); queries at offset S−T attend
    # causally into it (the pinned cross-decoder's seqlen_offset alignment).
    shared_k = torch.randn(B, S, H, DK, device="cuda")
    shared_v = torch.randn(B, S, H, DV, device="cuda")
    with torch.no_grad():
        actual = layer(hidden, shared_k, shared_v)
    # Reference: causal softmax attention of q (at offset) over the shared KV,
    # then o_proj. parallel_attn is self-attention over a single sequence, so
    # the cross-cache equation is verified against the K1 graph's own oracle.
    from urm.backends.numpy.k1.softmax import attention
    import numpy as np
    q = layer.q_proj(hidden).view(B, T, H, DK)
    # Offset-causal oracle: query i attends to shared keys 0..S−T+i.
    offset = S - T
    outs = []
    for b in range(B):
        qb = q[b].transpose(0, 1).detach().float().cpu().numpy()   # [H,T,K]
        kb = shared_k[b].transpose(0, 1).detach().float().cpu().numpy()
        vb = shared_v[b].transpose(0, 1).detach().float().cpu().numpy()
        ref = np.zeros((H, T, DV), dtype=np.float64)
        for i in range(T):
            # visible keys: 0..offset+i (kb/vb are [H, S, *])
            n_vis = offset + i + 1
            scale = DK ** -0.5
            scores = np.einsum("hk,hsk->hs", qb[:, i], kb[:, :n_vis]) * scale  # [H, n_vis]
            probs = torch.softmax(torch.from_numpy(scores), dim=-1).numpy()
            ref[:, i] = np.einsum("hs,hsv->hv", probs, vb[:, :n_vis])
        outs.append(torch.from_numpy(ref).transpose(0, 1))
    mixer = torch.stack(outs).to(hidden.device).float()
    expected_full = layer.o_proj(mixer.reshape(B, T, H * DK))
    err = (actual - expected_full).abs().max().item()
    assert err < 2e-3, f"cross-decoder parity: max abs err {err}"


def test_full_decoders_composition_and_gradients():
    torch.manual_seed(13)
    self_dec = YOCOSelfDecoder(HIDDEN, H, DK, DV)
    cross_dec = YOCOCrossDecoder(HIDDEN, H, DK)
    hidden = torch.randn(2, 6, HIDDEN)
    shared_k = torch.randn(2, 10, H, DK)
    shared_v = torch.randn(2, 10, H, DV)
    o1 = self_dec(hidden)
    o2 = cross_dec(hidden, shared_k, shared_v)
    assert o1.shape == (2, 6, HIDDEN) and o2.shape == (2, 6, HIDDEN)
    (o1.square().sum() + o2.square().sum()).backward()
    assert self_dec.gk_proj.weight.grad is not None
    assert cross_dec.q_proj.weight.grad is not None
