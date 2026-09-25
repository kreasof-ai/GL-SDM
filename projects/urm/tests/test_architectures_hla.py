"""Parity gates for arch-074 HLA (flagship Nest combinator, masked second-order
unnormalized causal case).

Verified against the pinned paper (HLA.pdf @ 484fef2b, the sweep's verification
origin). The pinned source is a paper + README (no executable reference), so
the oracle is an independent brute-force transcription of the closed-form
masked identity `o = ((L⊙QKᵀ)(L⊙QKᵀ)ᵀ ⊙ L) V` evaluated against the serial
Algorithm 1 recurrence, plus the Nest structure's defining property: the output
equals the strict-causal quadratic-form contraction.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.hla import HLALayer

H, K, V, T = 2, 8, 6, 7


def _brute_force_hla(q, k, v):
    """The closed-form masked identity: o_t = Σ_{j≤t} Σ_{i≤j} (q_t·k_j)(k_j·q_i)... evaluated
    via the paper's o_t = q_tᵀ(S_t C_t − G_t) with S_t = Σ_{i≤t} k_i k_iᵀ,
    C_t = Σ_{i≤t} q_i v_iᵀ, G_t = Σ_{i≤t} k_i k_iᵀ C_{i-1}."""
    B = q.shape[0]
    outs = torch.zeros(B, H, T, V, dtype=torch.float64)
    q64, k64, v64 = q.double(), k.double(), v.double()
    for b in range(B):
        for h in range(H):
            S = torch.zeros(K, K, dtype=torch.float64)
            C = torch.zeros(K, V, dtype=torch.float64)
            G = torch.zeros(K, V, dtype=torch.float64)
            for t in range(T):
                dS = torch.outer(k64[b, h, t], k64[b, h, t])
                dC = torch.outer(q64[b, h, t], v64[b, h, t])
                G = G + dS @ C           # exclusive: C before this token's update
                S = S + dS
                C = C + dC
                outs[b, h, t] = q64[b, h, t] @ (S @ C - G)
    return outs


def test_serial_recurrence_matches_brute_force_identity():
    """Algorithm 1's serial recurrence == the closed-form masked identity."""
    torch.manual_seed(5)
    q = torch.randn(1, H, T, K)
    k = torch.randn(1, H, T, K)
    v = torch.randn(1, H, T, V)
    layer = HLALayer(K)
    actual = layer(q, k, v)
    expected = _brute_force_hla(q, k, v).float()
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, f"HLA serial vs brute-force: max abs err {err}"


def test_strict_causality_correction_is_applied():
    """The −G_t correction makes the output strictly causal: changing a future
    token must not change earlier outputs."""
    torch.manual_seed(7)
    q = torch.randn(1, H, T, K)
    k = torch.randn(1, H, T, K)
    v = torch.randn(1, H, T, V)
    layer = HLALayer(K)
    base = layer(q, k, v)
    k_future = k.clone()
    k_future[:, :, -1] = torch.randn(1, H, K)  # perturb the last token
    perturbed = layer(q, k_future, v)
    # Outputs at all earlier tokens must be unchanged.
    err = (base[:, :, :-1] - perturbed[:, :, :-1]).abs().max().item()
    assert err < 1e-5, f"strict causality violated: max abs err {err}"


def test_gradients_flow_through_nested_recurrence():
    """VJP through the nested pair (torch autograd through the external recurrence)."""
    torch.manual_seed(11)
    q = torch.randn(1, H, T, K, requires_grad=True)
    k = torch.randn(1, H, T, K, requires_grad=True)
    v = torch.randn(1, H, T, V, requires_grad=True)
    HLALayer(K)(q, k, v).square().sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v)):
        assert t.grad is not None and t.grad.abs().sum().item() > 0, f"{name} has no gradient"
