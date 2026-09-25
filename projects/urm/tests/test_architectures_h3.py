"""Parity gates for arch-076 H3 mixer.

Verified against the pinned H3 module (src/models/ssm/h3.py @ 5c4d06b5, the
sweep's verification origin): the URM external module transcribes the pinned
non-fast path (two causal FFT convs, pointwise multiply, q-contraction) and is
checked against the pinned module on identical projections and the pinned
S4D/shift SSM kernels.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.h3_mixer import H3MixerLayer
from benchmarks.comparators.h3 import h3_adapter

D_MODEL, HEAD_DIM = 16, 4


def _build_pair(seed: int, L: int = 8):
    torch.manual_seed(seed)
    pinned, _id = h3_adapter(d_model=D_MODEL, head_dim=HEAD_DIM, l_max=L)
    urm = H3MixerLayer(D_MODEL, HEAD_DIM)
    with torch.no_grad():
        urm.q_proj.weight.copy_(pinned.q_proj.weight)
        urm.q_proj.bias.copy_(pinned.q_proj.bias)
        urm.k_proj.weight.copy_(pinned.k_proj.weight)
        urm.k_proj.bias.copy_(pinned.k_proj.bias)
        urm.v_proj.weight.copy_(pinned.v_proj.weight)
        urm.v_proj.bias.copy_(pinned.v_proj.bias)
        urm.ssm_k_D.copy_(pinned.ssm_k_D)
        urm.D.copy_(pinned.D)
        urm.output_linear.weight.copy_(pinned.output_linear.weight)
        urm.output_linear.bias.copy_(pinned.output_linear.bias)
    u = torch.randn(2, L, D_MODEL)
    return pinned, urm, u


def _kernels(pinned, L: int):
    """Produce the pinned SSM kernels (shift + S4D) at the module's length.

    The pinned S4D SSKernel emits length 2·L (its forward computes the kernel
    over the full conv length); use the emitted length, not L, when reshaping.
    """
    with torch.no_grad():
        ssm_kernel, _ = pinned.kernel(L=L, state=None, rate=1.0)   # (C H L_kernel)
        ssm_k_kernel, _ = pinned.ssm_k_kernel(L=L, state=None, rate=1.0)
    # Pinned rearranges '1 h l -> h l' / '1 c l -> c l' via rearrange.
    ssm_kernel = ssm_kernel.reshape(ssm_kernel.shape[-2], ssm_kernel.shape[-1])
    ssm_k_kernel = ssm_k_kernel.reshape(ssm_k_kernel.shape[-2], ssm_k_kernel.shape[-1])
    return ssm_kernel, ssm_k_kernel


def test_layer_matches_pinned_h3():
    pinned, urm, u = _build_pair(seed=5)
    L = u.shape[1]
    ssm_kernel, ssm_k_kernel = _kernels(pinned, L)
    with torch.no_grad():
        expected = pinned(u)
        actual = urm(u, ssm_kernel, ssm_k_kernel)
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, f"layer parity: max abs err {err}"


def test_gradients_flow_through_projections():
    pinned, urm, u = _build_pair(seed=11)
    L = u.shape[1]
    ssm_kernel, ssm_k_kernel = _kernels(pinned, L)
    urm(u, ssm_kernel, ssm_k_kernel).square().sum().backward()
    assert urm.q_proj.weight.grad is not None
    assert urm.output_linear.weight.grad is not None
    assert urm.D.grad is not None and urm.D.grad.abs().sum().item() > 0
