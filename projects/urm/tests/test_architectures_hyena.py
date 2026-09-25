"""Parity gates for arch-077 Hyena operator.

The pinned HyenaOperator is runnable only with the CUDA fftconv extension (its
reference path has a bias-layout broadcast bug that makes the non-fused
operator non-executable at every configuration — verified empirically during
this batch). The sweep's verdict (external FFT convs, no U2 realization) is
confirmed against the source. These gates therefore verify:

1. The URM module's implicit filter matches the pinned HyenaFilter (AST-
   extracted, running against the pinned source) on identical parameters.
2. The URM operator's composition (in_proj → short conv → per-order gate +
   causal FFT conv → gate by x_0 → out_proj) matches an equation-level
   transcription driven by the pinned filter's kernels.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from architectures.hyena_operator import HyenaOperatorLayer, _fftconv_ref
from benchmarks.comparators.hyena import hyena_pinned_filter

D_MODEL, L_MAX, FILTER_ORDER = 16, 16, 16


def _copy_filter(urm_filter, pinned_filter):
    with torch.no_grad():
        urm_filter.bias.copy_(pinned_filter.bias)
        for urm_lin, pin_lin in zip(
            (m for m in urm_filter.implicit_filter if isinstance(m, torch.nn.Linear)),
            (m for m in pinned_filter.implicit_filter if isinstance(m, torch.nn.Linear)),
        ):
            urm_lin.weight.copy_(pin_lin.weight)
            if pin_lin.bias is not None:
                urm_lin.bias.copy_(pin_lin.bias)
        for urm_sin, pin_sin in zip(
            (m for m in urm_filter.implicit_filter if hasattr(m, "freq")),
            (m for m in pinned_filter.implicit_filter if hasattr(m, "freq")),
        ):
            urm_sin.freq.copy_(pin_sin.freq)


def test_implicit_filter_matches_pinned_hyena_filter():
    """The implicit long filter (pos emb → Sin MLP → ExpMod) matches the pinned source."""
    torch.manual_seed(5)
    pinned, _id = hyena_pinned_filter(D_MODEL, order=FILTER_ORDER, seq_len=L_MAX)
    from architectures.hyena_operator import _HyenaFilter
    urm_filter = _HyenaFilter(D_MODEL, order=FILTER_ORDER, seq_len=L_MAX)
    _copy_filter(urm_filter, pinned)
    with torch.no_grad():
        expected = pinned.filter(L_MAX)
        actual = urm_filter.filter(L_MAX)
    err = (actual - expected).abs().max().item()
    assert err < 1e-5, f"filter parity: max abs err {err}"


def _build_pair(seed: int):
    torch.manual_seed(seed)
    pinned_filter, _id = hyena_pinned_filter(D_MODEL, order=FILTER_ORDER, seq_len=L_MAX)
    urm = HyenaOperatorLayer(D_MODEL, L_MAX, filter_order=FILTER_ORDER)
    _copy_filter(urm.filter_fn, pinned_filter)
    return pinned_filter, urm


def test_operator_matches_equation_transcription_with_pinned_kernels():
    """The operator composition matches the pinned equation, driven by the
    pinned filter's generated kernel and bias."""
    pinned_filter, urm = _build_pair(seed=7)
    torch.manual_seed(100)
    u = torch.randn(2, L_MAX, D_MODEL)

    with torch.no_grad():
        # Equation-level transcription of the pinned forward, using the pinned
        # filter's kernel/bias and the URM module's projections/short conv.
        k = pinned_filter.filter(L_MAX)          # [1, L, d_model]
        k = k[0].t()                             # [d_model, L]  (order-2: single kernel)
        bias = pinned_filter.bias                # [d_model]

        u2 = urm.in_proj(u).transpose(1, 2)      # [b, 3d, l]
        uc = urm.short_filter(u2)[..., :L_MAX]   # pinned slices to l_filter (= l here)
        x0, x1, v = uc.split(D_MODEL, dim=1)
        v = v * x1                               # gate multiply (dropout p=0)
        v = _fftconv_ref(v, k, bias)             # causal FFT conv + bias
        y = v * x0                               # final gate
        expected = urm.out_proj(y.transpose(1, 2))

        actual = urm(u)
    err = (actual - expected).abs().max().item()
    assert err < 1e-4, f"operator parity: max abs err {err}"


def test_gradients_flow_through_filter_and_projections():
    _pinned_filter, urm = _build_pair(seed=11)
    u = torch.randn(2, L_MAX, D_MODEL)
    urm(u).square().sum().backward()
    assert urm.in_proj.weight.grad is not None
    assert urm.out_proj.weight.grad is not None
    assert urm.filter_fn.bias.grad is not None and urm.filter_fn.bias.grad.abs().sum().item() > 0
