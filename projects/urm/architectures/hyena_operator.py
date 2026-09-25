"""External model module: Hyena operator (arch-077).

Verified composition-now row: the mixer is entirely external ordinary operators
— the implicit long filter (positional embedding → sine-activation MLP →
exponential modulation), a depthwise short conv, causal FFT convs, and gate
multiplies — and needs NO URM kernel call (sweep row-077, verified against
src/models/sequence/hyena.py @ 02220c69). No compact U2 realization is claimed.

Composition (transcribed from the pinned HyenaOperator.forward): in_proj, the
depthwise short conv, the per-order recurrence (v ← dropout(v ⊙ x_i) then the
implicit-filter causal FFT conv v ← fftconv_ref(v, k[o], bias[o])), the final
gate y = (v ⊙ x_0), and the out_proj are all external. The implicit filter is
the pinned HyenaFilter (positional embedding + Sin MLP + ExponentialModulation),
also transcribed here so the URM module is self-contained; parity is against
the pinned operator on identical parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Sin(nn.Module):
    def __init__(self, dim: int, w: float = 10.0):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim))

    def forward(self, x):
        return torch.sin(self.freq * x)


class _PositionalEmbedding(nn.Module):
    """Transcribed pinned PositionalEmbedding (complex exponential bands)."""

    def __init__(self, emb_dim: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        t = torch.linspace(0, 1, seq_len)[None, :, None]
        bands = (emb_dim - 1) // 2
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len
        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.register_buffer("z", z)
        self.register_buffer("t", t)

    def forward(self, L: int):
        return self.z[:, :L], self.t[:, :L]


class _ExponentialModulation(nn.Module):
    """Transcribed pinned ExponentialModulation."""

    def __init__(self, d_model: int, fast_decay_pct=0.3, slow_decay_pct=1.5,
                 target=1e-2, shift=0.0, modulate=True):
        super().__init__()
        self.modulate = modulate
        self.shift = shift
        max_decay = math.log(target) / fast_decay_pct
        min_decay = math.log(target) / slow_decay_pct
        deltas = torch.linspace(min_decay, max_decay, d_model)[None, None]
        self.register_buffer("deltas", deltas)

    def forward(self, t, x):
        if self.modulate:
            decay = torch.exp(-t * self.deltas.abs())
            x = x * (decay + self.shift)
        return x


class _HyenaFilter(nn.Module):
    """Transcribed pinned HyenaFilter (implicit long filter)."""

    def __init__(self, d_model: int, emb_dim=3, order=16, seq_len=1024,
                 num_inner_mlps=2, w=1.0):
        super().__init__()
        self.d_model = d_model
        self.bias = nn.Parameter(torch.randn(d_model))
        self.pos_emb = _PositionalEmbedding(emb_dim, seq_len)
        layers = [nn.Linear(emb_dim, order), _Sin(order, w)]
        for _ in range(num_inner_mlps):
            layers += [nn.Linear(order, order), _Sin(order, w)]
        layers += [nn.Linear(order, d_model, bias=False)]
        self.implicit_filter = nn.Sequential(*layers)
        self.modulation = _ExponentialModulation(d_model)

    def filter(self, L: int):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        return self.modulation(t, h)

    def forward(self, x, L, k=None, bias=None):
        """Transcribed pinned HyenaFilter.forward (non-fused fftconv_ref path).

        ``x`` is ``[b, d, l]`` (3D); the fftconv bias is the per-channel bias.
        """
        if k is None:
            k = self.filter(L)
        if bias is None:
            bias = self.bias
        return _fftconv_ref(x, k, bias)


def _fftconv_ref(u: torch.Tensor, k: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Transcribed pinned fftconv_ref (no gelu, no dropout mask, no k_rev)."""
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]
    return (y + u * D.unsqueeze(-1)).to(dtype=u.dtype)


class HyenaOperatorLayer(torch.nn.Module):
    """One Hyena operator (order-2, single block/head): fully external composition."""

    def __init__(self, d_model: int, l_max: int, filter_order: int = 16,
                 short_filter_order: int = 3):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.order = 2
        self.in_proj = nn.Linear(d_model, (self.order + 1) * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        total_width = d_model * (self.order + 1)
        self.short_filter = nn.Conv1d(
            total_width, total_width, short_filter_order,
            groups=total_width, padding=short_filter_order - 1,
        )
        self.filter_fn = _HyenaFilter(d_model, order=filter_order, seq_len=l_max)
        self.dropout = nn.Dropout(0.0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """``u`` ``[B, L, d_model]`` → same (order-2, num_heads=1, num_blocks=1).

        Transcribed from the pinned HyenaOperator.forward (non-fused path).
        """
        from einops import rearrange

        l = u.size(-2)
        l_filter = min(l, self.l_max)
        u = rearrange(self.in_proj(u), 'b l d -> b d l')
        uc = self.short_filter(u)[..., :l_filter]
        # Pinned: 'b (ho v) (z l) -> b ho v z l', z=num_blocks=1, ho=num_heads=1,
        # v=head_dim*(order+1)=d_model*(order+1).
        uc = rearrange(uc, 'b (ho v) (z l) -> b ho v z l', z=1, ho=1)
        *x, v = uc.split(self.d_model, dim=2)                   # x0, x1, v: [b,1,d,1,l]

        k = self.filter_fn.filter(l_filter)                    # [c=1, l, d*(order-1)]
        k = rearrange(k, 'c l (v o) -> c o v l', v=self.d_model, o=self.order - 1)[0]
        bias = rearrange(self.filter_fn.bias, '(v o) -> o v',
                         v=self.d_model, o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            v = self.dropout(v * x_i)
            # Pinned: filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None])
            # where bias[o] is [d]; the fftconv is over the 3D channel layout.
            v3 = rearrange(v, 'b 1 d 1 l -> b d l')
            v3 = self.filter_fn(v3, l_filter, k=k[o], bias=bias[o])
            v = rearrange(v3, 'b d l -> b 1 d 1 l')
        y = rearrange(v * x[0], 'b h v z l -> b (z l) (h v)', z=1, h=1)
        return self.out_proj(y)


__all__ = ["HyenaOperatorLayer"]
