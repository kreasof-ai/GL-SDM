"""External model module: H3 mixer (arch-076).

Verified composition-now row: the mixer is composed entirely of external
ordinary operators — q/k/v projections, two causal FFT convolutions (shift-SSM
on k, then S4D on the k⊗v outer-product layout), pointwise multiplies, and the
q-contraction (mul_sum over the query head dim) — and needs NO URM kernel call
(sweep row-076, verified against src/models/ssm/h3.py @ 5c4d06b5). A U2
state-space rewrite is separately blocked; this module is the honest external
FFT composition, transcribed from the pinned non-fast (``use_fast_fftconv=
False``) path. The S4D/shift SSKernel modules are themselves external; parity
is established by running the pinned H3 module with this module's parameters
(and vice versa) on identical inputs.
"""

from __future__ import annotations

import torch


def _b_hd_l__to__b_d1_1_h_l(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """einops 'b (h d1) l -> b d1 1 h l' with d1 = head_dim."""
    b, hd, l = x.shape
    h = hd // head_dim
    return x.view(b, h, head_dim, l).permute(0, 2, 1, 3).unsqueeze(2)


def _b_hd_l__to__b_1_d2_h_l(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    """einops 'b (h d2) l -> b 1 d2 h l' with d2 = head_dim."""
    b, hd, l = x.shape
    h = hd // head_dim
    return x.view(b, h, head_dim, l).permute(0, 2, 1, 3).unsqueeze(1)


def mul_sum(q: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The pinned mul_sum: (q * y).sum(dim=1)."""
    return (q * y).sum(dim=1)


class H3MixerLayer(torch.nn.Module):
    """One H3 mixer layer: external projections + FFT convs + q contraction.

    ``d_model`` channels, grouped into ``H = d_model // head_dim`` heads. The
    two SSM kernels (shift kernel ``ssm_k_kernel`` ``[d_model, L]`` and the S4D
    kernel ``ssm_kernel`` ``[H, L]``) are supplied per forward — they are the
    external SSKernel modules' output.
    """

    def __init__(self, d_model: int, head_dim: int) -> None:
        super().__init__()
        if d_model % head_dim != 0:
            raise ValueError("d_model must be divisible by head_dim")
        self.d_model = d_model
        self.head_dim = head_dim
        self.H = d_model // head_dim
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.ssm_k_D = torch.nn.Parameter(torch.randn(d_model))
        self.D = torch.nn.Parameter(torch.randn(self.H))
        self.output_linear = torch.nn.Linear(d_model, d_model)

    def forward(
        self,
        u: torch.Tensor,
        ssm_kernel: torch.Tensor,
        ssm_k_kernel: torch.Tensor,
    ) -> torch.Tensor:
        """``u`` ``[B, L, d_model]`` → same. Transcribed pinned non-fast path."""
        B, L, _ = u.shape
        L_kernel = ssm_kernel.shape[-1]
        fft_size = L_kernel + L

        q = self.q_proj(u).transpose(1, 2)   # [B, d_model, L]
        k = self.k_proj(u).transpose(1, 2)
        v = self.v_proj(u).transpose(1, 2)

        # First causal FFT conv: shift-SSM on k, + D_k·k skip.
        ssm_k_kernel_f = torch.fft.rfft(ssm_k_kernel, n=fft_size)      # [d_model, fft]
        k_f = torch.fft.rfft(k.to(ssm_k_kernel.dtype), n=fft_size)     # [B, d_model, fft]
        shift_k_out = torch.fft.irfft(ssm_k_kernel_f * k_f, n=fft_size)[..., :L]
        k = shift_k_out + self.ssm_k_D.view(-1, 1) * k

        # Outer-product layout kv [b, d1, d2, h, l], second causal FFT conv (S4D).
        kv = _b_hd_l__to__b_d1_1_h_l(k, self.head_dim) * _b_hd_l__to__b_1_d2_h_l(v, self.head_dim)
        kv_f = torch.fft.rfft(kv.to(ssm_kernel.dtype), n=fft_size) / fft_size
        ssm_kernel_f = torch.fft.rfft(ssm_kernel, n=fft_size)          # [H, fft]
        y = torch.fft.irfft(kv_f * ssm_kernel_f, n=fft_size, norm="forward")[..., :L]
        y = y + kv * self.D.view(1, 1, 1, self.H, 1)

        # q contraction: mul_sum over the d1 axis, then output projection.
        q_l = _b_hd_l__to__b_d1_1_h_l(q, self.head_dim)                # [b, d1, 1, h, l]
        if self.head_dim > 1:
            y = mul_sum(q_l, y)                                        # [b, d2, h, l]
            # Pinned einops 'b d h l -> b (d h) l': d (v-factor) outer.
            y = y.reshape(B, y.shape[1] * y.shape[2], L)               # [b, (d h), l]
        else:
            y = (y * q_l).reshape(B, self.H, L)
        y = y.transpose(1, 2)                                          # [B, L, d_model]
        return self.output_linear(y)


__all__ = ["H3MixerLayer", "mul_sum"]
