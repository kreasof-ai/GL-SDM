"""Causal RoPE/SDPA attention, shared by local context and Transformer baseline.

This mixer has projections, QK normalization and an output gate. The baseline
block adds its MLP; GL-SDM applies this mixer once before global reasoning.
"""
import torch
from torch import nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """ATMA RoPE softmax surround, used in every block with ordinary PyTorch."""
    def __init__(self, cfg, layer_idx):
        super().__init__()
        dim, dk = cfg["hidden_size"], cfg["head_dim"]
        self.head_dim, self.num_heads = dk, dim // dk
        self.num_kv_heads = cfg.get("num_key_value_heads", self.num_heads)
        if self.num_heads % self.num_kv_heads or dk % 4:
            raise ValueError("head_dim must be divisible by 4; KV heads must divide query heads")
        self.q = nn.Linear(dim, 2 * dim)
        self.k = nn.Linear(dim, self.num_kv_heads * dk)
        self.v = nn.Linear(dim, self.num_kv_heads * dk)
        self.proj = nn.Linear(dim, dim)
        freq = (1 / 1024) ** torch.linspace(0, 1, dk // 4)
        self.register_buffer("angular_freq", torch.cat((freq, torch.zeros_like(freq))))
        self.reference = False

    def _apply(self, fn, recurse=True):
        # ATMA's rotary frequencies stay FP32 even with BF16 projections.
        frequencies = self.angular_freq
        super()._apply(fn, recurse)
        self.angular_freq = frequencies.to(device=self.angular_freq.device)
        return self

    def rotary(self, x, offset):
        positions = torch.arange(offset, offset + x.shape[1], device=x.device, dtype=torch.float32)
        theta = torch.outer(positions, self.angular_freq.float())[None, :, None]
        a, b = x.float().chunk(2, -1)
        return torch.cat((a * theta.cos() + b * theta.sin(), -a * theta.sin() + b * theta.cos()), -1).to(x.dtype)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, gate = self.q(x).view(B, T, self.num_heads, 2 * self.head_dim).chunk(2, -1)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view_as(k).transpose(1, 2)
        offset = 0 if cache is None or not cache else cache["k"].shape[2]
        q = self.rotary(F.rms_norm(q, (self.head_dim,)), offset).transpose(1, 2)
        k = self.rotary(F.rms_norm(k, (self.head_dim,)), offset).transpose(1, 2)
        if cache is not None:
            if cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
            cache.update(k=k, v=v)
        groups = self.num_heads // self.num_kv_heads
        if groups != 1:
            k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        # Cached multi-token prefill needs an offset causal mask. SDPA's default
        # causal mask is aligned to the upper left, which is wrong here.
        mask = None
        if offset or self.reference:
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= torch.arange(offset, offset + T, device=x.device)[:, None]
        if self.reference:
            scores = (q.float() @ k.float().transpose(-1, -2)) * self.head_dim ** -0.5
            out = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.float()
            out = out.to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.proj(out.transpose(1, 2).reshape(B, T, D) * gate.reshape(B, T, D).sigmoid())
