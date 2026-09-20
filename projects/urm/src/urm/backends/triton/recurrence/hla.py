"""Fused FP32 streaming HLA second-order forward and reverse scans."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _hla_forward_kernel(
    Q, K, V, Y, S_HIST, C_HIST, G_HIST,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BD: tl.constexpr, BV: tl.constexpr,
):
    head = tl.program_id(0)
    b = head // H
    h = head % H
    rd = tl.arange(0, BD)
    rv = tl.arange(0, BV)
    mask_d = rd < D
    mask_v = rv < DV
    s = tl.zeros((BD, BD), tl.float32)
    c = tl.zeros((BD, BV), tl.float32)
    g = tl.zeros((BD, BV), tl.float32)
    for t in range(T):
        q = tl.load(Q + ((b * T + t) * H + h) * D + rd, mask=mask_d, other=0.0)
        k = tl.load(K + ((b * T + t) * H + h) * D + rd, mask=mask_d, other=0.0)
        v = tl.load(V + ((b * T + t) * H + h) * DV + rv, mask=mask_v, other=0.0)
        c_prev = c
        s = s + k[:, None] * k[None, :]
        c = c + q[:, None] * v[None, :]
        k_c_prev = tl.sum(k[:, None] * c_prev, axis=0)
        g = g + k[:, None] * k_c_prev[None, :]
        q_s = tl.sum(q[:, None] * s, axis=0)
        q_sc = tl.sum(q_s[:, None] * c, axis=0)
        q_g = tl.sum(q[:, None] * g, axis=0)
        y = q_sc - q_g
        y_offset = ((b * T + t) * H + h) * DV + rv
        tl.store(Y + y_offset, y, mask=mask_v)

        s_offset = (((b * T + t) * H + h) * D * D
                    + rd[:, None] * D + rd[None, :])
        c_offset = (((b * T + t) * H + h) * D * DV
                    + rd[:, None] * DV + rv[None, :])
        tl.store(S_HIST + s_offset, s, mask=mask_d[:, None] & mask_d[None, :])
        tl.store(C_HIST + c_offset, c, mask=mask_d[:, None] & mask_v[None, :])
        tl.store(G_HIST + c_offset, g, mask=mask_d[:, None] & mask_v[None, :])


@triton.jit
def _hla_backward_kernel(
    Q, K, V, DY, S_HIST, C_HIST, G_HIST, DQ, DK, DV_OUT,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
    BD: tl.constexpr, BV: tl.constexpr,
):
    head = tl.program_id(0)
    b = head // H
    h = head % H
    rd = tl.arange(0, BD)
    rv = tl.arange(0, BV)
    mask_d = rd < D
    mask_v = rv < DV
    adj_s = tl.zeros((BD, BD), tl.float32)
    adj_c = tl.zeros((BD, BV), tl.float32)
    adj_g = tl.zeros((BD, BV), tl.float32)
    for reverse_index in range(T):
        t = T - 1 - reverse_index
        q = tl.load(Q + ((b * T + t) * H + h) * D + rd, mask=mask_d, other=0.0)
        k = tl.load(K + ((b * T + t) * H + h) * D + rd, mask=mask_d, other=0.0)
        v = tl.load(V + ((b * T + t) * H + h) * DV + rv, mask=mask_v, other=0.0)
        dy = tl.load(DY + ((b * T + t) * H + h) * DV + rv, mask=mask_v, other=0.0)
        s_offset = (((b * T + t) * H + h) * D * D
                    + rd[:, None] * D + rd[None, :])
        c_offset = (((b * T + t) * H + h) * D * DV
                    + rd[:, None] * DV + rv[None, :])
        s = tl.load(S_HIST + s_offset, mask=mask_d[:, None] & mask_d[None, :], other=0.0)
        c = tl.load(C_HIST + c_offset, mask=mask_d[:, None] & mask_v[None, :], other=0.0)
        g = tl.load(G_HIST + c_offset, mask=mask_d[:, None] & mask_v[None, :], other=0.0)
        if t > 0:
            c_prev = tl.load(C_HIST + c_offset - H * D * DV,
                             mask=mask_d[:, None] & mask_v[None, :], other=0.0)
        else:
            c_prev = tl.zeros((BD, BV), tl.float32)

        c_dy = tl.sum(c * dy[None, :], axis=1)
        s_transpose_q = tl.sum(s * q[:, None], axis=0)
        q_grad = tl.sum(s * c_dy[None, :], axis=1) - tl.sum(g * dy[None, :], axis=1)
        adj_s = adj_s + q[:, None] * c_dy[None, :]
        adj_c = adj_c + s_transpose_q[:, None] * dy[None, :]
        adj_g = adj_g - q[:, None] * dy[None, :]

        k_c_prev = tl.sum(k[:, None] * c_prev, axis=0)
        grad_k_from_g = tl.sum(adj_g * k_c_prev[None, :], axis=1)
        grad_inner = tl.sum(adj_g * k[:, None], axis=0)
        grad_k_from_g = grad_k_from_g + tl.sum(c_prev * grad_inner[None, :], axis=1)
        grad_c_from_g = k[:, None] * grad_inner[None, :]

        grad_q_from_c = tl.sum(adj_c * v[None, :], axis=1)
        grad_v = tl.sum(adj_c * q[:, None], axis=0)
        grad_k_from_s = tl.sum((adj_s + tl.trans(adj_s)) * k[None, :], axis=1)
        grad_k = grad_k_from_s + grad_k_from_g
        adj_c = adj_c + grad_c_from_g

        base_q = ((b * T + t) * H + h) * D + rd
        base_k = ((b * T + t) * H + h) * D + rd
        base_v = ((b * T + t) * H + h) * DV + rv
        tl.store(DQ + base_q, q_grad + grad_q_from_c, mask=mask_d)
        tl.store(DK + base_k, grad_k, mask=mask_d)
        tl.store(DV_OUT + base_v, grad_v, mask=mask_v)


class _HLASecondOrder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value):
        if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
            raise ValueError("HLA Triton expects BTHD query/key/value tensors")
        if query.shape[:3] != value.shape[:3]:
            raise ValueError("HLA Triton batch/time/head dimensions must match")
        if not (query.dtype == key.dtype == value.dtype == torch.float32):
            raise TypeError("HLA Triton currently supports float32 tensors")
        if not (query.device == key.device == value.device):
            raise ValueError("HLA Triton tensors must share a device")
        if not query.is_cuda:
            raise ValueError("HLA Triton requires CUDA tensors")
        batch, sequence, heads, key_dim = query.shape
        value_dim = value.shape[-1]
        if max(key_dim, value_dim) > 32:
            raise ValueError("HLA Triton currently supports key/value dimensions up to 32")
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        output = torch.empty(
            (batch, sequence, heads, value_dim), device=query.device, dtype=query.dtype
        )
        s_hist = torch.empty(
            (batch, sequence, heads, key_dim, key_dim), device=query.device, dtype=torch.float32
        )
        c_hist = torch.empty(
            (batch, sequence, heads, key_dim, value_dim), device=query.device, dtype=torch.float32
        )
        g_hist = torch.empty_like(c_hist)
        block_d, block_v = triton.next_power_of_2(key_dim), triton.next_power_of_2(value_dim)
        _hla_forward_kernel[(batch * heads,)](
            query, key, value, output, s_hist, c_hist, g_hist,
            sequence, heads, key_dim, value_dim, block_d, block_v,
            num_warps=4,
        )
        ctx.save_for_backward(query, key, value, s_hist, c_hist, g_hist)
        ctx.dimensions = (batch, sequence, heads, key_dim, value_dim, block_d, block_v)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, s_hist, c_hist, g_hist = ctx.saved_tensors
        batch, sequence, heads, key_dim, value_dim, block_d, block_v = ctx.dimensions
        dq = torch.empty_like(query)
        dk = torch.empty_like(key)
        dv = torch.empty_like(value)
        _hla_backward_kernel[(batch * heads,)](
            query, key, value, grad_output.contiguous(), s_hist, c_hist, g_hist,
            dq, dk, dv, sequence, heads, key_dim, value_dim, block_d, block_v,
            num_warps=4,
        )
        return dq, dk, dv


def hla_second_order_triton(query, key, value):
    return _HLASecondOrder.apply(query, key, value)
