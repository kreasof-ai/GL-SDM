"""Empirical MFU and throughput demonstration for Dual-Form SDM Reparameterization.

Measures:
1. Isolated sequence mixer execution latency, Tensor Core throughput, and MFU.
2. Full model-level pretraining step time, tokens/second, and model MFU on NVIDIA A10G.
3. Comparative analysis against URM Native v0, Pinned Upstream SDM, and SDPA.
"""

from __future__ import annotations

import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from urm.backends.dual_form_sdm import DualFormSDMFunction, dual_form_sdm
from urm.pretraining import (
    PretrainingConfig,
    URMDecoderLM,
    DecoderBlock,
    FP32AdamW,
    semantic_training_flops,
)


class DualFormSparseMemoryMixer(nn.Module):
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        c, h, f, d = config.width, config.heads, config.factor_extent, config.value_dim

        self.score = nn.Linear(c, h * 2 * f, bias=config.bias)
        self.read_score_bias = nn.Parameter(torch.zeros(h, 2 * f))
        self.write_score_bias = nn.Parameter(torch.empty(h, 2 * f))
        nn.init.normal_(self.write_score_bias, std=0.002)
        self.value_gate = nn.Linear(c, h * (d + 2), bias=config.bias)
        self.output = nn.Linear(c, c, bias=config.bias)

        self.register_buffer(
            "persistent_memory",
            torch.zeros(config.parallel, config.slots_per_partition, d),
            persistent=False,
        )
        self._pending_state = None

    def reset_state(self) -> None:
        self.persistent_memory.zero_()
        self._pending_state = None

    @torch.no_grad()
    def detach_state(self) -> None:
        if self._pending_state is not None:
            self.persistent_memory.copy_(self._pending_state.detach())
            self._pending_state = None

    def _project(self, x):
        b, t, _ = x.shape
        h, f, d = self.config.heads, self.config.factor_extent, self.config.value_dim
        common = self.score(x).view(b, t, h, 2 * f).permute(0, 2, 1, 3)
        read_scores = (common + self.read_score_bias[None, :, None]).reshape(
            b * h, t, 2 * f
        )
        write_scores = (common + self.write_score_bias[None, :, None]).reshape(
            b * h, t, 2 * f
        )
        projected = self.value_gate(x).view(b, t, h, d + 2).permute(0, 2, 1, 3)
        values = projected[..., :d].reshape(b * h, t, d).contiguous()
        beta = torch.sigmoid(projected[..., d : d + 1]).reshape(b * h, t, 1)
        log_decay = -F.softplus(projected[..., d + 1 :]).reshape(b * h, t, 1)
        return (
            read_scores.contiguous(),
            write_scores.contiguous(),
            values,
            beta.contiguous(),
            log_decay.contiguous(),
        )

    def _route(self, scores, k_top, factor_extent):
        P, T, two_f = scores.shape
        f = factor_extent
        row_scores = scores[:, :, :f]
        col_scores = scores[:, :, f:]

        k_factor = round(k_top ** 0.5)
        _, top_rows = torch.topk(row_scores, k_factor, dim=-1)
        _, top_cols = torch.topk(col_scores, k_factor, dim=-1)

        indices = (top_rows.unsqueeze(-1) * f + top_cols.unsqueeze(-2)).reshape(P, T, k_top)
        row_w = F.softmax(row_scores.gather(-1, top_rows), dim=-1)
        col_w = F.softmax(col_scores.gather(-1, top_cols), dim=-1)
        weights = (row_w.unsqueeze(-1) * col_w.unsqueeze(-2)).reshape(P, T, k_top)
        return indices, weights

    def forward(self, x):
        b, t, c = x.shape
        read_scores, write_scores, values, beta, log_decay = self._project(x)

        write_indices, write_weights = self._route(write_scores, self.config.writes, self.config.factor_extent)
        read_indices, read_weights = self._route(read_scores, self.config.reads, self.config.factor_extent)

        memory = self.persistent_memory

        readings, final_memory = dual_form_sdm(
            memory,
            read_indices,
            read_weights,
            write_indices=write_indices,
            write_weights=write_weights,
            values=values,
            beta=beta,
            log_decay=log_decay,
        )
        self._pending_state = final_memory

        out = readings.view(b, self.config.heads, t, self.config.value_dim).permute(0, 2, 1, 3).reshape(b, t, c)
        return self.output(out)


class DualFormDecoderLM(nn.Module):
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.sequence_length, config.width)

        self.blocks = nn.ModuleList()
        for _ in range(config.layers):
            block = DecoderBlock(config, "sdpa")
            block.mixer = DualFormSparseMemoryMixer(config)
            self.blocks.append(block)

        self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token.weight

    def reset_state(self):
        for block in self.blocks:
            block.mixer.reset_state()

    def detach_state(self):
        for block in self.blocks:
            block.mixer.detach_state()

    def forward(self, tokens, targets=None):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token(tokens) + self.position(positions)[None]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def benchmark_isolated_kernel():
    P, T, S, D, W, R = 12, 1024, 4096, 64, 64, 64
    device = "cuda"
    dtype = torch.bfloat16

    print("\n--- 1. ISOLATED SEQUENCE MIXER BENCHMARK ---")
    memory = torch.randn(P, S, D, device=device, dtype=dtype, requires_grad=True)
    wi = torch.zeros(P, T, W, device=device, dtype=torch.int64)
    ri = torch.zeros(P, T, R, device=device, dtype=torch.int64)
    for p in range(P):
        for t in range(T):
            wi[p, t] = torch.randperm(S, device=device)[:W]
            ri[p, t] = torch.randperm(S, device=device)[:R]
    wi, _ = wi.sort(dim=-1)
    ri, _ = ri.sort(dim=-1)

    w = torch.rand(P, T, W, device=device, dtype=dtype, requires_grad=True)
    q = torch.rand(P, T, R, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(P, T, D, device=device, dtype=dtype, requires_grad=True)
    b = torch.rand(P, T, 1, device=device, dtype=dtype, requires_grad=True)
    g = (-torch.rand(P, T, 1, device=device, dtype=dtype) * 0.05).requires_grad_()

    # Warmup
    y, m = dual_form_sdm(memory, ri, q, write_indices=wi, write_weights=w, values=v, beta=b, log_decay=g)
    loss = y.sum() + m.sum()
    loss.backward()
    torch.cuda.synchronize()

    # Forward
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        y, m = dual_form_sdm(memory, ri, q, write_indices=wi, write_weights=w, values=v, beta=b, log_decay=g)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / iters * 1000.0

    # Total (Forward + Backward)
    t0 = time.perf_counter()
    for _ in range(iters):
        y, m = dual_form_sdm(memory, ri, q, write_indices=wi, write_weights=w, values=v, beta=b, log_decay=g)
        loss = y.sum() + m.sum()
        loss.backward()
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) / iters * 1000.0
    bwd_ms = total_ms - fwd_ms

    # FLOPs in GEMMs:
    # A = W @ W^T: 2 * P * T * S * T = 2 * 12 * 1024 * 4096 * 1024 = 103.08 GFLOPs
    # Omega = Q @ W^T: 103.08 GFLOPs
    # Y = Omega @ Delta: 2 * P * T * T * D = 1.61 GFLOPs
    # Backward GEMMs: dW_curr, dW_prev, dQ_curr: 3 * 103.08 = 309.24 GFLOPs
    # dDelta_read, dA, dOmega: ~5 GFLOPs
    # Total GEMM FLOPs in forward+backward = ~522 GFLOPs
    gemm_gflops = 522.0
    achieved_tflops = gemm_gflops / total_ms
    a10g_peak_tflops = 66.16599973571093
    kernel_mfu = achieved_tflops / a10g_peak_tflops

    print(f"  Dual-Form Forward Latency:       {fwd_ms:.2f} ms")
    print(f"  Dual-Form Backward Latency:      {bwd_ms:.2f} ms")
    print(f"  Dual-Form Total Mixer Step:      {total_ms:.2f} ms")
    print(f"  Speedup vs Native v0 (~87 ms):   {87.0 / total_ms:.2f}x")
    print(f"  Tensor Core Throughput:          {achieved_tflops:.2f} TFLOP/s")
    print(f"  Mixer MFU (vs A10G 66.2 TF):     {kernel_mfu * 100:.2f}%")


def benchmark_pretraining_step():
    config = PretrainingConfig()
    device = "cuda"

    print("\n--- 2. FULL MODEL OPTIMIZER-STEP BENCHMARK (124.65M DECODER) ---")
    model = DualFormDecoderLM(config).to(device=device, dtype=torch.bfloat16)
    optimizer = FP32AdamW(model.parameters())

    tokens = torch.randint(0, config.vocab_size, (1, config.sequence_length), device=device)
    target = torch.randint(0, config.vocab_size, (1, config.sequence_length), device=device)

    # Warmup
    for _ in range(config.gradient_accumulation):
        _, loss = model(tokens, target)
        loss.backward()
        model.detach_state()
    optimizer.step()
    optimizer.zero_grad()
    model.reset_state()
    torch.cuda.synchronize()

    # Measured steps
    num_steps = 3
    step_times = []
    for _ in range(num_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(config.gradient_accumulation):
            _, loss = model(tokens, target)
            loss.backward()
            model.detach_state()
        optimizer.step()
        optimizer.zero_grad()
        model.reset_state()
        torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000.0)

    step_times.sort()
    median_ms = step_times[len(step_times) // 2]
    tokens_per_sec = 4096.0 / (median_ms / 1000.0)

    flops = semantic_training_flops(config, "urm_native")
    achieved_tflops = (flops.useful_total / (median_ms / 1000.0)) / 1e12
    a10g_peak_tflops = 66.16599973571093
    model_mfu = achieved_tflops / a10g_peak_tflops

    print(f"  Step Latency (Median):           {median_ms:.2f} ms")
    print(f"  Training Throughput:             {tokens_per_sec:.1f} tokens/s")
    print(f"  Useful TFLOP/s:                  {achieved_tflops:.2f} TFLOP/s")
    print(f"  Model MFU (vs A10G 66.2 TF):     {model_mfu * 100:.2f}%")
    print(f"  Speedup vs Native v0 (4070 ms):  {4070.0 / median_ms:.2f}x")
    print(f"  Speedup vs Upstream (1675 ms):   {1675.0 / median_ms:.2f}x")


if __name__ == "__main__":
    benchmark_isolated_kernel()
    benchmark_pretraining_step()
