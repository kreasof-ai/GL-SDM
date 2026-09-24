"""Proof-of-concept: model-level training MFU with a URM-native recipe mixer.

Drops a single URM-native recipe (here ``gla``, which already has a native
backward) into the frozen 100M decoder LM in place of the sparse-memory mixer,
and measures end-to-end model-level training MFU against the measured bf16
tensor-core peak. This validates the RecipeMixer bridge before generalizing to
all 62 recipes.

Run: PYTHONPATH=src:benchmarks python benchmarks/model_level_poc.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIMITS = PROJECT_ROOT / "results" / "device-limits.json"


def _torch():
    import torch

    return torch


def _bf16_peak_tflops() -> float:
    limits = json.loads(DEVICE_LIMITS.read_text())
    return limits["bf16_tensor_core"]["bf16_tensor_core_tfps_measured"]


class RecipeMixerGLA:
    """Minimal nn.Module wiring the native GLA recurrence into [B,T,C]->[B,T,C].

    Projects hidden -> q,k,v,log_decay, runs the native compiled plan, projects
    back. log_decay is produced through a softplus-negation so the gate stays in
    the valid (decaying) range and is differentiable.
    """

    def __new__(cls, config, recipe_name: str = "gla", dtype: str = "bfloat16"):
        torch = _torch()
        import torch.nn as nn

        from urm.compiler.pipeline import MixerBackend, MixerIntent, compile_mixer
        from urm.frontend.recipes import named_mixer_recipe

        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                c, h, d = config.width, config.heads, config.value_dim
                self.h, self.d = h, d
                self.qkv = nn.Linear(c, 3 * h * d, bias=config.bias)
                self.decay = nn.Linear(c, h * d, bias=config.bias)
                self.output = nn.Linear(h * d, c, bias=config.bias)
                recipe = named_mixer_recipe(recipe_name)
                self._plan = compile_mixer(
                    recipe, intent=MixerIntent.TRAINING,
                    backend=MixerBackend.NATIVE, dtype=dtype,
                )
                self._dtype = getattr(torch, dtype)

            def forward(self, x):
                b, t, c = x.shape
                h, d = self.h, self.d
                qkv = self.qkv(x).view(b, t, 3, h, d)
                q = qkv[:, :, 0].contiguous()
                k = qkv[:, :, 1].contiguous()
                v = qkv[:, :, 2].contiguous()
                log_decay = -nn.functional.softplus(self.decay(x)).view(b, t, h, d)
                out = self._plan.execute(
                    query=q.to(self._dtype), key=k.to(self._dtype),
                    value=v.to(self._dtype), log_decay=log_decay.to(self._dtype),
                ).output
                return self.output(out.reshape(b, t, h * d).to(x.dtype))

        return _M()


def main() -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("model-level PoC requires CUDA")
    import torch.nn as nn

    from train.loop import PretrainingConfig

    # Frozen 100M config (from pretraining_step.toml), reduced steps for the PoC.
    config = PretrainingConfig(
        vocab_size=50304, sequence_length=1024, layers=12, width=768, heads=12,
        value_dim=64, mlp_ratio=4, microbatch=1, gradient_accumulation=4,
        dropout=0.0, bias=False,
    )

    # Build the model with the GLA recipe mixer swapped in.
    from train.loop import URMDecoderLM, MLP

    class GLABlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.norm1 = nn.LayerNorm(config.width)
            self.norm2 = nn.LayerNorm(config.width)
            self.mixer = RecipeMixerGLA(config)
            self.mlp = MLP(config)

        def forward(self, x):
            x = x + self.mixer(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class GLALM(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.token = nn.Embedding(config.vocab_size, config.width)
            self.position = nn.Embedding(config.sequence_length, config.width)
            self.blocks = nn.ModuleList(GLABlock(config) for _ in range(config.layers))
            self.norm = nn.LayerNorm(config.width)
            self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
            self.lm_head.weight = self.token.weight
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, tokens, targets):
            b, t = tokens.shape
            pos = torch.arange(t, device=tokens.device)
            x = self.token(tokens) + self.position(pos)[None]
            for block in self.blocks:
                x = block(x)
            logits = self.lm_head(self.norm(x))
            loss = nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.view(-1)
            )
            return logits, loss

    model = GLALM(config).cuda().to(torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)

    # Synthetic FineWeb-format tokens (random) for the PoC timing lane; the real
    # harness swaps in the pinned FineWeb shard. MFU is data-agnostic.
    rng = np.random.default_rng(0)
    b, t = config.microbatch, config.sequence_length
    accum = config.gradient_accumulation
    batches = [
        (
            torch.from_numpy(rng.integers(0, config.vocab_size, (b, t)).astype(np.int64)).cuda(),
            torch.from_numpy(rng.integers(0, config.vocab_size, (b, t)).astype(np.int64)).cuda(),
        )
        for _ in range(accum)
    ]

    # Model FLOPs per step: 6 * params * tokens (fwd+bwd transformer rule-of-thumb),
    # using the actual parameter count. This is the standard speedrun numerator.
    tokens_per_step = b * t * accum
    model_flops = 6.0 * n_params * tokens_per_step

    def step():
        optimizer.zero_grad(set_to_none=True)
        for tokens, targets in batches:
            _, loss = model(tokens, targets)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    peak = _bf16_peak_tflops()
    # warmup
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    median_s = times[len(times) // 2]
    mfu = model_flops / median_s / (peak * 1e12)
    tok_per_s = tokens_per_step / median_s
    print(f"recipe=gla  params={n_params/1e6:.1f}M  seq={t}  accum={accum}")
    print(f"median step: {median_s*1e3:.1f} ms   throughput: {tok_per_s:,.0f} tok/s")
    print(f"model FLOPs/step: {model_flops/1e12:.2f} TFLOP   measured bf16 peak: {peak:.1f} TFLOP/s")
    print(f"MODEL TRAINING MFU: {mfu*100:.2f}%")


if __name__ == "__main__":
    main()
