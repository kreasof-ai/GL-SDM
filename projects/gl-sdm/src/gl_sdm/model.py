"""ATMA's embed/blocks/norm/proj and summed-loss model contract."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from gl_sdm.baselines.block import Block
from gl_sdm.baselines.cache import FLACache
from gl_sdm.layers.common import RMSNorm
from gl_sdm.runtime.compilation import compiled


def _head_loss(x, targets, norm, weight):
    x = F.rms_norm(x, (x.shape[-1],), norm, eps=1e-6)
    logits = F.linear(x.to(weight.dtype), weight).float()
    logits = 15 * logits * (logits.square() + 225).rsqrt()
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")


_compiled_head_loss = compiled(_head_loss)


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.is_layer_stack = cfg["arch_type"] == "gl_sdm" and "gl_layer_pattern" in cfg
        dim, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
        if layers < 1 or dim % cfg["head_dim"]:
            raise ValueError("positive layer count and hidden_size divisible by head_dim required")
        self.embed = nn.Embedding(cfg["vocab_size"], dim)
        if self.is_layer_stack:
            from gl_sdm.layers.stack import make_layers
            from gl_sdm.memory import MemoryBank
            self.blocks = nn.ModuleList(make_layers(cfg))
            self.bank = MemoryBank(dim // cfg["head_dim"], cfg["gl_slots"], cfg["head_dim"])
        elif cfg["arch_type"] == "gl_sdm":
            from gl_sdm.layers.global_memory import GlobalMemoryBlock
            if layers != 1:
                raise ValueError("GL-SDM uses one tied block; set num_hidden_layers=1 and gl_max_steps for depth")
            self.blocks = nn.ModuleList([GlobalMemoryBlock(cfg)])
        else:
            self.blocks = nn.ModuleList(Block(cfg, i) for i in range(layers))
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(dim, cfg["vocab_size"], bias=False)
        self.num_attn_layers = layers
        self.to(dtype=getattr(torch, cfg.get("dtype", "bfloat16")))

    def head(self, x):
        logits = self.proj(self.norm(x).to(self.proj.weight.dtype)).float()
        return 15 * logits * (logits.square() + 15 ** 2).rsqrt()

    def hidden(self, inputs, cache=None):
        x = self.embed(inputs)
        if self.cfg.get("residual_dtype") == "float32":
            x = x.float()
        if self.is_layer_stack:
            from gl_sdm.layers.stack import forward
            return forward(self, x, None if cache is None else cache[0])
        reg, align = x.new_zeros(()), x.new_zeros(())
        for i, block in enumerate(self.blocks):
            state = cache if self.cfg["arch_type"] == "gdn2" else (None if cache is None else cache[i])
            x, r, a = block(x, state)
            reg, align = reg + r, align + a
        return x, reg / len(self.blocks), align / len(self.blocks)

    def forward(self, inputs, targets):
        x, reg, align = self.hidden(inputs)
        if self.cfg["arch_type"] == "gl_sdm" and self.cfg.get("gl_compile", False) and x.is_cuda and not any(b.attn.reference for b in self.blocks):
            return _compiled_head_loss(x, targets, self.norm.weight, self.proj.weight), reg, align
        logits = self.head(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"), reg, align

    def new_cache(self, batch_size):
        if self.cfg["arch_type"] == "gl_sdm":
            return [(self.bank if self.is_layer_stack else self.blocks[0].bank).new_cache(batch_size)]
        if self.cfg["arch_type"] == "gdn2":
            return FLACache()
        if self.cfg["arch_type"] == "transformer":
            return [{} for _ in self.blocks]
        p = self.embed.weight
        return [b.attn.create_kv_cache(batch_size, 0, p.dtype, p.device) for b in self.blocks]

    @torch.inference_mode()
    def prefill(self, inputs, cache=None):
        if self.training:
            raise RuntimeError("call model.eval() before inference")
        cache = self.new_cache(inputs.shape[0]) if cache is None else cache
        x, _, _ = self.hidden(inputs, cache)
        return self.head(x), cache

    @torch.inference_mode()
    def decode(self, inputs, cache):
        if inputs.shape[1] != 1:
            raise ValueError("decode expects one token per request")
        return self.prefill(inputs, cache)

    @contextmanager
    def reference(self):
        """Explicit oracle execution; never selected by training or inference."""
        if self.cfg["arch_type"] == "gdn2":
            from gl_sdm.baselines.reference import gdn2
            with patch("fla.layers.gdn2.chunk_gdn2", gdn2), patch("fla.layers.gdn2.fused_recurrent_gdn2", gdn2):
                yield
        else:
            try:
                for block in self.blocks:
                    block.attn.reference = not (self.is_layer_stack and hasattr(block.attn, "window") and self.cfg.get("dtype") != "float32")
                    if hasattr(block, "local_context"):
                        # BF16 memory integration shares its SDPA surround;
                        # discrete routes amplify tiny dense-vs-flash rounding.
                        # FP32 checks independently exercise dense attention.
                        block.local_context.reference = self.cfg.get("dtype") == "float32"
                yield
            finally:
                for block in self.blocks:
                    block.attn.reference = False
                    if hasattr(block, "local_context"):
                        block.local_context.reference = False


def create_model(cfg):
    return Model(cfg)
