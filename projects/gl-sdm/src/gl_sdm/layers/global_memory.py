"""GL-SDM module composition and configuration.

Weights remain registered here to preserve checkpoint keys. Execution schedules
are in chunk.py and token.py; sparse memory operations are in memory/.
"""
from torch import nn
from gl_sdm.memory import MemoryBank
from .common import MLP, RMSNorm
from .router import GlobalRouter


class GlobalMemoryBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["hidden_size"]
        self.max_steps = cfg.get("gl_max_steps", 8)
        self.reasoning = cfg.get("gl_reasoning", "adaptive")
        self.write_policy = cfg.get("gl_write_policy", "merged")
        self.halt_epsilon = cfg.get("gl_halt_epsilon", 0.01)
        if self.max_steps < 1 or self.reasoning not in {"fixed", "adaptive"} or self.write_policy not in {"merged", "final", "every_step"} or not 0 < self.halt_epsilon < 1:
            raise ValueError("invalid GL-SDM depth, halting or write policy")
        self.attn = GlobalRouter(cfg)
        self.bank = MemoryBank(self.attn.heads, self.attn.slots, self.attn.head_dim)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.mlp = MLP(dim)
        self.input_proj = nn.Linear(dim, dim)
        self.halt = nn.Linear(dim, 1) if self.reasoning == "adaptive" else None
        if self.halt is not None:
            nn.init.constant_(self.halt.bias, cfg.get("gl_halt_bias", -2.0))
        self.reg_mode, self.sketch_dim = cfg.get("reg_mode", "baseline"), cfg.get("sketch_dim", 64)
        self.last_depth = None
        self.chunk_size = cfg.get("gl_chunk_size", 1)
        self.compile_dense = cfg.get("gl_compile", False)
        if self.chunk_size < 1:
            raise ValueError("gl_chunk_size must be positive")
        if self.chunk_size > 1:
            if self.write_policy == "every_step":
                raise ValueError("chunk transactions require merged or final writes for causality")
            if max(self.attn.num_reads, self.attn.num_writes) > self.attn.half:
                raise ValueError("URM chunk routes cannot exceed the product-key factor extent")
            if self.attn.backend == "urm" and self.chunk_size > 2048:
                raise ValueError("frozen URM supports at most 2048 tokens per chunk")
            from .attention import Transformer
            self.local_context = Transformer(cfg, 0)
            self.local_norm = RMSNorm(dim)

    def forward(self, inputs, cache=None):
        # Chunk mode: local attention once, repeated global reads, boundary commit.
        # Token mode: the earlier sequential transaction control.
        if self.chunk_size > 1:
            from .chunk import forward
        else:
            from .token import forward
        return forward(self, inputs, cache)

    def reasoning_metrics(self, depth=None):
        depth = (self.last_depth if depth is None else depth).float()
        return {"mean_reasoning_steps": depth.mean().item(), "p95_reasoning_steps": depth.quantile(0.95).item(), "max_reasoning_steps": depth.max().item(), "executed_reasoner_calls": int(depth.sum().item()), "gl_write_policy": self.write_policy}
