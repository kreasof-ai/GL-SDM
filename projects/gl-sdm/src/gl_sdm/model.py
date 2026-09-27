"""ATMA's embed/blocks/norm/proj and summed-loss model contract."""
from contextlib import contextmanager
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from .mixers import make_mixer
from .regularization import sigreg


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), eps=1e-6)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 8 * dim)
        self.proj = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x, gate = self.fc(x).chunk(2, -1)
        return self.proj(gate * x.relu().square())


class FLACache:
    """Minimal FLA layer-cache protocol, kept per request, never on the model."""
    def __init__(self):
        self.states = []

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index]

    def update(self, layer_idx, offset=1, **state):
        while len(self.states) <= layer_idx:
            self.states.append(None)
        self.states[layer_idx] = state
        return state


class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.arch_type = cfg["arch_type"]
        self.attn = make_mixer(cfg, layer_idx)
        self.norm1, self.norm2 = RMSNorm(cfg["hidden_size"]), RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg["hidden_size"])
        self.reg_mode, self.sketch_dim = cfg.get("reg_mode", "baseline"), cfg.get("sketch_dim", 64)

    def forward(self, x, cache=None):
        z = self.norm1(x)
        if self.arch_type == "gdn2":
            mixed = self.attn(z, past_key_values=cache, use_cache=cache is not None)[0]
        elif self.arch_type == "sdm":
            mixed = self.attn(z, cache=cache)[0]
        else:
            mixed = self.attn(z, cache=cache)
        x = x + mixed
        x = x + self.mlp(self.norm2(x))
        return x, sigreg(x, self.reg_mode, self.sketch_dim), x.new_zeros(())


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        dim, layers = cfg["hidden_size"], cfg["num_hidden_layers"]
        if layers < 1 or dim % cfg["head_dim"]:
            raise ValueError("positive layer count and hidden_size divisible by head_dim required")
        self.embed = nn.Embedding(cfg["vocab_size"], dim)
        if cfg["arch_type"] == "gl_sdm":
            from .global_model import GlobalMemoryBlock
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
        logits = self.proj(self.norm(x)).float()
        return 15 * logits * (logits.square() + 15 ** 2).rsqrt()

    def hidden(self, inputs, cache=None):
        x = self.embed(inputs)
        reg, align = x.new_zeros(()), x.new_zeros(())
        for i, block in enumerate(self.blocks):
            state = cache if self.cfg["arch_type"] == "gdn2" else (None if cache is None else cache[i])
            x, r, a = block(x, state)
            reg, align = reg + r, align + a
        return x, reg / len(self.blocks), align / len(self.blocks)

    def forward(self, inputs, targets):
        x, reg, align = self.hidden(inputs)
        logits = self.head(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"), reg, align

    def new_cache(self, batch_size):
        if self.cfg["arch_type"] == "gl_sdm":
            return [self.blocks[0].bank.new_cache(batch_size)]
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
            from .reference import gdn2
            with patch("fla.layers.gdn2.chunk_gdn2", gdn2), patch("fla.layers.gdn2.fused_recurrent_gdn2", gdn2):
                yield
        else:
            try:
                for block in self.blocks:
                    block.attn.reference = True
                yield
            finally:
                for block in self.blocks:
                    block.attn.reference = False


def create_model(cfg):
    return Model(cfg)
