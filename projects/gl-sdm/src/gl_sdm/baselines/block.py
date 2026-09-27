"""Common ATMA residual block for the three baseline mixers."""
from torch import nn
from gl_sdm.layers.common import RMSNorm, MLP
from gl_sdm.layers.regularization import sigreg
from gl_sdm.baselines.mixers import make_mixer


class Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.arch_type = cfg["arch_type"]
        self.attn = make_mixer(cfg, layer_idx)
        self.norm1, self.norm2 = RMSNorm(cfg["hidden_size"]), RMSNorm(cfg["hidden_size"])
        self.mlp = MLP(cfg["hidden_size"], cfg.get("intermediate_size"))
        self.reg_mode, self.sketch_dim = cfg.get("reg_mode", "baseline"), cfg.get("sketch_dim", 64)

    def forward(self, x, cache=None):
        z = self.norm1(x).to(self.mlp.fc.weight.dtype)
        if self.arch_type == "gdn2":
            mixed = self.attn(z, past_key_values=cache, use_cache=cache is not None)[0]
        elif self.arch_type == "sdm":
            mixed = self.attn(z, cache=cache)[0]
        else:
            mixed = self.attn(z, cache=cache)
        x = x + mixed
        x = x + self.mlp(self.norm2(x).to(self.mlp.fc.weight.dtype))
        return x, sigreg(x, self.reg_mode, self.sketch_dim), x.new_zeros(())
