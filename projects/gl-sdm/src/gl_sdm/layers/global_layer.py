"""One untied global read/MLP layer; the model owns its shared memory bank."""
import torch
from torch import nn
import torch.nn.functional as F
from gl_sdm.memory import routing
from gl_sdm.memory.state import WriteProposal
from gl_sdm.runtime.dense import proposal_delta
from .common import MLP, RMSNorm
from .router import GlobalRouter
from .regularization import sigreg


class GlobalLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["hidden_size"]
        self.attn = GlobalRouter(cfg)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.mlp = MLP(dim, cfg.get("intermediate_size"))
        self.reg_mode, self.sketch_dim = cfg.get("reg_mode", "baseline"), cfg.get("sketch_dim", 64)

    def forward(self, x, snapshot, physical=None):
        B, T, _ = x.shape
        router = self.attn
        dtype = router.q.weight.dtype
        scores = router.q(self.norm1(x).to(dtype)).float().view(B, T, router.heads, -1)
        readings, _, _ = routing.routed_read(self, snapshot.values, scores, router.num_reads, physical)
        mixed = router.proj(router.read_norm(readings.to(dtype)).flatten(-2))
        x = x + mixed
        x = x + self.mlp(self.norm2(x).to(dtype))
        return x, sigreg(x, self.reg_mode, self.sketch_dim), x.new_zeros(())

    def propose(self, x, snapshot, offset, layer_idx, chunk_size, num_layers, physical=None):
        """Sum token/layer deltas, each evaluated against the same snapshot.

        There is no division by token count or global-layer count. Canonical
        collision order is address, absolute position within chunk, then layer.
        """
        B, T, _ = x.shape
        router = self.attn
        z = self.norm1(x)
        scores = router.k(z.to(router.k.weight.dtype)).float().view(B, T, router.heads, -1)
        prediction, index, weights = routing.routed_read(self, snapshot.values, scores, router.num_writes, physical)
        rows = routing.gather_rows(snapshot.values, index)
        addresses = routing.absolute_addresses(snapshot.values, index)
        dense = z.to(router.v.weight.dtype)
        target = router.v(dense).view(B, T, router.heads, router.head_dim)
        beta = router.beta(dense).float().sigmoid().unsqueeze(-1)
        decay = -F.softplus(router.decay(dense).float() - 4).unsqueeze(-1)
        mass = torch.ones((B, T), device=x.device)
        delta = proposal_delta(rows, weights, prediction, target, beta, decay, mass)
        positions = torch.arange(offset, offset + T, device=x.device).view(1, T, 1, 1)
        keys = addresses * (chunk_size * num_layers) + positions * num_layers + layer_idx
        return WriteProposal(snapshot.version, addresses.flatten(), delta.flatten(0, 3), snapshot.lineage, keys.flatten())
