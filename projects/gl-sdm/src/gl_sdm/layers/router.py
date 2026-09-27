"""Product-key projections and token-clock retrieval/proposal equations.

The chunk executor uses these same weights with memory.routing instead.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from gl_sdm.memory import read, propose_write
from .common import RMSNorm


class GlobalRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, dk = cfg["hidden_size"], cfg["head_dim"]
        self.heads, self.head_dim = dim // dk, dk
        self.slots = cfg.get("gl_slots", 1024)
        self.half = math.isqrt(self.slots)
        self.num_reads, self.num_writes = cfg.get("gl_reads", 8), cfg.get("gl_writes", 8)
        if self.half ** 2 != self.slots or not 1 <= min(self.num_reads, self.num_writes) or max(self.num_reads, self.num_writes) > self.slots:
            raise ValueError("GL-SDM requires square slots and route sizes in [1, slots]")
        self.q = nn.Linear(dim, self.heads * 2 * self.half)
        self.k = nn.Linear(dim, self.heads * 2 * self.half)
        self.v = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, self.heads)
        self.decay = nn.Linear(dim, self.heads)
        self.proj = nn.Linear(dim, dim)
        self.read_norm = RMSNorm(dk)
        self.backend = cfg.get("gl_memory_backend", "torch")
        self.experimental_large_route = cfg.get("gl_urm_large_route_override", False)
        if self.backend not in {"torch", "urm"}:
            raise ValueError("gl_memory_backend must be torch or urm")
        if self.backend == "urm":
            from gl_sdm.memory.backends.urm import verify_dependency
            verify_dependency()
        self.reference = False

    def route(self, projection, hidden, count):
        scores = projection(hidden).float().view(-1, self.heads, 2, self.half)
        if self.backend == "urm" and not self.reference:
            from gl_sdm.memory.backends.token import route
            return route(scores.flatten(-2), count)
        k = min(count, self.half)
        # Stable tie-breaking: smaller sub-key/product index wins equal scores.
        order = scores.argsort(dim=-1, descending=True, stable=True)[..., :k]
        top = scores.gather(-1, order)
        combined = (top[:, :, 0, :, None] + top[:, :, 1, None, :]).flatten(-2)
        product = (order[:, :, 0, :, None] * self.half + order[:, :, 1, None, :]).flatten(-2)
        # Lexicographic address order before score ranking makes product ties
        # deterministic even when sub-key logits differ but sums are equal.
        by_address = product.argsort(dim=-1, stable=True)
        product, combined = product.gather(-1, by_address), combined.gather(-1, by_address)
        chosen = combined.argsort(dim=-1, descending=True, stable=True)[..., :count]
        indices = product.gather(-1, chosen)
        # Canonical address order is required by URM's supplied-route read.
        order = indices.argsort(dim=-1, stable=True)
        indices = indices.gather(-1, order)
        logits = combined.gather(-1, chosen).gather(-1, order)
        weights = logits.softmax(-1)
        return indices, weights

    def retrieve(self, view, requests, hidden):
        idx, weights = self.route(self.q, hidden, self.num_reads)
        values = read(view, requests, idx, weights, self.reference, self.backend).to(hidden.dtype)
        return self.proj(self.read_norm(values).flatten(1))

    def propose(self, view, requests, hidden, mass):
        idx, weights = self.route(self.k, hidden, self.num_writes)
        shape = (hidden.shape[0], self.heads, self.head_dim)
        target = self.v(hidden).view(shape)
        beta = self.beta(hidden).float().sigmoid().unsqueeze(-1)
        g = -F.softplus(self.decay(hidden).float() - 4).unsqueeze(-1)
        return propose_write(view, requests, idx, weights, target, beta, g, mass, self.reference, self.backend)
