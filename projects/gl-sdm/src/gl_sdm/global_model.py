"""GL-SDM: a tied reasoner, one global bank and token transactions.

This is a PyTorch architecture implementation, not an optimized serving kernel.
Tokens execute in causal order. Recurrence is contained inside one ATMA block,
so a caller traversing embed/blocks/norm/proj cannot leak future-token writes.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .memory import MemoryBank, read, propose_write, merge, commit
from .model import MLP, RMSNorm
from .regularization import sigreg


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
        self.reference = False

    def route(self, projection, hidden, count):
        scores = projection(hidden).float().view(-1, self.heads, 2, self.half)
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
        weights = combined.gather(-1, chosen).softmax(-1)
        return indices, weights

    def retrieve(self, view, requests, hidden):
        idx, weights = self.route(self.q, hidden, self.num_reads)
        values = read(view, requests, idx, weights, self.reference).to(hidden.dtype)
        return self.proj(self.read_norm(values).flatten(1))

    def propose(self, view, requests, hidden, mass):
        idx, weights = self.route(self.k, hidden, self.num_writes)
        shape = (hidden.shape[0], self.heads, self.head_dim)
        target = self.v(hidden).view(shape)
        beta = self.beta(hidden).float().sigmoid().unsqueeze(-1)
        g = -F.softplus(self.decay(hidden).float() - 4).unsqueeze(-1)
        return propose_write(view, requests, idx, weights, target, beta, g, mass, self.reference)


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

    def forward(self, inputs, cache=None):
        B, T, _ = inputs.shape
        if B < 1 or T < 1:
            raise ValueError("GL-SDM requires a nonempty token batch")
        if cache is None:
            state = self.bank.snapshot(B)
        else:
            self.bank.validate_cache(cache, B)
            state = cache.view
        outputs, depths, ponder = [], [], []
        scale = self.max_steps ** -0.5
        for token in range(T):
            snapshot = state
            x = inputs[:, token]
            h = x
            condition = self.input_proj(x) * scale
            active = torch.ones(B, device=x.device, dtype=torch.bool)
            accumulated = torch.zeros(B, device=x.device)
            weighted = torch.zeros_like(x)
            depth = torch.zeros(B, device=x.device, dtype=torch.int64)
            remainder = torch.zeros(B, device=x.device)
            proposals = []
            for step in range(self.max_steps):
                requests = active.nonzero().flatten()
                if requests.numel() == 0:
                    break
                view = state if self.write_policy == "every_step" else snapshot
                current = h[requests]
                mixed = self.attn.retrieve(view, requests, self.norm1(current))
                updated = current + condition[requests] + mixed * scale
                updated = updated + self.mlp(self.norm2(updated)) * scale
                h = h.index_copy(0, requests, updated)
                depth = depth.index_add(0, requests, torch.ones_like(requests))
                if self.halt is None:
                    mass = x.new_full((requests.numel(),), 1 / self.max_steps, dtype=torch.float32)
                    stopped = torch.full_like(mass, step == self.max_steps - 1, dtype=torch.bool)
                else:
                    p = self.halt(self.norm1(updated)).float().sigmoid().flatten()
                    remaining = 1 - accumulated[requests]
                    stopped = (accumulated[requests] + p >= 1 - self.halt_epsilon) | (step == self.max_steps - 1)
                    mass = torch.where(stopped, remaining, p)
                    accumulated = accumulated.index_add(0, requests, mass)
                    weighted = weighted.index_add(0, requests, (updated.float() * mass[:, None]).to(x.dtype))
                    remainder = remainder.index_copy(0, requests, torch.where(stopped, mass, remainder[requests]))
                write_mass = stopped.float() if self.write_policy == "final" else mass
                proposal = self.attn.propose(view, requests, self.norm1(updated), write_mass)
                if self.write_policy == "every_step":
                    state = commit(view, merge(view, [proposal]), self.attn.reference)
                else:
                    proposals.append(proposal)
                active = active.index_copy(0, requests, ~stopped)
            if self.write_policy != "every_step":
                state = commit(snapshot, merge(snapshot, proposals), self.attn.reference)
            outputs.append(h if self.halt is None else weighted)
            depths.append(depth)
            ponder.append(depth.float() + remainder)
        if cache is not None:
            cache.view = state
            cache.tokens += T
        hidden = torch.stack(outputs, 1)
        self.last_depth = torch.stack(depths, 1).detach()
        # Third auxiliary loss is a summed ACT ponder cost; the training runner
        # weights it explicitly with auxiliary_loss_weight.
        auxiliary = torch.stack(ponder, 1).sum() if self.halt is not None else hidden.new_zeros(())
        return hidden, sigreg(hidden, self.reg_mode, self.sketch_dim), auxiliary

    def reasoning_metrics(self, depth=None):
        depth = (self.last_depth if depth is None else depth).float()
        return {"mean_reasoning_steps": depth.mean().item(), "p95_reasoning_steps": depth.quantile(0.95).item(), "max_reasoning_steps": depth.max().item(), "executed_reasoner_calls": int(depth.sum().item()), "gl_write_policy": self.write_policy}
