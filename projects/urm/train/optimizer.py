"""Optimizers for the URM training harness (ATMA pattern).

Muon (Newton-Schulz orthogonalized momentum) for the >=2-D block weights and AdamW
(with fp32 master state) for the embeddings/head/1-D params — the modded-nanogpt /
ATMA split. Both are ordinary torch optimizers over a model's ``parameters()``; they
carry no mixer-specific knowledge.
"""

from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 12) -> Tensor:
    """Newton-Schulz orthogonalization (bf16 iterations, not wallclock-tuned)."""
    assert G.ndim >= 2
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 2, -1.5, 0.5
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def muon_update(grad: Tensor, momentum: Tensor, mu: float = 0.95, nesterov: bool = True) -> Tensor:
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class Muon(torch.optim.Optimizer):
    """Muon for the >=2-D block weights (modded-nanogpt / ATMA)."""

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0, mu: float = 0.95):
        params = list(params)
        assert params and all(isinstance(p, torch.nn.Parameter) for p in params)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, mu=mu))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:  # skip params unused this step (e.g. dropped branches)
                    continue
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"])


def build_optimizers(model: torch.nn.Module, *, muon_lr: float = 0.005,
                     embed_lr: float = 0.02, head_lr: float = 0.002,
                     scalar_lr: float = 0.003, weight_decay: float = 0.01):
    """The AdamW (embed/head/1-D) + Muon (block >=2-D) split used across the harness.

    Asserts every model parameter is claimed by exactly one optimizer (the ATMA
    coverage check), so no parameter silently trains at the wrong rate or not at all.
    """
    named = dict(model.named_parameters())
    embed_params = [p for n, p in named.items() if n.startswith("token.") or n.startswith("position.")]
    head_params = [p for n, p in named.items() if n.startswith("lm_head.")]
    block_matrix = [p for n, p in named.items()
                    if n.startswith("blocks.") and p.ndim >= 2]
    others = [p for n, p in named.items()
              if p not in {*embed_params, *head_params, *block_matrix}]

    adamw_groups = []
    if embed_params:
        adamw_groups.append(dict(params=embed_params, lr=embed_lr))
    if head_params:
        adamw_groups.append(dict(params=head_params, lr=head_lr))
    if others:
        adamw_groups.append(dict(params=others, lr=scalar_lr))
    optimizers = []
    if adamw_groups:
        optimizers.append(torch.optim.AdamW(adamw_groups, betas=(0.8, 0.95),
                                            eps=1e-10, weight_decay=0, fused=torch.cuda.is_available()))
    if block_matrix:
        optimizers.append(Muon(block_matrix, lr=muon_lr, weight_decay=weight_decay))

    claimed = {p for opt in optimizers for group in opt.param_groups for p in group["params"]}
    assert claimed == set(model.parameters()), (
        f"optimizer coverage mismatch: {len(set(model.parameters()) - claimed)} params unclaimed"
    )
    return optimizers


__all__ = ["Muon", "build_optimizers", "muon_update", "zeropower_via_newtonschulz5"]
