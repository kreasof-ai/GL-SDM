"""Sequential token-clock control; chunk.py is the parallel chunk model.

Each token completes its reasoning and transaction before the next token.
This control has no local attention mixer.
"""
import torch
from gl_sdm.memory import merge, commit
from .regularization import sigreg


def forward(block, inputs, cache=None):
    B, T, _ = inputs.shape
    if B < 1 or T < 1:
        raise ValueError("GL-SDM requires a nonempty token batch")
    if cache is None:
        state = block.bank.snapshot(B)
    else:
        block.bank.validate_cache(cache, B)
        state = cache.view
    outputs, depths, ponder = [], [], []
    scale = block.max_steps ** -0.5
    fixed_requests = torch.arange(B, device=inputs.device) if block.halt is None else None
    fixed_mass = inputs.new_full((B,), 1 / block.max_steps, dtype=torch.float32) if block.halt is None else None
    for token in range(T):
        snapshot = state
        x = inputs[:, token]
        h = x
        condition = block.input_proj(x) * scale
        active = torch.ones(B, device=x.device, dtype=torch.bool)
        accumulated = torch.zeros(B, device=x.device)
        weighted = torch.zeros_like(x)
        depth = torch.full((B,), block.max_steps if block.halt is None else 0, device=x.device, dtype=torch.int64)
        remainder = torch.zeros(B, device=x.device)
        proposals = []
        for step in range(block.max_steps):
            # Fixed depth has no data-dependent request selection. Avoid a
            # CUDA-to-host nonzero synchronization at every reasoning step.
            requests = fixed_requests if block.halt is None else active.nonzero().flatten()
            if requests.numel() == 0:
                break
            view = state if block.write_policy == "every_step" else snapshot
            current = h if block.halt is None else h[requests]
            mixed = block.attn.retrieve(view, requests, block.norm1(current))
            updated = current + (condition if block.halt is None else condition[requests]) + mixed * scale
            updated = updated + block.mlp(block.norm2(updated)) * scale
            h = updated if block.halt is None else h.index_copy(0, requests, updated)
            if block.halt is None:
                mass = fixed_mass
                stopped = torch.full_like(mass, step == block.max_steps - 1, dtype=torch.bool)
            else:
                depth = depth.index_add(0, requests, torch.ones_like(requests))
                p = block.halt(block.norm1(updated)).float().sigmoid().flatten()
                remaining = 1 - accumulated[requests]
                stopped = (accumulated[requests] + p >= 1 - block.halt_epsilon) | (step == block.max_steps - 1)
                mass = torch.where(stopped, remaining, p)
                accumulated = accumulated.index_add(0, requests, mass)
                weighted = weighted.index_add(0, requests, (updated.float() * mass[:, None]).to(x.dtype))
                remainder = remainder.index_copy(0, requests, torch.where(stopped, mass, remainder[requests]))
            write_mass = stopped.float() if block.write_policy == "final" else mass
            proposal = block.attn.propose(view, requests, block.norm1(updated), write_mass)
            if block.write_policy == "every_step":
                state = commit(view, merge(view, [proposal]), block.attn.reference, block.attn.backend)
            else:
                proposals.append(proposal)
            if block.halt is not None:
                active = active.index_copy(0, requests, ~stopped)
        if block.write_policy != "every_step":
            state = commit(snapshot, merge(snapshot, proposals), block.attn.reference, block.attn.backend)
        outputs.append(h if block.halt is None else weighted)
        depths.append(depth)
        ponder.append(depth.float() + remainder)
    if cache is not None:
        cache.view = state
        cache.tokens += T
    hidden = torch.stack(outputs, 1)
    block.last_depth = torch.stack(depths, 1).detach()
    # Third auxiliary loss is a summed ACT ponder cost; the training runner
    # weights it explicitly with auxiliary_loss_weight.
    auxiliary = torch.stack(ponder, 1).sum() if block.halt is not None else hidden.new_zeros(())
    return hidden, sigreg(hidden, block.reg_mode, block.sketch_dim), auxiliary
