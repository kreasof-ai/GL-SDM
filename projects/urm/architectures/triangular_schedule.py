"""Block forward substitution using the existing public triangular-solve op.

Cross-block corrections are ordinary matmuls; dependencies within each block
stay in the native solve. No inverse or backend kernel is replaced.
"""
from __future__ import annotations

import torch


def solve_in_blocks(plan, probs, beta, value, output_name, block_size=32):
    tokens = value.shape[-2]
    if tokens <= block_size:
        return plan.execute(probs=probs, beta=beta, value=value)[output_name]
    outputs = []
    with torch.autocast(value.device.type, enabled=False):
        probs, beta, value = probs.float(), beta.float(), value.float()
        # Factor all diagonal blocks in one public native call. The small
        # inverse blocks let the cross-block schedule use batched matmuls rather
        # than launching another serial kernel for every block.
        import torch.nn.functional as F
        blocks = (tokens + block_size - 1) // block_size
        padding = blocks * block_size - tokens
        padded_probs = F.pad(probs, (0, padding, 0, padding))
        batch, heads = probs.shape[:2]
        diagonal = padded_probs.reshape(batch, heads, blocks, block_size, blocks, block_size)
        diagonal = diagonal.diagonal(dim1=2, dim2=4).permute(0, 1, 4, 2, 3)
        diagonal = diagonal.reshape(batch * heads * blocks, 1, block_size, block_size).contiguous()
        block_beta = F.pad(beta, (0, padding)).reshape(batch * heads * blocks, 1, block_size)
        eye = torch.eye(block_size, device=value.device).expand_as(diagonal).contiguous()
        inverses = plan.execute(probs=diagonal, beta=block_beta, value=eye)[output_name]
        inverses = inverses.reshape(batch, heads, blocks, block_size, block_size)
        for start in range(0, tokens, block_size):
            end = min(start + block_size, tokens)
            rhs = value[..., start:end, :]
            if start:
                prefix = torch.cat(outputs, dim=-2)
                rhs = rhs - beta[..., start:end, None] * (
                    probs[..., start:end, :start] @ prefix
                )
            size = end - start
            outputs.append(inverses[:, :, start // block_size, :size, :size] @ rhs)
    return torch.cat(outputs, dim=-2)
