"""Pure tensor arithmetic and its compiled execution adapters.

ChunkOps binds the existing block weights without registering new modules or
changing checkpoint keys. Compilation preserves the eager rounding casts.
"""
import torch
import torch.nn.functional as F
from gl_sdm.runtime.compilation import compiled


def proposal_delta(rows, weights, prediction, target, beta, log_decay, mass):
    decay = log_decay.exp()
    error = beta * (target.float() - decay * prediction)
    return mass[..., None, None, None] * ((decay[..., None] - 1) * rows + weights[..., None] * error[..., None, :])


def dense_step(current, condition, readings, scale, norm2, fcw, fcb, pw, pb,
               read_norm, ow, ob, norm1, hw, hb, head_dim):
    # Pure tensor arithmetic keeps model parameter names and checkpoint ABI.
    r = F.rms_norm(readings.to(current.dtype), (head_dim,), read_norm, eps=1e-6)
    updated = current + condition + F.linear(r.flatten(-2), ow, ob) * scale
    z = F.rms_norm(updated, (updated.shape[-1],), norm2, eps=1e-6)
    x, gate = F.linear(z, fcw, fcb).chunk(2, -1)
    updated = updated + F.linear(gate * x.relu().square(), pw, pb) * scale
    z = F.rms_norm(updated, (updated.shape[-1],), norm1, eps=1e-6)
    halt = F.linear(z, hw, hb).float().sigmoid().flatten() if hw is not None else None
    return updated, z, halt


def write_coefficients(z, weight, bias, heads, dim, score_width):
    pieces = F.linear(z, weight, bias).split([heads * score_width, heads * dim, heads, heads], -1)
    shape = z.shape[:-1]
    return (pieces[0].float().reshape(*shape, heads, score_width),
        pieces[1].reshape(*shape, heads, dim), pieces[2].float().sigmoid().unsqueeze(-1),
        -F.softplus(pieces[3].float() - 4).unsqueeze(-1))


_compiled_step = compiled(dense_step)
_compiled_delta = compiled(proposal_delta)
_compiled_write = compiled(write_coefficients)


class ChunkOps:
    """Choose eager/compiled arithmetic once per chunk forward call."""
    def __init__(self, block, inputs):
        self.block = block
        use_compiled = block.compile_dense and inputs.is_cuda and not block.attn.reference
        self.step_fn = _compiled_step if use_compiled else dense_step
        self.delta = _compiled_delta if use_compiled else proposal_delta
        self.write_fn = _compiled_write if use_compiled else write_coefficients
        router = block.attn
        projections = [router.k, router.v, router.beta, router.decay]
        self.write_weight = torch.cat([p.weight for p in projections])
        self.write_bias = torch.cat([p.bias for p in projections])

    def step(self, current, condition, readings, scale):
        block, router = self.block, self.block.attn
        return self.step_fn(current, condition, readings, scale,
            block.norm2.weight, block.mlp.fc.weight, block.mlp.fc.bias,
            block.mlp.proj.weight, block.mlp.proj.bias, router.read_norm.weight,
            router.proj.weight, router.proj.bias, block.norm1.weight,
            None if block.halt is None else block.halt.weight,
            None if block.halt is None else block.halt.bias, router.head_dim)

    def write(self, normalized):
        router = self.block.attn
        return self.write_fn(normalized, self.write_weight, self.write_bias,
            router.heads, router.head_dim, 2 * router.half)
