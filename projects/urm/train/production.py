"""Trainable adapters for pinned production ops, with matched external frontends."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from architectures.mamba import Mamba2K2Layer
from architectures.raven import RavenLayer
from train.registry import _RWKV7Adapter
from architectures.sdm_memory import SparseDeltaMemoryLayer


class SDMProduction(SparseDeltaMemoryLayer):
    """Shared native routes/projections with the clean pinned CUDA state kernel."""

    def __init__(self, model_dim, num_heads, head_dim, intent="training",
                 target="reference", batch_size=None):
        if batch_size is None:
            raise ValueError("SDM requires the microbatch batch size at construction")
        super().__init__(model_dim, num_heads, head_dim, 256, 8, 8, batch_size,
                         intent=intent, target="native", execution="upstream-cuda",
                         chunk_size=64)


class Mamba2Production(Mamba2K2Layer):
    def __init__(self, model_dim, num_heads, head_dim, intent="training", target="reference"):
        super().__init__(model_dim, num_heads, head_dim, 64, intent=intent, target=target)
        # The inherited reference plan is unused; keep frontend projections in
        # the compiled surround, with only SSD itself behind an eager boundary.
        self._plan = None
        from extra.comparators.mamba2_kernel import ssd_kernel
        self._kernel = torch.compiler.disable(ssd_kernel())

    def forward(self, hidden):
        batch, tokens, width = hidden.shape
        heads, dim, state = self.n_heads, self.head_dim, self.d_state
        x, dt, b, c = self.in_proj(hidden).split([width, heads, state, state], dim=-1)
        dt = F.softplus(dt + self.dt_bias)
        out = self._kernel(
            x.reshape(batch, tokens, heads, dim), dt, -self.A_log.exp(),
            b.reshape(batch, tokens, 1, state), c.reshape(batch, tokens, 1, state),
            chunk_size=32,
        )
        return out.reshape(batch, tokens, width)


class _RWKV7Kernel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        from extra.comparators.fla_k2 import fla_op
        self._kernel = torch.compiler.disable(fla_op("fla.ops.rwkv7.chunk_rwkv7"))
        self.dim = dim

    def forward(self, r, w, k, v, a, b):
        out, _ = self._kernel(
            r=r, w=w, k=k, v=v, a=a.to(r.dtype), b=b.to(r.dtype),
            scale=self.dim ** -0.5, chunk_size=16,
        )
        return out


class RWKV7Production(_RWKV7Adapter):
    def __init__(self, model_dim, num_heads, head_dim, intent="training", target="reference"):
        super().__init__(model_dim, num_heads, head_dim, intent, target)
        self._mixer = _RWKV7Kernel(head_dim)


class RavenProduction(RavenLayer):
    def __init__(self, model_dim, num_heads, head_dim, intent="training", target="reference"):
        super().__init__(model_dim, num_heads, head_dim, head_dim, 8, 2,
                         intent=intent, target=target)
        from extra.comparators.fla_k2 import fla_op
        self._kernel = torch.compiler.disable(fla_op("fla.ops.gsa.chunk_gsa"))
        self._stage2 = None  # The original fused production kernel owns both stages.

    def _mixer(self, q, k, v, s, g, scale):
        # GSA's backward tensor-core dot requires at least sixteen slots. Equal
        # duplication leaves each slot logit/state unchanged, splits its softmax
        # mass equally, and preserves the output and all original gradients.
        def time_first(x):
            return x.transpose(1, 2).contiguous()
        out, _ = self._kernel(
            q=time_first(q), k=time_first(k), v=time_first(v),
            s=time_first(torch.cat((s, s), dim=-1)).to(q.dtype),
            g=time_first(torch.cat((g, g), dim=-1)).float(), scale=scale,
        )
        return out.transpose(1, 2)


class _LogLinearKernel(torch.nn.Module):
    num_levels = 4

    def __init__(self):
        super().__init__()
        import importlib
        import triton
        from extra.comparators.fla_k2 import fla_k2_source_identity
        fla_k2_source_identity()
        module = importlib.import_module("fla.ops.log_linear_attn.chunk")
        # Use bf16 operands and a one-stage pipeline on Ampere. The default
        # fp32/two-stage launches exceed the A10G shared-memory limit.
        for entry in vars(module).values():
            kernel = entry
            while hasattr(kernel, "fn"):
                if isinstance(kernel, triton.runtime.Autotuner):
                    kernel.configs = [triton.Config(c.kwargs, num_warps=c.num_warps,
                                                   num_stages=1)
                                      for c in kernel.configs]
                    kernel.cache.clear()
                    break
                kernel = kernel.fn
        self._kernel = torch.compiler.disable(module.chunk_log_linear_attn)

    def forward(self, q, k, v, g, level_scales):
        import math
        batch, tokens, heads, dim = q.shape
        def group(x, dtype=torch.bfloat16):
            return x.transpose(1, 2).reshape(batch * heads, tokens, 1, x.shape[-1]).to(dtype)
        extra_levels = max(0, math.ceil(math.log2(tokens)) + 1 - self.num_levels)
        # The capped native bank retains older aligned blocks in its last slot;
        # all their dyadic levels use the last scale rather than zero padding.
        scales = torch.cat((level_scales, level_scales[..., -1:].expand(
            *level_scales.shape[:-1], extra_levels)), dim=-1)
        out, _ = self._kernel(group(q), group(k), group(v),
                              g.transpose(1, 2).reshape(batch * heads, tokens, 1).float(),
                              group(scales))
        return out.reshape(batch, heads, tokens, dim).transpose(1, 2).to(q.dtype)


def log_linear_production(model_dim, num_heads, head_dim, intent="training", target="reference"):
    from train.registry import _op, _log_linear_ops
    builder = _op("log_linear_attention.BankedLogLinearMixer", layout="bthd",
                  extra=_log_linear_ops, gate_out_dim="heads_dim", num_levels=4)
    layer = builder(model_dim, num_heads, head_dim, intent, target)
    layer._mixer = _LogLinearKernel()
    return layer


def log_linear_mamba2_production(model_dim, num_heads, head_dim, intent="training", target="reference"):
    from train.registry import get_mixer
    layer = get_mixer("log_linear_mamba2").builder(model_dim, num_heads, head_dim, intent, target)
    layer._layer._mixer = _LogLinearKernel()
    return layer


class _DeltaKernel(torch.nn.Module):
    def __init__(self, row):
        super().__init__()
        from extra.comparators.fla_k2 import fla_op
        paths = {
            "comba": "fla.ops.comba.chunk_comba",
            "gdn2": "fla.ops.gdn2.chunk_gdn2",
            "gated_delta_product": "fla.ops.gated_delta_product.chunk_gated_delta_product",
            "dplr": "fla.ops.generalized_delta_rule.dplr.chunk_dplr_delta_rule",
        }
        self.row = row
        self._kernel = torch.compiler.disable(fla_op(paths[row]))

    def forward(self, q, k, v, *args, **kwargs):
        def time_first(x):
            return x.transpose(1, 2).contiguous()
        dtype = q.dtype if q.dtype != torch.float32 else torch.bfloat16
        operands = dict(q=time_first(q).to(dtype), k=time_first(k).to(dtype),
                        v=time_first(v).to(dtype))
        if self.row == "comba":
            operands.update(p=time_first(kwargs["p"]).to(dtype),
                            g=time_first(kwargs["g"]).float(),
                            beta=time_first(kwargs["beta"]).to(dtype))
        elif self.row == "gdn2":
            operands.update(g=time_first(kwargs["g"]).float(),
                            b=time_first(kwargs["erase_gate"]).to(dtype),
                            w=time_first(kwargs["write_gate"]).to(dtype))
        elif self.row == "gated_delta_product":
            g, beta, scale = args
            operands.update(g=time_first(g).float(), beta=time_first(beta).to(dtype),
                            scale=scale, num_householder=k.shape[2] // q.shape[2])
        else:
            alpha, beta, gk = args
            operands.update(a=time_first(alpha).to(dtype), b=time_first(beta).to(dtype),
                            gk=time_first(gk).float(), chunk_size=16)
        out, _ = self._kernel(**operands)
        out = out.transpose(1, 2).to(q.dtype)
        return (out, None) if self.row == "dplr" else out


def delta_production(row):
    def build(model_dim, num_heads, head_dim, intent="training", target="reference"):
        from train.registry import get_mixer
        layer = get_mixer(row).builder(model_dim, num_heads, head_dim, intent, target)
        layer._mixer = _DeltaKernel(row)
        return layer
    return build


class _SDPAPlan:
    def __init__(self):
        self._kernel = F.scaled_dot_product_attention

    def execute(self, *, query, key, value):
        out = self._kernel(query.transpose(1, 2), key.transpose(1, 2),
                           value.transpose(1, 2), is_causal=True)
        return {"output": out.transpose(1, 2)}


def samba_attention_production(model_dim, num_heads, head_dim, intent="training", target="reference"):
    from architectures.samba_attention import SambaAttentionLayer
    layer = SambaAttentionLayer(model_dim, num_heads, head_dim,
                                intent=intent, target=target)
    layer._plan = _SDPAPlan()
    return layer
