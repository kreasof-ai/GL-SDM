"""Plain SDPA surround and thin adapters to the authors' SDM and FLA layers."""
import torch
from torch import nn
import torch.nn.functional as F
from .upstream import sdm_layer, verified_import


class Transformer(nn.Module):
    """ATMA RoPE softmax surround, used in every block with ordinary PyTorch."""
    def __init__(self, cfg, layer_idx):
        super().__init__()
        dim, dk = cfg["hidden_size"], cfg["head_dim"]
        self.head_dim, self.num_heads = dk, dim // dk
        self.num_kv_heads = cfg.get("num_key_value_heads", self.num_heads)
        if self.num_heads % self.num_kv_heads or dk % 4:
            raise ValueError("head_dim must be divisible by 4; KV heads must divide query heads")
        self.q = nn.Linear(dim, 2 * dim)
        self.k = nn.Linear(dim, self.num_kv_heads * dk)
        self.v = nn.Linear(dim, self.num_kv_heads * dk)
        self.proj = nn.Linear(dim, dim)
        freq = (1 / 1024) ** torch.linspace(0, 1, dk // 4)
        self.register_buffer("angular_freq", torch.cat((freq, torch.zeros_like(freq))))
        self.reference = False

    def _apply(self, fn, recurse=True):
        # ATMA's rotary frequencies stay FP32 even with BF16 projections.
        frequencies = self.angular_freq
        super()._apply(fn, recurse)
        self.angular_freq = frequencies.to(device=self.angular_freq.device)
        return self

    def rotary(self, x, offset):
        positions = torch.arange(offset, offset + x.shape[1], device=x.device, dtype=torch.float32)
        theta = torch.outer(positions, self.angular_freq.float())[None, :, None]
        a, b = x.float().chunk(2, -1)
        return torch.cat((a * theta.cos() + b * theta.sin(), -a * theta.sin() + b * theta.cos()), -1).to(x.dtype)

    def forward(self, x, cache=None):
        B, T, D = x.shape
        q, gate = self.q(x).view(B, T, self.num_heads, 2 * self.head_dim).chunk(2, -1)
        k = self.k(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v(x).view_as(k).transpose(1, 2)
        offset = 0 if cache is None or not cache else cache["k"].shape[2]
        q = self.rotary(F.rms_norm(q, (self.head_dim,)), offset).transpose(1, 2)
        k = self.rotary(F.rms_norm(k, (self.head_dim,)), offset).transpose(1, 2)
        if cache is not None:
            if cache:
                k, v = torch.cat((cache["k"], k), 2), torch.cat((cache["v"], v), 2)
            cache.update(k=k, v=v)
        groups = self.num_heads // self.num_kv_heads
        if groups != 1:
            k, v = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
        # Cached multi-token prefill needs an offset causal mask. SDPA's default
        # causal mask is aligned to the upper left, which is wrong here.
        mask = None
        if offset or self.reference:
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= torch.arange(offset, offset + T, device=x.device)[:, None]
        if self.reference:
            scores = (q.float() @ k.float().transpose(-1, -2)) * self.head_dim ** -0.5
            out = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ v.float()
            out = out.to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        return self.proj(out.transpose(1, 2).reshape(B, T, D) * gate.reshape(B, T, D).sigmoid())


def make_sdm(cfg, layer_idx):
    upstream, _ = sdm_layer()
    ops = verified_import("sdm", "lingua.sparse_delta_memory.memory_ops")

    class SDM(upstream.SparseDeltaMemory):
        # Only adapt the launch ABI: preserve the full upstream projections,
        # gates, router, learned initial memory and output processing.
        reference = False

        def gated_write_read(self, memory, ki, kw, v, beta, g, qi, qw, grad_final_memory=None):
            P, T, D = v.shape
            if self.reference:
                from .reference import sdm
                output, final = sdm(memory, ki, kw, v, beta, g, qi, qw)
                return output, final.to(memory.dtype)
            if not self.training and T == 1:
                with torch.autocast(v.device.type, enabled=False):
                    out = ops.fused_decode_step(memory, ki[:, 0], kw[:, 0], v[:, 0], beta[:, 0], g[:, 0], qi[:, 0], qw[:, 0], use_delta_rule=True, normalize_memory=False, key_weighted_decay=False)
                return out.view(P, 1, D), memory
            chunk = self.args.memory_block_size
            pad = (-T) % chunk
            # The upstream layer adds partition offsets before this call. Pad
            # each partition to its own first slot; padding is an identity write.
            offsets = torch.arange(P, device=v.device)[:, None, None] * self.slots_per_head
            def index(z):
                return (F.pad(z - offsets, (0, 0, 0, pad)) + offsets).reshape(-1, z.shape[-1]).contiguous()
            def operand(z):
                return F.pad(z, (0, 0, 0, pad)).reshape(-1, z.shape[-1]).contiguous()
            args = (memory, index(ki), operand(kw), operand(v), operand(beta), operand(g), index(qi), operand(qw))
            with torch.autocast(v.device.type, enabled=False):
                if self.training:
                    out, _ = ops.GatedSparseMemoryWriteRead.apply(*args, chunk, True, self.slots_per_head, P, False, "none", grad_final_memory)
                else:
                    out = ops.gated_sparse_memory_write_read_inference_v2(*args, chunk, self.slots_per_head, P)
            return out.view(P, T + pad, D)[:, :T], memory

    args = upstream.SparseDeltaMemoryArgs(
        dim=cfg["hidden_size"], num_heads=cfg["hidden_size"] // cfg["head_dim"],
        slots_per_head=cfg.get("sdm_slots", 1024), num_reads=cfg.get("sdm_reads", 8),
        num_writes=cfg.get("sdm_writes", 8), memory_block_size=cfg.get("sdm_chunk", 64),
        backprop_on_memory=True, output_gate=cfg.get("sdm_output_gate", True),
        normalize_readings=cfg.get("sdm_normalize_readings", True), query_batchnorm=False,
    )
    if args.memory_block_size < 2 or args.num_reads > args.slots_per_head or args.num_writes > args.slots_per_head:
        raise ValueError("invalid SDM chunk or route size")
    layer = SDM(args, layer_idx)
    layer.init_weights()
    return layer


def make_mixer(cfg, layer_idx):
    arch = cfg["arch_type"]
    if arch == "transformer":
        return Transformer(cfg, layer_idx)
    if arch == "sdm":
        return make_sdm(cfg, layer_idx)
    if arch == "gdn2":
        return verified_import("fla", "fla.layers.gdn2").GatedDeltaNet2(
            hidden_size=cfg["hidden_size"], head_dim=cfg["head_dim"],
            num_heads=cfg["hidden_size"] // cfg["head_dim"], expand_v=1.0,
            use_short_conv=cfg.get("gdn2_short_conv", True), conv_size=cfg.get("gdn2_conv_size", 4),
            mode="chunk", layer_idx=layer_idx,
        )
    raise ValueError(f"unknown arch_type={arch!r}; choose transformer, sdm, gdn2")
