"""Plain SDPA surround and thin adapters to the authors' SDM and FLA layers."""
import torch
import torch.nn.functional as F
from .upstream import sdm_layer, verified_import


from gl_sdm.layers.attention import Transformer


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
                from gl_sdm.baselines.reference import sdm
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
