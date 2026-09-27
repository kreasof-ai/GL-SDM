"""External model module: BitAttention (arch-014).

Verified composition-now row: external FusedBitLinear projections produce
q/k/v (and the output projection), and the mixer is plain causal softmax
attention (U1.S) with a head map. The sweep (row-014) confirmed against
fla/layers/bitattn.py @ 864a87f6 that the BitLinear quantization and its
gradient policy live entirely in the external module; the K1 call sees
ordinary dense q/k/v tensors.

The projection equation is transcribed from the pinned fla source
(fla/modules/fused_bitlinear.py):

    x_norm = RMSNorm(x)                       # fp32 norm, eps pinned at 1e-8
    x_q    = activation_quant(x_norm)         # per-token 8-bit round/clip
    w_q    = weight_quant(w)                  # per-tensor 1.58-bit round/clip
    y      = x_q @ w_q^T

with the pinned straight-through gradient policy (identity through the
rounding, real gradients through RMSNorm). The mixer is the same typed K1
``weighted_reduce`` graph as the head-map cluster, executed through the public
compile path. Rotary embedding and KV-cache/decode modes of the pinned source
are external residual work (sweep blocker), not claimed here.
"""

from __future__ import annotations

import torch

from urm.compiler.normalize.graph import normalize_graph_document
from urm.compiler.pipeline import CompilationIntent, compile_graph
from urm.frontend.recipes import load_graph_recipe_document


def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Per-token 8-bit quantization, transcribed from the pinned fla source."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    return (x * scale).round().clamp(-128, 127) / scale


def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor 1.58-bit quantization, transcribed from the pinned fla source."""
    scale = 1.0 / w.abs().mean().clamp(min=1e-5)
    return (w * scale).round().clamp(-1, 1) / scale


class BitLinear(torch.nn.Linear):
    """RMSNorm + quantized linear, matching the pinned fla BitLinear semantics.

    Forward values are the closed form of the pinned fused kernel; gradients
    use the pinned straight-through policy (identity through the rounding
    operators, real gradients through the RMS norm and both quant operands).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, norm_eps: float = 1e-8):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = torch.nn.RMSNorm(in_features, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x.to(torch.float32)).to(x.dtype)
        # Straight-through estimators: values quantized, gradients identity.
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
        return torch.nn.functional.linear(x_quant, w_quant, self.bias)


class BitAttentionLayer(torch.nn.Module):
    """One BitAttention layer: BitLinear Q/K/V/O projections + typed K1 mixer.

    Mirrors the pinned fla BitAttention module contract for the prefill path:
    causal softmax over dense q/k/v with a query→KV head map (GQA-capable).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        *,
        target: str = "reference",
        intent: str = "inference",
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        kv_dim = num_kv_heads * self.head_dim

        self.q_proj = BitLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = BitLinear(hidden_size, kv_dim, bias=False)
        self.v_proj = BitLinear(hidden_size, kv_dim, bias=False)
        self.o_proj = BitLinear(hidden_size, hidden_size, bias=False)

        if num_kv_heads == num_heads:
            head_map = "equal"
        elif num_kv_heads == 1:
            head_map = "single"
        else:
            head_map = "grouped"
        params: dict[str, object] = {
            "query_domain": "sequence",
            "source_domain": "sequence",
            "selection": "dense",
            "normalization": "softmax",
            "capacity_policy": "dropless",
            "deterministic": True,
            "causal": True,
            "head_map": head_map,
        }
        if head_map == "grouped":
            params["group_size"] = num_heads // num_kv_heads

        document = {
            "schema_version": 2,
            "name": "bit_attention_mixer",
            "kind": "kernel_fragment",
            "graph": {
                "inputs": [
                    {"name": "query", "dtype": "float32",
                     "shape": ["B", "T", num_heads, self.head_dim]},
                    {"name": "key", "dtype": "float32",
                     "shape": ["B", "S", num_kv_heads, self.head_dim]},
                    {"name": "value", "dtype": "float32",
                     "shape": ["B", "S", num_kv_heads, self.head_dim]},
                ],
                "nodes": [
                    {
                        "id": "attend",
                        "op": "weighted_reduce",
                        "inputs": ["query", "key", "value"],
                        "outputs": ["output"],
                        "params": params,
                    }
                ],
                "outputs": ["output"],
            },
        }
        recipe = load_graph_recipe_document(document)
        program = normalize_graph_document(recipe.document)
        self._plan = compile_graph(
            program, target=target, intent=CompilationIntent(intent)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill path of the pinned contract: ``[B, T, hidden_size]`` → same."""
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim)
        out = self._plan.execute(query=q, key=k, value=v)["output"]
        return self.o_proj(out.reshape(B, T, self.num_heads * self.head_dim))


__all__ = ["BitAttentionLayer", "BitLinear", "activation_quant", "weight_quant"]
