"""External model module: Pattention / parameter-token attention (arch-057).

Verified composition-now row as a typed-domain composition (sweep row-057,
against megatron/model/tokenformer.py @ 4d56c73f): the mixer is attention over
a fixed parameter-token source domain — ``scores = query @ key_paramᵀ ×
scale``; a parameter-domain nonlinear normalizer over the parameter tokens
(``softmax`` = exp+L1-normalize×count; ``gelu_l2_norm``; ``l2_norm_gelu``);
``output = norm(scores) @ value_param``. The parameter tokens are learned
``nn.Parameter`` tensors, so gradients flow to them through the contraction.

The pinned score stage is a plain inner product (NO softmax denominator), so
this module composes typed ``matmul`` calls around the external normalizer —
the closed K1 softmax contract does not cover a non-softmax score stage, and
forcing one would be a descriptor hack (sweep row-057's correction).
Parameter-token construction/reparameterization and the model replacement
(Pattention replacing QKV/output projections and the MLP) are external; "one
attention contraction equals MLP/MoE" is explicitly not claimed (sweep
blocker).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def nonlinear_norm_func(inputs: torch.Tensor, normalize_type: str, dim: int = -1) -> torch.Tensor:
    """Transcribed from the pinned tokenformer.py nonlinear_norm_func."""
    if normalize_type == "softmax":
        # Pinned: softmax = exp then L1-normalize scaled by the domain count.
        nonlinear_outputs = torch.exp(inputs)
        return nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
    if normalize_type == "gelu_l2_norm":
        nonlinear_outputs = F.gelu(inputs)
        return nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
    if normalize_type == "l2_norm_gelu":
        norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
        return F.gelu(norm_outputs)
    raise NotImplementedError(f"unknown normalize_type {normalize_type!r}")


class PattentionLayer(torch.nn.Module):
    """One Pattention layer: parameter tokens + typed matmuls + external normalizer.

    The query is the input; key/value are learned parameter tokens
    ``[param_token_num, key/value_dim]``. The two contractions (scores and
    output) are typed ``matmul`` ops; the parameter-domain normalizer is the
    external nonlinear stage.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        param_token_num: int,
        *,
        norm_activation_type: str = "softmax",
    ) -> None:
        super().__init__()
        if norm_activation_type not in ("softmax", "gelu_l2_norm", "l2_norm_gelu"):
            raise ValueError(f"unsupported norm_activation_type {norm_activation_type!r}")
        self.param_token_num = param_token_num
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = norm_activation_type

        self.key_param_tokens = torch.nn.Parameter(torch.rand(param_token_num, input_channels))
        self.value_param_tokens = torch.nn.Parameter(torch.rand(param_token_num, output_channels))

    def forward(
        self,
        inputs: torch.Tensor,
        router_index: torch.Tensor | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """``inputs`` ``[..., L, key_dim]`` → ``[..., L, value_dim]``.

        ``router_index`` selects a subset of parameter tokens (the pinned MoE
        mode); ``scale`` is the pinned ``scale_factor`` (default 1).
        """
        query = inputs
        if router_index is None:
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]

        scale_factor = 1 if scale is None else scale
        # Typed contraction: scores = query @ keyᵀ × scale (plain inner product).
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        # External parameter-domain nonlinear normalizer.
        attn_weight = nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        # Typed contraction: output = norm(scores) @ value.
        return torch.matmul(attn_weight, value)


__all__ = ["PattentionLayer", "nonlinear_norm_func"]
