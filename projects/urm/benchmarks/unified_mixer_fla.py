"""Parity and paired profiles for unified mixer plans against pinned FLA calls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import fla
from fla.ops.delta_rule import chunk_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gla import chunk_gla
from fla.ops.linear_attn import chunk_linear_attn
from fla.ops.simple_gla import chunk_simple_gla
from fla.ops.retention import fused_chunk_retention
from fla.ops.lightning_attn import chunk_lightning_attn
from fla.ops.simple_gla import fused_chunk_simple_gla
from fla.ops.gdn2 import chunk_gdn2
from fla.ops.based import fused_chunk_based
from fla.ops.rebased import parallel_rebased
from fla.ops.log_linear_attn import chunk_log_linear_attn
from fla.ops.kda import chunk_kda
from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.rwkv4 import fused_recurrent_rwkv4
from fla.ops.rwkv6 import fused_recurrent_rwkv6
from fla.ops.momentum_delta_rule.chunk import chunk_momentum_delta_rule
from fla.ops.rwkv7 import chunk_rwkv7
from fla.ops.dsa.naive import naive_dsa
from fla.ops.nsa.parallel import parallel_nsa
from fla.ops.forgetting_attn.parallel import parallel_forgetting_attn
from fla.ops.parallax.parallel import parallel_parallax
from fla.ops.wall_attn.parallel import parallel_wall_attn
from fla.ops.moba.parallel import parallel_moba
from fla.ops.path_attn.parallel import parallel_path_attn
from fla.ops.gated_oja_rule import chunk_gated_oja_rule
from fla.ops.comba import chunk_comba
from fla.ops.precond_gated_delta_rule.chunk import chunk_precond_gated_delta_rule
from fla.ops.precond_kda.chunk import chunk_precond_kda
from fla.ops.abc import chunk_abc
from fla.ops.gsa import chunk_gsa

from measurement import quantile
from provenance import provenance, write_artifact
from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer
from urm.frontend.mixer_recipes import named_mixer_recipe

EXPECTED_FLA_REVISION = "864a87f6ce5be8828bef81eb22baafd41937cdf2"
EXPECTED_FLASH_ATTN_REVISION = "1bda8f9290cd48d030f1516f0e680cd464ef3554"
RECIPES = (
    "linear_attention",
    "simple_gla",
    "gla",
    "rodimus_gla_core",
    "retention_core",
    "lightning_attention_core",
    "lightnet_gla_core",
    "hgrn2_ssm_core",
    "delta_net",
    "gated_delta_net",
    "gdn2_core",
    "gated_oja_core",
    "comba_core",
    "pgdn_core",
    "pkda_core",
    "abc_core",
    "gsa_core",
    "based_attention_core",
    "rebased_attention_core",
    "log_linear_attention_core",
    "kda_core",
    "gated_delta_product_core",
    "generalized_delta_iplr_core",
    "generalized_delta_dplr_core",
    "rwkv4_memory_core",
    "rwkv6_memory_core",
    "momentum_delta_core",
    "mesa_net_core",
    "titans_linear_memory_core",
    "ttt_linear_core",
    "rwkv7_transition_core",
    "path_attention_core",
    "deltaformer_attention_core",
    "fox",
    "parallax_attention_core",
    "wall_attention_core",
    "moba_selected_attention_core",
    "dsa_attention_core",
    "nsa_selected_attention_core",
)


def _loaded_revision() -> str | None:
    root = Path(fla.__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _moba_flash_attn_identity() -> dict[str, object]:
    import flash_attn
    import flash_attn_2_cuda

    package_file = Path(flash_attn.__file__).resolve()
    repository = next(
        (parent for parent in package_file.parents if (parent / ".git").exists()),
        None,
    )
    if repository is None:
        raise RuntimeError("could not identify the nested FlashAttention source checkout")
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    if revision != EXPECTED_FLASH_ATTN_REVISION or dirty:
        raise RuntimeError(
            "MoBA requires a clean pinned FlashAttention source checkout; "
            f"got {revision} with dirty={bool(dirty)}"
        )
    extension = Path(flash_attn_2_cuda.__file__).resolve()
    identity_path = extension.parent.parent / "build_identity.json"
    if not identity_path.is_file():
        raise RuntimeError(
            "MoBA requires build_identity.json beside the narrow comparator extension; "
            "run benchmarks/build_flash_attn_moba.py"
        )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("extension_sha256") is None or identity.get("revision") != revision:
        raise RuntimeError("nested FlashAttention build identity does not match its source")
    return {
        "repository": "https://github.com/Dao-AILab/flash-attention",
        "revision": revision,
        "source_checkout": str(repository),
        "python_module": str(package_file),
        "extension": str(extension),
        "extension_sha256": identity["extension_sha256"],
        "source_api_sha256": identity["source_api_sha256"],
        "narrowed_dispatch_sha256": identity["narrowed_dispatch_sha256"],
        "kernel_source_sha256": identity["kernel_source_sha256"],
        "supported_contract": identity["supported_contract"],
    }


def _moba_attention_mask(
    query: torch.Tensor, key: torch.Tensor, chunk_size: int, topk: int
) -> torch.Tensor:
    batch, sequence, heads, _ = query.shape
    if batch != 1 or sequence % chunk_size:
        raise ValueError("the MoBA profile requires one evenly chunked sequence")
    num_chunks = sequence // chunk_size
    target_chunks = num_chunks - 1
    block_keys = key[0].view(num_chunks, chunk_size, heads, -1)[:-1].mean(dim=1).float()
    gate = torch.einsum("nhd,thd->nht", block_keys, query[0].float())
    positions = torch.arange(sequence, device=query.device)
    block_ends = (torch.arange(target_chunks, device=query.device) + 1) * chunk_size
    gate.masked_fill_(positions[None, None, :] < block_ends[:, None, None], -float("inf"))
    selected_count = min(topk - 1, target_chunks)
    selected_indices = torch.topk(
        gate, k=selected_count, dim=0, largest=True, sorted=False
    ).indices
    finite_routes = ~torch.isinf(gate)
    selected = torch.zeros_like(finite_routes).scatter_(0, selected_indices, True)
    selected &= finite_routes
    local_causal = (
        (positions[:, None] // chunk_size == positions[None, :] // chunk_size)
        & (positions[None, :] <= positions[:, None])
    )
    mask = local_causal.view(1, 1, sequence, sequence).expand(
        batch, heads, -1, -1
    ).clone()
    for chunk_idx in range(target_chunks):
        start = chunk_idx * chunk_size
        stop = start + chunk_size
        mask[:, :, :, start:stop] |= selected[chunk_idx].view(
            1, heads, sequence, 1
        )
    return mask


def _inputs(recipe: str, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if recipe == "ttt_linear_core":
        batch, sequence, heads, dim = 1, 64, 2, 16
        dtype = torch.bfloat16
        return {
            "query": (torch.randn(batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator) * 0.1).requires_grad_(),
            "key": torch.nn.functional.normalize(
                torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator), dim=-1
            ).to(dtype).requires_grad_(),
            "value": (torch.randn(batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator) * 0.1).requires_grad_(),
            "w": (torch.ones(heads, dim, device="cuda", dtype=dtype) + torch.randn(heads, dim, device="cuda", dtype=dtype, generator=generator) * 0.01).requires_grad_(),
            "b": (torch.randn(heads, dim, device="cuda", dtype=dtype, generator=generator) * 0.01).requires_grad_(),
            "eta": (torch.randn(batch, sequence, heads, 1, device="cuda", dtype=dtype, generator=generator) * 0.005).requires_grad_(),
            "initial_state": (torch.randn(batch, heads, dim, dim, device="cuda", generator=generator) * 0.01).requires_grad_(),
            "initial_state_bias": (torch.randn(batch, heads, 1, dim, device="cuda", generator=generator) * 0.01).requires_grad_(),
        }
    if recipe == "titans_linear_memory_core":
        batch, sequence, heads, dim = 1, 64, 1, 16
        query = torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator) * 0.1
        key = torch.nn.functional.normalize(
            torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator), dim=-1
        )
        value = torch.randn(batch, sequence, heads, dim, device="cuda", generator=generator) * 0.1
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "w": (torch.ones(heads, dim, device="cuda") + torch.randn(heads, dim, device="cuda", generator=generator) * 0.01).requires_grad_(),
            "b": (torch.randn(heads, dim, device="cuda", generator=generator) * 0.01).requires_grad_(),
            "theta": (torch.rand(batch, sequence, heads, 1, device="cuda", generator=generator) * 0.05 + 0.05).requires_grad_(),
            "alpha": (torch.rand(batch, sequence, heads, 1, device="cuda", generator=generator) * 0.05 + 0.05).requires_grad_(),
            "eta": (torch.rand(batch, sequence, heads, 1, device="cuda", generator=generator) * 0.05 + 0.9).requires_grad_(),
            "initial_state": (torch.randn(batch, heads, dim, dim, device="cuda", generator=generator) * 0.01).requires_grad_(),
        }
    if recipe == "rodimus_gla_core":
        batch, sequence, heads, key_dim, value_dim = 1, 64, 1, 64, 128
        dtype = torch.bfloat16
        query = torch.randn(
            batch, sequence, heads, key_dim, device="cuda", dtype=dtype,
            generator=generator,
        ) * 0.1
        key = torch.nn.functional.normalize(
            torch.randn(
                batch, sequence, heads, key_dim, device="cuda", dtype=torch.float32,
                generator=generator,
            ), dim=-1,
        ).to(dtype)
        value = torch.randn(
            batch, sequence, heads, value_dim, device="cuda", dtype=dtype,
            generator=generator,
        ) * 0.1
        log_decay = -torch.rand(
            batch, sequence, heads, key_dim, device="cuda", dtype=dtype,
            generator=generator,
        ) * 0.03
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "log_decay": log_decay.requires_grad_(),
        }
    if recipe == "mesa_net_core":
        batch, sequence, heads, key_dim = 1, 64, 2, 16
        dtype = torch.bfloat16
        query = torch.randn(
            batch, sequence, heads, key_dim, device="cuda", dtype=dtype,
            generator=generator,
        ) * 0.1
        key = torch.nn.functional.normalize(
            torch.randn(
                batch, sequence, heads, key_dim, device="cuda",
                dtype=torch.float32, generator=generator,
            ), dim=-1,
        ).to(dtype)
        value = torch.randn(
            batch, sequence, heads, key_dim, device="cuda", dtype=dtype,
            generator=generator,
        ) * 0.1
        log_decay = -torch.rand(
            batch, sequence, heads, device="cuda", generator=generator,
        ) * 0.03
        beta = torch.rand(
            batch, sequence, heads, device="cuda", generator=generator,
        ) * 0.4 + 0.2
        lamb = torch.nn.functional.softplus(
            torch.randn(heads, key_dim, device="cuda", generator=generator)
        ) + 1.0
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "log_decay": log_decay.requires_grad_(),
            "beta": beta.requires_grad_(),
            "lamb": lamb.requires_grad_(),
        }
    if recipe == "deltaformer_attention_core":
        batch, sequence, heads, dim = 1, 64, 2, 32
        dtype = torch.bfloat16
        query = torch.randn(
            batch, sequence, heads, dim, device="cuda", dtype=dtype, generator=generator
        ) * 0.1
        key = torch.randn(
            query.shape, device="cuda", dtype=dtype, generator=generator
        ) * 0.1
        value = torch.randn(
            query.shape, device="cuda", dtype=dtype, generator=generator
        ) * 0.1
        beta = torch.sigmoid(
            torch.randn(batch, sequence, heads, device="cuda", generator=generator)
        ).to(dtype)
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "beta": beta.requires_grad_(),
        }
    if recipe == "path_attention_core":
        batch, sequence, query_heads, key_heads, key_dim = 1, 128, 4, 2, 32
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, query_heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn(batch, sequence, key_heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        value = torch.randn_like(key) * 0.1
        weight = torch.nn.functional.normalize(
            torch.randn(batch, sequence, key_heads, key_dim, device="cuda", generator=generator),
            dim=-1,
        )
        beta = torch.rand(batch, sequence, key_heads, device="cuda", generator=generator) * 2
        gate = torch.nn.functional.logsigmoid(
            torch.randn(batch, sequence, query_heads, device="cuda", generator=generator)
        )
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "w": weight.requires_grad_(),
            "beta": beta.requires_grad_(),
            "g": gate.requires_grad_(),
        }
    if recipe == "gated_oja_core":
        batch, sequence, heads, key_dim, value_dim = 1, 256, 2, 16, 16
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn_like(query) * 0.1
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        gate = -torch.rand(batch, sequence, heads, value_dim, device="cuda", generator=generator) * 0.03
        beta = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", generator=generator))
        initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.01
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "gv": gate.requires_grad_(),
            "beta": beta.requires_grad_(),
            "initial_state": initial_state.requires_grad_(),
        }
    if recipe == "comba_core":
        batch, sequence, heads, key_dim, value_dim = 1, 256, 2, 16, 16
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn_like(query) * 0.1
        prediction_key = torch.randn_like(query) * 0.1
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        log_decay = -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
        beta = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", generator=generator))
        initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.01
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "p": prediction_key.requires_grad_(),
            "g": log_decay.requires_grad_(),
            "beta": beta.requires_grad_(),
            "initial_state": initial_state.requires_grad_(),
        }
    if recipe == "pgdn_core":
        batch, sequence, heads, key_dim, value_dim = 1, 256, 2, 16, 16
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn_like(query) * 0.1
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        g_atk = -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
        gate = -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
        beta_atk = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", generator=generator))
        beta = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", generator=generator))
        initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.01
        initial_A_state = torch.rand(batch, heads, key_dim, device="cuda", generator=generator) * 0.01
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "g_atk": g_atk.requires_grad_(),
            "g": gate.requires_grad_(),
            "beta_atk": beta_atk.requires_grad_(),
            "beta": beta.requires_grad_(),
            "initial_state": initial_state.requires_grad_(),
            "initial_A_state": initial_A_state.requires_grad_(),
        }
    if recipe == "pkda_core":
        batch, sequence, heads, key_dim, value_dim = 1, 256, 2, 16, 16
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn_like(query) * 0.1
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        gate = -torch.rand(batch, sequence, heads, key_dim, device="cuda", generator=generator) * 0.03
        g_atk = -torch.rand(batch, sequence, heads, device="cuda", generator=generator) * 0.03
        beta_atk = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", generator=generator))
        beta = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", generator=generator))
        initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=torch.float32, generator=generator) * 0.01
        initial_A_state = torch.rand(batch, heads, key_dim, device="cuda", generator=generator) * 0.01
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "g": gate.requires_grad_(),
            "g_atk": g_atk.requires_grad_(),
            "beta_atk": beta_atk.requires_grad_(),
            "beta": beta.requires_grad_(),
            "initial_state": initial_state.requires_grad_(),
            "initial_A_state": initial_A_state.requires_grad_(),
        }
    if recipe in {"abc_core", "gsa_core"}:
        batch, sequence, key_heads, query_heads, key_dim, value_dim, slots = 1, 128, 2, 2, 32, 32, 16
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, query_heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn(batch, sequence, key_heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        value = torch.randn(batch, sequence, key_heads, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        initial_key_state = torch.randn(batch, key_heads, key_dim, slots, device="cuda", dtype=torch.float32, generator=generator) * 0.01
        initial_value_state = torch.randn(batch, key_heads, slots, value_dim, device="cuda", dtype=torch.float32, generator=generator) * 0.01
        if recipe == "abc_core":
            return {
                "query": query.requires_grad_(),
                "key": key.requires_grad_(),
                "value": value.requires_grad_(),
                "slot_logits": (torch.randn(batch, sequence, key_heads, slots, device="cuda", dtype=dtype, generator=generator) * 0.1).requires_grad_(),
                "initial_key_state": initial_key_state.requires_grad_(),
                "initial_value_state": initial_value_state.requires_grad_(),
                "_output_only": True,
            }
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "slot_weights": torch.sigmoid(torch.randn(batch, sequence, key_heads, slots, device="cuda", dtype=dtype, generator=generator)).requires_grad_(),
            "log_decay": (torch.nn.functional.logsigmoid(torch.randn(batch, sequence, key_heads, slots, device="cuda", dtype=dtype, generator=generator)) * 0.05).requires_grad_(),
            "initial_key_state": initial_key_state.requires_grad_(),
            "initial_value_state": initial_value_state.requires_grad_(),
            "_output_only": True,
        }
    if recipe == "dsa_attention_core":
        batch, sequence, heads, key_dim, value_dim, topk = 1, 128, 2, 32, 16, 16
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator)
        key = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator)
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", dtype=dtype, generator=generator)
        positions = torch.arange(sequence, device="cuda")
        indices = (torch.rand(batch, sequence, topk, device="cuda", generator=generator) * (positions[None, :, None] + 1)).long()
        attention_mask = torch.zeros(batch, sequence, sequence, dtype=torch.bool, device="cuda")
        attention_mask.scatter_(2, indices, True)
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "attention_mask": attention_mask[:, None],
            "indices": indices,
        }
    if recipe == "nsa_selected_attention_core":
        batch, sequence, query_heads, key_dim, value_dim = 1, 128, 16, 32, 32
        block_size, block_topk = 32, 2
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, query_heads, key_dim, device="cuda", dtype=dtype, generator=generator)
        key = torch.randn(batch, sequence, 1, key_dim, device="cuda", dtype=dtype, generator=generator)
        value = torch.randn(batch, sequence, 1, value_dim, device="cuda", dtype=dtype, generator=generator)
        positions = torch.arange(sequence, device="cuda")
        current_block = (positions // block_size).view(1, sequence, 1, 1)
        block_indices = torch.cat(
            (
                current_block.expand(batch, -1, -1, -1),
                (current_block - 1).expand(batch, -1, -1, -1),
            ),
            dim=-1,
        )
        block_counts = (positions // block_size + 1).clamp(max=block_topk)
        block_counts = block_counts.view(batch, sequence, 1)
        block_tokens = (
            block_indices[:, :, 0, :, None] * block_size
            + torch.arange(block_size, device="cuda")
        )
        valid_tokens = (
            (block_tokens <= positions[None, :, None, None])
            & (block_tokens < sequence)
            & (
                torch.arange(block_topk, device="cuda")[None, None, :]
                < block_counts
            )[..., None]
        )
        key_positions = torch.arange(sequence, device="cuda")
        attention_mask = (
            (key_positions[None, None, None, None, :] == block_tokens[..., None])
            & valid_tokens[..., None]
        ).any(dim=(2, 3))
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "attention_mask": attention_mask[:, None],
            "block_indices": block_indices,
            "block_counts": block_counts,
            "g_slc": torch.ones(batch, sequence, query_heads, device="cuda", dtype=dtype),
        }
    if recipe == "moba_selected_attention_core":
        batch, sequence, heads, key_dim = 1, 128, 2, 32
        chunk_size, topk = 32, 3
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator)
        key = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator)
        value = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator)
        attention_mask = _moba_attention_mask(query, key, chunk_size, topk)
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "attention_mask": attention_mask,
            "cu_seqlens": torch.tensor([0, sequence], device="cuda", dtype=torch.int32),
            "max_seqlen": sequence,
            "chunk_size": chunk_size,
            "topk": topk,
        }
    if recipe == "rwkv4_memory_core":
        batch, sequence, channels = 1, 1024, 512
        w = (-2.0 + torch.randn(channels, device="cuda", generator=generator) * 0.05)
        u = torch.randn(channels, device="cuda", generator=generator) * 0.1
        key = torch.randn(batch, sequence, channels, device="cuda", generator=generator) * 0.1
        value = torch.randn(batch, sequence, channels, device="cuda", generator=generator) * 0.1
        state = torch.stack((
            torch.randn(batch, channels, device="cuda", generator=generator) * 0.1,
            torch.rand(batch, channels, device="cuda", generator=generator) + 0.5,
            torch.randn(batch, channels, device="cuda", generator=generator) * 0.1,
        ), dim=1).unsqueeze(2)
        return {
            "w": w.requires_grad_(),
            "u": u.requires_grad_(),
            "k": key.requires_grad_(),
            "v": value.requires_grad_(),
            "state": state.requires_grad_(),
        }
    if recipe == "rwkv6_memory_core":
        batch, sequence, heads, key_dim, value_dim = 1, 512, 2, 32, 32
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", generator=generator) * 0.1
        key = torch.randn(batch, sequence, heads, key_dim, device="cuda", generator=generator) * 0.1
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", generator=generator) * 0.1
        log_decay = -torch.rand(batch, sequence, heads, key_dim, device="cuda", generator=generator) * 0.05
        bonus = torch.randn(heads, key_dim, device="cuda", generator=generator) * 0.1
        initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", generator=generator) * 0.05
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "log_decay": log_decay.requires_grad_(),
            "bonus": bonus.requires_grad_(),
            "initial_state": initial_state.requires_grad_(),
        }
    if recipe == "momentum_delta_core":
        batch, sequence, heads, key_dim, value_dim = 1, 256, 2, 32, 32
        dtype = torch.bfloat16
        query = torch.randn(batch, sequence, heads, key_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        key = torch.randn_like(query) * 0.1
        p = torch.randn_like(query) * 0.1
        value = torch.randn(batch, sequence, heads, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.1
        log_alpha = -torch.rand(batch, sequence, heads, device="cuda", dtype=dtype, generator=generator) * 0.03
        log_mu = -torch.rand(batch, sequence, heads, device="cuda", dtype=dtype, generator=generator) * 0.03
        # Keep the coupled two-state recurrence numerically stable over a
        # multi-chunk parity/profile workload while exercising a varying gate.
        beta = 0.10 + torch.rand(batch, sequence, heads, device="cuda", dtype=dtype, generator=generator) * 0.02
        eta = torch.sigmoid(torch.randn(batch, sequence, heads, device="cuda", dtype=dtype, generator=generator))
        initial_state = torch.randn(batch, heads, key_dim, value_dim, device="cuda", dtype=dtype, generator=generator) * 0.01
        initial_normalizer_state = torch.randn_like(initial_state) * 0.01
        return {
            "query": query.requires_grad_(),
            "key": key.requires_grad_(),
            "value": value.requires_grad_(),
            "p": p.requires_grad_(),
            "log_alpha": log_alpha.requires_grad_(),
            "log_mu": log_mu.requires_grad_(),
            "beta": beta.requires_grad_(),
            "eta": eta.requires_grad_(),
            "initial_state": initial_state.requires_grad_(),
            "initial_normalizer_state": initial_normalizer_state.requires_grad_(),
        }
    sequence = (
        4096
        if recipe == "retention_core"
        else 1024
        if recipe == "lightning_attention_core"
        else 70
        if recipe == "log_linear_attention_core"
        else 128
        if recipe == "fox"
        else 128
        if recipe == "parallax_attention_core"
        else 256
        if recipe in {"generalized_delta_iplr_core", "generalized_delta_dplr_core", "rwkv7_transition_core"}
        else 64
    )
    dtype = (
        torch.float32
        if recipe in {
            "simple_gla",
            "gla",
            "retention_core",
            "lightning_attention_core",
            "lightnet_gla_core",
            "hgrn2_ssm_core",
            "gdn2_core",
            "based_attention_core",
            "rebased_attention_core",
            "log_linear_attention_core",
            "kda_core",
            "generalized_delta_iplr_core",
            "wall_attention_core",
            "rwkv6_memory_core",
        }
        else torch.bfloat16
    )
    heads, query_heads, key_dim, value_dim = (
        (2, 1, 64, 16)
        if recipe == "log_linear_attention_core"
        else (2, 2, 64, 16)
        if recipe == "kda_core"
        else (2, 2, 16, 8)
        if recipe == "gated_delta_product_core"
        else (2, 2, 32, 32)
        if recipe in {"generalized_delta_iplr_core", "generalized_delta_dplr_core", "rwkv7_transition_core", "parallax_attention_core"}
        else (1, 2, 32, 16)
        if recipe == "wall_attention_core"
        else (2, 2, 16, 16)
        if recipe == "gdn2_core"
        else (2, 2, 8, 8)
        if recipe in {"based_attention_core", "rebased_attention_core"}
        else (4, 4, 32, 32)
    )
    query = torch.randn(
        (1, sequence, query_heads, key_dim),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    key_shape = (
        (1, sequence, heads, key_dim)
        if recipe == "wall_attention_core"
        else query.shape
    )
    key = torch.randn(
        key_shape, device="cuda", dtype=query.dtype, generator=generator
    )
    if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core"}:
        query = query * 0.1
        key = key * 0.1
    key_normalizer = None
    if recipe == "lightnet_gla_core":
        query = torch.nn.functional.silu(query)
        key_normalizer = key.float().logcumsumexp(dim=1)
        key = torch.exp(key.float() - key_normalizer).to(dtype)
    if recipe in {"delta_net", "gated_delta_net"}:
        key = torch.nn.functional.normalize(key.float(), dim=-1).to(dtype)
    value = torch.randn(
        (1, sequence, heads, value_dim),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core"}:
        value = value * 0.1
    if recipe in {"log_linear_attention_core", "kda_core"}:
        query = query * 0.1
        key = key * 0.1
        value = value * 0.1
    if recipe in {"delta_net", "gated_delta_net"}:
        value = value * 0.1
    operands = {
        "query": query.requires_grad_(),
        "key": key.requires_grad_(),
        "value": value.requires_grad_(),
    }
    if recipe == "parallax_attention_core":
        operands["r"] = (
            torch.randn(query.shape, device="cuda", dtype=dtype, generator=generator)
            * 0.1
        ).requires_grad_()
    if recipe == "wall_attention_core":
        operands["g"] = (
            -torch.rand(
                (1, sequence, query_heads, key_dim),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.03
        ).requires_grad_()
    if recipe == "simple_gla":
        gate_shape = (1, sequence, heads)
    elif recipe == "gla":
        gate_shape = (1, sequence, heads, key_dim)
    elif recipe in {"hgrn2_ssm_core", "lightnet_gla_core"}:
        gate_shape = (1, sequence, heads, key_dim)
    elif recipe == "gdn2_core":
        gate_shape = (1, sequence, heads, key_dim)
    elif recipe == "retention_core":
        gate_shape = (4,)
    elif recipe == "lightning_attention_core":
        gate_shape = (4,)
    elif recipe == "gated_delta_net":
        gate_shape = (1, sequence, heads)
    elif recipe == "log_linear_attention_core":
        gate_shape = (1, sequence, heads)
    elif recipe == "kda_core":
        gate_shape = (1, sequence, heads, key_dim)
    elif recipe == "gated_delta_product_core":
        gate_shape = (1, sequence, heads)
    elif recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core"}:
        gate_shape = (1, sequence, heads, key_dim)
    elif recipe == "fox":
        gate_shape = (1, sequence, query_heads)
    else:
        gate_shape = None
    if gate_shape is not None:
        log_decay = (
            -torch.rand(
                gate_shape,
                device="cuda",
                dtype=(
                    dtype
                    if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core", "fox"}
                    else torch.float32
                ),
                generator=generator,
            )
            * (0.1 if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core", "fox"} else 0.25)
        )
        operands["log_decay"] = log_decay.requires_grad_()
    if recipe == "log_linear_attention_core":
        operands["log_decay"] = (
            -torch.rand((1, sequence, heads), device="cuda", generator=generator) * 0.03
        ).requires_grad_()
        operands["level_scales"] = (
            torch.rand((1, sequence, heads, 8), device="cuda", generator=generator) * 0.2
        ).requires_grad_()
    if recipe == "kda_core":
        operands["log_decay"] = (
            -torch.rand((1, sequence, heads, key_dim), device="cuda", generator=generator) * 0.1
        ).requires_grad_()
        operands["beta"] = torch.sigmoid(
            torch.randn((1, sequence, heads), device="cuda", generator=generator)
        ).requires_grad_()
    if recipe == "gated_delta_product_core":
        ranks = 2
        operands["beta"] = torch.sigmoid(
            torch.randn(
                (1, sequence, ranks, heads),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
        ).requires_grad_()
        operands["update_keys"] = torch.nn.functional.normalize(
            torch.randn(
                (1, sequence, ranks, heads, key_dim),
                device="cuda",
                dtype=dtype,
                generator=generator,
            ),
            dim=-1,
        ).requires_grad_()
        operands["update_values"] = (
            torch.randn(
                (1, sequence, ranks, heads, value_dim),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * 0.1
        ).requires_grad_()
    if recipe in {"generalized_delta_iplr_core", "generalized_delta_dplr_core", "rwkv7_transition_core"}:
        operands["transition_alpha"] = (
            torch.randn(
                (1, sequence, heads, key_dim),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * (0.005 if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core"} else 0.03)
        ).requires_grad_()
        operands["transition_beta"] = (
            torch.randn(
                (1, sequence, heads, key_dim),
                device="cuda",
                dtype=dtype,
                generator=generator,
            )
            * (0.005 if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core"} else 0.03)
        ).requires_grad_()
    if recipe in {"delta_net", "gated_delta_net"}:
        beta = torch.sigmoid(
            torch.randn((1, sequence, heads), device="cuda", generator=generator)
        )
        operands["beta"] = beta.requires_grad_()
    if recipe == "gdn2_core":
        operands["erase_gate"] = torch.sigmoid(
            torch.randn((1, sequence, heads, key_dim), device="cuda", generator=generator)
        ).requires_grad_()
        operands["write_gate"] = torch.sigmoid(
            torch.randn((1, sequence, heads, value_dim), device="cuda", generator=generator)
        ).requires_grad_()
    if recipe in {
        "simple_gla",
        "gla",
        "retention_core",
        "lightning_attention_core",
        "lightnet_gla_core",
        "gdn2_core",
        "kda_core",
        "gated_delta_product_core",
        "generalized_delta_iplr_core",
        "generalized_delta_dplr_core",
        "rwkv7_transition_core",
    }:
        initial_state = torch.randn(
            (1, heads, key_dim, value_dim),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        operands["initial_state"] = initial_state.requires_grad_()
    if recipe == "lightnet_gla_core":
        operands["log_decay"] = (
            torch.cat((key_normalizer[:, :1], key_normalizer[:, :-1]), dim=1)
            - key_normalizer
        ).contiguous().requires_grad_()
    if recipe == "retention_core":
        # RetNet's upstream layer constructs one data-independent log decay
        # value per head. Its derivative is not a trainable input.
        gamma = (
            1
            - query.new_tensor(2.0, dtype=torch.float32).pow(
                -5.0 - query.new_tensor(range(heads), dtype=torch.float32)
            )
        ).log()
        operands["log_decay"] = gamma.contiguous()
    if recipe == "lightning_attention_core":
        layer_idx, num_layers = 3, 12
        gamma = -(8.0 / 4 * (1 - layer_idx / num_layers)) * torch.arange(
            heads, device="cuda", dtype=torch.float32
        )
        operands["log_decay"] = gamma
    return operands


def _direct(recipe: str, inputs: dict[str, torch.Tensor]):
    if recipe == "ttt_linear_core":
        from fla.ops.ttt.chunk import chunk_ttt_linear

        output, state, normalizer_state = chunk_ttt_linear(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["w"].contiguous(),
            inputs["b"].contiguous(),
            inputs["eta"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            chunk_size=16,
            initial_state=inputs["initial_state"].contiguous(),
            initial_state_bias=inputs["initial_state_bias"].contiguous(),
            output_final_state=True,
        )
        return output, (state, normalizer_state)
    if recipe == "titans_linear_memory_core":
        from fla.ops.titans.naive import chunk_titans_linear_ref

        return chunk_titans_linear_ref(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["w"].contiguous(),
            inputs["b"].contiguous(),
            inputs["theta"].contiguous(),
            inputs["alpha"].contiguous(),
            inputs["eta"].contiguous(),
            eps=1e-6,
            chunk_size=16,
            initial_state=inputs["initial_state"].contiguous(),
            output_final_state=True,
            use_chunk=True,
        )
    if recipe == "mesa_net_core":
        from fla.ops.mesa_net.chunk import chunk_mesa_net

        output, h_kk, h_kv = chunk_mesa_net(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["log_decay"].contiguous(),
            inputs["beta"].contiguous(),
            inputs["lamb"].contiguous(),
            output_final_state=True,
            max_CG_iteration=30,
            use_qk_l2norm_in_kernel=False,
        )
        return output, (h_kk, h_kv)
    if recipe == "rodimus_gla_core":
        return chunk_gla(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["log_decay"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            output_final_state=True,
            state_v_first=True,
        )
    if recipe == "deltaformer_attention_core":
        from fla.ops.deltaformer import deltaformer_attn

        return deltaformer_attn(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["beta"].contiguous(),
            C=32,
        ), None
    if recipe == "abc_core":
        output, final_state = chunk_abc(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["slot_logits"].contiguous(),
            initial_state=(inputs["initial_key_state"].contiguous(), inputs["initial_value_state"].contiguous()),
            output_final_state=True,
        )
        return output, tuple(final_state)
    if recipe == "gsa_core":
        output, final_state = chunk_gsa(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["slot_weights"].contiguous(),
            inputs["log_decay"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=(inputs["initial_key_state"].contiguous(), inputs["initial_value_state"].contiguous()),
            output_final_state=True,
            checkpoint_level=0,
        )
        return output, tuple(final_state)
    if recipe == "pgdn_core":
        output, final_state, final_A_state = chunk_precond_gated_delta_rule(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["g_atk"].contiguous(),
            inputs["g"].contiguous(),
            inputs["beta_atk"].contiguous(),
            inputs["beta"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=inputs["initial_state"].contiguous(),
            initial_A_state=inputs["initial_A_state"].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            x=1.5,
            eps=1e-6,
            log_atk_scale=None,
        )
        return output, (final_state, final_A_state)
    if recipe == "pkda_core":
        output, final_state, final_A_state = chunk_precond_kda(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["g"].contiguous(),
            inputs["g_atk"].contiguous(),
            inputs["beta_atk"].contiguous(),
            inputs["beta"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=inputs["initial_state"].contiguous(),
            initial_A_state=inputs["initial_A_state"].contiguous(),
            output_final_state=True,
            use_gate_in_kernel=False,
            safe_gate=False,
            x=1.5,
            eps=1e-6,
            log_atk_scale=None,
        )
        return output, (final_state, final_A_state)
    if recipe == "comba_core":
        return chunk_comba(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["p"].contiguous(),
            inputs["g"].contiguous(),
            beta=inputs["beta"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=inputs["initial_state"].contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    if recipe == "gated_oja_core":
        return chunk_gated_oja_rule(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["gv"].contiguous(),
            inputs["beta"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=inputs["initial_state"].contiguous(),
            output_final_state=True,
            use_q_l2norm=False,
            use_k_l2norm=False,
            chunk_size=64,
        )
    if recipe == "path_attention_core":
        output, _ = parallel_path_attn(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["w"].contiguous(),
            inputs["beta"].contiguous(),
            inputs["g"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
        )
        return output, None
    if recipe == "rwkv4_memory_core":
        return fused_recurrent_rwkv4(
            inputs["w"].contiguous(),
            inputs["u"].contiguous(),
            inputs["k"].contiguous(),
            inputs["v"].contiguous(),
            inputs["state"].contiguous(),
        )
    if recipe == "rwkv6_memory_core":
        return fused_recurrent_rwkv6(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["log_decay"].contiguous(),
            inputs["bonus"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=inputs["initial_state"].contiguous(),
            output_final_state=True,
        )
    if recipe == "momentum_delta_core":
        output, final_state = chunk_momentum_delta_rule(
            inputs["query"].contiguous(),
            inputs["key"].contiguous(),
            inputs["value"].contiguous(),
            inputs["log_alpha"].contiguous(),
            inputs["log_mu"].contiguous(),
            p=inputs["p"].contiguous(),
            beta=inputs["beta"].contiguous(),
            eta=inputs["eta"].contiguous(),
            scale=inputs["query"].shape[-1] ** -0.5,
            initial_state=torch.stack(
                (
                    inputs["initial_state"].contiguous(),
                    inputs["initial_normalizer_state"].contiguous(),
                )
            ),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_p_times_alpha=False,
            chunk_size=64,
        )
        return output, (final_state[0], final_state[1])
    if recipe == "fox":
        return parallel_forgetting_attn(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["log_decay"],
            scale=inputs["query"].shape[-1] ** -0.5,
        ), None
    if recipe == "parallax_attention_core":
        return parallel_parallax(
            inputs["query"],
            inputs["r"],
            inputs["key"],
            inputs["value"],
            scale=inputs["query"].shape[-1] ** -0.5,
        ), None
    if recipe == "wall_attention_core":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
        return parallel_wall_attn(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["g"],
            scale=inputs["query"].shape[-1] ** -0.5,
        ), None
    if recipe == "dsa_attention_core":
        return naive_dsa(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            q_idx=None,
            k_idx=None,
            indices=inputs["indices"],
            topk=inputs["indices"].shape[-1],
            scale=inputs["query"].shape[-1] ** -0.5,
        ), None
    if recipe == "nsa_selected_attention_core":
        return parallel_nsa(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            g_slc=inputs["g_slc"],
            block_indices=inputs["block_indices"],
            block_counts=inputs["block_counts"],
            block_size=32,
            window_size=0,
            scale=inputs["query"].shape[-1] ** -0.5,
        ), None
    if recipe == "moba_selected_attention_core":
        return parallel_moba(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["cu_seqlens"],
            max_seqlen=inputs["max_seqlen"],
            chunk_size=inputs["chunk_size"],
            topk=inputs["topk"],
        ), None
    query, key, value = inputs["query"], inputs["key"], inputs["value"]
    if recipe in {"simple_gla", "gla", "lightnet_gla_core"}:
        fn = chunk_simple_gla if recipe == "simple_gla" else chunk_gla
        output, state = fn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g=inputs["log_decay"].contiguous(),
            scale=1.0,
            initial_state=inputs["initial_state"],
            output_final_state=True,
        )
        return output, state
    if recipe == "retention_core":
        return fused_chunk_simple_gla(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g_gamma=inputs["log_decay"],
            scale=1.0,
            initial_state=inputs["initial_state"],
            output_final_state=True,
        )
    if recipe == "lightning_attention_core":
        return chunk_simple_gla(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g_gamma=inputs["log_decay"],
            scale=1.0,
            initial_state=inputs["initial_state"],
            output_final_state=True,
        )
    if recipe == "hgrn2_ssm_core":
        return chunk_gla(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g=inputs["log_decay"].contiguous(),
            scale=1.0,
            output_final_state=True,
            state_v_first=False,
        )
    if recipe == "linear_attention":
        q = (
            torch.where(query.float() > 0, query.float(), torch.expm1(query.float()))
            + 1
        )
        k = torch.where(key.float() > 0, key.float(), torch.expm1(key.float())) + 1
        output, state = chunk_linear_attn(
            q.to(query.dtype).contiguous(),
            k.to(key.dtype).contiguous(),
            value.contiguous(),
            scale=1.0,
            output_final_state=True,
            normalize=True,
        )
        kv_state, normalizer_state = state
        if normalizer_state.ndim == 4 and normalizer_state.shape[1] == 1:
            normalizer_state = normalizer_state.squeeze(1)
        return output, (kv_state, normalizer_state)
    if recipe == "delta_net":
        return chunk_delta_rule(
            query,
            key,
            value,
            inputs["beta"],
            scale=1.0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    if recipe == "generalized_delta_iplr_core":
        from fla.ops.generalized_delta_rule.iplr import fused_recurrent_iplr_delta_rule

        return fused_recurrent_iplr_delta_rule(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            inputs["transition_alpha"].contiguous(),
            inputs["transition_beta"].contiguous(),
            scale=query.shape[-1] ** -0.5,
            initial_state=inputs["initial_state"],
            output_final_state=True,
        )
    if recipe == "generalized_delta_dplr_core":
        from fla.ops.generalized_delta_rule.dplr import chunk_dplr_delta_rule

        return chunk_dplr_delta_rule(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            inputs["transition_alpha"].contiguous(),
            inputs["transition_beta"].contiguous(),
            inputs["log_decay"].contiguous(),
            scale=query.shape[-1] ** -0.5,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            chunk_size=16,
        )
    if recipe == "rwkv7_transition_core":
        return chunk_rwkv7(
            r=query.contiguous(),
            w=inputs["log_decay"].contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            a=inputs["transition_alpha"].contiguous(),
            b=inputs["transition_beta"].contiguous(),
            scale=1.0,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            safe_gate=True,
            lower_bound=-0.6065306597126334,
            chunk_size=64,
        )
    if recipe == "gated_delta_net":
        return chunk_gated_delta_rule(
            query,
            key,
            value,
            inputs["log_decay"],
            inputs["beta"],
            scale=1.0,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=False,
        )
    if recipe == "gdn2_core":
        return chunk_gdn2(
            query,
            key,
            value,
            inputs["log_decay"],
            inputs["erase_gate"],
            inputs["write_gate"],
            scale=query.shape[-1] ** -0.5,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    if recipe == "based_attention_core":
        return (
            fused_chunk_based(
                query,
                key,
                value,
                scale=query.shape[-1] ** -0.5,
                use_norm=True,
            ),
            None,
        )
    if recipe == "rebased_attention_core":
        return (
            parallel_rebased(
                query,
                key,
                value,
                eps=1e-6,
                use_scale=True,
                use_normalize=True,
            ),
            None,
        )
    if recipe == "log_linear_attention_core":
        return chunk_log_linear_attn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            inputs["log_decay"].contiguous(),
            inputs["level_scales"].contiguous(),
            output_final_state=True,
        )
    if recipe == "kda_core":
        return chunk_kda(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            inputs["log_decay"].contiguous(),
            inputs["beta"].contiguous(),
            scale=query.shape[-1] ** -0.5,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            use_beta_sigmoid_in_kernel=False,
            state_v_first=False,
        )
    if recipe == "gated_delta_product_core":
        batch, sequence, ranks, heads, key_dim = inputs["update_keys"].shape
        value_dim = inputs["update_values"].shape[-1]
        return chunk_gated_delta_product(
            query.contiguous(),
            inputs["update_keys"].reshape(
                batch, sequence * ranks, heads, key_dim
            ).contiguous(),
            inputs["update_values"].reshape(
                batch, sequence * ranks, heads, value_dim
            ).contiguous(),
            inputs["log_decay"].contiguous(),
            inputs["beta"].reshape(batch, sequence * ranks, heads).contiguous(),
            num_householder=ranks,
            scale=key_dim**-0.5,
            initial_state=inputs["initial_state"],
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
    raise ValueError(f"unsupported FLA benchmark recipe: {recipe}")


def _equation_direct(recipe: str, inputs: dict[str, torch.Tensor]):
    if recipe == "ttt_linear_core":
        from fla.ops.ttt.naive import chunk_ttt_linear_ref

        output, state, normalizer_state = chunk_ttt_linear_ref(
            inputs["query"],
            inputs["key"],
            inputs["value"],
            inputs["w"],
            inputs["b"],
            inputs["eta"],
            mini_batch_size=16,
            initial_state=inputs["initial_state"],
            initial_state_bias=inputs["initial_state_bias"],
            output_final_state=True,
        )
        return output, (state, normalizer_state)
    return _direct(recipe, inputs)


def _architecture_entrypoint(recipe: str, inputs: dict[str, torch.Tensor]):
    query, key, value = inputs["query"], inputs["key"], inputs["value"]
    if recipe == "retention_core":
        return fused_chunk_retention(
            query.contiguous(), key.contiguous(), value.contiguous(),
            scale=1.0, initial_state=inputs["initial_state"], output_final_state=True,
        )
    if recipe == "lightning_attention_core":
        return chunk_lightning_attn(
            query.contiguous(), key.contiguous(), value.contiguous(),
            layer_idx=3, num_layers=12, scale=1.0,
            initial_state=inputs["initial_state"], output_final_state=True,
        )
    raise ValueError(f"no architecture wrapper registered for {recipe!r}")


def _compiled(plan, inputs: dict[str, torch.Tensor]):
    execute_inputs = {
        name: value
        for name, value in inputs.items()
        if name not in {"indices", "block_indices", "block_counts", "g_slc", "_output_only"}
    }
    if plan.spec.name == "fox" and plan.backend is MixerBackend.REFERENCE:
        cumulative_decay = inputs["log_decay"].float().cumsum(dim=1).transpose(1, 2)
        execute_inputs.pop("log_decay")
        execute_inputs["score_bias"] = (
            cumulative_decay.unsqueeze(-1) - cumulative_decay.unsqueeze(-2)
        )
    if plan.spec.name == "moba_selected_attention_core" and plan.backend is MixerBackend.REFERENCE:
        for name in ("cu_seqlens", "max_seqlen", "chunk_size", "topk"):
            execute_inputs.pop(name)
    result = plan.execute(**execute_inputs)
    state = result.final_state
    if result.final_normalizer_state is not None:
        state = (state, result.final_normalizer_state)
    return result.output, state


def _loss(
    output: torch.Tensor, state: torch.Tensor, *, include_state: bool = True
) -> torch.Tensor:
    loss = output.square().mean()
    if not include_state:
        return loss
    if hasattr(state, "ht"):
        states = (state.ht,)
    else:
        states = state if isinstance(state, tuple) else (state,)
    return loss + sum(item.square().mean() for item in states if item is not None)


def _state_components(state):
    if state is None:
        return ()
    if isinstance(state, tuple):
        return state
    fields = getattr(state, "__dataclass_fields__", None)
    if fields is not None:
        return tuple(getattr(state, name) for name in fields)
    return (state,)


def _state_max_errors(direct_state, compiled_state) -> list[float]:
    direct_states = _state_components(direct_state)
    compiled_states = _state_components(compiled_state)
    return [
        (left.float() - right.float()).abs().max().item()
        for left, right in zip(direct_states, compiled_states, strict=True)
    ]


def _clear_grads(inputs: dict[str, torch.Tensor]) -> None:
    for tensor in inputs.values():
        if isinstance(tensor, torch.Tensor):
            tensor.grad = None


def _clone_inputs(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_(tensor.requires_grad)
        if isinstance(tensor, torch.Tensor)
        else tensor
        for name, tensor in inputs.items()
    }


def _forward_backward(call, inputs: dict[str, torch.Tensor]):
    _clear_grads(inputs)
    output, state = call(inputs)
    _loss(output, state, include_state="bonus" not in inputs and not inputs.get("_output_only", False)).backward()
    return output, state, tuple(
        inputs[name].grad if isinstance(inputs[name], torch.Tensor) else None
        for name in inputs
    )


def _time_one(
    call, inputs: dict[str, torch.Tensor], backward: bool
) -> tuple[float, float]:
    _clear_grads(inputs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    output, state = call(inputs)
    if backward:
        _loss(output, state, include_state="bonus" not in inputs and not inputs.get("_output_only", False)).backward()
    end_event.record()
    torch.cuda.synchronize()
    return (
        time.perf_counter() - start_wall,
        start_event.elapsed_time(end_event) / 1000.0,
    )


def _summarize(samples: list[float]) -> dict[str, object]:
    return {
        "sample_count": len(samples),
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": quantile(samples, 0.95) * 1000,
        "raw_samples_ms": [sample * 1000 for sample in samples],
    }


def _measure_pair(
    direct, compiled, direct_inputs, compiled_inputs, pairs: int, warmup: int
) -> dict[str, object]:
    # Capture each first invocation before any autotuning warmup.
    cold_direct = _time_one(direct, direct_inputs, backward=False)
    cold_compiled = _time_one(compiled, compiled_inputs, backward=False)
    for _ in range(warmup):
        _time_one(direct, direct_inputs, backward=False)
        _time_one(compiled, compiled_inputs, backward=False)
        _time_one(direct, direct_inputs, backward=True)
        _time_one(compiled, compiled_inputs, backward=True)

    measurements: dict[str, object] = {}
    for label, backward in (("forward", False), ("forward_backward", True)):
        direct_wall: list[float] = []
        compiled_wall: list[float] = []
        direct_device: list[float] = []
        compiled_device: list[float] = []
        overhead: list[float] = []
        order: list[str] = []
        for index in range(pairs):
            first, second = (
                ("direct", "compiled") if index % 2 == 0 else ("compiled", "direct")
            )
            order.append(first + second)
            for name in (first, second):
                if name == "direct":
                    wall, device = _time_one(direct, direct_inputs, backward)
                    direct_wall.append(wall)
                    direct_device.append(device)
                else:
                    wall, device = _time_one(compiled, compiled_inputs, backward)
                    compiled_wall.append(wall)
                    compiled_device.append(device)
            paired_index = len(overhead)
            d = direct_wall[paired_index]
            c = compiled_wall[paired_index]
            overhead.append((c - d) / d if d else 0.0)
        measurements[label] = {
            "direct_wall": _summarize(direct_wall),
            "compiled_wall": _summarize(compiled_wall),
            "direct_device": _summarize(direct_device),
            "compiled_device": _summarize(compiled_device),
            "paired_compiled_overhead_fraction": {
                "median": statistics.median(overhead),
                "p95": quantile(overhead, 0.95),
                "raw_samples": overhead,
                "gate": {
                    "limit_fraction": 0.10,
                    "pass": statistics.median(overhead) <= 0.10,
                },
            },
            "pair_order": order,
        }
    return {
        "first_direct_forward_call_ms": {
            "wall": cold_direct[0] * 1000,
            "device": cold_direct[1] * 1000,
        },
        "first_compiled_forward_call_after_direct_autotune_ms": {
            "wall": cold_compiled[0] * 1000,
            "device": cold_compiled[1] * 1000,
        },
        "warmup_calls_per_backend_per_mode": warmup,
        "paired_samples_per_mode": pairs,
        "measurements": measurements,
    }


def run(
    pairs: int,
    warmup: int,
    output: Path,
    recipes: tuple[str, ...] = RECIPES,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the FLA unified mixer profile requires CUDA")
    revision = _loaded_revision()
    if revision != EXPECTED_FLA_REVISION:
        raise RuntimeError(
            "loaded FLA source must match the register pin "
            f"{EXPECTED_FLA_REVISION}, got {revision!r} at {fla.__file__}"
        )
    nested_flash_attention = (
        _moba_flash_attn_identity()
        if "moba_selected_attention_core" in recipes
        else None
    )

    cases: dict[str, object] = {}
    for recipe in recipes:
        operands = _inputs(recipe, seed=44018 if recipe == "simple_gla" else 44019)
        direct_inputs = _clone_inputs(operands)
        compiled_inputs = _clone_inputs(operands)
        plan_started = time.perf_counter()
        plan = compile_mixer(
            named_mixer_recipe(recipe),
            backend=MixerBackend.LIBRARY,
            intent=MixerIntent.TRAINING,
            dtype=(
                "float32"
                if recipe in {
                    "simple_gla",
                    "gla",
                    "retention_core",
                    "lightning_attention_core",
                    "lightnet_gla_core",
                    "hgrn2_ssm_core",
                    "gdn2_core",
                    "based_attention_core",
                    "rebased_attention_core",
                    "log_linear_attention_core",
                    "kda_core",
                    "generalized_delta_iplr_core",
                    "wall_attention_core",
                    "rwkv4_memory_core",
                    "rwkv6_memory_core",
                    "titans_linear_memory_core",
                }
                else "bfloat16"
            ),
        )
        plan_build_ms = (time.perf_counter() - plan_started) * 1000
        direct = lambda values, recipe=recipe: _direct(recipe, values)
        compiled = lambda values, plan=plan: _compiled(plan, values)
        primary_input = operands["k"] if recipe == "rwkv4_memory_core" else operands["query"]
        value_atol = 2e-2 if primary_input.dtype == torch.bfloat16 else 1e-6
        gradient_atol = 2e-2 if primary_input.dtype == torch.bfloat16 else 2e-6
        state_atol = 2e-2 if primary_input.dtype == torch.bfloat16 else 2e-2
        if recipe == "rwkv4_memory_core":
            value_atol = gradient_atol = state_atol = 5e-6
        if recipe in {
            "log_linear_attention_core",
            "kda_core",
            "gated_delta_product_core",
            "generalized_delta_iplr_core",
            "generalized_delta_dplr_core",
            "rwkv4_memory_core",
            "rwkv6_memory_core",
            "momentum_delta_core",
            "mesa_net_core",
            "titans_linear_memory_core",
            "ttt_linear_core",
            "rwkv7_transition_core",
            "deltaformer_attention_core",
        }:
            value_atol = 1e-4
            gradient_atol = 2e-3
            state_atol = 3e-4

        performance = _measure_pair(
            direct, compiled, direct_inputs, compiled_inputs, pairs, warmup
        )
        direct_result = _forward_backward(direct, direct_inputs)
        compiled_result = _forward_backward(compiled, compiled_inputs)
        output_error = (direct_result[0] - compiled_result[0]).abs().max().item()
        state_errors = _state_max_errors(direct_result[1], compiled_result[1])
        gradient_errors = [
            0.0 if left is None and right is None
            else float("inf") if left is None or right is None
            else (left - right).abs().max().item()
            for left, right in zip(direct_result[2], compiled_result[2], strict=True)
        ]
        reference_equation = None
        if recipe in {
            "log_linear_attention_core",
            "kda_core",
            "gated_delta_product_core",
            "generalized_delta_iplr_core",
            "generalized_delta_dplr_core",
            "rwkv4_memory_core",
            "rwkv6_memory_core",
            "momentum_delta_core",
            "mesa_net_core",
            "titans_linear_memory_core",
            "ttt_linear_core",
            "rwkv7_transition_core",
            "path_attention_core",
            "rodimus_gla_core",
            "deltaformer_attention_core",
            "gated_oja_core",
            "comba_core",
            "pgdn_core",
            "pkda_core",
            "abc_core",
            "gsa_core",
            "fox",
            "parallax_attention_core",
            "wall_attention_core",
            "dsa_attention_core",
            "nsa_selected_attention_core",
        }:
            reference_inputs = _clone_inputs(operands)
            reference_plan = compile_mixer(
                named_mixer_recipe(recipe),
                backend=MixerBackend.REFERENCE,
                intent=MixerIntent.TRAINING,
                dtype=(
                    "bfloat16"
                    if recipe in {"gated_delta_product_core", "generalized_delta_dplr_core", "rwkv7_transition_core", "fox", "parallax_attention_core", "dsa_attention_core", "nsa_selected_attention_core", "momentum_delta_core", "mesa_net_core", "ttt_linear_core", "path_attention_core", "rodimus_gla_core", "deltaformer_attention_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                    else "float32"
                ),
            )
            equation_result = _compiled(reference_plan, reference_inputs)
            equation_gradients = _forward_backward(
                lambda values: _compiled(reference_plan, values), reference_inputs
            )
            upstream_inputs = _clone_inputs(operands)
            upstream_result = _forward_backward(
                lambda values: _equation_direct(recipe, values), upstream_inputs
            )
            equation_errors = _state_max_errors(
                equation_result[1], upstream_result[1]
            )
            equation_gradient_errors = [
                None
                if recipe in {"abc_core", "gsa_core"}
                and name in {"initial_key_state", "initial_value_state"}
                else None
                if actual is None and expected is None
                else float("inf")
                if actual is None or expected is None
                else (actual - expected).abs().max().item()
                for name, actual, expected in zip(
                    operands,
                    equation_gradients[2],
                    upstream_result[2],
                    strict=True,
                )
            ]
            equation_output_atol = (
                1e-4
                if recipe == "log_linear_attention_core"
                else 2e-2
                if recipe in {"dsa_attention_core", "nsa_selected_attention_core", "fox", "parallax_attention_core", "momentum_delta_core", "path_attention_core", "rodimus_gla_core", "deltaformer_attention_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                else 5e-3
                if recipe == "wall_attention_core"
                else 2e-2
                if recipe == "gated_delta_product_core"
                else 5e-3
                if recipe == "rwkv7_transition_core"
                else 2e-3
                if recipe == "generalized_delta_dplr_core"
                else 3e-4
            )
            equation_state_atol = (
                3e-4
                if recipe == "log_linear_attention_core"
                else 1e-2
                if recipe == "gated_delta_product_core"
                else 2e-2
                if recipe == "momentum_delta_core"
                else 2e-2
                if recipe in {"rodimus_gla_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                else 1e-3
                if recipe in {"generalized_delta_dplr_core", "rwkv7_transition_core"}
                else 5e-4
            )
            equation_gradient_atol = (
                2e-2
                if recipe in {"gated_delta_product_core", "generalized_delta_dplr_core", "rwkv7_transition_core", "fox", "parallax_attention_core", "wall_attention_core", "dsa_attention_core", "nsa_selected_attention_core", "momentum_delta_core", "path_attention_core", "rodimus_gla_core", "deltaformer_attention_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                else 2e-3
            )
            equation_output_rtol = (
                1e-3
                if recipe == "log_linear_attention_core"
                else 2e-2
                if recipe in {"dsa_attention_core", "nsa_selected_attention_core", "fox", "parallax_attention_core", "momentum_delta_core", "path_attention_core", "rodimus_gla_core", "deltaformer_attention_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                else 5e-3
                if recipe == "wall_attention_core"
                else 2e-2
                if recipe in {"gated_delta_product_core", "generalized_delta_dplr_core", "rwkv7_transition_core"}
                else 3e-3
            )
            if recipe == "mesa_net_core":
                equation_output_atol = 3e-3
                equation_output_rtol = 3e-3
                equation_state_atol = 3e-3
                equation_gradient_atol = 2e-2
            if recipe == "titans_linear_memory_core":
                equation_output_atol = 3e-5
                equation_output_rtol = 3e-4
                equation_state_atol = 3e-5
                equation_gradient_atol = 3e-3
            if recipe == "ttt_linear_core":
                equation_output_atol = 1e-2
                equation_output_rtol = 1e-2
                equation_state_atol = 1e-2
                equation_gradient_atol = 2e-2
            torch.testing.assert_close(
                equation_result[0].float(),
                upstream_result[0].float(),
                atol=equation_output_atol,
                rtol=equation_output_rtol,
            )
            for actual, expected in zip(
                _state_components(equation_result[1]),
                _state_components(upstream_result[1]),
                strict=True,
            ):
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=equation_state_atol,
                    rtol=(
                        2e-2
                        if recipe == "ttt_linear_core"
                        else 2e-2
                        if recipe in {"gated_delta_product_core", "generalized_delta_dplr_core", "rwkv7_transition_core", "fox", "parallax_attention_core", "wall_attention_core", "momentum_delta_core", "path_attention_core", "rodimus_gla_core", "deltaformer_attention_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                        else 3e-3
                    ),
                )
            for name, actual, expected in zip(
                operands,
                equation_gradients[2],
                upstream_result[2],
                strict=True,
            ):
                if recipe in {"abc_core", "gsa_core"} and name in {
                    "initial_key_state",
                    "initial_value_state",
                }:
                    continue
                if actual is None and expected is None:
                    continue
                if actual is None or expected is None:
                    raise AssertionError("reference/upstream gradient presence differs")
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=equation_gradient_atol,
                    rtol=(
                        1e-2
                        if recipe == "titans_linear_memory_core"
                        else 2e-2
                        if recipe == "ttt_linear_core"
                        else
                        3e-2
                        if recipe in {"gated_delta_product_core", "generalized_delta_dplr_core", "rwkv7_transition_core", "fox", "parallax_attention_core", "wall_attention_core", "momentum_delta_core", "path_attention_core", "rodimus_gla_core", "deltaformer_attention_core", "gated_oja_core", "comba_core", "pgdn_core", "pkda_core", "abc_core", "gsa_core"}
                        else 3e-3
                    ),
                )
            reference_equation = {
                "status": "pass",
                "anchor": reference_plan.anchor,
                "upstream_reference_callable": (
                    "fla.ops.ttt.naive.chunk_ttt_linear_ref"
                    if recipe == "ttt_linear_core"
                    else None
                ),
                "test": (
                    "tests/test_unified_mixer.py::test_titans_linear_memory_core_matches_pinned_upstream_outputs_state_and_gradients"
                    if recipe == "titans_linear_memory_core"
                    else "tests/test_unified_mixer.py::test_ttt_linear_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "ttt_linear_core"
                    else "tests/test_unified_mixer.py::test_log_linear_core_matches_pinned_upstream_outputs_state_and_gradients"
                    if recipe == "log_linear_attention_core"
                    else "tests/test_unified_mixer.py::test_gated_delta_product_core_matches_pinned_upstream_and_library_anchor"
                    if recipe == "gated_delta_product_core"
                    else "tests/test_unified_mixer.py::test_generalized_delta_transition_cores_match_pinned_upstream"
                    if recipe in {"generalized_delta_iplr_core", "generalized_delta_dplr_core", "rwkv7_transition_core"}
                    else "tests/test_unified_mixer.py::test_rwkv4_memory_core_matches_pinned_upstream_and_library_anchor"
                    if recipe == "rwkv4_memory_core"
                    else "tests/test_unified_mixer.py::test_rwkv6_memory_core_matches_pinned_upstream_outputs_and_gradients"
                    if recipe == "rwkv6_memory_core"
                    else "tests/test_unified_mixer.py::test_momentum_delta_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "momentum_delta_core"
                    else "tests/test_unified_mixer.py::test_mesa_net_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "mesa_net_core"
                    else "tests/test_unified_mixer.py::test_path_attention_core_matches_pinned_upstream_outputs_and_gradients"
                    if recipe == "path_attention_core"
                    else "tests/test_unified_mixer.py::test_rodimus_gla_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "rodimus_gla_core"
                    else "tests/test_unified_mixer.py::test_deltaformer_attention_core_matches_pinned_upstream_outputs_and_gradients"
                    if recipe == "deltaformer_attention_core"
                    else "tests/test_unified_mixer.py::test_gated_oja_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "gated_oja_core"
                    else "tests/test_unified_mixer.py::test_comba_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "comba_core"
                    else "tests/test_unified_mixer.py::test_pgdn_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "pgdn_core"
                    else "tests/test_unified_mixer.py::test_pkda_core_matches_pinned_upstream_outputs_states_and_gradients"
                    if recipe == "pkda_core"
                    else "tests/test_unified_mixer.py::test_slot_attention_cores_match_pinned_upstream_outputs_states_and_gradients"
                    if recipe in {"abc_core", "gsa_core"}
                    else "tests/test_unified_mixer.py::test_dsa_attention_core_matches_pinned_upstream_with_precomputed_routes"
                    if recipe == "dsa_attention_core"
                    else "tests/test_unified_mixer.py::test_nsa_selected_attention_core_matches_pinned_upstream_routes"
                    if recipe == "nsa_selected_attention_core"
                    else "tests/test_unified_mixer.py::test_fox_attention_core_matches_pinned_upstream_outputs_and_gradients"
                    if recipe == "fox"
                    else "tests/test_unified_mixer.py::test_parallax_attention_core_matches_pinned_upstream_outputs_and_gradients"
                    if recipe == "parallax_attention_core"
                    else "tests/test_unified_mixer.py::test_wall_attention_core_matches_pinned_upstream_outputs_and_gradients"
                    if recipe == "wall_attention_core"
                    else "tests/test_unified_mixer.py::test_kda_core_matches_pinned_upstream_and_library_anchor"
                ),
                "output_max_abs_error": (
                    equation_result[0] - upstream_result[0]
                ).abs().max().item(),
                "final_state_max_abs_errors": equation_errors,
                "input_gradient_max_abs_errors": dict(
                    zip(operands, equation_gradient_errors, strict=True)
                ),
                "tolerances": {
                    "output_atol": equation_output_atol,
                    "state_atol": equation_state_atol,
                    "gradient_atol": equation_gradient_atol,
                },
            }
        torch.testing.assert_close(
            compiled_result[0], direct_result[0], atol=value_atol, rtol=value_atol
        )
        for actual, expected in zip(
            _state_components(compiled_result[1]),
            _state_components(direct_result[1]),
            strict=True,
        ):
            if actual is None and expected is None:
                continue
            if actual is None or expected is None:
                raise AssertionError("upstream and compiler state presence differs")
            torch.testing.assert_close(
                actual, expected, atol=state_atol, rtol=state_atol
            )
        for actual, expected in zip(compiled_result[2], direct_result[2], strict=True):
            if actual is None and expected is None:
                continue
            if actual is None or expected is None:
                raise AssertionError("upstream and compiler gradient presence differs")
            torch.testing.assert_close(
                actual, expected, atol=gradient_atol, rtol=gradient_atol
            )

        wrapper_parity = None
        if recipe in {"retention_core", "lightning_attention_core"}:
            with torch.no_grad():
                wrapper_output, wrapper_state = _architecture_entrypoint(
                    recipe, direct_inputs
                )
            wrapper_output_error = (
                wrapper_output.float() - direct_result[0].detach().float()
            ).abs().max().item()
            wrapper_state_error = _state_max_errors(
                wrapper_state, direct_result[1]
            )
            torch.testing.assert_close(
                wrapper_output,
                direct_result[0].detach(),
                atol=value_atol,
                rtol=value_atol,
            )
            for actual, expected in zip(
                _state_components(wrapper_state),
                _state_components(direct_result[1]),
                strict=True,
            ):
                torch.testing.assert_close(
                    actual, expected.detach(), atol=value_atol, rtol=value_atol
                )
            wrapper_parity = {
                "status": "pass",
                "output_max_abs_error": wrapper_output_error,
                "final_state_max_abs_errors": wrapper_state_error,
            }

        cases[recipe] = {
            "architecture_ids": {
                "linear_attention": ["arch-015"],
                "retention_core": ["arch-017"],
                "lightning_attention_core": ["arch-016"],
                "lightnet_gla_core": ["arch-022"],
                "simple_gla": ["arch-018", "arch-052"],
                "gla": ["arch-019"],
                "delta_net": ["arch-025"],
                "gated_delta_net": ["arch-026"],
                "hgrn2_ssm_core": ["arch-024"],
                "gdn2_core": ["arch-027"],
                "based_attention_core": ["arch-020"],
                "rebased_attention_core": ["arch-021"],
                "log_linear_attention_core": ["arch-009", "arch-046"],
                "kda_core": ["arch-028"],
                "gated_delta_product_core": ["arch-029"],
                "generalized_delta_iplr_core": ["arch-031"],
                "generalized_delta_dplr_core": ["arch-032"],
                "rwkv4_memory_core": ["arch-040"],
                "rwkv6_memory_core": ["arch-041"],
                "momentum_delta_core": ["arch-030"],
                "mesa_net_core": ["arch-038"],
                "titans_linear_memory_core": ["arch-039"],
                "ttt_linear_core": ["arch-055"],
                "path_attention_core": ["arch-010"],
                "rodimus_gla_core": ["arch-036"],
                "deltaformer_attention_core": ["arch-013"],
                "gated_oja_core": ["arch-033"],
                "comba_core": ["arch-037"],
                "pgdn_core": ["arch-034"],
                "pkda_core": ["arch-035"],
                "abc_core": ["arch-048"],
                "gsa_core": ["arch-049", "arch-050"],
                "rwkv7_transition_core": ["arch-042"],
                "fox": ["arch-008"],
                "parallax_attention_core": ["arch-012"],
                "wall_attention_core": ["arch-011"],
                "dsa_attention_core": ["arch-007"],
                "nsa_selected_attention_core": ["arch-005"],
                "moba_selected_attention_core": ["arch-006"],
            }[recipe],
            "semantic_scope": (
                "GDN-2 gated delta state recurrence; Q/K/V projections, gate production, normalization and output projection excluded"
                if recipe == "gdn2_core"
                else "dyadic level-scaled LogLinear recurrence; Mamba-2 projections, dt/level transforms, partial-chunk cache ABI and full layer excluded"
                if recipe == "log_linear_attention_core"
                else "KDA key-channel-decayed delta recurrence with 1/sqrt(K) read scaling; source gate/frontend and output layer excluded"
                if recipe == "kda_core"
                else "Gated DeltaProduct ordered rank-R gated delta recurrence with 1/sqrt(K) read scaling; per-update projections and gate production excluded"
                if recipe == "gated_delta_product_core"
                else "IPLR additive key-value write with low-rank left transition; full-layer factors/projections excluded"
                if recipe == "generalized_delta_iplr_core"
                else "DPLR additive key-value write with diagonal-plus-low-rank left transition; full-layer factors/projections excluded"
                if recipe == "generalized_delta_dplr_core"
                else "RWKV-4 max-shifted log-sum-exp time-mix state core; frontend time-mix and output gate excluded"
                if recipe == "rwkv4_memory_core"
                else "RWKV-6 key-channel-decayed matrix recurrence with static bonus read correction; time-mix frontend, projections and output gate excluded"
                if recipe == "rwkv6_memory_core"
                else "Momentum DeltaNet coupled fast-weight and momentum matrix-state recurrence; model-side normalization variants, frontend and projections excluded"
                if recipe == "momentum_delta_core"
                else "MesaNet dual covariance-state recurrence with a per-token regularized solve; model-side Q/K normalization, lambda production, and streaming-state qualification remain external"
                if recipe == "mesa_net_core"
                else "Titans chunked associative-memory inner update with learned reconstruction loss and normalized readout; outer attention, projection/gate production, memory hierarchy and complete model block excluded"
                if recipe == "titans_linear_memory_core"
                else "TTT-Linear chunkwise learned inner-loss update with matrix/bias states; projections, MLP variant and complete layer excluded"
                if recipe == "ttt_linear_core"
                else "PaTH chunkwise triangular q/k transformation and gated causal softmax; model-side projections, short convolution and cache/decode integration excluded"
                if recipe == "path_attention_core"
                else "Rodimus BF16 key-channel GLA with V-first state and source-default 1/sqrt(K) read scale; gate/input projections, short convolution, normalization and output projection excluded"
                if recipe == "rodimus_gla_core"
                else "DeltaFormer two-stage causal value correction followed by softmax attention; q/k/v/beta projections and the complete layer excluded"
                if recipe == "deltaformer_attention_core"
                else "Gated Oja matrix recurrence with value-channel decay and key residual correction; gate generation and full layer excluded"
                if recipe == "gated_oja_core"
                else "COMBA head-decayed dual-key delta recurrence; model-side feature projection and full layer excluded"
                if recipe == "comba_core"
                else "Preconditioned gated delta recurrence with a learned ATK diagonal key metric; q/k normalization and full layer excluded"
                if recipe == "pgdn_core"
                else "Preconditioned KDA key-channel delta recurrence with explicit learned ATK metric and decay gates; frontend and full layer excluded"
                if recipe == "pkda_core"
                else "ABC two-stage slot recurrence with logcumsumexp-normalized slot weights; slot frontend and full layer excluded"
                if recipe == "abc_core"
                else "GSA two-stage slot recurrence with explicit slot weights and forget gates; frontend and full layer excluded"
                if recipe == "gsa_core"
                else "RWKV-7 DPLR state transition and additive KV write; model-side factors and projections excluded"
                if recipe == "rwkv7_transition_core"
                else "DSA causal softmax under precomputed selected-token routes; indexer and projection/frontend excluded"
                if recipe == "dsa_attention_core"
                else "NSA selected-block attention under precomputed block routes; compression, indexer and gate branches excluded"
                if recipe == "nsa_selected_attention_core"
                else "FoX forget-gated causal softmax under an additive pairwise log-decay bias; Q/K/V projections and gate production excluded"
                if recipe == "fox"
                else "Parallax causal attention with secondary-query correction; Q/R/K/V projections and feature transforms excluded"
                if recipe == "parallax_attention_core"
                else "Wall causal attention with per-channel gate-scaled q/k logits; Q/K/V/gate production and optional scalar sink/window inputs excluded"
                if recipe == "wall_attention_core"
                else "causal polynomial feature-state attention; Q/K/V projections and architecture-level normalization excluded"
                if recipe in {"based_attention_core", "rebased_attention_core"}
                else "K2 recurrent operator; gate and projection production excluded"
            ),
            "shape": {
                "batch": 1,
                "sequence": int(primary_input.shape[1]),
                "heads": (
                    int(operands["query"].shape[2])
                    if recipe == "nsa_selected_attention_core"
                    else int(operands["value"].shape[2])
                    if recipe != "rwkv4_memory_core"
                    else None
                ),
                "query_heads": (
                    int(operands["query"].shape[2])
                    if recipe != "rwkv4_memory_core"
                    else None
                ),
                "key_value_heads": (
                    int(operands["key"].shape[2])
                    if recipe != "rwkv4_memory_core"
                    else None
                ),
                "key_dim": int(operands["query"].shape[-1]) if recipe != "rwkv4_memory_core" else None,
                "value_dim": int(operands["value"].shape[-1]) if recipe != "rwkv4_memory_core" else None,
                "channels": int(primary_input.shape[-1]) if recipe == "rwkv4_memory_core" else None,
                "block_size": 32 if recipe == "nsa_selected_attention_core" else None,
                "selected_block_slots": (
                    int(operands["block_indices"].shape[-1])
                    if recipe == "nsa_selected_attention_core"
                    else None
                ),
                "selected_block_count_min": (
                    int(operands["block_counts"].min().item())
                    if recipe == "nsa_selected_attention_core"
                    else None
                ),
                "selected_block_count_max": (
                    int(operands["block_counts"].max().item())
                    if recipe == "nsa_selected_attention_core"
                    else None
                ),
                "dtype": str(primary_input.dtype).removeprefix("torch."),
                "update_rank": (
                    int(operands["update_keys"].shape[2])
                    if recipe == "gated_delta_product_core"
                    else None
                ),
            },
            "upstream_callable": (
                {
                    "linear_attention": "fla.ops.linear_attn.chunk.chunk_linear_attn",
                    "retention_core": "fla.ops.simple_gla.fused_chunk.fused_chunk_simple_gla",
                    "lightning_attention_core": "fla.ops.simple_gla.chunk.chunk_simple_gla",
                    "lightnet_gla_core": "fla.ops.gla.chunk.chunk_gla",
                    "simple_gla": "fla.ops.simple_gla.chunk.chunk_simple_gla",
                    "gla": "fla.ops.gla.chunk.chunk_gla",
                    "delta_net": "fla.ops.delta_rule.chunk.chunk_delta_rule",
                    "gated_delta_net": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
                    "hgrn2_ssm_core": "fla.ops.gla.chunk.chunk_gla",
                    "gdn2_core": "fla.ops.gdn2.chunk.chunk_gdn2",
                    "based_attention_core": "fla.ops.based.fused_chunk.fused_chunk_based",
                    "rebased_attention_core": "fla.ops.rebased.parallel.parallel_rebased",
                    "log_linear_attention_core": "fla.ops.log_linear_attn.chunk.chunk_log_linear_attn",
                    "kda_core": "fla.ops.kda.chunk.chunk_kda",
                    "gated_delta_product_core": "fla.ops.gated_delta_product.chunk.chunk_gated_delta_product",
                    "generalized_delta_iplr_core": "fla.ops.generalized_delta_rule.iplr.fused_recurrent.fused_recurrent_iplr_delta_rule",
                    "generalized_delta_dplr_core": "fla.ops.generalized_delta_rule.dplr.chunk.chunk_dplr_delta_rule",
                    "rwkv4_memory_core": "fla.ops.rwkv4.fused_recurrent.fused_recurrent_rwkv4",
                    "rwkv6_memory_core": "fla.ops.rwkv6.fused_recurrent.fused_recurrent_rwkv6",
                    "momentum_delta_core": "fla.ops.momentum_delta_rule.chunk.chunk_momentum_delta_rule",
                    "mesa_net_core": "fla.ops.mesa_net.chunk.chunk_mesa_net",
                    "titans_linear_memory_core": "fla.ops.titans.naive.chunk_titans_linear_ref (use_chunk=True)",
                    "ttt_linear_core": "fla.ops.ttt.chunk.chunk_ttt_linear",
                    "path_attention_core": "fla.ops.path_attn.parallel.parallel_path_attn",
                    "rodimus_gla_core": "fla.ops.gla.chunk.chunk_gla",
                    "deltaformer_attention_core": "fla.ops.deltaformer.parallel.deltaformer_attn",
                    "gated_oja_core": "fla.ops.gated_oja_rule.chunk.chunk_gated_oja_rule",
                    "comba_core": "fla.ops.comba.chunk.chunk_comba",
                    "pgdn_core": "fla.ops.precond_gated_delta_rule.chunk.chunk_precond_gated_delta_rule",
                    "pkda_core": "fla.ops.precond_kda.chunk.chunk_precond_kda",
                    "abc_core": "fla.ops.abc.chunk.chunk_abc",
                    "gsa_core": "fla.ops.gsa.chunk.chunk_gsa",
                    "rwkv7_transition_core": "fla.ops.rwkv7.chunk.chunk_rwkv7",
                    "dsa_attention_core": "fla.ops.dsa.naive.naive_dsa",
                    "nsa_selected_attention_core": "fla.ops.nsa.parallel.parallel_nsa",
                    "fox": "fla.ops.forgetting_attn.parallel.parallel_forgetting_attn",
                    "parallax_attention_core": "fla.ops.parallax.parallel.parallel_parallax",
                    "wall_attention_core": "fla.ops.wall_attn.parallel.parallel_wall_attn",
                    "moba_selected_attention_core": "fla.ops.moba.parallel.parallel_moba",
                }[recipe]
            ),
            "architecture_entrypoint": {
                "retention_core": "fla.ops.retention.fused_chunk.fused_chunk_retention",
                "lightning_attention_core": "fla.ops.lightning_attn.chunk.chunk_lightning_attn",
            }.get(recipe),
            "compiled_anchor": plan.anchor,
            "compiler_plan_build_ms": plan_build_ms,
            "parity": {
                "status": "pass",
                "output_max_abs_error": output_error,
                "final_state_max_abs_errors": state_errors,
                "input_gradient_max_abs_errors": dict(
                    zip(operands, gradient_errors, strict=True)
                ),
                "tolerances": {
                    "output_atol": value_atol,
                    "state_atol": state_atol,
                    "gradient_atol": (
                        gradient_atol
                    ),
                },
            },
            "reference_equation_parity": reference_equation,
            "architecture_entrypoint_parity": wrapper_parity,
            "performance": performance,
        }

    config = {"recipes": recipes, "pairs": pairs, "warmup": warmup}
    payload = {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "purpose": "compare unified mixer compiler plan overhead with direct pinned FLA operators",
        "upstream": {
            "repository": "https://github.com/fla-org/flash-linear-attention",
            "revision": revision,
            "module_version": getattr(fla, "__version__", None),
            "distribution_version": importlib.metadata.version(
                "flash-linear-attention"
            ),
            "fla_core_distribution_version": importlib.metadata.version("fla-core"),
            "loaded_module": fla.__file__,
            "nested_flash_attention": nested_flash_attention,
        },
        "provenance": provenance(
            "PYTHONPATH=/path/to/fla-checkout:src python benchmarks/unified_mixer_fla.py",
            config,
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "methodology": {
            "timed_work": "one forward call or one forward plus backward on preallocated leaf inputs",
            "sampling": "paired interleaved direct/compiled calls, order alternates, synchronized wall and CUDA event timing",
            "warmup": warmup,
            "pairs": pairs,
            "overhead": "median of per-pair (compiled-direct)/direct fractions",
            "overhead_gate_fraction": 0.10,
            "overhead_gate_rule": "median compiled slowdown must not exceed 10%; speedups pass",
            "interpretation": "the library backend calls the same upstream kernel; measures URM compiler-plan and validation overhead, not a new kernel speedup",
        },
        "cases": cases,
    }
    write_artifact(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=21)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--only", choices=RECIPES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/unified-mixer/fla-gated-additive.json"),
    )
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup < 0:
        parser.error("--pairs must be positive and --warmup nonnegative")
    run(
        args.pairs,
        args.warmup,
        args.output,
        recipes=(args.only,) if args.only else RECIPES,
    )


if __name__ == "__main__":
    main()
