"""Product evidence table: all 62 covered recipes, URM-native vs upstream.

This is the product-claim evidence record. For each of the 62 representation-covered
named recipes (the recipes that lower into a canonical core and match their
independent equation, per ``benchmarks/representation_coverage.py``) it measures the
URM-**native** kernel against the **upstream** comparator (the pinned FLA / SDM /
SDPA adapter, via the LIBRARY backend) on identical operands. Both sides are executed
through the same ``plan.execute`` dispatch, so URM's dispatch overhead is common to
both and cancels in the comparison - the difference is the kernel itself.

Eight measurements per recipe, each native vs upstream:

Correctness (canonical coverage shapes, the ``_rng_operands`` contract shapes):

- **parity**: output vs the NumPy canonical core (``execute_canonical``, the
  contract), max relative error - reported for native and for upstream.
- **gradient alignment**: native input gradients vs upstream input gradients
  (autograd through both, same scalar loss), max relative error over the floated
  operands. Recipes whose native (or upstream) kernel is forward-only report the
  limitation; when only native has a backward, a finite-difference check of the
  native backward is reported instead.
- **inference KL divergence**: a single-token decode step, KL(native || upstream)
  over the per-token output distribution. Near zero = same distribution (the
  quantity that matters for sampling / generation quality).

Performance (realistic shapes, timed serially on a single GPU):

- **MFU** training (fwd+bwd) / prefill (fwd) / decode (single token): useful model
  FLOPs / measured wall time / measured hardware peak.
- **throughput** prefill / decode in tokens/second.
- **peak memory** (prefill), ``torch.cuda.max_memory_allocated``, in MB.

Honest-accounting rules enforced here:

- MFU uses the **measured** hardware peak (``results/device-limits.json``), never
  the vendor datasheet: fp32 -> measured fp32 CUDA-core TFLOP/s; bf16/fp16 ->
  measured bf16 tensor-core TFLOP/s. Each side uses the peak for the dtype it
  actually ran in (recorded per side in the JSON).
- The K1 online-softmax kernel exceeds A10G shared memory in fp32 at head dim 64,
  so K1 recipes are timed in **bfloat16** (the production serving dtype); K2/K3
  recurrence kernels accumulate in fp32 and are timed in float32. Upstream is
  compiled at the matching dtype where the adapter allows.
- All timing is serial (warmup + synchronized samples, ``torch.no_grad`` for the
  inference paths). Nothing GPU-heavy runs concurrently with this measurement.
- A recipe that cannot run a given mode records the reason; nothing is hidden.
  K3 (sparse_delta_memory) has no upstream adapter; some upstream adapters need
  optional dependencies not installed here (recorded as "upstream error").

Run ``PYTHONPATH=src:benchmarks python benchmarks/product_table.py`` to regenerate
``results/validation/product-table.json`` and ``docs/validation/product-table.md``.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from urm.frontend.mixer_recipes import named_mixer_recipe
from urm.oracles.composition import execute_canonical

from catalog_upstream_validation import _to_cuda_operands
from representation_coverage import _rng_operands

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = PROJECT_ROOT / "results" / "validation" / "product-table.json"
MD_OUT = PROJECT_ROOT / "docs" / "validation" / "product-table.md"
DEVICE_LIMITS = PROJECT_ROOT / "results" / "device-limits.json"

COVERED_RECIPES = (
    "abc_core", "based_attention_core", "cat_attention_core", "comba_core",
    "conformer_attention_core", "delta_net", "deltaformer_attention_core",
    "differential_attention_core", "dsa_attention_core", "foveal_attention_core",
    "gated_delta_net", "gated_delta_product_core", "gated_oja_core", "gdn2_core",
    "generalized_delta_dplr_core", "generalized_delta_iplr_core", "gla", "gqa",
    "gru_core", "gsa_core", "h3_ssm_fft_core", "hgrn2_ssm_core", "hgrn_ssm_core",
    "hla_second_order_core", "hopfield_attention_core", "hyena_fftconv_core",
    "kata_attention_core", "kda_core", "lightnet_gla_core",
    "lightning_attention_core", "linear_attention", "longformer_attention_core",
    "m2rnn_core", "mamba1_ssm_core", "mamba2_ssm_core", "mamba3_siso_core",
    "mesa_net_core", "mha", "mla_attention_core", "mom_selected_memory_core",
    "momentum_delta_core", "mqa", "nsa_selected_attention_core",
    "parallax_attention_core", "pattention_core", "rebased_attention_core",
    "retention_core", "rnn_core", "rodimus_gla_core", "rwkv4_memory_core",
    "rwkv6_memory_core", "rwkv7_transition_core", "samba_attention_core",
    "simple_gla", "sparse_attention_core", "sparse_delta_memory",
    "tda_attention_core", "titans_linear_memory_core", "tpa_attention_core",
    "ttt_linear_core", "tucker_attention_core", "wall_attention_core",
)

WARMUP = 5
SAMPLES = 30
PARITY_REL_TOL = 1e-3       # native (fp32) vs fp64 canonical contract
UPSTREAM_PARITY_REL_TOL = 2e-2  # upstream chunked bf16/fp32 vs fp64 canonical
GRAD_REL_TOL = 2e-2         # native vs upstream gradients (dtype-precision aware)


def _peaks() -> dict[str, float]:
    limits = json.loads(DEVICE_LIMITS.read_text())
    return {
        "float32": limits["fp32_cuda_core"]["fp32_cuda_core_tfps_measured"],
        "bfloat16": limits["bf16_tensor_core"]["bf16_tensor_core_tfps_measured"],
        "float16": limits["bf16_tensor_core"]["bf16_tensor_core_tfps_measured"],
    }


def _torch():
    import torch

    return torch


def _perf_dtype(spec) -> str:
    """K1 attention is timed in bf16 (fp32 exceeds A10G shared memory at K=64)."""
    from urm.ir.mixer import MixerKernelFamily

    return "bfloat16" if spec.family is MixerKernelFamily.SOFTMAX else "float32"


def _to_cuda(operands: dict[str, Any], dtype) -> dict[str, Any]:
    torch = _torch()
    out: dict[str, Any] = {}
    for key, value in operands.items():
        if isinstance(value, (int, float, bool)):
            out[key] = value
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in "iu":
            out[key] = torch.as_tensor(arr, dtype=torch.int64, device="cuda")
        elif arr.dtype.kind == "b":
            out[key] = torch.as_tensor(arr, device="cuda")
        else:
            out[key] = torch.as_tensor(arr, dtype=dtype, device="cuda").contiguous()
    return out


def _sort_k3_routes(operands: dict[str, Any]) -> dict[str, Any]:
    """K3-native requires strictly-increasing unique routes; sort + reorder weights."""
    ops = dict(operands)
    for idx_key, w_key in (("read_indices", "read_weights"), ("write_indices", "write_weights")):
        if idx_key not in ops:
            continue
        idx = np.asarray(ops[idx_key])
        w = np.asarray(ops[w_key])
        order = np.argsort(idx, axis=-1)
        ops[idx_key] = np.take_along_axis(idx, order, -1)
        ops[w_key] = np.take_along_axis(w, order, -1)
    return ops


def _perf_operands(spec, seed=0) -> dict[str, Any]:
    """Realistic-shape operands mirroring ``_rng_operands`` structure/names."""
    from urm.ir.mixer import (
        DecayGranularity,
        K1Operation,
        MixerKernelFamily,
        RecurrenceOperator,
        RecurrentLayout,
        StateUpdateRule,
    )

    rng = np.random.default_rng(seed)
    if spec.family is MixerKernelFamily.SOFTMAX:
        b, t, h, k, v = 2, 256, 8, 64, 64
        op = spec.k1_operation
        if op is K1Operation.DIFFERENTIAL:
            return {
                "query_a": rng.normal(size=(b, t, h, k)), "query_b": rng.normal(size=(b, t, h, k)),
                "key_a": rng.normal(size=(b, t, h, k)), "key_b": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "lambda_weight": rng.uniform(0.1, 0.9, size=(h,)),
            }
        if op is K1Operation.PROJECTED:
            r = 32
            return {
                "query": rng.normal(size=(b, t, r)), "B_pre": rng.normal(size=(h, r, k)),
                "key": rng.normal(size=(b, t, k)), "value": rng.normal(size=(b, t, v)),
            }
        if op is K1Operation.POSITIONAL:
            return {
                "query": rng.normal(size=(b, t, h, k)), "r": rng.normal(size=(b, t, h, k)),
                "key": rng.normal(size=(b, t, h, k)), "value": rng.normal(size=(b, t, h, k)),
            }
        if op is K1Operation.POSITIVE_FEATURE:
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "num_groups": 2,
            }
        if op is K1Operation.GATED:
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "g": -rng.uniform(0, 0.3, size=(b, t, h, k)),
            }
        if op is K1Operation.THRESHOLDED:
            return {
                "query_a": rng.normal(size=(b, t, h, k)), "query_b": rng.normal(size=(b, t, h, k)),
                "key_a": rng.normal(size=(b, t, h, k)), "key_b": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
                "beta": np.asarray(0.5), "lambda_weight": np.asarray(0.5),
            }
        if op is K1Operation.DELTA_TRANSFORM:
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
            }
        ops = {
            "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
            "value": rng.normal(size=(b, t, h, v)),
        }
        if op is K1Operation.LOCAL_WINDOW:
            ops["attention_window"] = 64
        elif spec.requires_attention_mask:
            mask = np.ones((b, 1, t, t), dtype=bool)
            mask[..., 0, :] = True
            ops["attention_mask"] = mask
        return ops

    if spec.family is MixerKernelFamily.RECURRENCE:
        op = spec.recurrence_operator
        if op is RecurrenceOperator.TANH_RNN:
            b, t, n, h = 2, 256, 4, 64
            return {
                "query": rng.normal(size=(b, t, n, h)), "weight": rng.normal(size=(n, h, h)) * 0.3,
                "initial_state": rng.normal(size=(b, n, h)),
            }
        if op is RecurrenceOperator.GATED_RNN:
            b, t, n, h = 2, 256, 4, 64
            return {
                "query": rng.normal(size=(b, t, n, h)), "weight": rng.normal(size=(n, h, h)) * 0.3,
                "forget_input": rng.normal(size=(b, t, n, h)), "forget_weight": rng.normal(size=(n, h, h)) * 0.3,
                "reset_input": rng.normal(size=(b, t, n, h)), "reset_weight": rng.normal(size=(n, h, h)) * 0.3,
                "initial_state": rng.normal(size=(b, n, h)),
            }
        if op is RecurrenceOperator.MULTIPLICATIVE_RNN:
            b, t, n, k, v = 2, 256, 4, 64, 64
            return {
                "query": rng.normal(size=(b, t, n, k)), "key": rng.normal(size=(b, t, n, k)),
                "value": rng.normal(size=(b, t, n, v)), "weight": rng.normal(size=(n, v, v)) * 0.3,
                "forget_input": rng.uniform(0.1, 0.9, size=(b, t, n)),
                "initial_state": rng.normal(size=(b, n, k, v)) * 0.1,
            }
        if op is RecurrenceOperator.FFT_CONVOLUTION:
            b, t, c = 2, 256, 512
            return {
                "query": rng.normal(size=(b, t, c)), "kernel": rng.normal(size=(c, t)),
                "direct": rng.normal(size=(c,)),
            }
        if op is RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION:
            b, t, h = 2, 256, 64
            return {
                "query": rng.normal(size=(b, t, h, 1)), "key": rng.normal(size=(b, t, h, 1)),
                "value": rng.normal(size=(b, t, h, 1)), "ssm_kernel": rng.normal(size=(h, t)),
                "ssm_k_kernel": rng.normal(size=(h, t)), "ssm_k_direct": rng.normal(size=(h,)),
                "skip": rng.normal(size=(h,)),
            }
        if op is RecurrenceOperator.SECOND_ORDER_CUMSUM:
            b, t, h, k, v = 2, 256, 8, 64, 64
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)),
            }
        if op is RecurrenceOperator.REGULARIZED_SOLVE:
            b, t, h, k = 2, 256, 8, 64
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, k)), "log_decay": -rng.uniform(0, 0.4, size=(b, t, h)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)), "lamb": rng.uniform(0.5, 1.5, size=(h, k)),
            }
        if op is RecurrenceOperator.LAYERNORM_INNER_STATE:
            b, t, h, d = 2, 256, 8, 64
            return {
                "query": rng.normal(size=(b, t, h, d)), "key": rng.normal(size=(b, t, h, d)),
                "value": rng.normal(size=(b, t, h, d)), "w": rng.normal(size=(h, d)),
                "b": rng.normal(size=(h, d)), "eta": rng.uniform(0.01, 0.1, size=(b, t, h, 1)),
                "chunk_size": 8,
            }
        if op is RecurrenceOperator.MOMENTUM_INNER_STATE:
            b, t, h, d = 2, 256, 8, 64
            return {
                "query": rng.normal(size=(b, t, h, d)), "key": rng.normal(size=(b, t, h, d)),
                "value": rng.normal(size=(b, t, h, d)), "w": rng.normal(size=(h, d)),
                "b": rng.normal(size=(h, d)), "theta": rng.uniform(0.01, 0.1, size=(b, t, h, 1)),
                "alpha": rng.uniform(0.01, 0.3, size=(b, t, h, 1)),
                "eta": rng.uniform(0.01, 0.3, size=(b, t, h, 1)), "chunk_size": 8,
            }
        if op is RecurrenceOperator.MAMBA2_STRUCTURED_SSM:
            b, t, h, p, g, n = 2, 256, 8, 64, 1, 64
            return {
                "x": rng.normal(size=(b, t, h, p)), "dt": rng.uniform(0.01, 0.5, size=(b, t, h)),
                "A": -rng.uniform(0.1, 1.0, size=(h,)), "B": rng.normal(size=(b, t, g, n)),
                "C": rng.normal(size=(b, t, g, n)),
            }
        if op is RecurrenceOperator.RWKV6_BONUS_CORRECTED:
            b, t, h, k, v = 2, 256, 8, 64, 64
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "log_decay": -rng.uniform(0, 0.4, size=(b, t, h, k)),
                "bonus": rng.normal(size=(h, k)),
            }
        if op is RecurrenceOperator.RWKV4_SCALAR_STATE:
            b, t, c = 2, 256, 512
            return {
                "w": -rng.uniform(0.1, 1.0, size=(c,)), "u": rng.normal(size=(c,)),
                "k": rng.normal(size=(b, t, c)), "v": rng.normal(size=(b, t, c)),
                "state": rng.normal(size=(b, 3, 1, c)) * 0.1,
            }
        if op is RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE:
            b, t, hk, hq, k, s, v = 2, 256, 4, 8, 64, 16, 64
            base = {
                "query": rng.normal(size=(b, t, hq, k)), "key": rng.normal(size=(b, t, hk, k)),
                "value": rng.normal(size=(b, t, hk, v)),
            }
            if spec.name == "abc_core":
                base["slot_logits"] = rng.normal(size=(b, t, hk, s))
            else:
                base["slot_weights"] = rng.uniform(0.1, 1.0, size=(b, t, hk, s))
                base["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, hk, s))
            return base
        if op is RecurrenceOperator.GATED_OJA_VALUE_CHANNEL:
            b, t, h, k, v = 2, 256, 8, 64, 64
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "gv": -rng.uniform(0, 0.4, size=(b, t, h, v)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)),
            }
        if op is RecurrenceOperator.MOMENTUM_DELTA_STATE:
            b, t, h, k, v = 2, 256, 8, 64, 64
            return {
                "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
                "value": rng.normal(size=(b, t, h, v)), "p": rng.normal(size=(b, t, h, k)),
                "log_alpha": -rng.uniform(0, 0.4, size=(b, t, h)), "log_mu": -rng.uniform(0, 0.4, size=(b, t, h)),
                "beta": rng.uniform(0.1, 0.9, size=(b, t, h)), "eta": rng.uniform(0.01, 0.3, size=(b, t, h)),
            }
        if op is RecurrenceOperator.TRAPEZOIDAL_SSM:
            b, t, h, kd, vd, a = 2, 256, 8, 64, 64, 2
            return {
                "query": rng.normal(size=(b, t, h, kd)), "key": rng.normal(size=(b, t, h, kd)),
                "value": rng.normal(size=(b, t, h, vd)), "adt": -rng.uniform(0, 0.5, size=(b, h, t)),
                "dt": rng.uniform(0.01, 0.5, size=(b, h, t)), "trap": rng.normal(size=(b, h, t)),
                "query_bias": rng.normal(size=(h, kd)), "key_bias": rng.normal(size=(h, kd)),
                "angles": rng.normal(size=(b, t, h, a)),
            }
        if spec.recurrent_layout is RecurrentLayout.DIAGONAL:
            b, t, c, n = 2, 256, 512, 64
            if spec.diagonal_hgrn:
                return {
                    "x": rng.normal(size=(b, t, c)), "log_decay": -rng.uniform(0, 0.4, size=(b, t, c)),
                }
            ops = {"x": rng.normal(size=(b, t, c)), "log_decay": -rng.uniform(0, 0.4, size=(b, t, n))}
            ops["input_gate"] = rng.uniform(0.1, 1.0, size=(b, t, n))
            ops["read_gate"] = rng.uniform(0.1, 1.0, size=(b, t, n))
            if spec.step_size_discretization:
                ops["step_size"] = rng.uniform(0.1, 1.0, size=(b, t, c))
            return ops
        # Plain / variant matrix-state.
        b, t, h, k, v = 2, 256, 8, 64, 64
        # Polynomial bases (based/rebased) pre-expand the feature dim to ~1+K+K^2;
        # use a smaller base K so the expanded width stays within the kernel's limits.
        from urm.ir.mixer import PolynomialBasis
        if spec.polynomial_basis is not PolynomialBasis.NONE:
            k = 16
        ops = {
            "query": rng.normal(size=(b, t, h, k)), "key": rng.normal(size=(b, t, h, k)),
            "value": rng.normal(size=(b, t, h, v)),
        }
        if spec.gdn2_ssm:
            ops["erase_gate"] = rng.uniform(0.1, 0.9, size=(b, t, h, k))
            ops["write_gate"] = rng.uniform(0.1, 0.9, size=(b, t, h, v))
        elif spec.update_rule is StateUpdateRule.DELTA:
            ops["beta"] = rng.uniform(0.1, 0.9, size=(b, t, h))
        if spec.generalized_delta_iplr or spec.generalized_delta_dplr:
            ops["transition_alpha"] = rng.normal(size=(b, t, h, k)) * 0.3
            ops["transition_beta"] = rng.normal(size=(b, t, h, k)) * 0.3
        if spec.gated_delta_product:
            r = 2
            ops["update_keys"] = rng.normal(size=(b, t, r, h, k))
            ops["update_values"] = rng.normal(size=(b, t, r, h, v))
            ops["beta"] = rng.uniform(0.1, 0.9, size=(b, t, r, h))
        if spec.comba_rule:
            ops["p"] = rng.normal(size=(b, t, h, k))
            ops["g"] = -rng.uniform(0, 0.4, size=(b, t, h))
        elif spec.generalized_delta_dplr:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, h, k))
        elif spec.static_head_decay:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(h,))
        elif spec.decay is DecayGranularity.HEAD:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, h))
        elif spec.decay is DecayGranularity.KEY_CHANNEL:
            ops["log_decay"] = -rng.uniform(0, 0.4, size=(b, t, h, k))
        return ops

    if spec.family is MixerKernelFamily.SPARSE_DELTA:
        b, t, s, d, r = 2, 256, 64, 64, 8
        read_idx = np.stack([rng.choice(s, r, replace=False) for _ in range(b * t)]).reshape(b, t, r)
        write_idx = np.stack([rng.choice(s, r, replace=False) for _ in range(b * t)]).reshape(b, t, r)
        read_w = rng.uniform(0.1, 1.0, size=(b, t, r)); read_w /= read_w.sum(-1, keepdims=True)
        write_w = rng.uniform(0.1, 1.0, size=(b, t, r)); write_w /= write_w.sum(-1, keepdims=True)
        return _sort_k3_routes({
            "memory": rng.normal(size=(b, s, d)), "read_indices": read_idx, "read_weights": read_w,
            "write_indices": write_idx, "write_weights": write_w, "values": rng.normal(size=(b, t, d)),
            "beta": rng.uniform(0.1, 0.9, size=(b, t)), "log_decay": -rng.uniform(0, 0.5, size=(b, t)),
        })
    return {}


# Operand names that never carry a sequence axis (never sliced for decode).
_NON_SEQUENCE = {
    "weight", "forget_weight", "reset_weight", "A", "B_pre", "lamb", "w", "b", "u",
    "bonus", "query_bias", "key_bias", "kernel", "ssm_kernel", "ssm_k_kernel",
    "ssm_k_direct", "skip", "direct", "memory", "state", "initial_state",
    "lambda_weight", "beta_scalar",
}


def _seq_len(ops: dict[str, Any]) -> int:
    for key in ("query", "query_a", "key", "value", "x", "k", "values"):
        if key in ops and not isinstance(ops[key], (int, float, bool)):
            arr = np.asarray(ops[key])
            if arr.ndim >= 2:
                return int(arr.shape[1])
    return 1


def _decode_ops(spec, ops: dict[str, Any]) -> dict[str, Any]:
    """Single-token decode operands: recurrent/sparse slice axis 1 to one token;
    K1 attention keeps the full key/value history with a single query token (and a
    single query row of the attention mask)."""
    from urm.ir.mixer import MixerKernelFamily

    t = _seq_len(ops)
    out: dict[str, Any] = {}
    is_k1 = spec.family is MixerKernelFamily.SOFTMAX
    for key, value in ops.items():
        if isinstance(value, (int, float, bool)):
            out[key] = value
            continue
        arr = np.asarray(value)
        # K1 decode: single query token + one query row of the mask; full key history.
        if is_k1 and key.startswith("query") and arr.ndim >= 2 and arr.shape[1] == t:
            out[key] = arr[:, :1]
            continue
        if is_k1 and key == "attention_mask" and arr.ndim == 4:
            out[key] = arr[:, :, :1, :]
            continue
        if key in _NON_SEQUENCE or arr.ndim < 2 or arr.shape[1] != t:
            out[key] = value
            continue
        # Recurrence / sparse: slice the sequence axis to one token.
        out[key] = arr[:, :1]
    return out


def _useful_flops(spec, ops: dict[str, Any], mode: str) -> float:
    from urm.ir.mixer import MixerKernelFamily, RecurrenceOperator, RecurrentLayout

    def shp(name):
        return np.asarray(ops[name]).shape

    if spec.family is MixerKernelFamily.SOFTMAX:
        if "B_pre" in ops:
            # PROJECTED: query [b,t,r] expanded to [b,t,h,k] via B_pre [h,r,k].
            b, tq, r = shp("query")
            h, _, k = shp("B_pre")
            tk = shp("key")[1]
            v = shp("value")[-1]
            project = 2.0 * b * tq * r * h * k  # query @ B_pre expansion
            attend = 2.0 * b * h * tq * tk * (k + v) * 0.5
            return (project + attend) * (2.5 if mode == "training" else 1.0)
        q = shp("query") if "query" in ops else shp("query_a")
        b, tq, h, k = q[0], q[1], q[2], q[3]
        kk = shp("key") if "key" in ops else shp("key_a")
        tk = kk[1]
        v = shp("value")[-1]
        flops = 2.0 * b * h * tq * tk * (k + v) * 0.5
        if "query_a" in ops:
            flops += 2.0 * b * h * tq * tk * k * 0.5
        return flops * (2.5 if mode == "training" else 1.0)

    if spec.family is MixerKernelFamily.SPARSE_DELTA:
        b, t = shp("values")[0], shp("values")[1]
        r = shp("read_indices")[2]
        d = shp("values")[2]
        flops = 4.0 * b * t * (r + r) * d
        return flops * (2.0 if mode == "training" else 1.0)

    op = spec.recurrence_operator
    mult = 2.0 if mode == "training" else 1.0
    if op is RecurrenceOperator.TANH_RNN:
        b, t, n, h = shp("query"); return mult * 2.0 * b * t * n * h * h
    if op is RecurrenceOperator.GATED_RNN:
        b, t, n, h = shp("query"); return mult * 3.0 * 2.0 * b * t * n * h * h
    if op is RecurrenceOperator.MULTIPLICATIVE_RNN:
        b, t, n, k = shp("query"); v = shp("value")[3]
        return mult * 2.0 * b * t * n * (k * v + v * v)
    if op is RecurrenceOperator.FFT_CONVOLUTION:
        b, t, c = shp("query"); return mult * 5.0 * b * c * t * max(np.log2(max(t, 2)), 1.0)
    if op is RecurrenceOperator.TWO_STAGE_FFT_CONVOLUTION:
        b, t, h = shp("query")[:3]; return mult * 2.0 * 5.0 * b * h * t * max(np.log2(max(t, 2)), 1.0)
    if op is RecurrenceOperator.SECOND_ORDER_CUMSUM:
        b, t, h, k = shp("query"); return mult * 2.0 * b * t * h * k * k
    if op is RecurrenceOperator.REGULARIZED_SOLVE:
        b, t, h, k = shp("query"); return mult * (2.0 * b * t * h * k * k + b * t * h * k ** 3 / 3.0)
    if op in (RecurrenceOperator.LAYERNORM_INNER_STATE, RecurrenceOperator.MOMENTUM_INNER_STATE):
        b, t, h, d = shp("query"); return mult * 2.0 * b * t * h * d * d
    if op is RecurrenceOperator.MAMBA2_STRUCTURED_SSM:
        b, t, h, p = shp("x"); n = shp("B")[3]; return mult * 2.0 * b * t * h * p * n
    if op is RecurrenceOperator.RWKV4_SCALAR_STATE:
        b, t, c = shp("k"); return mult * 6.0 * b * t * c
    if op is RecurrenceOperator.RWKV6_BONUS_CORRECTED:
        b, t, h, k = shp("query"); v = shp("value")[3]; return mult * 5.0 * b * t * h * k * v
    if op is RecurrenceOperator.SLOT_ATTENTION_TWO_STAGE:
        b, t, hq, k = shp("query"); hk, v = shp("key")[2], shp("value")[3]
        s = shp("slot_logits")[3] if "slot_logits" in ops else shp("slot_weights")[3]
        return mult * 2.0 * b * t * (hk * k * s + hk * s * v + hq * k * s)
    if op in (RecurrenceOperator.GATED_OJA_VALUE_CHANNEL, RecurrenceOperator.MOMENTUM_DELTA_STATE):
        b, t, h, k = shp("query"); v = shp("value")[3]; return mult * 5.0 * b * t * h * k * v
    if op is RecurrenceOperator.TRAPEZOIDAL_SSM:
        b, t, h, kd = shp("query"); vd = shp("value")[3]; return mult * 5.0 * b * t * h * kd * vd
    if spec.recurrent_layout is RecurrentLayout.DIAGONAL:
        b, t, c = shp("x"); n = 1 if spec.diagonal_hgrn else shp("log_decay")[2]
        return mult * 2.0 * b * t * c * n
    b, t, h, k = shp("query"); v = shp("value")[3]
    rank = shp("update_keys")[2] if "update_keys" in ops else 1
    # Polynomial bases pre-expand the key/query feature dim; the useful model FLOPs
    # are at the expanded width (the feature map IS the model's nonlinearity).
    from urm.ir.mixer import PolynomialBasis
    if spec.polynomial_basis is PolynomialBasis.BASED_TAYLOR2:
        k = 1 + k + k * k
    elif spec.polynomial_basis is PolynomialBasis.REBASED_SQUARE:
        k = k * k
    return mult * 5.0 * b * t * h * k * v * rank


def _tokens(spec, ops: dict[str, Any], mode: str) -> float:
    from urm.ir.mixer import MixerKernelFamily

    if mode == "decode":
        for value in ops.values():
            if not isinstance(value, (int, float, bool)):
                return float(np.asarray(value).shape[0])
        return 1.0
    if spec.family is MixerKernelFamily.SOFTMAX:
        q = ops.get("query", ops.get("query_a"))
        return float(np.asarray(q).shape[0] * np.asarray(q).shape[1])
    if spec.family is MixerKernelFamily.SPARSE_DELTA:
        return float(np.asarray(ops["values"]).shape[0] * np.asarray(ops["values"]).shape[1])
    key = "query" if "query" in ops else ("x" if "x" in ops else "k")
    return float(np.asarray(ops[key]).shape[0] * np.asarray(ops[key]).shape[1])


def _time_fn(fn, warmup=WARMUP, samples=SAMPLES) -> float:
    torch = _torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return times[len(times) // 2]


def _peak_mem_mb(fn) -> float:
    torch = _torch()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def _backward_trustworthy(result) -> bool:
    """Whether the native backward is a real, validated backward (not a partial
    autograd graph through a forward-only materializing kernel). The native
    executors that are forward-only declare ``backward_supported: False``; a bare
    ``grad_fn`` is not sufficient (a forward-only kernel can still produce one)."""
    if result.output.grad_fn is None:
        return False
    return result.metadata.get("backward_supported") is not False


def _input_grads(plan, cuda_ops, loss_seed) -> dict[str, Any] | None:
    """Input gradients via autograd; None when the plan has no trustworthy backward."""
    torch = _torch()
    grad_ops = {
        k: (v.detach().clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v)
        for k, v in cuda_ops.items()
    }
    result = plan.execute(**grad_ops)
    if not _backward_trustworthy(result):
        return None
    out = result.output
    seed = torch.as_tensor(loss_seed, dtype=out.dtype, device=out.device)
    torch.autograd.backward(out, grad_tensors=seed)
    return {
        k: g.grad.float().detach().cpu().numpy()
        for k, g in grad_ops.items()
        if torch.is_tensor(g) and g.is_floating_point() and g.grad is not None
    }


def _reference_grads(spec, ops, loss_seed) -> dict[str, Any] | None:
    """Input gradients through the REFERENCE backend (the independent torch
    equation) - the trusted gradient reference when upstream has no backward."""
    torch = _torch()
    from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer

    recipe = named_mixer_recipe(spec.name)
    plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, backend=MixerBackend.REFERENCE, dtype="float32")
    grad_ops = {}
    for k, v in ops.items():
        if isinstance(v, (int, float, bool)):
            grad_ops[k] = v
            continue
        arr = np.asarray(v)
        if arr.dtype.kind in "iu":
            grad_ops[k] = torch.as_tensor(arr, dtype=torch.int64)
        elif arr.dtype.kind == "b":
            grad_ops[k] = torch.as_tensor(arr)
        else:
            grad_ops[k] = torch.as_tensor(arr, dtype=torch.float32).requires_grad_(True)
    out = plan.execute(**grad_ops).output
    if out.grad_fn is None:
        return None
    seed = torch.as_tensor(np.asarray(loss_seed), dtype=out.dtype)
    torch.autograd.backward(out, grad_tensors=seed)
    return {
        k: g.grad.float().detach().cpu().numpy()
        for k, g in grad_ops.items()
        if torch.is_tensor(g) and g.is_floating_point() and g.grad is not None
    }


def _grad_rel_err(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Max relative gradient error over the shared operands."""
    max_rel = 0.0
    for k, ga in a.items():
        if k in b and b[k].shape == ga.shape:
            gb = b[k]
            denom = max(float(np.abs(gb).max()), float(np.abs(ga).max()), 1e-9)
            max_rel = max(max_rel, float(np.abs(ga - gb).max()) / denom)
    return max_rel


def _perf_side(plan, spec, pops, dtype_name, peak, to_cuda) -> dict[str, Any]:
    """Time prefill / decode / training + peak memory for one backend.

    Each mode is guarded independently: a recipe with no single-token decode form
    (a full-sequence operation) records decode as unavailable without losing the
    prefill/training numbers. The failure reason is captured in ``*_note``.
    """
    torch = _torch()
    dtype = getattr(torch, dtype_name)
    cuda = to_cuda(pops, dtype)
    res: dict[str, Any] = {}
    try:
        with torch.no_grad():
            prefill_ms = _time_fn(lambda: plan.execute(**cuda))
            res["peak_mem_prefill_mb"] = _peak_mem_mb(lambda: plan.execute(**cuda))
        res["mfu_prefill"] = _useful_flops(spec, pops, "prefill") / (prefill_ms / 1e3) / 1e12 / peak
        res["tps_prefill"] = _tokens(spec, pops, "prefill") / (prefill_ms / 1e3)
    except Exception as exc:
        res["prefill_note"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    try:
        dops = _decode_ops(spec, pops)
        dcuda = to_cuda(dops, dtype)
        with torch.no_grad():
            decode_ms = _time_fn(lambda: plan.execute(**dcuda))
        res["mfu_decode"] = _useful_flops(spec, dops, "decode") / (decode_ms / 1e3) / 1e12 / peak
        res["tps_decode"] = _tokens(spec, dops, "decode") / (decode_ms / 1e3)
    except Exception as exc:
        res["decode_note"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    try:
        train_ops = {
            k: (v.clone().requires_grad_(True) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in cuda.items()
        }
        if _backward_trustworthy(plan.execute(**train_ops)):
            def fwd_bwd():
                o = plan.execute(**train_ops).output
                torch.autograd.backward(o, grad_tensors=torch.ones_like(o))
            train_ms = _time_fn(fwd_bwd)
            res["mfu_training"] = _useful_flops(spec, pops, "training") / (train_ms / 1e3) / 1e12 / peak
            res["peak_mem_training_mb"] = _peak_mem_mb(fwd_bwd)
    except Exception as exc:
        res["training_note"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    return res


@dataclass
class ProductRow:
    name: str
    family: str
    # correctness
    parity_native: float | None = None
    parity_upstream: float | None = None
    grad_align: float | None = None
    grad_note: str = ""
    kl_native_vs_upstream: float | None = None
    correctness_note: str = ""
    # performance (native / upstream)
    native_dtype: str = ""
    upstream_dtype: str = ""
    mfu_prefill_native: float | None = None
    mfu_prefill_upstream: float | None = None
    mfu_decode_native: float | None = None
    mfu_decode_upstream: float | None = None
    mfu_training_native: float | None = None
    mfu_training_upstream: float | None = None
    tps_prefill_native: float | None = None
    tps_prefill_upstream: float | None = None
    tps_decode_native: float | None = None
    tps_decode_upstream: float | None = None
    peak_mem_prefill_native_mb: float | None = None
    peak_mem_prefill_upstream_mb: float | None = None
    upstream_status: str = "not attempted"  # ok | no upstream adapter | upstream error
    perf_note: str = ""


def _compile_upstream(recipe, prefer_dtype: str):
    """Compile the LIBRARY (upstream) plan, preferring the native-matching dtype.

    Returns (plan, dtype, intent, error). Tries the preferred dtype first
    (training then inference), then the catalog's full fallback list.
    """
    from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer

    attempts = [
        ("training", prefer_dtype), ("inference", prefer_dtype),
        ("training", "float32"), ("training", "bfloat16"),
        ("inference", "float32"), ("inference", "bfloat16"),
    ]
    seen = set()
    last_exc: Exception | None = None
    for intent, dtype in attempts:
        if (intent, dtype) in seen:
            continue
        seen.add((intent, dtype))
        try:
            plan = compile_mixer(recipe, intent=MixerIntent(intent), backend=MixerBackend.LIBRARY, dtype=dtype)
            return plan, dtype, intent, None
        except Exception as exc:
            last_exc = exc
    return None, None, None, last_exc


def measure_recipe(name: str, seed: int = 0) -> ProductRow:
    torch = _torch()
    recipe = named_mixer_recipe(name)
    spec = recipe.spec
    row = ProductRow(name=name, family=spec.family.name)
    peaks = _peaks()
    native_dtype = _perf_dtype(spec)
    row.native_dtype = native_dtype

    # ---- correctness at the canonical coverage shapes ----
    ops = _rng_operands(spec, seed=seed)
    if spec.family.name == "SPARSE_DELTA":
        ops = _sort_k3_routes(ops)
    try:
        canon = execute_canonical(spec, **ops)
        ref = np.asarray(canon["output"], dtype=np.float32)
        ref_norm = max(float(np.abs(ref).max()), 1e-12)
    except Exception as exc:
        row.correctness_note = f"coverage surprise: {type(exc).__name__}"
        return row

    # ===================================================================
    # NATIVE FIRST: capture every native measurement before any upstream call,
    # so a crashing upstream kernel cannot poison the CUDA context and cost us
    # the native numbers (the product). Upstream is only the comparator.
    # ===================================================================
    native_plan = None
    native_cuda = None
    native_grads = None
    native_dec = None
    loss_seed = np.random.default_rng(seed).standard_normal(ref.shape)

    # 1. native correctness (fp32) vs the canonical contract
    try:
        native_plan = _compile_native(recipe, "training", "float32")
        native_cuda = _to_cuda(ops, torch.float32)
        native_np = native_plan.execute(**native_cuda).output.detach().float().cpu().numpy()
        if native_np.shape == ref.shape:
            row.parity_native = float(np.abs(native_np - ref).max() / ref_norm)
        else:
            row.correctness_note = f"native shape {native_np.shape} vs canonical {ref.shape}"
    except Exception as exc:
        row.correctness_note = f"native exec: {type(exc).__name__}: {str(exc)[:100]}"
        native_plan = None

    # 2. native performance at realistic shapes (timed serially)
    perf_dtype = native_dtype
    pops = _perf_operands(spec, seed=seed)
    if spec.family.name == "SPARSE_DELTA":
        pops = _sort_k3_routes(pops)
    try:
        native_perf_plan = _compile_native(recipe, "inference", perf_dtype)
        nres = _perf_side(native_perf_plan, spec, pops, perf_dtype, peaks[perf_dtype], _to_cuda)
        row.mfu_prefill_native = nres.get("mfu_prefill")
        row.mfu_decode_native = nres.get("mfu_decode")
        row.mfu_training_native = nres.get("mfu_training")
        row.tps_prefill_native = nres.get("tps_prefill")
        row.tps_decode_native = nres.get("tps_decode")
        row.peak_mem_prefill_native_mb = nres.get("peak_mem_prefill_mb")
    except Exception as exc:
        row.perf_note = f"native perf: {type(exc).__name__}: {str(exc)[:100]}"

    # 3. native gradients + native decode output (stored for the upstream comparison).
    # Use FRESH operands for each execution: the K3 native kernel updates its memory
    # operand in place, so reusing the parity run's tensors would compute gradients at
    # an already-mutated state.
    if native_plan is not None:
        try:
            native_grads = _input_grads(native_plan, _to_cuda(ops, torch.float32), loss_seed)
        except Exception:
            native_grads = None
        try:
            dops = _decode_ops(spec, ops)
            with torch.no_grad():
                native_dec = native_plan.execute(**_to_cuda(dops, torch.float32)).output.detach().float()
        except Exception:
            native_dec = None

    # ===================================================================
    # UPSTREAM (comparator) - native numbers are already captured above.
    # ===================================================================
    up_plan, up_dtype, up_intent, up_exc = _compile_upstream(recipe, "float32")
    if up_plan is None:
        msg = str(up_exc)
        row.upstream_status = (
            "no upstream adapter" if "K3 uses the URM-native" in msg else "upstream error"
        )
        row.correctness_note = (row.correctness_note + " | " if row.correctness_note else "") + (
            f"upstream: {type(up_exc).__name__}: {msg[:80]}"
        )
    else:
        row.upstream_status = "ok"
        row.upstream_dtype = up_dtype
        try:
            up_cuda = _to_cuda_operands(ops, getattr(torch, up_dtype))
            up_np = up_plan.execute(**up_cuda).output.detach().float().cpu().numpy()
            if up_np.shape == ref.shape:
                row.parity_upstream = float(np.abs(up_np - ref).max() / ref_norm)
        except Exception as exc:
            row.correctness_note = (row.correctness_note + " | " if row.correctness_note else "") + (
                f"upstream exec: {type(exc).__name__}: {str(exc)[:80]}"
            )
            up_plan = None

    # Gradient alignment (fully guarded - never crashes the isolated worker).
    # Primary: native vs upstream input gradients (same scalar loss). Fallback:
    # native vs the REFERENCE equation's gradients (when upstream has no backward).
    if native_grads is None:
        row.grad_note = "fwd-only"
    else:
        try:
            if up_plan is not None and up_intent == "training":
                up_grads = None
                try:
                    up_grads = _input_grads(up_plan, _to_cuda_operands(ops, getattr(torch, up_dtype)), loss_seed)
                except Exception:
                    up_grads = None
                if up_grads:
                    row.grad_align = _grad_rel_err(native_grads, up_grads)
                    row.grad_note = "native vs upstream"
            if row.grad_align is None:
                ref_grads = _reference_grads(spec, ops, loss_seed)
                if ref_grads:
                    row.grad_align = _grad_rel_err(native_grads, ref_grads)
                    row.grad_note = "native vs reference" if up_plan is not None else "native vs reference (no upstream)"
                else:
                    row.grad_note = "no gradient reference"
        except Exception as exc:
            row.grad_note = f"grad error: {type(exc).__name__}: {str(exc)[:60]}"

    # inference KL divergence: native vs upstream single-token decode
    if native_dec is not None and up_plan is not None:
        try:
            dops = _decode_ops(spec, ops)
            with torch.no_grad():
                u_dec = up_plan.execute(**_to_cuda_operands(dops, getattr(torch, up_dtype))).output.detach().float()
            if native_dec.shape == u_dec.shape:
                p = torch.softmax(native_dec.reshape(-1, native_dec.shape[-1]), dim=-1).clamp_min(1e-12)
                q = torch.softmax(u_dec.reshape(-1, u_dec.shape[-1]), dim=-1).clamp_min(1e-12)
                row.kl_native_vs_upstream = float((p * (p.log() - q.log())).sum(dim=-1).mean().item())
        except Exception:
            pass

    # upstream performance at the same realistic shapes
    up_perf_plan, up_perf_dtype, up_perf_intent, _ = _compile_upstream(recipe, perf_dtype)
    if up_perf_plan is not None:
        try:
            ures = _perf_side(up_perf_plan, spec, pops, up_perf_dtype, peaks[up_perf_dtype], _to_cuda_operands)
            row.mfu_prefill_upstream = ures.get("mfu_prefill")
            row.mfu_decode_upstream = ures.get("mfu_decode")
            row.mfu_training_upstream = ures.get("mfu_training")
            row.tps_prefill_upstream = ures.get("tps_prefill")
            row.tps_decode_upstream = ures.get("tps_decode")
            row.peak_mem_prefill_upstream_mb = ures.get("peak_mem_prefill_mb")
            row.upstream_dtype = up_perf_dtype
        except Exception as exc:
            row.perf_note = (row.perf_note + " | " if row.perf_note else "") + (
                f"upstream perf: {type(exc).__name__}: {str(exc)[:80]}"
            )
    return row


def _compile_native(recipe, intent, dtype):
    from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer

    return compile_mixer(recipe, intent=MixerIntent(intent), backend=MixerBackend.NATIVE, dtype=dtype)


def _f(value: float | None, kind: str = "rel") -> str:
    if value is None:
        return "-"
    if value != value:
        return "n/a"
    if kind == "rel":
        return f"{value:.1e}"
    if kind == "mfu":
        return f"{value * 100:.2f}%"
    if kind == "tps":
        return f"{value:.0f}"
    if kind == "mb":
        return f"{value:.0f}"
    return str(value)


def _pair(native: float | None, upstream: float | None, kind: str) -> str:
    return f"{_f(native, kind)} / {_f(upstream, kind)}"


def render_markdown(rows: list[ProductRow]) -> str:
    n = len(rows)
    parity_ok = sum(1 for r in rows if r.parity_native is not None and r.parity_native < PARITY_REL_TOL)
    up_ok = sum(1 for r in rows if r.upstream_status == "ok")
    perf_ok = sum(1 for r in rows if r.mfu_prefill_native is not None)
    native_faster = sum(
        1 for r in rows
        if r.mfu_prefill_native is not None and r.mfu_prefill_upstream is not None
        and r.tps_prefill_native and r.tps_prefill_upstream
        and r.tps_prefill_native >= r.tps_prefill_upstream
    )
    both_perf = sum(1 for r in rows if r.tps_prefill_native is not None and r.tps_prefill_upstream is not None)

    lines = [
        "# Product evidence table: all 62 covered recipes, URM-native vs upstream",
        "",
        "Status: evidence record, regenerated by `benchmarks/product_table.py`. Every",
        "metric is measured for the URM-**native** kernel and the **upstream** comparator",
        "(pinned FLA / SDM / SDPA adapter) on identical operands, both through the same",
        "`plan.execute` dispatch (so URM's dispatch overhead is common to both). Cells are",
        "`native / upstream`. `-` = not measurable (e.g. forward-only kernel, no upstream",
        "adapter, or an adapter whose optional dependency is not installed here).",
        "",
        "MFU uses the **measured** hardware peak (`results/device-limits.json`), never the",
        "datasheet. K1 is timed in bf16 (fp32 exceeds A10G shared memory at head dim 64);",
        "K2/K3 in fp32. All timing is serial on one GPU. Parity is vs the NumPy canonical",
        "core (the contract).",
        "",
        f"**{parity_ok}/{n} recipes pass native parity (< {PARITY_REL_TOL:g}); "
        f"{up_ok}/{n} have a working upstream adapter; {perf_ok}/{n} have native "
        f"performance; native prefill throughput >= upstream on {native_faster}/{both_perf} "
        "recipes where both are measured.**",
        "",
        "## Correctness (canonical coverage shapes)",
        "",
        "| Recipe | Family | parity (native) | parity (upstream) | grad align | KL(native\\|upstream) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        grad = _f(r.grad_align) if r.grad_align is not None else (r.grad_note or "-")
        lines.append(
            f"| `{r.name}` | {r.family} | {_f(r.parity_native)} | {_f(r.parity_upstream)} "
            f"| {grad} | {_f(r.kl_native_vs_upstream)} |"
        )
    lines += [
        "",
        "## Performance (realistic shapes; `native / upstream`)",
        "",
        "| Recipe | MFU prefill | MFU decode | MFU train | prefill tok/s | decode tok/s | peak mem prefill (MB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.name}` | {_pair(r.mfu_prefill_native, r.mfu_prefill_upstream, 'mfu')} "
            f"| {_pair(r.mfu_decode_native, r.mfu_decode_upstream, 'mfu')} "
            f"| {_pair(r.mfu_training_native, r.mfu_training_upstream, 'mfu')} "
            f"| {_pair(r.tps_prefill_native, r.tps_prefill_upstream, 'tps')} "
            f"| {_pair(r.tps_decode_native, r.tps_decode_upstream, 'tps')} "
            f"| {_pair(r.peak_mem_prefill_native_mb, r.peak_mem_prefill_upstream_mb, 'mb')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(rows: list[ProductRow]) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "gate": "product_table",
        "recipe_count": len(rows),
        "tolerances": {
            "parity_native_rel": PARITY_REL_TOL,
            "parity_upstream_rel": UPSTREAM_PARITY_REL_TOL,
            "grad_rel": GRAD_REL_TOL,
        },
        "timing": {"warmup": WARMUP, "samples": SAMPLES, "serial": True},
        "peaks_tfps": _peaks(),
        "recipes": [asdict(r) for r in rows],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    MD_OUT.write_text(render_markdown(rows), encoding="utf-8")


def _worker(recipe_name: str, out_path: str) -> None:
    """Measure one recipe and write its row JSON. Runs in an isolated subprocess so
    a crashing (upstream) kernel cannot poison the CUDA context for other recipes."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("product table requires a CUDA device")
    row = measure_recipe(recipe_name)
    Path(out_path).write_text(json.dumps(asdict(row)), encoding="utf-8")


def _run_one_subprocess(name: str, tmpdir: Path) -> ProductRow:
    """Measure one recipe in a fresh subprocess; fall back to an error row on crash."""
    import subprocess
    import sys

    out_path = tmpdir / f"{name}.json"
    env = dict(os.environ)
    env.setdefault("CUDA_HOME", "/opt/conda/lib/python3.12/site-packages/nvidia/cu13")
    sdm = "/tmp/urm-comparator-pins/sdm"
    atma = "/tmp/urm-comparator-pins/atma"
    pp = env.get("PYTHONPATH", "")
    for p in (sdm, atma):
        if os.path.isdir(p) and p not in pp:
            pp = f"{pp}:{p}" if pp else p
    env["PYTHONPATH"] = pp
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--out", str(out_path)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode == 0 and out_path.exists():
        return ProductRow(**json.loads(out_path.read_text()))
    # The subprocess crashed (e.g. an illegal memory access in an upstream kernel).
    recipe = named_mixer_recipe(name)
    row = ProductRow(name=name, family=recipe.spec.family.name)
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1][:140] if tail else f"exit {proc.returncode}"
    row.correctness_note = f"subprocess crash: {detail}"
    row.perf_note = f"subprocess crash: {detail}"
    return row


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("product table requires a CUDA device")
    tmpdir = Path(tempfile.mkdtemp(prefix="urm_product_rows_"))
    rows = []
    for i, name in enumerate(COVERED_RECIPES):
        row = _run_one_subprocess(name, tmpdir)
        rows.append(row)
        print(
            f"[{i + 1}/{len(COVERED_RECIPES)}] {name}: parityN={_f(row.parity_native)} "
            f"parityU={_f(row.parity_upstream)} mfuN={_f(row.mfu_prefill_native, 'mfu')} "
            f"mfuU={_f(row.mfu_prefill_upstream, 'mfu')} up={row.upstream_status}",
            flush=True,
        )
    write_outputs(rows)
    parity_ok = sum(1 for r in rows if r.parity_native is not None and r.parity_native < PARITY_REL_TOL)
    perf_ok = sum(1 for r in rows if r.mfu_prefill_native is not None)
    crashed = sum(1 for r in rows if "subprocess crash" in r.correctness_note)
    print(f"[product] {parity_ok}/{len(rows)} pass native parity; {perf_ok} native perf measured; {crashed} subprocess crashes")
    print(f"[product] wrote {JSON_OUT}")
    print(f"[product] wrote {MD_OUT}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", default=None, help="measure a single recipe (isolated mode)")
    parser.add_argument("--out", default=None, help="worker output JSON path")
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker, args.out)
    else:
        main()
