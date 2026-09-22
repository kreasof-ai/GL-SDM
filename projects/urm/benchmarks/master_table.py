"""Master coverage table: all 62 covered recipes, native vs upstream, model-level.

This is the master product-evidence record, superseding the per-recipe kernel table
for the performance claim. Each recipe is dropped into the frozen ~100M decoder LM
(12 layers, 768 width, 12 heads, value_dim 64, seq 1024, FineWeb-Edu tokens) as a
``RecipeMixer`` - a projection stack that produces the recipe's operands, runs the
URM-native compiled plan (or the upstream comparator plan), and projects back. The
SAME projection stack feeds both backends, so the comparison is apples-to-apples:
only the mixer kernel differs.

Columns (each native vs upstream where an upstream adapter exists):

- **gradient parity**: max rel err of native input grads vs upstream (or reference).
- **parameter parity after 10 training steps**: max rel err of the two models'
  parameters after 10 identical AdamW steps from the same init on the same FineWeb
  batches.
- **training MFU**: 6 * params * tokens / step_time / measured bf16 peak.
- **prefill MFU / MBU / throughput** and **decode MFU / MBU / throughput**: model
  forward at the stated batch/sequence; MFU vs measured bf16 peak, MBU vs measured
  HBM bandwidth.
- **inference KL div**: KL(native || upstream) over next-token logits.
- **peak memory**: torch.cuda.max_memory_allocated for training, prefill, decode,
  and long-sequence (2K..32K) prefill.

Inference scaling: prefill/decode throughput measured at sequence lengths
1K,2K,4K,8K,16K,32K (bs=1) and at bs 256-512 (to push decode compute/bandwidth
bound). All GPU timing is serial; each recipe runs in an isolated subprocess.

Run: PYTHONPATH=src:benchmarks python benchmarks/master_table.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from urm.frontend.mixer_recipes import named_mixer_recipe

from product_table import COVERED_RECIPES, _sort_k3_routes
from representation_coverage import _rng_operands

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = PROJECT_ROOT / "results" / "validation" / "master-table.json"
MD_OUT = PROJECT_ROOT / "docs" / "validation" / "master-table.md"
DEVICE_LIMITS = PROJECT_ROOT / "results" / "device-limits.json"
FINEWEB = PROJECT_ROOT / "data" / "finewebedu_train_000001.bin"

# Frozen 100M model config (matches pretraining_step.toml).
MODEL = dict(vocab_size=50304, sequence_length=1024, layers=12, width=768,
             heads=12, value_dim=64, mlp_ratio=4)
TRAIN_DTYPE = "bfloat16"
TRAIN_STEPS = 10
GRAD_REL_TOL = 2e-2

# Inference sweep.
PREFILL_SEQ_LENS = [1024, 2048, 4096, 8192, 16384, 32768]
DECODE_BATCH = 256  # push decode toward compute/bandwidth bound
DECODE_BATCH_HI = 512


def _torch():
    import torch

    return torch


def _peaks() -> dict[str, float]:
    limits = json.loads(DEVICE_LIMITS.read_text())
    return {
        "bf16_tflops": limits["bf16_tensor_core"]["bf16_tensor_core_tfps_measured"],
        "fp32_tflops": limits["fp32_cuda_core"]["fp32_cuda_core_tfps_measured"],
        "hbm_gbps": limits["bandwidth"]["sustainable_gbps"],
    }


# ---------------------------------------------------------------------------
# RecipeMixer: project the model hidden state [B,T,C] into a recipe's operands,
# run the compiled plan (native or upstream), project back to [B,T,C].
#
# Operand groups (from representation_coverage._rng_operands signatures):
#  - qkv attention / matrix-state: query,key,value [B,T,H,K/V]
#  - gated/decay variants add: log_decay, beta, gates, transition_alpha/beta, ...
#  - distinguished RNN/SSM: weight matrices + sequence inputs
#  - K3 sparse: integer routes + weights + values + persistent memory
# The SAME projection stack feeds native and upstream; only the kernel differs.
# ---------------------------------------------------------------------------

# Operand name -> (kind, per-head dim multiplier). kind: 'seq' (project from x,
# sequence-varying), 'static' (learned weight, no seq), 'scalar' (python number).
# dims are expressed in units of value_dim d (per head) unless absolute.


def _recipe_operand_plan(spec) -> dict[str, str]:
    """Classify each operand the recipe needs into a projection role."""
    ops = _rng_operands(spec, seed=0)
    roles: dict[str, str] = {}
    for name, value in ops.items():
        if isinstance(value, (int, float, bool)):
            roles[name] = "scalar"
            continue
        arr = np.asarray(value)
        if arr.dtype.kind in "iu":
            roles[name] = "route"  # integer indices (K3)
        elif arr.ndim >= 3 and arr.shape[1] >= 1 and name not in _STATIC_WEIGHTS:
            roles[name] = "seq"
        else:
            roles[name] = "static"
    return roles


# Operand names that are learned static weights (not sequence-varying projections).
_STATIC_WEIGHTS = {
    "weight", "forget_weight", "reset_weight", "A", "B_pre", "lamb", "w", "b", "u",
    "bonus", "query_bias", "key_bias", "kernel", "ssm_kernel", "ssm_k_kernel",
    "ssm_k_direct", "skip", "direct", "lambda_weight",
}
# Gate-like sequence operands that must be squashed into a valid range.
_NEG_LOG_GATES = {"log_decay", "log_alpha", "log_mu", "adt", "g", "gv"}
_POS_UNIT_GATES = {"beta", "eta", "theta", "alpha", "input_gate", "read_gate",
                   "step_size", "dt", "trap", "erase_gate", "write_gate",
                   "slot_weights", "forget_input", "reset_input", "angles"}

# Recipes whose NATIVE kernel requires fp32 operands (the mixer runs in fp32; the
# rest of the model stays bf16). Detected empirically by the probe.
# NOTE: the masked K1 recipes (cat/dsa/nsa/sparse) are NOT here - their fp32
# online-softmax kernel exceeds A10G shared memory at value_dim 64, so they run in
# bf16 (the additive-mask gradient path is dtype-tolerant at inference/training).
_FP32_NATIVE_RECIPES = {
    "abc_core", "gsa_core",  # slot_attention_two_stage requires fp32
    "mesa_net_core",  # regularized_solve fp32
    "momentum_delta_core",  # momentum_delta fp32
    "gated_oja_core",  # gated_oja fp32
    "hla_second_order_core",  # HLA fp32
    "h3_ssm_fft_core",  # H3 FFT fp32
    "hyena_fftconv_core",  # Hyena FFT fp32
    "ttt_linear_core",  # layernorm_inner_state fp32
    "titans_linear_memory_core",  # momentum_inner_state fp32
    # Distinguished RNN/SSM: the native backward recomputes the scan in fp32 and
    # mixes fp32 state with the (cast) operands, so the mixer runs in fp32.
    "rnn_core", "gru_core", "m2rnn_core",
    "rwkv4_memory_core", "rwkv6_memory_core",
    "mamba1_ssm_core", "mamba2_ssm_core", "mamba3_siso_core",
}


# Upstream adapters whose FLA/pinned kernel requires fp32 GATE operands (the plan
# compiles at bf16, but specific gate/schedule operands must be fp32).
_UPSTREAM_FP32_OPERANDS = {
    "comba_core": {"log_decay", "beta", "g"},
    "gated_oja_core": {"gv", "beta"},
    "mesa_net_core": {"log_decay", "beta", "lamb"},
    "hla_second_order_core": set(),  # blocked: HLA supports key/value dim <= 32
    "lightning_attention_core": set(),  # needs log_decay [H] static head
    "retention_core": set(),  # needs log_decay [H] static head
}


def _mixer_dtype(recipe_name: str, backend: str) -> str:
    if backend == "native" and recipe_name in _FP32_NATIVE_RECIPES:
        return "float32"
    return TRAIN_DTYPE


# K3 (sparse_delta_memory) persistent-memory slot count. The recipe spec carries
# no slots/reads/writes fields; the mixer fixes a realistic geometry (the K3
# native kernel requires strictly-increasing unique routes within each token, so
# routes_per_token < slots).
_K3_SLOTS = 64
_K3_ROUTES = 8
_TUCKER_RANK = 32  # low-rank query width for tucker_attention_core
# Operand names that get neither a generic projection nor a static parameter
# (integer routes, persistent/initial state, or adapter-derived). Everything else
# a specialized adapter uses is projected or learned via the explicit width/shape
# maps in ``_proj_width``/``_static_shape``.
_SPECIAL_ONLY = {
    "memory", "state", "initial_state", "initial_states", "initial_normalizer_state",
    "read_indices", "write_indices", "attention_mask",
}


# Cache of mixer CLASSES so all layers of a model share one type. Dynamo guards on
# the mixer's type identity; a fresh closure class per layer would force a recompile
# per layer. Sharing one class object per (recipe, backend, dtype) means a
# static-shape training step compiles exactly once.
_MIXER_CLASS_CACHE: dict[tuple, type] = {}


def build_recipe_mixer(config, recipe_name: str, backend: str, dtype: str | None = None):
    """Build an nn.Module mapping [B,T,C]->[B,T,C] for the recipe on the backend.

    backend: "native" (URM NATIVE plan) or "upstream" (LIBRARY adapter plan).
    Returns None when the backend cannot serve the recipe (e.g. K3 upstream).
    The mixer runs in its required dtype (fp32 for some native kernels); the
    surrounding model stays in the training dtype.
    """
    torch = _torch()
    import torch.nn as nn

    from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer

    if dtype is None:
        dtype = _mixer_dtype(recipe_name, backend)
    recipe = named_mixer_recipe(recipe_name)
    spec = recipe.spec
    mb = MixerBackend.NATIVE if backend == "native" else MixerBackend.LIBRARY
    # Try the recipe's required dtype first, then fall back across dtypes so a
    # working upstream adapter is not missed just because it needs fp32 (or bf16).
    # Native uses its single required dtype.
    if backend == "native":
        dtype_attempts = [dtype]
    else:
        dtype_attempts = [dtype, "float32", "bfloat16"]
    plan = None
    seen = set()
    for dt in dtype_attempts:
        if dt in seen:
            continue
        seen.add(dt)
        try:
            plan = compile_mixer(recipe, intent=MixerIntent.TRAINING, backend=mb, dtype=dt)
            dtype = dt
            break
        except Exception:
            plan = None
    if plan is None:
        return None

    roles = _recipe_operand_plan(spec)
    # Reference operand shapes (tiny) tell us per-operand head/dim structure.
    ref_ops = _rng_operands(spec, seed=0)
    if spec.family.name == "SPARSE_DELTA":
        ref_ops = _sort_k3_routes(ref_ops)

    # Reuse one class object per (recipe, backend, dtype) so every layer shares a
    # single mixer type and Dynamo compiles the static-shape step once.
    cache_key = (recipe_name, backend, dtype)
    if cache_key in _MIXER_CLASS_CACHE:
        return _MIXER_CLASS_CACHE[cache_key](
            config, spec, plan, roles, ref_ops, getattr(torch, dtype), backend
        )

    class RecipeMixer(nn.Module):
        def __init__(self, config, spec, plan, roles, ref_ops, dtype_t, backend):
            super().__init__()
            self.config = config
            self.spec = spec
            self._plan = plan
            self._dtype = dtype_t
            self._backend = backend
            self._roles = roles
            self._ref_ops = ref_ops
            c = config.width
            h = config.heads
            d = config.value_dim
            # Polynomial bases (based/rebased) expand the feature dim to ~1+K+K^2,
            # which exceeds the native kernel's shared-memory limit at K=64. Use a
            # smaller base key dim so the expanded width stays within limits.
            if getattr(spec, "polynomial_basis", None) is not None and \
                    getattr(spec.polynomial_basis, "name", "NONE") != "NONE":
                d = 16
            self.h, self.d = h, d
            # One projection per sequence operand; learned params for static ones.
            self.proj = nn.ModuleDict()
            self.static = nn.ParameterDict()
            self._static_names = []  # avoid nn.Module reserved-name collisions
            # Reference per-head key/value dims, to normalize each operand's rank.
            ref_k = self._ref_dim(ref_ops, "key", "query")
            ref_v = self._ref_dim(ref_ops, "value")
            # K3 geometry must be set before _proj_width reads it.
            if spec.family.name == "SPARSE_DELTA":
                self._k3_slots = _K3_SLOTS
                self._k3_routes = _K3_ROUTES
            for name, role in roles.items():
                if role == "seq":
                    ref = np.asarray(ref_ops[name])
                    # Explicit projection width (adapter-aware); falls back to the
                    # generic per-head rank for the standard QKV recipes.
                    width = self._proj_width(name, ref, ref_k, ref_v)
                    if width is None:
                        continue  # built by a specialized adapter without a projection
                    safe = "p_values" if name == "values" else name  # dict-method collision
                    self.proj[safe] = nn.Linear(c, width, bias=config.bias)
                    self.proj[safe]._urm_rank = width // h if width % h == 0 else 0
                elif role == "static":
                    needed = self._static_needed()
                    if needed is not None and name not in needed:
                        continue  # adapter-driven recipe: only build the statics it reads
                    ref = np.asarray(ref_ops[name])
                    # learned static weight shaped like the reference (heads->h)
                    shape = self._static_shape(ref, name)
                    safe = "w_" + name  # avoid nn.Module reserved names (values, etc.)
                    self.static[safe] = nn.Parameter(torch.randn(shape) * 0.02)
                    self._static_names.append((name, safe))
            self.output = nn.Linear(h * d, c, bias=config.bias)
            # Static head decay (lightning/retention): log_decay is a per-head [H]
            # schedule. The upstream adapters require it FIXED (non-trainable); the
            # native kernel accepts a trainable parameter.
            if getattr(spec, "static_head_decay", False):
                init = torch.randn(h) * 0.02
                if backend == "upstream":
                    self.register_buffer("w_log_decay", init)
                else:
                    self.static["w_log_decay"] = nn.Parameter(init)
                if "log_decay" in self.proj:
                    del self.proj["log_decay"]
            # K3 persistent memory: a stateful buffer the native kernel updates in
            # place. Reset/detach between steps via reset_state()/detach_state().
            if spec.family.name == "SPARSE_DELTA":
                # beta/log_decay classify as static (ndim=2) but are per-token
                # projections here; build them explicitly and drop the static copies.
                self.proj["beta"] = nn.Linear(c, 1, bias=config.bias)
                self.proj["log_decay"] = nn.Linear(c, 1, bias=config.bias)
                for nm in ("beta", "log_decay"):
                    if "w_" + nm in self.static:
                        del self.static["w_" + nm]
                self.register_buffer(
                    "persistent_memory",
                    torch.zeros(1, _K3_SLOTS, d),
                    persistent=False,
                )
                self._pending_state = None

        @staticmethod
        def _ref_dim(ops, *names):
            for n in names:
                if n in ops and not isinstance(ops[n], (int, float, bool)):
                    a = np.asarray(ops[n])
                    if a.ndim >= 1:
                        return a.shape[-1]
            return None

        def _per_head_rank(self, name, ref, ref_k, ref_v):
            """Per-head last-dim rank of an operand, normalized to the model's dims.

            The reference tiny shapes use key_dim/value_dim; the model uses
            value_dim d. An operand's per-head rank is its last-dim expressed in
            units of the reference key/value dim, scaled to the model's d.
            """
            d = self.d
            if ref.ndim <= 3:  # [B,T,H] or [B,T,C] -> rank 1 per head (a gate/schedule)
                return 1
            last = ref.shape[-1]
            # value-like operands scale with value_dim; key-like with key_dim.
            if name in ("value", "write_gate", "gv") and ref_v:
                return max(1, round(last * d / ref_v))
            if ref_k:
                return max(1, round(last * d / ref_k))
            return last

        def _static_needed(self):
            """Static operands the adapter actually reads (others are skipped)."""
            name = self.spec.name
            rop = getattr(self.spec, "recurrence_operator", None)
            rname = rop.name if rop is not None else ""
            if rname == "TANH_RNN":
                return {"weight"}
            if rname == "GATED_RNN":
                return {"weight", "forget_weight", "reset_weight"}
            if rname == "MULTIPLICATIVE_RNN":
                return {"weight"}
            if rname == "RWKV4_SCALAR_STATE":
                return {"w", "u"}
            if rname == "RWKV6_BONUS_CORRECTED":
                return {"bonus"}
            if rname == "MAMBA2_STRUCTURED_SSM":
                return {"A"}
            if rname == "TRAPEZOIDAL_SSM":
                return {"query_bias", "key_bias"}
            if rname == "REGULARIZED_SOLVE":
                return {"lamb"}
            if rname in ("LAYERNORM_INNER_STATE", "MOMENTUM_INNER_STATE"):
                return {"w", "b"}
            if rname == "FFT_CONVOLUTION":
                return {"kernel", "direct"}
            if rname == "TWO_STAGE_FFT_CONVOLUTION":
                return {"ssm_kernel", "ssm_k_kernel", "ssm_k_direct", "skip"}
            if name in ("differential_attention_core", "tda_attention_core"):
                return {"lambda_weight"}
            if name == "tucker_attention_core":
                return {"B_pre"}
            return None  # generic path: build every static operand

        def _proj_width(self, name, ref, ref_k, ref_v):
            """Explicit projection output width for a sequence operand, or None to
            skip the projection (the adapter builds the operand without one).

            Default (None returned only for adapter-handled operands): the generic
            path projects hidden -> h * per_head_rank. The distinguished adapters
            need exact widths (e.g. channel-domain [B,T,c], head-major [B,T,h],
            multi-rank [B,T,r*h*d], low-rank tucker query [B,T,R]).
            """
            h, d, c = self.h, self.d, self.config.width
            rname = getattr(self.spec.recurrence_operator, "name", "") if getattr(
                self.spec, "recurrence_operator", None) is not None else ""
            # Persistent/initial state and integer routes: no projection.
            if name in _SPECIAL_ONLY:
                return None
            # Channel-domain (no head axis): project to c.
            if rname == "RWKV4_SCALAR_STATE" and name in ("k", "v"):
                return c
            if rname == "FFT_CONVOLUTION" and name == "query":
                return c
            if rname == "PLAIN_LINEAR_RECURRENCE" and getattr(
                self.spec, "step_size_discretization", False
            ):
                if name == "x":
                    return c
                if name == "step_size":
                    return c
                return h  # input_gate/read_gate/log_decay -> [B,T,h]
            # Head-major schedules [B,T,h] (mamba2 dt, mamba3 adt/dt/trap).
            if rname == "MAMBA2_STRUCTURED_SSM":
                if name == "x":
                    return h * d
                if name == "dt":
                    return h
                if name in ("B", "C"):
                    return d  # [B,T,1,d]
            if rname == "TRAPEZOIDAL_SSM":
                if name in ("query", "key", "value"):
                    return h * d
                if name in ("adt", "dt", "trap"):
                    return h
                if name == "angles":
                    return h * max(1, d // 2)
            # Head-major gates [B,T,h] for the inner-state/solve recurrences.
            if rname in ("REGULARIZED_SOLVE",) and name in ("log_decay", "beta"):
                return h
            if rname in ("LAYERNORM_INNER_STATE", "MOMENTUM_INNER_STATE") and name in (
                "eta", "theta", "alpha",
            ):
                return h
            # H3 two-stage FFT: query/key/value are [B,T,h,1].
            if rname == "TWO_STAGE_FFT_CONVOLUTION" and name in ("query", "key", "value"):
                return h
            # m2rnn forget_input is [B,T,h] (scalar per head).
            if rname == "MULTIPLICATIVE_RNN" and name == "forget_input":
                return h
            # gated_delta_product multi-rank updates.
            if getattr(self.spec, "gated_delta_product", False):
                r = np.asarray(self._ref_ops["update_keys"]).shape[2]
                if name in ("update_keys", "update_values"):
                    return r * h * d
                if name == "beta":
                    return r * h
                if name == "log_decay":
                    return h
            # tucker: low-rank query (no head dim), key/value [B,T,d] (no head dim).
            if self.spec.name == "tucker_attention_core":
                if name == "query":
                    return _TUCKER_RANK
                if name in ("key", "value"):
                    return d
            # K3: routes are integer (no projection); weights/values/beta/log_decay.
            if self.spec.family.name == "SPARSE_DELTA":
                r = self._k3_routes
                if name in ("read_weights", "write_weights"):
                    return r
                if name == "values":
                    return d
                if name in ("beta", "log_decay"):
                    return 1
                return None  # routes/memory: no projection
            # Generic path: standard QKV-style per-head rank.
            rank = self._per_head_rank(name, ref, ref_k, ref_v)
            return h * rank

        def _static_shape(self, ref, name=None):
            # Map a reference static weight to model dims. Heuristic by ndim:
            #  [H]        -> [h]            (per-head scalar: lambda_weight, A, skip)
            #  [H,K]      -> [h, d]         (per-head vector: lamb, w, b, bonus, biases)
            #  [N,H,H]    -> [n, d, d]      (RNN weight matrices; n = ref head count)
            #  [C,T] / [C](channel-domain FFT kernels) -> [c, T] / [c]
            h, d, c = self.h, self.d, self.config.width
            if name == "B_pre":  # tucker low-rank expansion [h, R, d]
                return (h, _TUCKER_RANK, d)
            if name in ("w", "u") and getattr(
                self.spec.recurrence_operator, "name", ""
            ) == "RWKV4_SCALAR_STATE":
                return (c,)  # rwkv4 per-channel decay/bonus [c]
            if name in ("kernel", "ssm_kernel", "ssm_k_kernel"):
                # FFT conv filter [C, T] / [h, T]; T = max sequence length.
                rows = c if name == "kernel" else h
                return (rows, self.config.sequence_length)
            if name == "direct":  # Hyena pointwise direct term [c]
                return (c,)
            shp = list(ref.shape)
            if ref.ndim == 1:
                return (c,) if ref.shape[0] > 4 * h else (h,)
            if ref.ndim == 2:
                if ref.shape[0] <= 4 * h:  # [H,K] per-head
                    return (h, d)
                return (c, ref.shape[1])  # [C,T] FFT kernel
            if ref.ndim == 3:  # [N,H,H] RNN weight -> [h, d, d] (one matrix per head)
                return (h, d, d)
            return tuple(shp)

        def _build_operands(self, x):
            b, t, c = x.shape
            h, d = self.h, self.d
            # Specialized operand construction for recipes the generic projection
            # cannot serve (distinguished RNN/SSM/FFT/diagonal, K3 sparse, K1
            # masked/differential). Returns a complete operand dict or None.
            special = self._special_operands(x)
            if special is not None:
                return special
            operands: dict[str, Any] = {}
            for name, role in self._roles.items():
                ref = np.asarray(self._ref_ops[name])
                if role == "seq":
                    rank = self.proj[name]._urm_rank
                    y = self.proj[name](x).view(b, t, h, rank)
                    if name in _NEG_LOG_GATES:
                        y = -nn.functional.softplus(y)
                    elif name in _POS_UNIT_GATES:
                        y = torch.sigmoid(y)
                    # collapse head-count mismatch (e.g. key/value with fewer heads)
                    operands[name] = self._fit_seq(y, ref)
                elif role == "static":
                    operands[name] = self.static["w_" + name]
                elif role == "scalar":
                    operands[name] = ref.item() if ref.ndim == 0 else ref
                elif role == "route":
                    operands[name] = self._routes(b, t, ref)
            return operands

        def _fit_seq(self, y, ref):
            # y is [B,T,H,rank]. Match the reference's head count (dim 2) and drop
            # the trailing rank-1 dim for [B,T,H] gate/schedule operands.
            if ref.ndim >= 3:
                want_h = ref.shape[2]
                have_h = y.shape[2]
                if have_h != want_h and want_h >= 1:
                    idx = torch.arange(want_h, device=y.device) % have_h
                    y = y.index_select(2, idx)
            if ref.ndim == 3 and y.shape[-1] == 1:
                y = y.squeeze(-1)  # [B,T,H] gate
            return y

        def _routes(self, b, t, ref):
            # K3 integer routes: random distinct routes per token (sorted for native).
            r = ref.shape[-1]
            s = self.spec.slots if hasattr(self.spec, "slots") else 64
            idx = torch.randint(0, max(s, r), (b, t, r), device="cuda")
            return idx.sort(dim=-1).values.to(torch.int64)

        def _special_operands(self, x):
            """Per-recipe operand construction for non-generic signatures.

            Returns a complete operand dict, or None to fall through to the generic
            projection path. Handles: distinguished RNN/SSM (weight matrices),
            FFT/diagonal channel-domain convs, K3 sparse (routes + persistent
            memory), K1 masked/differential (mask/lambda/paired operands).
            """
            torch = _torch()
            import torch.nn.functional as F
            b, t, c = x.shape
            h, d = self.h, self.d
            name = self.spec.name
            rop = getattr(self.spec, "recurrence_operator", None)
            k1op = getattr(self.spec, "k1_operation", None)
            dev = x.device

            # --- K1 masked attention: build a causal [B,1,T,T] mask ---
            if k1op is not None and self.spec.requires_attention_mask:
                qkv = self._qkv(x)
                mask = torch.ones(b, 1, t, t, dtype=torch.bool, device=dev).tril()
                return {**qkv, "attention_mask": mask}
            # --- K1 positional (parallax): add a rotary/position term r ---
            if name == "parallax_attention_core":
                qkv = self._qkv(x)
                out = dict(qkv)
                out["r"] = self._proj_extra(x, "r")
                return out

            # --- static head decay (lightning/retention): log_decay is a learned
            # [H] per-head vector, not a per-token projection ---
            if getattr(self.spec, "static_head_decay", False):
                qkv = self._qkv(x)
                # log_decay is a buffer (upstream) or a Parameter in self.static (native).
                ld = getattr(self, "w_log_decay", None)
                if ld is None:
                    ld = self.static["w_log_decay"]
                log_decay = -F.softplus(ld)  # [H]
                return {**qkv, "log_decay": log_decay}

            # --- distinguished RNN/SSM/FFT/diagonal/K3/K1-specialized adapters ---
            return self._distinguished_operands(x)

        def _qkv(self, x):
            b, t, _ = x.shape
            h, d = self.h, self.d
            return {
                "query": self.proj["query"](x).view(b, t, h, -1),
                "key": self.proj["key"](x).view(b, t, h, -1),
                "value": self.proj["value"](x).view(b, t, h, -1),
            }

        def _proj_extra(self, x, name):
            b, t, _ = x.shape
            h = self.h
            rank = self.proj[name]._urm_rank
            return self.proj[name](x).view(b, t, h, rank)

        def _seq(self, x, name, rank=None):
            """Project x -> [B,T,H,rank] (rank defaults to value_dim d)."""
            b, t, _ = x.shape
            h, d = self.h, self.d
            r = self.d if rank is None else rank
            return self.proj[name](x).view(b, t, h, r)

        def _static(self, name):
            return self.static["w_" + name]

        def _distinguished_operands(self, x):
            """Operand adapters for the non-generic distinguished recipes.

            Returns a complete operand dict, or None to fall through to the generic
            projection path. Each branch mirrors the recipe's ``_rng_operands``
            signature at the model's dims (h heads, d value_dim, c width).
            """
            torch = _torch()
            import torch.nn.functional as F
            b, t, c = x.shape
            h, d = self.h, self.d
            name = self.spec.name
            rop = getattr(self.spec, "recurrence_operator", None)
            dev = x.device
            f32 = torch.float32

            # ---- distinguished RNN (weight matrices + sequence inputs) ----
            if rop is not None and rop.name == "TANH_RNN":  # rnn_core
                return {
                    "query": self._seq(x, "query"),
                    "weight": self._static("weight") * 0.3,
                    "initial_state": torch.zeros(b, h, d, device=dev, dtype=x.dtype),
                }
            if rop is not None and rop.name == "GATED_RNN":  # gru_core
                return {
                    "query": self._seq(x, "query"),
                    "weight": self._static("weight") * 0.3,
                    "forget_input": self._seq(x, "forget_input"),
                    "forget_weight": self._static("forget_weight") * 0.3,
                    "reset_input": self._seq(x, "reset_input"),
                    "reset_weight": self._static("reset_weight") * 0.3,
                    "initial_state": torch.zeros(b, h, d, device=dev, dtype=x.dtype),
                }
            if rop is not None and rop.name == "MULTIPLICATIVE_RNN":  # m2rnn_core
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "weight": self._static("weight") * 0.3,
                    "forget_input": torch.sigmoid(self._seq(x, "forget_input", 1)).squeeze(-1),
                    "initial_state": torch.zeros(b, h, d, d, device=dev, dtype=x.dtype),
                }
            if rop is not None and rop.name == "RWKV4_SCALAR_STATE":  # rwkv4_memory_core
                # Channel-domain: k/v [B,T,c]; w/u [c]; state [B,3,1,c].
                return {
                    "w": -F.softplus(self._static("w")),
                    "u": self._static("u"),
                    "k": self.proj["k"](x),
                    "v": self.proj["v"](x),
                    "state": torch.zeros(b, 3, 1, c, device=dev, dtype=x.dtype),
                }
            if rop is not None and rop.name == "RWKV6_BONUS_CORRECTED":  # rwkv6_memory_core
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "log_decay": -F.softplus(self._seq(x, "log_decay")),
                    "bonus": self._static("bonus"),
                }
            if rop is not None and rop.name == "MAMBA2_STRUCTURED_SSM":  # mamba2_ssm_core
                # x [B,T,h,d]; dt [B,T,h]; A [h]; B/C [B,T,1,n_state=d].
                return {
                    "x": self._seq(x, "x"),
                    "dt": torch.sigmoid(self._seq(x, "dt", 1)).squeeze(-1) * 0.5,
                    "A": -F.softplus(self._static("A")),
                    "B": self.proj["B"](x).view(b, t, 1, d),
                    "C": self.proj["C"](x).view(b, t, 1, d),
                }
            if rop is not None and rop.name == "TRAPEZOIDAL_SSM":  # mamba3_siso_core
                # adt/dt/trap are [B,H,T] (head-major); angles [B,T,h,a] with a=d//2.
                a = max(1, d // 2)
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "adt": -F.softplus(self.proj["adt"](x).view(b, t, h, 1)).squeeze(-1).permute(0, 2, 1).contiguous(),
                    "dt": (torch.sigmoid(self.proj["dt"](x).view(b, t, h, 1)).squeeze(-1) * 0.5).permute(0, 2, 1).contiguous(),
                    "trap": self.proj["trap"](x).view(b, t, h, 1).squeeze(-1).permute(0, 2, 1).contiguous(),
                    "query_bias": self._static("query_bias"),
                    "key_bias": self._static("key_bias"),
                    "angles": self.proj["angles"](x).view(b, t, h, a),
                }

            # ---- inner-state / solve recurrences ----
            if rop is not None and rop.name == "REGULARIZED_SOLVE":  # mesa_net_core
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "log_decay": -F.softplus(self._seq(x, "log_decay", 1)).squeeze(-1),
                    "beta": torch.sigmoid(self._seq(x, "beta", 1)).squeeze(-1),
                    "lamb": F.softplus(self._static("lamb")) + 0.5,
                }
            if rop is not None and rop.name == "LAYERNORM_INNER_STATE":  # ttt_linear_core
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "w": self._static("w"),
                    "b": self._static("b"),
                    "eta": torch.sigmoid(self._seq(x, "eta", 1)) * 0.1,
                    "chunk_size": 8,
                }
            if rop is not None and rop.name == "MOMENTUM_INNER_STATE":  # titans_linear_memory_core
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "w": self._static("w"),
                    "b": self._static("b"),
                    "theta": torch.sigmoid(self._seq(x, "theta", 1)) * 0.1,
                    "alpha": torch.sigmoid(self._seq(x, "alpha", 1)) * 0.3,
                    "eta": torch.sigmoid(self._seq(x, "eta", 1)) * 0.3,
                    "chunk_size": 8,
                }

            # ---- diagonal SSM (mamba1 step-size; hgrn handled generically) ----
            if rop is not None and rop.name == "PLAIN_LINEAR_RECURRENCE" and getattr(
                self.spec, "step_size_discretization", False
            ):  # mamba1_ssm_core
                # x [B,T,c]; input/read_gate [B,T,d]; log_decay [B,T,d]; step_size [B,T,c] fp32.
                return {
                    "x": self.proj["x"](x),
                    "input_gate": torch.sigmoid(self._seq(x, "input_gate", 1)).squeeze(-1),
                    "read_gate": torch.sigmoid(self._seq(x, "read_gate", 1)).squeeze(-1),
                    "log_decay": -F.softplus(self._seq(x, "log_decay", 1)).squeeze(-1),
                    "step_size": torch.sigmoid(self.proj["step_size"](x)).to(f32),
                }

            # ---- FFT / channel-domain convolutions ----
            if rop is not None and rop.name == "FFT_CONVOLUTION":  # hyena_fftconv_core
                # query [B,T,c]; kernel [c,T] (T = runtime seq); direct [c].
                return {
                    "query": self.proj["query"](x),
                    "kernel": self._static("kernel")[:, :t],
                    "direct": self._static("direct"),
                }
            if rop is not None and rop.name == "TWO_STAGE_FFT_CONVOLUTION":  # h3_ssm_fft_core
                # query/key/value [B,T,h,1]; kernels [h,T]; ssm_k_direct/skip [h].
                return {
                    "query": self._seq(x, "query", 1),
                    "key": self._seq(x, "key", 1),
                    "value": self._seq(x, "value", 1),
                    "ssm_kernel": self._static("ssm_kernel")[:, :t],
                    "ssm_k_kernel": self._static("ssm_k_kernel")[:, :t],
                    "ssm_k_direct": self._static("ssm_k_direct"),
                    "skip": self._static("skip"),
                }

            # ---- K1 differential / thresholded (paired Q/K) ----
            if name in ("differential_attention_core", "tda_attention_core"):
                out = {
                    "query_a": self._proj_extra(x, "query_a"),
                    "query_b": self._proj_extra(x, "query_b"),
                    "key_a": self._proj_extra(x, "key_a"),
                    "key_b": self._proj_extra(x, "key_b"),
                    "value": self._proj_extra(x, "value"),
                }
                lw = self._static("lambda_weight")
                out["lambda_weight"] = (
                    torch.sigmoid(lw) if np.asarray(self._ref_ops["lambda_weight"]).ndim > 0
                    else float(torch.sigmoid(lw.detach().reshape(())))
                )
                if "beta" in self._roles:
                    out["beta"] = 0.5
                return out

            # ---- K1 projected (tucker): low-rank query, no head dim ----
            if name == "tucker_attention_core":
                r = self.proj["query"].out_features  # low-rank query width
                return {
                    "query": self.proj["query"](x),  # [B,T,R]
                    "B_pre": self._static("B_pre"),
                    "key": self.proj["key"](x),  # [B,T,K] (no head dim)
                    "value": self.proj["value"](x),  # [B,T,V]
                }

            # ---- gated_delta_product: multi-rank updates ----
            if getattr(self.spec, "gated_delta_product", False):
                r = np.asarray(self._ref_ops["update_keys"]).shape[2]  # ranks per token
                return {
                    "query": self._seq(x, "query"),
                    "key": self._seq(x, "key"),
                    "value": self._seq(x, "value"),
                    "update_keys": self.proj["update_keys"](x).view(b, t, r, h, d),
                    "update_values": self.proj["update_values"](x).view(b, t, r, h, d),
                    "beta": torch.sigmoid(self.proj["beta"](x).view(b, t, r, h)),
                    "log_decay": -F.softplus(self._seq(x, "log_decay", 1)).squeeze(-1),
                }

            # ---- K3 sparse delta memory (persistent state) ----
            if self.spec.family.name == "SPARSE_DELTA":
                return self._k3_operands(x)

            return None

        def _k3_operands(self, x):
            """K3 sparse-delta-memory operands: persistent memory + integer routes.

            Routes are integer (no grad), strictly increasing per token (native
            requirement). read/write weights are softmax-normalized (grad); values,
            beta, log_decay are projected (grad). memory is the persistent buffer.
            """
            torch = _torch()
            import torch.nn.functional as F
            b, t, _ = x.shape
            d = self.d
            s, r = self._k3_slots, self._k3_routes
            dev = x.device
            # Expand the persistent memory to this batch (state is per-batch-slot).
            if self.persistent_memory.shape[0] != b:
                self.persistent_memory = torch.zeros(b, s, d, device=dev, dtype=self._dtype)
            memory = self.persistent_memory
            # Integer routes: distinct, sorted ascending per token (native contract).
            # Vectorized: argsort of random scores, take the first r, then sort.
            def _routes():
                scores = torch.rand(b * t, s, device=dev)
                idx = scores.argsort(dim=-1)[:, :r].sort(dim=-1).values
                return idx.view(b, t, r).to(torch.int64)
            read_idx = _routes()
            write_idx = _routes()
            # Normalized weights (softmax over the route axis), values, beta, log_decay.
            read_w = F.softmax(self.proj["read_weights"](x).view(b, t, r), dim=-1)
            write_w = F.softmax(self.proj["write_weights"](x).view(b, t, r), dim=-1)
            values = self.proj["p_values"](x).view(b, t, d)
            beta = torch.sigmoid(self.proj["beta"](x).view(b, t))
            log_decay = -F.softplus(self.proj["log_decay"](x).view(b, t))
            return {
                "memory": memory,
                "read_indices": read_idx,
                "read_weights": read_w,
                "write_indices": write_idx,
                "write_weights": write_w,
                "values": values,
                "beta": beta,
                "log_decay": log_decay,
            }

        def reset_state(self) -> None:
            """Zero the K3 persistent memory between training steps."""
            if hasattr(self, "persistent_memory"):
                self.persistent_memory.zero_()
                self._pending_state = None

        @torch.no_grad()
        def detach_state(self) -> None:
            """Fold the K3 updated memory (final_state) back into the persistent
            buffer, detached from the autograd graph (SparseMemoryMixer pattern)."""
            if hasattr(self, "persistent_memory"):
                if self._pending_state is not None:
                    self.persistent_memory.copy_(self._pending_state.detach())
                    self._pending_state = None
                else:
                    self.persistent_memory.detach_()

        def forward(self, x):
            b, t, c = x.shape
            in_dtype = x.dtype
            # Run the mixer in its required dtype (fp32 for some native kernels).
            x = x.to(self._dtype)
            out = self._mixer_core(x, b, t)
            return out.to(in_dtype)

        def _mixer_core(self, x, b, t):
            # The custom Triton kernel is not dynamo-traceable; run it eagerly as a
            # compile boundary so torch.compile fuses the surrounding model but calls
            # this as an opaque op.
            operands = self._build_operands(x)
            # Upstream adapters with fp32 gate requirements: cast those operands to
            # fp32 even though the plan runs at bf16.
            fp32_ops = _UPSTREAM_FP32_OPERANDS.get(self.spec.name, set())
            operands = {
                k: (
                    v.to(torch.float32)
                    if torch.is_tensor(v) and v.is_floating_point() and k in fp32_ops
                    else (v.to(self._dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
                )
                for k, v in operands.items()
            }
            out = self._run_plan(operands)
            h, d = self.h, self.d
            out = out.reshape(b, t, -1)
            # fit to h*d for the output projection
            if out.shape[-1] != h * d:
                idx = torch.arange(h * d, device=out.device) % out.shape[-1]
                out = out.index_select(-1, idx)
            return self.output(out.to(self._dtype))

    import torch as _t

    # Opaque boundary: dynamo compiles the model, calls the mixer kernel eagerly.
    # For K3 the native kernel returns the updated memory as final_state (it does
    # not mutate the persistent buffer in place); stash it as _pending_state so
    # detach_state() can fold it back into the buffer between steps.
    def _run_plan(self, operands):
        result = self._plan.execute(**operands)
        if self.spec.family.name == "SPARSE_DELTA" and result.final_state is not None:
            self._pending_state = result.final_state
        return result.output

    RecipeMixer._run_plan = _t._dynamo.disable(_run_plan)
    # Make the WHOLE mixer an opaque compile boundary: Dynamo compiles the
    # traceable shell (embeddings / norms / MLP / head) and calls the mixer eagerly.
    # This avoids the per-call recompile that a closure-defined mixer class triggers
    # (Dynamo cannot stable-guard a <locals> type), so a static-shape training step
    # compiles exactly once.
    RecipeMixer.forward = _t._dynamo.disable(RecipeMixer.forward)
    # Register the shared class so every layer of the model reuses one mixer type.
    _MIXER_CLASS_CACHE[cache_key] = RecipeMixer
    # The mixer's projections run in the recipe's required dtype.
    module = RecipeMixer(config, spec, plan, roles, ref_ops, getattr(torch, dtype), backend)
    return module.to(getattr(torch, dtype))


# ---------------------------------------------------------------------------
# Frozen 100M decoder LM with the recipe mixer swapped in.
# ---------------------------------------------------------------------------


def _build_model(config, recipe_name: str, backend: str, seed: int = 0):
    torch = _torch()
    import torch.nn as nn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(config.width)
            self.norm2 = nn.LayerNorm(config.width)
            self.mixer = build_recipe_mixer(config, recipe_name, backend)
            self.mlp = nn.Sequential(
                nn.Linear(config.width, config.mlp_ratio * config.width, bias=config.bias),
                nn.GELU(approximate="tanh"),
                nn.Linear(config.mlp_ratio * config.width, config.width, bias=config.bias),
            )

        def forward(self, x):
            x = x + self.mixer(self.norm1(x))
            return x + self.mlp(self.norm2(x))

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(config.vocab_size, config.width)
            self.position = nn.Embedding(config.sequence_length, config.width)
            self.blocks = nn.ModuleList(Block() for _ in range(config.layers))
            self.norm = nn.LayerNorm(config.width)
            self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
            self.lm_head.weight = self.token.weight
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, tokens, targets=None):
            b, t = tokens.shape
            pos = torch.arange(t, device=tokens.device).clamp(max=self.position.num_embeddings - 1)
            x = self.token(tokens) + self.position(pos)[None]
            for block in self.blocks:
                x = block(x)
            logits = self.lm_head(self.norm(x))
            loss = None
            if targets is not None:
                loss = nn.functional.cross_entropy(
                    logits.float().view(-1, logits.size(-1)), targets.view(-1)
                )
            return logits, loss

    model = LM().cuda().to(getattr(torch, TRAIN_DTYPE))
    # Restore each mixer's required dtype (the blanket bf16 cast overrides it).
    for block in model.blocks:
        md = _mixer_dtype(recipe_name, backend)
        block.mixer.to(getattr(torch, md))
    return model


def _fineweb_batches(config, count: int, seed: int = 0):
    """Prefetched FineWeb-Edu CUDA batches from the pinned-format shard."""
    torch = _torch()
    tokens = np.memmap(FINEWEB, dtype=np.uint16, mode="r", offset=1024)
    b, t = config.microbatch, config.sequence_length
    need = b * (t + 1)
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(count):
        start = int(rng.integers(0, len(tokens) - need))
        host = np.array(tokens[start:start + need], dtype=np.int64).reshape(b, t + 1)
        dev = torch.from_numpy(host).cuda()
        batches.append((dev[:, :-1], dev[:, 1:]))
    return batches


def _model_params(model) -> dict[str, Any]:
    return {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}


def _param_rel_err(a: dict, b: dict) -> float:
    worst = 0.0
    for k in a:
        if k in b and a[k].shape == b[k].shape and a[k].is_floating_point():
            denom = max(float(b[k].abs().max()), float(a[k].abs().max()), 1e-9)
            worst = max(worst, float((a[k] - b[k]).abs().max()) / denom)
    return worst


def measure_recipe_master(recipe_name: str) -> dict[str, Any]:
    """Full model-level measurement for one recipe, native vs upstream.

    Training uses torch.compile (fullgraph) to fuse the step; inference decode uses
    CUDA-graph capture/replay to eliminate launch overhead (the bandwidth-bound
    regime). These are the levers that take MFU/MBU from the overhead-bound floor
    toward the compute/bandwidth-bound targets.
    """
    torch = _torch()
    from urm.pretraining import PretrainingConfig

    config = PretrainingConfig(**MODEL)
    peaks = _peaks()
    row: dict[str, Any] = {"recipe": recipe_name, "family": named_mixer_recipe(recipe_name).spec.family.name}
    n_params = None

    # Identical FineWeb batches for both backends (deterministic, seed fixed).
    accum = config.gradient_accumulation
    batches = _fineweb_batches(config, (TRAIN_STEPS + 1) * accum, seed=0)
    tokens_per_step = config.microbatch * config.sequence_length * accum

    for backend in ("native", "upstream"):
        mixer = build_recipe_mixer(config, recipe_name, backend)
        if mixer is None:
            row[f"{backend}_status"] = "no adapter"
            continue
        try:
            # Same seed -> identical init for native and upstream (parity baseline).
            model = _build_model(config, recipe_name, backend, seed=0)
            n_params = sum(p.numel() for p in model.parameters())
            row["params_m"] = round(n_params / 1e6, 1)
            opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
            flops_per_step = 6.0 * n_params * tokens_per_step

            # torch.compile is OPT-IN (URM_COMPILE=1): for the custom native kernel
            # the mixer is the bottleneck (compiled ~= eager), and for upstream
            # SDPA/FLA the kernels are already fused so compile only adds overhead.
            # The lever that DOES help is CUDA-graph replay for decode (below).
            run_model = model
            if os.environ.get("URM_COMPILE") == "1":
                try:
                    torch._dynamo.config.recompile_limit = 64
                    run_model = torch.compile(model, fullgraph=False, dynamic=False)
                    tok, tgt = batches[0]
                    _, loss = run_model(tok, tgt)
                    loss.backward()
                    opt.zero_grad(set_to_none=True)
                    row[f"{backend}_compiled"] = True
                except Exception:
                    run_model = model
                    row[f"{backend}_compiled"] = False
            else:
                row[f"{backend}_compiled"] = False

            def train_step(step_idx):
                opt.zero_grad(set_to_none=True)
                for i in range(accum):
                    tok, tgt = batches[step_idx * accum + i]
                    _, loss = run_model(tok, tgt)
                    (loss / accum).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            train_step(0)  # warm (also compiles the mixer kernels)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for s in range(TRAIN_STEPS):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                train_step(s + 1)
                torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            train_peak_mb = torch.cuda.max_memory_allocated() / 1e6
            times.sort()
            step_s = times[len(times) // 2]
            row[f"{backend}_train_mfu"] = flops_per_step / step_s / (peaks["bf16_tflops"] * 1e12)
            row[f"{backend}_train_tok_s"] = tokens_per_step / step_s
            row[f"{backend}_train_peak_mem_mb"] = train_peak_mb
            row[f"{backend}_params"] = _model_params(model)
            # --- gradient parity: mixer-input gradient on a fixed batch ---
            tok, tgt = batches[0]
            opt.zero_grad(set_to_none=True)
            _mixer_input_grads(model)  # register hooks
            _, loss = model(tok, tgt)
            loss.backward()
            row[f"{backend}_migrad"] = _collect_mixer_input_grads(model)
            # --- inference KL: next-token logits on a fixed prompt ---
            model.eval()
            with torch.no_grad():
                logits, _ = model(tok)
            row[f"{backend}_logits"] = logits[0, -1].detach().float().cpu()
            row[f"{backend}_status"] = "ok"
            del model, opt
            torch.cuda.empty_cache()
        except Exception as exc:
            row[f"{backend}_status"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            torch.cuda.empty_cache()

    # --- parameter parity after 10 steps ---
    if "native_params" in row and "upstream_params" in row:
        row["param_parity_10step"] = _param_rel_err(row["native_params"], row["upstream_params"])
    row.pop("native_params", None)
    row.pop("upstream_params", None)

    # --- gradient parity (native vs upstream mixer-input grads) ---
    if "native_migrad" in row and "upstream_migrad" in row:
        row["grad_parity"] = _grad_list_rel_err(row["native_migrad"], row["upstream_migrad"])
    row.pop("native_migrad", None)
    row.pop("upstream_migrad", None)

    # --- inference KL divergence (native || upstream next-token distribution) ---
    if "native_logits" in row and "upstream_logits" in row:
        row["kl_div"] = _kl(row["native_logits"], row["upstream_logits"])
    row.pop("native_logits", None)
    row.pop("upstream_logits", None)
    return row


def _mixer_input_grads(model) -> list:
    """Gradient of the loss w.r.t. each block's mixer input (the norm1 output that
    feeds the mixer). Captured via a forward hook that retains the gradient. This is
    the gradient the mixer's backward must produce; comparing it native-vs-upstream
    measures whether the two mixers backprop the same signal."""
    torch = _torch()
    captured = []
    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            if out.requires_grad:
                out.retain_grad()
                captured.append(out)
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.norm1.register_forward_hook(make_hook(i)))
    # The caller runs forward+backward after registering; here we just return the
    # captured list (populated during the backward the caller performs).
    model._urm_migrad_tensors = captured
    model._urm_migrad_hooks = hooks
    return captured


def _collect_mixer_input_grads(model) -> list:
    grads = [t.grad.detach().float().cpu() for t in getattr(model, "_urm_migrad_tensors", []) if t.grad is not None]
    for h in getattr(model, "_urm_migrad_hooks", []):
        h.remove()
    return grads


def _grad_list_rel_err(a: list, b: list) -> float:
    worst = 0.0
    for ga, gb in zip(a, b):
        if ga.shape == gb.shape:
            denom = max(float(gb.abs().max()), float(ga.abs().max()), 1e-9)
            worst = max(worst, float((ga - gb).abs().max()) / denom)
    return worst


def _kl(native_logits, upstream_logits) -> float:
    torch = _torch()
    p = torch.softmax(native_logits, dim=-1).clamp_min(1e-12)
    q = torch.softmax(upstream_logits, dim=-1).clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum().item())
def _probe_recipe(recipe_name: str) -> dict[str, Any]:
    """Try to build the native + upstream mixer and run one fwd+bwd. Diagnostic."""
    torch = _torch()
    from urm.pretraining import PretrainingConfig

    config = PretrainingConfig(**{k: v for k, v in MODEL.items()})
    result: dict[str, Any] = {"recipe": recipe_name}
    x = torch.randn(1, 256, config.width, device="cuda", dtype=torch.bfloat16)
    for backend in ("native", "upstream"):
        try:
            mixer = build_recipe_mixer(config, recipe_name, backend)
            if mixer is None:
                result[backend] = "no plan"
                continue
            mixer = mixer.cuda().to(torch.bfloat16)
            out = mixer(x)
            loss = out.float().sum()
            loss.backward()
            result[backend] = f"ok out={tuple(out.shape)}"
        except Exception as exc:
            result[backend] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return result


# ---------------------------------------------------------------------------
# Inference sweep: prefill/decode MFU/MBU/throughput across seq lens and batch.
# ---------------------------------------------------------------------------


def _time_it(fn, warmup=3, samples=10) -> float:
    torch = _torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(samples):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def _infer_flops(n_params: int, batch: int, seq: int) -> float:
    """Forward-only model FLOPs (~2 * params * tokens)."""
    return 2.0 * n_params * batch * seq


def _cudagraph_callable(model, toks):
    """Capture the model forward into a CUDA graph and return a replay callable.

    Decode is launch-overhead-bound (single token, many small kernels); capturing
    the whole forward into a graph and replaying it removes per-kernel launch cost,
    which is what pushes decode toward the bandwidth/compute bound. Returns None if
    the model cannot be captured (e.g. a kernel does host sync / in-place CPU work).
    """
    torch = _torch()
    if os.environ.get("URM_NO_CUDAGRAPH") == "1":
        return None
    try:
        static_in = toks.clone()
        # warm up on a side stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(static_in)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph):
                static_out = model(static_in)

        def replay(new_toks):
            static_in.copy_(new_toks)
            graph.replay()
            return static_out

        return replay
    except Exception:
        return None


def measure_inference(recipe_name: str, n_params: int) -> dict[str, Any]:
    """Prefill/decode MFU/MBU/throughput + peak mem + KL, native vs upstream."""
    torch = _torch()
    from urm.pretraining import PretrainingConfig

    peaks = _peaks()
    out: dict[str, Any] = {}
    for backend in ("native", "upstream"):
        config = PretrainingConfig(**{**MODEL, "sequence_length": max(PREFILL_SEQ_LENS)})
        mixer = build_recipe_mixer(config, recipe_name, backend)
        if mixer is None:
            continue
        try:
            model = _build_model(config, recipe_name, backend, seed=0)
            model.eval()
            # --- prefill sweep: seq 1K..32K at bs=1 ---
            prefill = {}
            for seq in PREFILL_SEQ_LENS:
                toks = torch.randint(0, config.vocab_size, (1, seq), device="cuda")
                with torch.no_grad():
                    ms = _time_it(lambda: model(toks))
                    torch.cuda.reset_peak_memory_stats()
                    with torch.no_grad():
                        model(toks)
                    peak_mb = torch.cuda.max_memory_allocated() / 1e6
                fl = _infer_flops(n_params, 1, seq)
                prefill[seq] = {
                    "mfu": fl / ms / (peaks["bf16_tflops"] * 1e12),
                    "tok_s": seq / ms,
                    "ms": ms * 1e3,
                    "peak_mem_mb": peak_mb,
                    "mbu": (n_params * 2) / ms / (peaks["hbm_gbps"] * 1e9),  # weight-read bound
                }
            # --- decode: bs 256/512 single-token, CUDA-graph replayed ---
            decode = {}
            for bs in (DECODE_BATCH, DECODE_BATCH_HI):
                toks = torch.randint(0, config.vocab_size, (bs, 1), device="cuda")
                try:
                    # Prefer CUDA-graph replay (removes launch overhead); fall back
                    # to eager if the model cannot be captured.
                    runner = _cudagraph_callable(model, toks)
                    graphed = runner is not None
                    if runner is None:
                        runner = lambda t: model(t)
                    with torch.no_grad():
                        ms = _time_it(lambda: runner(toks))
                        torch.cuda.reset_peak_memory_stats()
                        with torch.no_grad():
                            runner(toks)
                        peak_mb = torch.cuda.max_memory_allocated() / 1e6
                    fl = _infer_flops(n_params, bs, 1)
                    decode[bs] = {
                        "mfu": fl / ms / (peaks["bf16_tflops"] * 1e12),
                        "tok_s": bs / ms,
                        "ms": ms * 1e3,
                        "peak_mem_mb": peak_mb,
                        "mbu": (n_params * 2) / ms / (peaks["hbm_gbps"] * 1e9),
                        "cuda_graph": graphed,
                    }
                except Exception:
                    decode[bs] = None
            out[f"{backend}_prefill"] = prefill
            out[f"{backend}_decode"] = decode
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            out[f"{backend}_infer_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
            torch.cuda.empty_cache()
    return out


def _run_one(name: str, tmpdir: Path) -> dict[str, Any]:
    """Run one recipe's full measurement in an isolated subprocess."""
    out_path = tmpdir / f"{name}.json"
    env = dict(os.environ)
    pp = env.get("PYTHONPATH", "")
    for p in ("/tmp/urm-comparator-pins/sdm", "/tmp/urm-comparator-pins/atma"):
        if os.path.isdir(p) and p not in pp:
            pp = f"{pp}:{p}" if pp else p
    env["PYTHONPATH"] = pp
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--out", str(out_path)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode == 0 and out_path.exists():
        return json.loads(out_path.read_text())
    recipe = named_mixer_recipe(name)
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = tail[-1][:140] if tail else f"exit {proc.returncode}"
    return {"recipe": name, "family": recipe.spec.family.name, "native_status": f"crash: {detail}"}


def _worker(name: str, out_path: str) -> None:
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("master table requires CUDA")
    row = measure_recipe_master(name)
    n_params = int(row.get("params_m", 131.5) * 1e6)
    try:
        row["inference"] = measure_inference(name, n_params)
    except Exception as exc:
        row["inference_error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    Path(out_path).write_text(json.dumps(row, default=str), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default=None, help="probe a single recipe")
    parser.add_argument("--probe-all", action="store_true", help="probe all recipes")
    parser.add_argument("--worker", default=None, help="measure a single recipe (isolated)")
    parser.add_argument("--out", default=None, help="worker output JSON path")
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker, args.out)
    elif args.probe:
        r = _probe_recipe(args.probe)
        print(f"{r['recipe']}:\n  native:   {r['native']}\n  upstream: {r['upstream']}")
    elif args.probe_all:
        for name in COVERED_RECIPES:
            r = _probe_recipe(name)
            ok_n = str(r["native"]).startswith("ok")
            ok_u = str(r["upstream"]).startswith("ok")
            print(f"{'OK ' if ok_n else 'N--'}{'UOK' if ok_u else 'U--'} {name}")
            if not ok_n:
                print(f"      native: {r['native']}")
            if not ok_u:
                print(f"      upstream: {r['upstream']}")
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="urm_master_rows_"))
        rows = []
        for i, name in enumerate(COVERED_RECIPES):
            row = _run_one(name, tmpdir)
            rows.append(row)
            print(f"[{i+1}/{len(COVERED_RECIPES)}] {name}: native={row.get('native_status')} upstream={row.get('upstream_status')}", flush=True)
        JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
        JSON_OUT.write_text(json.dumps({"schema_version": 1, "recipes": rows}, indent=2, default=str) + "\n")
        MD_OUT.parent.mkdir(parents=True, exist_ok=True)
        MD_OUT.write_text(render_markdown(rows), encoding="utf-8")
        print(f"[master] wrote {JSON_OUT}")
        print(f"[master] wrote {MD_OUT}")


def _pct(v):
    return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "-"


def _num(v):
    if isinstance(v, (int, float)):
        return f"{v:,.0f}" if v >= 100 else f"{v:.2f}"
    return "-"


def _pair(row, key, fmt):
    n = row.get(f"native_{key}")
    u = row.get(f"upstream_{key}")
    return f"{fmt(n)} / {fmt(u)}"


def render_markdown(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    native_ok = sum(1 for r in rows if r.get("native_status") == "ok")
    upstream_ok = sum(1 for r in rows if r.get("upstream_status") == "ok")
    lines = [
        "# Master coverage table: all 62 covered recipes, URM-native vs upstream (model-level)",
        "",
        "Status: master product-evidence record, regenerated by `benchmarks/master_table.py`.",
        "Every recipe is dropped into the frozen ~100M decoder LM (12 layers, 768 width,",
        "12 heads, value_dim 64, FineWeb-Edu tokens) as a `RecipeMixer`; the SAME projection",
        "stack feeds the native and upstream plans, so only the mixer kernel differs. Cells are",
        "`native / upstream`. `-` = not measurable (no upstream adapter, or a kernel limit).",
        "",
        "MFU uses the **measured** bf16 tensor-core peak; MBU uses the **measured** HBM",
        "bandwidth (`results/device-limits.json`). Training MFU = 6·params·tokens/step/peak.",
        "Decode uses CUDA-graph replay (launch-overhead-free); prefill/decode throughput is",
        "measured at seq 1K-32K (bs=1) and bs 256-512. All GPU timing is serial; each recipe",
        "runs in an isolated subprocess.",
        "",
        f"**{native_ok}/{n} recipes measured natively; {upstream_ok}/{n} have a working upstream adapter.**",
        "",
        "## Training (100M params, FineWeb, 10 steps)",
        "",
        "| Recipe | train MFU | train tok/s | param parity @10 | peak mem (MB) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['recipe']}` | {_pair(r, 'train_mfu', _pct)} "
            f"| {_pair(r, 'train_tok_s', _num)} "
            f"| {_num(r.get('param_parity_10step'))} "
            f"| {_pair(r, 'train_peak_mem_mb', _num)} |"
        )
    # Inference: prefill/decode at a reference seq/batch for the summary table.
    lines += [
        "",
        "## Inference (prefill @1K bs=1; decode @bs=512, CUDA-graph)",
        "",
        "| Recipe | prefill MFU | prefill tok/s | decode MFU | decode MBU | decode tok/s |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        inf = r.get("inference", {})
        np_ = inf.get("native_prefill", {}).get("1024", {})
        up_ = inf.get("upstream_prefill", {}).get("1024", {})
        nd = inf.get("native_decode", {}).get(str(DECODE_BATCH_HI)) or {}
        ud = inf.get("upstream_decode", {}).get(str(DECODE_BATCH_HI)) or {}
        lines.append(
            f"| `{r['recipe']}` | {_pct(np_.get('mfu'))} / {_pct(up_.get('mfu'))} "
            f"| {_num(np_.get('tok_s'))} / {_num(up_.get('tok_s'))} "
            f"| {_pct(nd.get('mfu'))} / {_pct(ud.get('mfu'))} "
            f"| {_pct(nd.get('mbu'))} / {_pct(ud.get('mbu'))} "
            f"| {_num(nd.get('tok_s'))} / {_num(ud.get('tok_s'))} |"
        )
    lines.append("")
    lines.append("Full per-sequence (1K-32K) and per-batch (256/512) detail, KL divergence,")
    lines.append("gradient parity, and long-sequence peak memory are in")
    lines.append(f"`results/validation/master-table.json`.")
    lines.append("")
    return "\n".join(lines)
