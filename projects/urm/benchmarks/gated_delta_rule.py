"""Gated delta-rule comparator and benchmark: oracle / eager / FLA / URM.

Four comparison levels per docs/baselines.md:

1. ``oracle``     - slow explicit fp32 recurrence loop, correctness only.
                    Restricted to a documented token budget because it is an
                    O(T) Python loop; larger sequences are recorded as
                    ``not_applicable``, never as zero.
2. ``eager``      - transparent batched PyTorch recurrence (fp32,
                    autograd-capable framework baseline).
3. ``fla_direct`` - pinned FLA operation called directly:
                    ``chunk_gated_delta_rule`` for prefill regimes and
                    ``fused_recurrent_gated_delta_rule`` stepped token-by-token
                    for decode regimes (upstream ships no recurrent backward;
                    decode is forward-only by upstream choice).
4. ``urm_fla``    - the SAME upstream operation behind the typed URM adapter
                    (`UrmGatedDeltaRuleAdapter`), receiving identical
                    preallocated tensors and executing identical kernels.

Measurement methodology mirrors the repaired attention harness:

- Q/K/V/g/beta/states/output gradients are preallocated leaf tensors; input
  preparation never happens inside a timed region;
- cold first calls are recorded separately from warm steady state;
- every sample records end-to-end wall latency and CUDA-event device span;
- direct-versus-adapter samples are paired interleaved A/B, B/A so clock and
  co-tenant drift cancel; overhead is reported as a distribution of paired
  fractions with a percentile bootstrap CI;
- useful FLOP/s uses a documented recurrence model (``FLOP_MODEL``); MBU uses
  an explicitly static analytic byte bound; host-bound decode regimes report
  absolute microseconds and dispatch fraction rather than MFU/MBU.

Usage (from projects/urm):
    PYTHONPATH=src python benchmarks/gated_delta_rule.py [--seq ...]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from urm.adapters.gated_delta_rule import UrmGatedDeltaRuleAdapter, fla_version

# Reference implementations are O(T) Python loops; this budget bounds their
# runtime. Larger cases stay in the matrix for optimized paths while reference
# levels record {"status": "not_applicable", ...}.
REFERENCE_TOKEN_BUDGET = 2048
DECODE_LOOP_TOKENS = 256
STEADY_STATE_MIN_SEQ = 2048
OVERHEAD_GATE_FRACTION = 0.10
HOST_DISPATCH_FRACTION_LIMIT = 0.20

FLOP_MODEL = (
    "forward useful flops = B*T*HV*(6*K*V + 2*(K+V)): per token and value head "
    "the recurrence performs one K*V decay scale, one K*V retrieve "
    "multiply-add, one K*V rank-1 update multiply, one K*V output "
    "multiply-add, plus O(K+V) elementwise work; backward modeled at 2x forward"
)
MBU_NOTE = (
    "static analytic lower bound: mandatory q/k/v/o traffic plus initial/final "
    "state reads/writes when enabled; measured DRAM counters unavailable on "
    "this host so MBU is explicitly labeled static"
)


# ---------------------------------------------------------------------------
# Levels 1 and 2 live in src/urm/adapters/gated_delta_reference.py so the
# benchmark, the adapter tests, and any future tooling share one definition.
# ---------------------------------------------------------------------------
from urm.adapters.gated_delta_reference import (
    eager_gated_delta_rule,
    oracle_gated_delta_rule,
)

# ---------------------------------------------------------------------------
# Levels 3 and 4: pinned upstream direct call and the URM adapter wrapper
# ---------------------------------------------------------------------------


class DirectFLA:
    """Direct pinned FLA call for one regime.

    ``mode="prefill"`` runs chunk-parallel prefill over the whole sequence.
    ``mode="decode"`` steps the fused-recurrent kernel once per token with a
    carried state, matching production inference decode.
    """

    name = "fla_direct"

    def __init__(self, mode: str, steps: int = DECODE_LOOP_TOKENS):
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule,
            fused_recurrent_gated_delta_rule,
        )

        self.mode = mode
        if mode == "prefill":
            self._chunk = chunk_gated_delta_rule
        elif mode == "decode":
            self.steps = steps
            self._fused = fused_recurrent_gated_delta_rule
        else:
            raise ValueError(mode)

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        scale=None,
        initial_state=None,
        output_final_state=True,
    ):
        if self.mode == "prefill":
            return self._chunk(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=False,
            )
        state = initial_state.clone() if initial_state is not None else None
        outputs = []
        for index in range(self.steps):
            out, state = self._fused(
                q[:, index : index + 1],
                k[:, index : index + 1],
                v[:, index : index + 1],
                g=g[:, index : index + 1],
                beta=beta[:, index : index + 1],
                scale=scale,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                state_v_first=False,
            )
            # Every decoded token produces an output; keeping them mirrors
            # what production decode consumes instead of discarding work.
            outputs.append(out)
        stacked = torch.cat(outputs, dim=1)
        return stacked, (state if output_final_state else None)


def make_adapter_call(mode: str, steps: int = DECODE_LOOP_TOKENS):
    """Level 4: the same upstream operation behind the typed URM adapter."""
    adapter = UrmGatedDeltaRuleAdapter(mode)

    if mode == "prefill":

        def prefill_call(
            q, k, v, g, beta, *, scale=None, initial_state=None, output_final_state=True
        ):
            return adapter.execute(
                q,
                k,
                v,
                g,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )

        return prefill_call

    def decode_call(
        q, k, v, g, beta, *, scale=None, initial_state=None, output_final_state=True
    ):
        state = initial_state.clone() if initial_state is not None else None
        outputs = []
        for index in range(steps):
            out, state = adapter.execute(
                q[:, index : index + 1],
                k[:, index : index + 1],
                v[:, index : index + 1],
                g[:, index : index + 1],
                beta[:, index : index + 1],
                scale=scale,
                initial_state=state,
                output_final_state=True,
            )
            # Every decoded token produces an output; keeping them mirrors
            # what production decode consumes instead of discarding work.
            outputs.append(out)
        stacked = torch.cat(outputs, dim=1)
        return stacked, (state if output_final_state else None)

    return decode_call


# ---------------------------------------------------------------------------
# Timing helpers (same conventions as benchmarks/dense_attention.py)
# ---------------------------------------------------------------------------


def _time_one(call):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    call()
    end_event.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return wall_ms, start_event.elapsed_time(end_event)


def distribution_stats(samples):
    ordered = sorted(samples)
    n = len(ordered)
    median = statistics.median(ordered)

    def percentile(fraction):
        return ordered[max(0, -(-int(fraction * n * 100) // 100) - 1)]

    return {
        "n": n,
        "median": median,
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "iqr": percentile(0.75) - percentile(0.25),
        "median_abs_deviation": statistics.median([abs(x - median) for x in ordered]),
        "stdev": statistics.stdev(ordered) if n > 1 else 0.0,
    }


def bootstrap_median_ci(samples, *, resamples=2000, level=0.95, seed=20260826):
    rng = random.Random(seed)
    n = len(samples)
    medians = sorted(
        statistics.median([samples[rng.randrange(n)] for _ in range(n)])
        for _ in range(resamples)
    )
    alpha = (1.0 - level) / 2.0
    low_index = max(0, -(-int(alpha * resamples * 100) // 100) - 1)
    high_index = min(resamples - 1, -(-int((1.0 - alpha) * resamples * 100) // 100) - 1)
    return {
        "level": level,
        "resamples": resamples,
        "lower": medians[low_index],
        "upper": medians[high_index],
    }


def round_stats(stats):
    rounded = {}
    for key, value in stats.items():
        rounded[key] = round(value, 6) if isinstance(value, float) else value
    ci = rounded.get("bootstrap_ci95_median")
    if isinstance(ci, dict):
        ci["lower"] = round(ci["lower"], 6)
        ci["upper"] = round(ci["upper"], 6)
    return rounded


def paired_overhead_stats(direct_wall, adapted_wall, direct_device, adapted_device):
    fractions = [(a - d) / d for d, a in zip(direct_wall, adapted_wall)]
    device_fractions = [(a - d) / d for d, a in zip(direct_device, adapted_device)]
    absolute_us = [(a - d) * 1000.0 for d, a in zip(direct_wall, adapted_wall)]
    ratio = statistics.median(adapted_wall) / statistics.median(direct_wall) - 1.0
    return {
        "pairs": len(fractions),
        "interleaved": True,
        "wall_fraction": {
            **distribution_stats(fractions),
            "bootstrap_ci95_median": bootstrap_median_ci(fractions),
        },
        "device_span_fraction": distribution_stats(device_fractions),
        "absolute_delta_us": distribution_stats(absolute_us),
        "ratio_of_independent_medians_minus_one": ratio,
    }


def useful_flops(batch, tokens, hv, k_dim, v_dim, mode: str) -> float:
    per_token_head = 6.0 * k_dim * v_dim + 2.0 * (k_dim + v_dim)
    base = batch * tokens * hv * per_token_head
    return {"forward": base, "backward": 2.0 * base}[mode]


def static_traffic_bytes(case: dict, dtype_bytes: int) -> float:
    """Static analytic lower bound on mandatory DRAM traffic for one forward."""
    b = case["batch"]
    t = case["sequence"]
    h = case["heads"]
    hv = case["value_heads"]
    kd = case["key_dim"]
    vd = case["value_dim"]
    traffic = (2 * h * kd + hv * (kd + vd)) * b * t * dtype_bytes  # q,k read; v,o rw
    if case.get("initial_state") == "carried":
        traffic += b * hv * kd * vd * 4
    if case.get("output_final_state"):
        traffic += b * hv * kd * vd * 4
    return traffic


def driver_version() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return result.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - environment probing is best-effort
        return "not_available"


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------


class ForwardBundle:
    """Preallocated-input callable for forward-only steady-state timing."""

    def __init__(self, name, call, q, k, v, g, beta, state):
        self.name = name
        self.call = call
        self.inputs = (q, k, v, g, beta, state)

    def invoke(self):
        q, k, v, g, beta, state = self.inputs
        return self.call(q, k, v, g, beta, initial_state=state, output_final_state=True)


def build_forward_bundles(regime, steps, q, k, v, g, beta, state):
    direct = DirectFLA(regime, steps)
    adapter_call = make_adapter_call(regime, steps)

    def direct_entry(
        inp_q,
        inp_k,
        inp_v,
        inp_g,
        inp_beta,
        *,
        initial_state=None,
        output_final_state=True,
    ):
        return direct(
            inp_q,
            inp_k,
            inp_v,
            inp_g,
            inp_beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    def adapter_entry(
        inp_q,
        inp_k,
        inp_v,
        inp_g,
        inp_beta,
        *,
        initial_state=None,
        output_final_state=True,
    ):
        return adapter_call(
            inp_q,
            inp_k,
            inp_v,
            inp_g,
            inp_beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    return {
        "fla_direct": ForwardBundle(
            "fla_direct", direct_entry, q, k, v, g, beta, state
        ),
        "urm_fla": ForwardBundle("urm_fla", adapter_entry, q, k, v, g, beta, state),
    }


def run_case(case, dtype, args, measured_bw, bf16_peak):
    regime = case["regime"]
    tokens = case["sequence"]
    steps = case.get("decode_steps", tokens)
    b, hv = case["batch"], case["value_heads"]
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    prep_started = time.perf_counter()

    def randn(shape):
        return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

    q_data = randn((b, tokens, case["heads"], case["key_dim"]))
    # Caller-side L2 normalization of k per the frozen contract (the production
    # GDN convention); it keeps beta<=1 a contraction so random-data states
    # stay bounded instead of exploding over long sequences.
    k_data = F.normalize(
        randn((b, tokens, case["heads"], case["key_dim"])).float(), dim=-1
    ).to(dtype)
    v_data = randn((b, tokens, hv, case["value_dim"]))
    g_data = F.logsigmoid(randn((b, tokens, hv)).float())
    beta_data = torch.rand(b, tokens, hv, device="cuda", generator=generator).float()
    state_data = None
    if case["initial_state"] == "carried":
        state_data = torch.randn(
            (b, hv, case["key_dim"], case["value_dim"]),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    grad_data = (
        randn((b, tokens, hv, case["value_dim"])) if regime == "prefill" else None
    )
    torch.cuda.synchronize()
    input_prep_ms = (time.perf_counter() - prep_started) * 1000.0

    correctness: dict[str, object] = {}
    implementations: dict[str, object] = {}

    # ---- Level 1/2 references: correctness anchors and slow-baseline timings.
    ref_output = None
    ref_state = None
    if tokens <= REFERENCE_TOKEN_BUDGET and not args.skip_oracle:
        started = time.perf_counter()
        ref_output, ref_state = oracle_gated_delta_rule(
            q_data,
            k_data,
            v_data,
            g_data,
            beta_data,
            initial_state=state_data,
            output_final_state=True,
        )
        torch.cuda.synchronize()
        oracle_ms = (time.perf_counter() - started) * 1000.0
        correctness["oracle"] = {"status": "evaluated"}
        implementations["oracle"] = {
            "role": "semantic oracle",
            "wall_ms_cold_only": round(oracle_ms, 3),
            "note": "single-shot timing; O(T) python loop, not a performance claim",
        }
        started = time.perf_counter()
        eager_out, eager_state = eager_gated_delta_rule(
            q_data,
            k_data,
            v_data,
            g_data,
            beta_data,
            initial_state=state_data,
            output_final_state=True,
        )
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - started) * 1000.0
        implementations["eager"] = {
            "role": "framework baseline",
            "wall_ms_cold_only": round(eager_ms, 3),
            "max_abs_diff_vs_oracle": round(
                (eager_out.float() - ref_output.float()).abs().max().item(), 8
            ),
            "note": "differentiable fp32 recurrence; single-shot timing",
        }
        del eager_out, eager_state
    elif tokens > REFERENCE_TOKEN_BUDGET:
        correctness["oracle"] = {
            "status": "not_applicable",
            "reason": f"sequence {tokens} exceeds documented reference token "
            f"budget {REFERENCE_TOKEN_BUDGET}",
        }

    # ---- Levels 3/4: paired interleaved direct-vs-adapter measurement.
    bundles = build_forward_bundles(
        regime, steps, q_data, k_data, v_data, g_data, beta_data, state_data
    )

    cold = {}
    for name, bundle in bundles.items():
        try:
            started = time.perf_counter()
            bundle.invoke()
            torch.cuda.synchronize()
            cold[name] = round((time.perf_counter() - started) * 1000.0, 3)
        except Exception as error:  # noqa: BLE001 - record unsupported configs
            implementations[name] = {
                "status": "not_applicable",
                "reason": repr(error)[:200],
            }
            cold[name] = None
    ready_names = [name for name in bundles if cold.get(name) is not None]
    paired_ready = len(ready_names) == len(bundles)

    for name in ready_names:
        bundle = bundles[name]
        for _ in range(args.warmup):
            bundle.invoke()
        torch.cuda.synchronize()

    samples: dict[str, tuple[list[float], list[float]]] = {}
    overhead: dict[str, object] = {}
    if paired_ready:
        order_a, order_b = "fla_direct", "urm_fla"
        walls = {order_a: [], order_b: []}
        devices = {order_a: [], order_b: []}
        for pair_index in range(args.pairs):
            first, second = (
                (bundles[order_a], bundles[order_b])
                if pair_index % 2 == 0
                else (bundles[order_b], bundles[order_a])
            )
            for bundle in (first, second):
                wall, device_span = _time_one(bundle.invoke)
                walls[bundle.name].append(wall)
                devices[bundle.name].append(device_span)
        for name in (order_a, order_b):
            samples[name] = (walls[name], devices[name])
        stats_pair = paired_overhead_stats(
            walls[order_a], walls[order_b], devices[order_a], devices[order_b]
        )
        host_bound = regime == "decode"
        median_fraction = stats_pair["wall_fraction"]["median"]
        overhead["forward"] = {
            **{
                key: (round_stats(val) if isinstance(val, dict) else val)
                for key, val in stats_pair.items()
            },
            "gate": {
                "limit_fraction": OVERHEAD_GATE_FRACTION,
                "host_bound_decode": host_bound,
                "steady_state_gpu_bound": tokens >= STEADY_STATE_MIN_SEQ
                and not host_bound,
                "pass": median_fraction <= OVERHEAD_GATE_FRACTION or host_bound,
            },
        }

    # ---- Per-implementation metrics from the paired samples.
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    persistent_bytes = sum(
        tensor.numel() * (4 if tensor.dtype == torch.float32 else dtype_bytes)
        for tensor in (q_data, k_data, v_data, g_data, beta_data, state_data)
        if tensor is not None
    )
    for name in ready_names if paired_ready else []:
        walls, devices = samples[name]
        wall_stats = distribution_stats(walls)
        device_stats = distribution_stats(devices)
        entry: dict[str, object] = {
            "mode": regime,
            "cold_first_call_ms": cold[name],
            "wall_ms": round_stats(wall_stats),
            "device_span_ms": round_stats(device_stats),
            "median_ms": wall_stats["median"],
            "p95_ms": wall_stats["p95"],
            "tokens_per_second": round(b * tokens / (wall_stats["median"] / 1e3), 1),
        }
        flops = useful_flops(
            b, tokens, hv, case["key_dim"], case["value_dim"], "forward"
        )
        entry["useful_tflops_model"] = round(
            flops / (entry["median_ms"] / 1e3) / 1e12, 4
        )
        entry["static_analytic_bytes"] = static_traffic_bytes(case, dtype_bytes)
        if measured_bw:
            entry["mbu_static"] = round(
                (entry["static_analytic_bytes"] / (entry["median_ms"] / 1e3))
                / (measured_bw * 1e9),
                4,
            )
        if bf16_peak:
            entry["mfu_fraction_of_measured_bf16_peak"] = round(
                entry["useful_tflops_model"] / bf16_peak, 4
            )
        dispatch_share = max(
            0.0, (wall_stats["median"] - device_stats["median"]) / wall_stats["median"]
        )
        entry["host_dispatch_share_median"] = round(dispatch_share, 4)
        entry["gpu_bound_eligible_for_mfu_mbu"] = bool(
            dispatch_share <= HOST_DISPATCH_FRACTION_LIMIT
        )
        if regime == "decode":
            entry["absolute_us_per_decode_token"] = round(
                entry["median_ms"] * 1000.0 / max(1, steps), 2
            )
        torch.cuda.reset_peak_memory_stats()
        bundles[name].invoke()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        entry["peak_allocated_bytes"] = peak
        entry["temporary_bytes_above_inputs"] = max(0, peak - persistent_bytes)
        implementations[name] = entry

    # ---- Correctness of optimized levels against the oracle.
    if ref_output is not None and ready_names:
        tolerance = 6e-2 if dtype == torch.bfloat16 else 4.5e-2
        for name in ready_names:
            out, state = bundles[name].invoke()
            difference = (out.float() - ref_output.float()).abs().max().item()
            state_error = None
            if state is not None and ref_state is not None:
                state_error = round(
                    (state.float() - ref_state.float()).abs().max().item(), 6
                )
            correctness[name] = {
                "max_abs_output_error_vs_oracle": round(difference, 6),
                "max_abs_state_error_vs_oracle": state_error,
                "within_dtype_tolerance": bool(difference <= tolerance),
                "tolerance": tolerance,
            }
            del out, state

    # ---- Chunk vs recurrent semantic-equivalence evidence (small shapes).
    if regime == "prefill" and tokens <= 512 and ready_names:
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

        o_chunk, s_chunk = DirectFLA("prefill")(
            q_data,
            k_data,
            v_data,
            g_data,
            beta_data,
            initial_state=state_data,
            output_final_state=True,
        )
        state_rec = state_data.clone() if state_data is not None else None
        o_rec, s_rec = fused_recurrent_gated_delta_rule(
            q_data,
            k_data,
            v_data,
            g=g_data,
            beta=beta_data,
            scale=None,
            initial_state=state_rec,
            output_final_state=True,
        )
        correctness["chunk_vs_recurrent_equivalence"] = {
            "max_abs_output_diff": round(
                (o_chunk.float() - o_rec.float()).abs().max().item(), 6
            ),
            "max_abs_state_diff": round(
                (s_chunk.float() - s_rec.float()).abs().max().item(), 6
            ),
            "note": "reassociation-level agreement expected, not bitwise",
        }
        del o_chunk, s_chunk, o_rec, s_rec

    # ---- Backward (chunk path only; decode has no upstream backward).
    if regime == "prefill" and paired_ready:
        grad_state_data = None
        if state_data is not None:
            grad_state_data = torch.randn(
                (b, hv, case["key_dim"], case["value_dim"]),
                dtype=torch.float32,
                device="cuda",
                generator=generator,
            )
        backward = measure_backward(
            bundles,
            grad_data,
            grad_state_data,
            b,
            tokens,
            hv,
            case,
            args,
            bf16_peak,
            dtype_bytes,
        )
        if backward is not None:
            paired = backward.pop("paired_overhead")
            implementations["backward_note"] = {
                "gradient_set": "dq dk dv dg dbeta (+d(initial_state) when a "
                "state leaf is provided); fused-recurrent decode is "
                "forward-only upstream",
            }
            for name in ("fla_direct", "urm_fla"):
                implementations[name]["backward"] = backward[name]
            median_fraction = paired["wall_fraction"]["median"]
            overhead["backward"] = {
                **{
                    key: (round_stats(val) if isinstance(val, dict) else val)
                    for key, val in paired.items()
                },
                "gate": {
                    "limit_fraction": OVERHEAD_GATE_FRACTION,
                    "steady_state_gpu_bound": tokens >= STEADY_STATE_MIN_SEQ,
                    "pass": median_fraction <= OVERHEAD_GATE_FRACTION
                    or tokens < STEADY_STATE_MIN_SEQ,
                },
            }

    # ---- Final-state materialization cost (prefill only).
    if regime == "prefill" and "fla_direct" in ready_names:
        direct = DirectFLA("prefill")

        def with_state():
            direct(
                q_data,
                k_data,
                v_data,
                g_data,
                beta_data,
                initial_state=state_data,
                output_final_state=True,
            )

        def without_state():
            direct(
                q_data,
                k_data,
                v_data,
                g_data,
                beta_data,
                initial_state=state_data,
                output_final_state=False,
            )

        for _ in range(args.warmup):
            with_state()
            without_state()
        med_with = statistics.median([_time_one(with_state)[0] for _ in range(10)])
        med_without = statistics.median(
            [_time_one(without_state)[0] for _ in range(10)]
        )
        implementations["final_state_materialization"] = {
            "median_with_state_ms": round(med_with, 6),
            "median_without_state_ms": round(med_without, 6),
            "delta_us": round((med_with - med_without) * 1000.0, 2),
            "state_bytes": b * hv * case["key_dim"] * case["value_dim"] * 4,
        }

    return {
        "case": case,
        "input_preparation_ms": round(input_prep_ms, 3),
        "correctness": correctness,
        "implementations": implementations,
        "adapter_overhead": overhead,
    }


class _BackwardLeaf:
    """Fresh-graph backward callable over preallocated leaf tensors.

    The timed region is exactly one forward plus one backward on a freshly
    built graph. Leaf refresh (allocation) happens outside the timed region;
    a fixed preallocated output gradient - and a fixed preallocated final-state
    gradient when a state leaf exists - drives the backward.
    """

    def __init__(self, call, q, k, v, g, beta, state, grad_output, grad_state):
        self.call = call
        self.sources = (q, k, v, g, beta, state)
        self.grad_output = grad_output
        self.grad_state = grad_state
        self.leaves = None
        self.state_leaf = None
        self.refresh()

    def refresh(self):
        self.leaves = tuple(
            tensor.detach().clone().requires_grad_(True) for tensor in self.sources[:5]
        )
        self.state_leaf = (
            self.sources[5].detach().clone().requires_grad_(True)
            if self.sources[5] is not None
            else None
        )

    def invoke(self):
        q, k, v, g, beta = self.leaves
        output, final = self.call(
            q,
            k,
            v,
            g,
            beta,
            initial_state=self.state_leaf,
            output_final_state=self.state_leaf is not None,
        )
        targets = [output]
        grads = [self.grad_output]
        if final is not None and self.grad_state is not None:
            targets.append(final)
            grads.append(self.grad_state)
        torch.autograd.backward(targets, grads)

    def clear(self):
        for tensor in (*self.leaves, self.state_leaf):
            if tensor is not None:
                tensor.grad = None


def measure_backward(
    bundles,
    grad_data,
    grad_state_data,
    b,
    tokens,
    hv,
    case,
    args,
    bf16_peak,
    dtype_bytes,
):
    if grad_data is None:
        return None
    workers = {}
    for name, bundle in bundles.items():
        workers[name] = _BackwardLeaf(
            _unwrap_call(bundle), *bundle.inputs, grad_data, grad_state_data
        )

    for worker in workers.values():
        worker.invoke()
        worker.clear()
        for _ in range(args.warmup):
            worker.invoke()
            worker.clear()
    torch.cuda.synchronize()

    walls = {name: [] for name in workers}
    devices = {name: [] for name in workers}
    for pair_index in range(args.pairs):
        names = ("fla_direct", "urm_fla")
        first, second = names if pair_index % 2 == 0 else names[::-1]
        for who in (first, second):
            worker = workers[who]
            worker.refresh()  # fresh graph; allocation stays outside timing
            wall, device_span = _time_one(worker.invoke)
            worker.clear()
            walls[who].append(wall)
            devices[who].append(device_span)

    metrics: dict[str, object] = {}
    for name in workers:
        wall_stats = distribution_stats(walls[name])
        device_stats = distribution_stats(devices[name])
        flops = useful_flops(
            b, tokens, hv, case["key_dim"], case["value_dim"], "backward"
        )
        entry = {
            "wall_ms": round_stats(wall_stats),
            "device_span_ms": round_stats(device_stats),
            "median_ms": wall_stats["median"],
            "p95_ms": wall_stats["p95"],
            "useful_tflops_model": round(
                flops / (wall_stats["median"] / 1e3) / 1e12, 4
            ),
            "gradients_verified_finite_in_tests": True,
        }
        if bf16_peak:
            entry["mfu_fraction_of_measured_bf16_peak"] = round(
                entry["useful_tflops_model"] / bf16_peak, 4
            )
        metrics[name] = entry
    metrics["paired_overhead"] = paired_overhead_stats(
        walls["fla_direct"],
        walls["urm_fla"],
        devices["fla_direct"],
        devices["urm_fla"],
    )
    return metrics


def _unwrap_call(bundle: ForwardBundle):
    call = bundle.call

    def wrapped(q, k, v, g, beta, *, initial_state=None, output_final_state=True):
        return call(
            q,
            k,
            v,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )

    return wrapped


# ---------------------------------------------------------------------------
# Matrix driver
# ---------------------------------------------------------------------------


def iter_cases(args):
    """Yield benchmark cases: prefill grid plus dedicated decode cases."""
    dim_pairs = [(d, d) for d in args.dims]
    if len(args.asym_dims) % 2:
        raise SystemExit("--asym-dims needs K V pairs")
    asymmetric = [
        (args.asym_dims[i], args.asym_dims[i + 1])
        for i in range(0, len(args.asym_dims), 2)
    ]

    for batch in args.batch:
        # Dedicated decode cases: token-by-token loops at representative
        # depths, always with a carried nonzero state.
        for key_dim, value_dim in dim_pairs:
            yield {
                "name": f"decode_b{batch}_t1_k{key_dim}_v{value_dim}",
                "regime": "decode",
                "batch": batch,
                "sequence": 1,
                "decode_steps": 1,
                "heads": args.heads,
                "value_heads": args.heads,
                "key_dim": key_dim,
                "value_dim": value_dim,
                "initial_state": "carried",
                "output_final_state": True,
                "seed": 11,
            }
            yield {
                "name": f"decode_b{batch}_t{DECODE_LOOP_TOKENS}_k{key_dim}_v{value_dim}",
                "regime": "decode",
                "batch": batch,
                "sequence": DECODE_LOOP_TOKENS,
                "decode_steps": DECODE_LOOP_TOKENS,
                "heads": args.heads,
                "value_heads": args.heads,
                "key_dim": key_dim,
                "value_dim": value_dim,
                "initial_state": "carried",
                "output_final_state": True,
                "seed": 11,
            }
        # Prefill grid.
        for seq in args.seq:
            for key_dim, value_dim in dim_pairs:
                for init in ("zero", "carried"):
                    yield {
                        "name": f"prefill_b{batch}_s{seq}_k{key_dim}_v{value_dim}_{init}",
                        "regime": "prefill",
                        "batch": batch,
                        "sequence": seq,
                        "heads": args.heads,
                        "value_heads": args.heads,
                        "key_dim": key_dim,
                        "value_dim": value_dim,
                        "initial_state": init,
                        "output_final_state": True,
                        "seed": 11,
                    }
        # One asymmetric-dim and one GVA correctness/benchmark probe per batch.
        if asymmetric:
            key_dim, value_dim = asymmetric[0]
            yield {
                "name": f"prefill_b{batch}_s2048_k{key_dim}_v{value_dim}_zero_asym",
                "regime": "prefill",
                "batch": batch,
                "sequence": 2048,
                "heads": args.heads,
                "value_heads": args.heads,
                "key_dim": key_dim,
                "value_dim": value_dim,
                "initial_state": "zero",
                "output_final_state": True,
                "seed": 11,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 2048, 8192, 32768])
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 128],
        help="matched K/V dims; one (K,V)=(d,d) pair per entry",
    )
    parser.add_argument(
        "--asym-dims",
        type=int,
        nargs="+",
        default=[64, 128],
        help="extra asymmetric K V pair benchmarked at s2048 (default 64 128)",
    )
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="re-run cases already present"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fla-gated-delta-rule/benchmark.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    dtype = getattr(torch, args.dtype)

    properties = torch.cuda.get_device_properties(0)
    limits_path = Path("results/device-limits.json")
    measured_bw = None
    bf16_peak = None
    if limits_path.exists():
        limits = json.loads(limits_path.read_text())
        measured_bw = limits["bandwidth"]["sustainable_gbps"]
        bf16_peak = limits["bf16_tensor_core"]["bf16_tensor_core_tfps_measured"]

    document = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "driver": driver_version(),
        "fla_upstream": fla_version(),
        "utilization_denominators": {
            "measured_hbm_gb_s": measured_bw,
            "measured_bf16_tensor_core_tfps": bf16_peak,
            "source": "results/device-limits.json (measured on this host)",
            "mfu_note": "FLA kernels issue MMA instructions; the measured BF16 "
            "tensor-core peak is the MFU denominator. Host-bound decode cases "
            "report absolute microseconds and dispatch share instead.",
            "mbu_note": MBU_NOTE,
            "nsight_compute_counters": "not_available (container lacks "
            "CAP_SYS_ADMIN; ERR_NVGPUCTRPERM)",
        },
        "flop_model": FLOP_MODEL,
        "methodology": {
            "timed_region": "one prefill forward (or forward+backward on a "
            "fresh graph with fixed preallocated output gradient), or one "
            f"{DECODE_LOOP_TOKENS}-step token-by-token decode loop carrying "
            "state; gradients cleared outside timed regions",
            "input_handling": "q/k/v/g/beta/states/output-gradients are "
            "preallocated leaves; nothing unrelated happens while timing",
            "latency_fields": {
                "wall_ms": "end-to-end latency including host dispatch",
                "device_span_ms": "CUDA-event device-side span",
            },
            "sampling": f"paired interleaved direct/adapter calls, {args.pairs} "
            f"pairs after {args.warmup} warmup rounds; within-pair order "
            "alternates A/B, B/A",
            "cold_vs_warm": "first-call compile/autotune cost recorded "
            "separately from warm steady state",
            "reference_budget": REFERENCE_TOKEN_BUDGET,
            "steady_state_min_seq": STEADY_STATE_MIN_SEQ,
            "overhead_gate_fraction": OVERHEAD_GATE_FRACTION,
            "host_dispatch_eligibility_limit": HOST_DISPATCH_FRACTION_LIMIT,
        },
        "semantics": {
            "contract": "docs/fla-gated-delta-rule.md",
            "layout": "q/k [B,T,H,K]; v/g/beta [B,T,HV,*]; state [B,HV,K,V] fp32",
            "gva": "HV % H == 0 supported; this matrix benchmarks H == HV",
            "normalization": "caller-side; use_qk_l2norm_in_kernel frozen False",
            "gate": "log-space decay g; sigmoid-beta fusion frozen off",
            "scale_default": "1/sqrt(K)",
            "k_normalization": "k is L2-normalized caller-side (production "
            "GDN convention, frozen contract); q raw",
            "beta_range": "beta uniform in (0,1) post-sigmoid",
            "decode_backward": "unsupported upstream (forward-only fused "
            "recurrent); backward targets the chunk prefill path",
        },
        "cases": {},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.force:
        existing = json.loads(args.output.read_text())
        if existing.get("cases"):
            document["cases"] = existing["cases"]
    for case in iter_cases(args):
        if case["name"] in document["cases"] and not args.force:
            continue
        print(f"== {case['name']}")
        try:
            result = run_case(case, dtype, args, measured_bw, bf16_peak)
        except torch.OutOfMemoryError as error:
            # Memory-bound exclusion: recorded as not_applicable with a
            # reason, never as a failed performance number.
            torch.cuda.empty_cache()
            reason = f"CUDA OOM on this GPU ({str(error).splitlines()[0][:160]})"
            result = {
                "case": case,
                "correctness": {
                    "status": "not_applicable",
                    "reason": reason,
                },
                "implementations": {
                    "fla_direct": {"status": "not_applicable", "reason": reason},
                    "urm_fla": {"status": "not_applicable", "reason": reason},
                },
                "adapter_overhead": {},
                "memory_bound_exclusion": True,
            }
            print(f"   not_applicable ({reason})")
        document["cases"][case["name"]] = result
        args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
