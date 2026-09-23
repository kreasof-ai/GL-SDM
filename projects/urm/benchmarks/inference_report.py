"""Inference throughput and MFU comparison: URM-native vs the upstream comparator.

The release gate records paired wall times (``direct_wall`` = upstream,
``compiled_wall`` = native) per case, dtype, and mode. This module turns those
into the two product-facing serving tables the acceptance contract calls for:

- **Inference throughput** (prefill and decode), in tokens/second, native vs
  upstream, with the paired overhead. Prefill processes the case's full prompt;
  decode is a single-token step against the persistent state.
- **MFU** (model FLOPs utilization): the useful model FLOPs of the operation
  divided by the measured wall time and the *measured* hardware peak (never the
  vendor datasheet), per the acceptance requirement to distinguish useful model
  FLOPs from implementation FLOPs and to use measured hardware denominators.

The FLOP model counts the operation's useful model FLOPs (the matmul / state
work), documented per family below. MFU is reported for the native and the
upstream path on the same operands, so the comparison is apples-to-apples.

Run ``python benchmarks/inference_report.py`` to regenerate
``docs/validation/inference-throughput.md`` from the committed artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "qualification"
DEVICE_LIMITS = PROJECT_ROOT / "results" / "device-limits.json"
DOC = PROJECT_ROOT / "docs" / "validation" / "inference-throughput.md"

# Measured hardware peaks (results/device-limits.json), never the vendor sheet.
def _peaks() -> dict[str, float]:
    limits = json.loads(DEVICE_LIMITS.read_text())
    return {
        "fp32_cuda_core_tfps": limits["fp32_cuda_core"]["fp32_cuda_core_tfps_measured"],
        "bf16_tensor_core_tfps": limits["bf16_tensor_core"]["bf16_tensor_core_tfps_measured"],
        "sustainable_gbps": limits["bandwidth"]["sustainable_gbps"],
    }


# --- Useful model FLOPs per family, per mode ---------------------------------
# These count the operation's useful model FLOPs (the matmul / state work), not
# implementation overhead. Documented so the MFU denominator is auditable.

def _k1_flops(case: dict, mode: str) -> float:
    """Normalized softmax attention: QK^T + PV matmuls.

    Forward/prefill: 2*B*H*Tq*Tk*(K+V); causal attends to ~half the keys on
    average, so the matmul work is halved. Decode (Tq=1): 2*B*H*Tk*(K+V) (the
    single query attends to the full causal history).
    """
    b, tq, tk = case["batch"], case["query_length"], case["key_length"]
    h, k, v = case["query_heads"], case["key_dim"], case["value_dim"]
    if mode == "decode":
        return 2.0 * b * h * tk * (k + v)  # single query, full history
    causal_factor = 0.5 if case.get("causal", True) else 1.0
    flops = 2.0 * b * h * tq * tk * (k + v) * causal_factor
    if mode == "forward_backward":
        flops *= 2.5  # backward recomputes scores + 3 gradient matmuls
    return flops


def _k2_matrix_flops(case: dict, mode: str) -> float:
    """Matrix-state gated-delta recurrence: per-token decay + retrieved + delta +
    state update + readout, ~5*B*H*K*V per token. Decode is a single token."""
    b, h, k, v = case["batch"], case["heads"], case["key_dim"], case["value_dim"]
    t = 1 if mode == "decode" else case["sequence"]
    flops = 5.0 * b * h * t * k * v
    if mode == "forward_backward":
        flops *= 2.0  # reverse scan
    return flops


def _k2_diagonal_flops(case: dict, mode: str) -> float:
    """Diagonal (per-channel) recurrence: ~2*B*C*N per token (update + readout).
    C = heads * key_dim, N = state width (1 for HGRN)."""
    b, h, k = case["batch"], case["heads"], case["key_dim"]
    c = h * k
    n = 1  # HGRN state width
    t = 1 if mode == "decode" else case["sequence"]
    flops = 2.0 * b * t * c * n
    if mode == "forward_backward":
        flops *= 2.0
    return flops


def _k3_flops(case: dict, mode: str) -> float:
    """Sparse delta state update: per-token read (read_width * value_dim) + write
    (write_width * value_dim) routed gather/scatter plus the delta rule."""
    b, s = case["batch"], case["slots"]
    rw, ww, v = case["read_width"], case["write_width"], case["value_dim"]
    t = 1 if mode == "decode" else case["sequence"]
    flops = 4.0 * b * t * (rw + ww) * v
    if mode == "forward_backward":
        flops *= 2.0
    return flops


_FAMILY_FLOPS = {
    "k1-mha": _k1_flops,
    "k1-gqa": _k1_flops,
    "k1-masked-variant": _k1_flops,
    "k2-diagonal-recurrence": _k2_diagonal_flops,
    "k2-gated-delta-recurrence": _k2_matrix_flops,
    "k3-sparse-state": _k3_flops,
}

# Map each workload to its qualification artifact.
_ARTIFACTS = {
    "k1-mha": "native-k1-mha.json",
    "k1-gqa": "native-k1-gqa.json",
    "k1-masked-variant": "native-k1-masked.json",
    "k2-diagonal-recurrence": "native-k2-hgrn.json",
    "k2-gated-delta-recurrence": "native-k2-gated-delta.json",
    "k3-sparse-state": "native-k2-gated-delta.json",  # placeholder; K3 lives elsewhere
}


def _load_matrix_cases() -> dict[str, dict]:
    matrix = json.loads((PROJECT_ROOT / "benchmarks" / "production-matrix.json").read_text())
    return {w["id"]: {c["id"]: c for c in w["cases"]} for w in matrix["workloads"]}


def _tokens_per_call(workload: str, case: dict, mode: str) -> float:
    """Tokens processed per call for throughput accounting."""
    if mode == "decode":
        return float(case["batch"])  # one token per sequence
    if workload.startswith("k1"):
        return float(case["batch"] * case["query_length"])
    return float(case["batch"] * case["sequence"])


def build_rows() -> list[dict]:
    """Build the per-(workload, case, dtype, mode) throughput + MFU rows."""
    peaks = _peaks()
    matrix_cases = _load_matrix_cases()
    rows = []
    artifact_files = {
        "k1-mha": "native-k1-mha.json",
        "k1-gqa": "native-k1-gqa.json",
        "k1-masked-variant": "native-k1-masked.json",
        "k2-diagonal-recurrence": "native-k2-hgrn.json",
        "k2-gated-delta-recurrence": "native-k2-gated-delta.json",
        "k3-sparse-state": None,  # K3 lives under results/unified-mixer
    }
    for workload, artifact_name in artifact_files.items():
        if workload == "k3-sparse-state":
            path = PROJECT_ROOT / "results" / "unified-mixer" / "sdm-k3.json"
        else:
            path = RESULTS / artifact_name
        if not path.exists():
            continue
        artifact = json.loads(path.read_text())
        flops_fn = _FAMILY_FLOPS[workload]
        for case_key, case_payload in artifact.get("cases", {}).items():
            case_id, _, dtype_name = case_key.partition("/")
            matrix_case = matrix_cases.get(workload, {}).get(case_id)
            if matrix_case is None:
                continue
            peak_tfps = (
                peaks["fp32_cuda_core_tfps"]
                if dtype_name == "float32"
                else peaks["bf16_tensor_core_tfps"]
            )
            for mode, measurement in case_payload.get("performance", {}).get(
                "measurements", {}
            ).items():
                direct_ms = measurement.get("direct_wall", {}).get("median_ms")
                compiled_ms = measurement.get("compiled_wall", {}).get("median_ms")
                if direct_ms is None or compiled_ms is None:
                    continue
                flops = flops_fn(matrix_case, mode)
                tokens = _tokens_per_call(workload, matrix_case, mode)
                overhead = measurement.get(
                    "paired_native_overhead_fraction",
                    measurement.get("paired_compiled_overhead_fraction", {}),
                )
                rows.append(
                    {
                        "workload": workload,
                        "case": case_id,
                        "dtype": dtype_name,
                        "mode": mode,
                        "upstream_ms": direct_ms,
                        "native_ms": compiled_ms,
                        "upstream_tokens_per_s": tokens / (direct_ms / 1e3),
                        "native_tokens_per_s": tokens / (compiled_ms / 1e3),
                        "native_overhead_median": overhead.get("median"),
                        "native_overhead_ci95_upper": overhead.get("ci95_upper"),
                        "gate_pass": overhead.get("gate", {}).get("pass"),
                        "useful_flops": flops,
                        "native_mfu": (flops / (compiled_ms / 1e3) / 1e12) / peak_tfps,
                        "upstream_mfu": (flops / (direct_ms / 1e3) / 1e12) / peak_tfps,
                        "peak_tfps_measured": peak_tfps,
                    }
                )
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "# Inference throughput and MFU: URM-native vs upstream",
        "",
        "Status: evidence record, regenerated from the committed release-gate",
        "artifacts by `benchmarks/inference_report.py`. This is the product-facing",
        "serving comparison the acceptance contract calls for: absolute inference",
        "throughput (tokens/second) and model FLOPs utilization (MFU) for the native",
        "kernel and the upstream comparator on the same operands, per case, dtype,",
        "and mode.",
        "",
        "MFU uses the **measured** hardware peak (`results/device-limits.json`), never",
        "the vendor datasheet, per the acceptance requirement to use measured hardware",
        "denominators. The FLOP model counts the operation's useful model FLOPs (the",
        "matmul / state work), documented in `benchmarks/inference_report.py`.",
        "",
        "## Prefill throughput (tokens/second)",
        "",
        "| Workload | Case | dtype | upstream tok/s | native tok/s | native overhead | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    prefill = [r for r in rows if r["mode"] == "prefill"]
    for r in prefill:
        gate = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['workload']} | {r['case']} | {r['dtype']} | "
            f"{r['upstream_tokens_per_s']:.0f} | {r['native_tokens_per_s']:.0f} | "
            f"{r['native_overhead_median']*100:+.1f}% | {gate} |"
        )
    lines += [
        "",
        "## Decode throughput (tokens/second)",
        "",
        "| Workload | Case | dtype | upstream tok/s | native tok/s | native overhead | gate |",
        "|---|---|---|---|---|---|---|",
    ]
    decode = [r for r in rows if r["mode"] == "decode"]
    for r in decode:
        gate = "PASS" if r["gate_pass"] else "FAIL"
        lines.append(
            f"| {r['workload']} | {r['case']} | {r['dtype']} | "
            f"{r['upstream_tokens_per_s']:.0f} | {r['native_tokens_per_s']:.0f} | "
            f"{r['native_overhead_median']*100:+.1f}% | {gate} |"
        )
    lines += [
        "",
        "## MFU: model FLOPs utilization (native vs upstream)",
        "",
        "MFU = useful model FLOPs / measured wall time / measured hardware peak.",
        "Reported for the forward and decode modes (the serving paths).",
        "",
        "| Workload | Case | dtype | mode | upstream MFU | native MFU | measured peak (TFLOP/s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["mode"] not in ("forward", "decode"):
            continue
        lines.append(
            f"| {r['workload']} | {r['case']} | {r['dtype']} | {r['mode']} | "
            f"{r['upstream_mfu']*100:.1f}% | {r['native_mfu']*100:.1f}% | "
            f"{r['peak_tfps_measured']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = build_rows()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_markdown(rows), encoding="utf-8")
    n_prefill = sum(1 for r in rows if r["mode"] == "prefill")
    n_decode = sum(1 for r in rows if r["mode"] == "decode")
    print(f"wrote {DOC} ({len(rows)} rows: {n_prefill} prefill, {n_decode} decode)")


if __name__ == "__main__":
    main()
