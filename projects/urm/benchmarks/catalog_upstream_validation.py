"""Catalog upstream validation: coverage, upstream parity, upstream performance.

For each of the 62 representation-covered named recipes (the recipes that lower
into a canonical core and match their independent equation, per
``benchmarks/representation_coverage.py``) this gate measures three things:

1. **Coverage** - the recipe still lowers into a canonical core
   (:func:`urm.oracles.composition.execute_canonical`). Any surprise (a covered
   recipe that no longer lowers) is recorded, never hidden.
2. **Upstream parity** - the recipe's LIBRARY plan (the pinned upstream FLA /
   SDM / SDPA adapter) is executed on CUDA and compared against the REFERENCE
   plan (the independent architecture equation) on identical operands. Pass
   when ``max_abs_err < 2e-2`` or ``max_rel_err < 1e-2``: upstream adapters run
   in float32/bfloat16 with chunked kernels whose accumulation order differs
   from the reference, so the criterion is looser than the FP64-composition
   coverage gate. A recipe whose LIBRARY compile declines is recorded as
   "no upstream adapter" with the decline reason.
3. **Upstream performance** - when the recipe compiles under BOTH NATIVE and
   LIBRARY, the two plans are timed on CUDA in paired interleaved samples
   (synchronized wall timing, order alternated) and the paired overhead
   fraction ``(native - library) / library`` median is recorded. Negative means
   the URM-native kernel is faster than the upstream call.

Some FLA K2 anchors require bfloat16 (or decline the training intent); the gate
retries LIBRARY compiles across ``(training, float32) -> (training, bfloat16)
-> (inference, float32) -> (inference, bfloat16)`` and records the dtype that
bound. Parity for a bfloat16 adapter is measured against the float32 reference
with the same tolerance criterion.

Run ``python benchmarks/catalog_upstream_validation.py`` to regenerate both
outputs:

- ``results/validation/catalog-upstream-validation.json`` (machine-readable)
- ``docs/validation/catalog-upstream-validation.md`` (summary tables)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The SDM upstream checkout needs CUDA_HOME and its pin on sys.path to import.
# Set defaults only - never hardcode-fail when the pin is unavailable.
os.environ.setdefault("CUDA_HOME", "/opt/conda/lib/python3.12/site-packages/nvidia/cu13")
_SDM_PIN = "/tmp/urm-comparator-pins/sdm"
if os.path.isdir(_SDM_PIN):
    import sys

    if _SDM_PIN not in sys.path:
        sys.path.append(_SDM_PIN)

from urm.frontend.mixer_recipes import named_mixer_recipe
from urm.oracles.composition import execute_canonical

from representation_coverage import _reference_execute, _rng_operands

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_OUT = PROJECT_ROOT / "results" / "validation" / "catalog-upstream-validation.json"
MD_OUT = PROJECT_ROOT / "docs" / "validation" / "catalog-upstream-validation.md"

# The 62 representation-covered recipes (representation_coverage.py contract).
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

# Parity criterion: upstream adapters run in float32/bfloat16 with chunked
# kernels whose accumulation order differs from the reference equation.
ABS_TOL = 2e-2
REL_TOL = 1e-2

# Paired timing: warmup calls per side, then paired interleaved samples.
WARMUP = 3
PAIRED_SAMPLES = 12

# LIBRARY compile attempts, in order. Some FLA K2 anchors require bfloat16;
# some K1 adapters are inference-only (their backward is not anchored).
_COMPILE_ATTEMPTS = (
    ("training", "float32"),
    ("training", "bfloat16"),
    ("inference", "float32"),
    ("inference", "bfloat16"),
)

# Operands the pinned FLA chunked kernels require in float32 even when the
# QKV path runs in bfloat16 (per-head/per-channel gates, decays, betas, step
# sizes, biases). This mirrors the operand construction in
# benchmarks/unified_mixer_fla.py. Operands that must share the QKV dtype
# (QKV-matched factors such as transition_alpha/beta, update_keys/values,
# slot_weights, p) are deliberately NOT here - those adapters require one
# shared dtype across query/key/value/factors.
_FLOAT32_GATE_OPERANDS = frozenset({
    "g", "beta", "log_decay", "erase_gate", "write_gate", "gv", "log_alpha",
    "log_mu", "eta", "theta", "alpha", "lamb", "dt", "step_size", "w", "b",
    "bonus", "r", "lambda_weight", "initial_state", "forget_input",
    "forget_weight", "reset_input", "reset_weight", "weight", "x", "A", "B",
    "C",
})


@dataclass
class RecipeValidation:
    name: str
    family: str
    coverage: bool = False
    coverage_note: str = ""
    upstream_status: str = "not attempted"  # pass | fail | no upstream adapter | upstream error
    upstream_anchor: str | None = None
    upstream_dtype: str | None = None
    upstream_abs_err: float | None = None
    upstream_rel_err: float | None = None
    upstream_note: str = ""
    native_status: str = "not attempted"  # available | no native kernel | native error
    native_anchor: str | None = None
    native_note: str = ""
    performance_overhead: float | None = None  # median (native - library) / library
    performance_native_ms: float | None = None
    performance_library_ms: float | None = None
    performance_samples: int = 0
    performance_note: str = ""


def _torch():
    import torch

    return torch


def _to_cuda_operands(operands: dict[str, Any], dtype) -> dict[str, Any]:
    """Convert NumPy operands to CUDA torch tensors for compiler execution.

    Integer-kind arrays (route indices) become int64; booleans stay boolean;
    Python scalars pass through. Floating operands become ``dtype`` on CUDA,
    except the FLA gate/decay/beta operands the pinned chunked kernels require
    in float32 (see ``_FLOAT32_GATE_OPERANDS``).
    """
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
            tensor_dtype = (
                torch.float32 if key in _FLOAT32_GATE_OPERANDS else dtype
            )
            out[key] = torch.as_tensor(arr, dtype=tensor_dtype, device="cuda").contiguous()
    return out


def _compile(recipe, backend, intent: str, dtype: str):
    from urm.compiler.unified_mixer import MixerBackend, MixerIntent, compile_mixer

    return compile_mixer(
        recipe, intent=MixerIntent(intent), backend=backend, dtype=dtype
    )


def _compile_library(recipe):
    """Try the LIBRARY compile attempts in order; return (plan, dtype, error)."""
    from urm.ir.mixer import MixerBackend

    last_exc: Exception | None = None
    for intent, dtype in _COMPILE_ATTEMPTS:
        try:
            return _compile(recipe, MixerBackend.LIBRARY, intent, dtype), dtype, None
        except Exception as exc:  # decline or adapter import failure
            last_exc = exc
    return None, None, last_exc


def _paired_overhead(native_fn, library_fn) -> tuple[float, float, float, int]:
    """Paired interleaved synchronized wall timing; median (native-lib)/lib."""
    torch = _torch()
    for fn in (native_fn, library_fn):
        for _ in range(WARMUP):
            fn()
    torch.cuda.synchronize()
    fractions: list[float] = []
    native_ms: list[float] = []
    library_ms: list[float] = []
    for i in range(PAIRED_SAMPLES):
        first, second = (native_fn, library_fn) if i % 2 == 0 else (library_fn, native_fn)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        first()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        second()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        a_ms = (t1 - t0) * 1e3
        b_ms = (t2 - t1) * 1e3
        n_ms, l_ms = (a_ms, b_ms) if i % 2 == 0 else (b_ms, a_ms)
        native_ms.append(n_ms)
        library_ms.append(l_ms)
        if l_ms > 0:
            fractions.append((n_ms - l_ms) / l_ms)
    if not fractions:
        raise RuntimeError("no positive library timing samples")
    fractions.sort()
    median = fractions[len(fractions) // 2] if len(fractions) % 2 else (
        fractions[len(fractions) // 2 - 1] + fractions[len(fractions) // 2]
    ) / 2
    return median, float(np.median(native_ms)), float(np.median(library_ms)), len(fractions)


def validate_recipe(name: str, seed: int = 0) -> RecipeValidation:
    from urm.ir.mixer import MixerBackend

    torch = _torch()
    recipe = named_mixer_recipe(name)
    spec = recipe.spec
    row = RecipeValidation(name=name, family=spec.family.name)
    operands = _rng_operands(spec, seed=seed)

    # 1. Coverage: the recipe lowers into a canonical core.
    try:
        execute_canonical(spec, **operands)
        row.coverage = True
    except Exception as exc:
        row.coverage_note = f"coverage surprise: {type(exc).__name__}: {exc}"

    # Reference output (independent equation, float32) for parity.
    reference = None
    try:
        reference = _reference_execute(spec, operands)
    except Exception as exc:
        row.upstream_status = "upstream error"
        row.upstream_note = f"reference unavailable: {type(exc).__name__}"
        return row

    # 2. Upstream parity: LIBRARY vs REFERENCE on identical operands.
    library_plan = None
    library_dtype_name: str | None = None
    try:
        library_plan, library_dtype_name, exc = _compile_library(recipe)
        if library_plan is None:
            raise exc if exc is not None else RuntimeError("compile declined")
        row.upstream_anchor = library_plan.anchor
        row.upstream_dtype = library_dtype_name
        lib_dtype = getattr(torch, library_dtype_name)
        cuda_ops = _to_cuda_operands(operands, lib_dtype)
        try:
            library_result = library_plan.execute(**cuda_ops)
        except Exception as first_exc:
            # Some FLA chunk backward paths require even head dims in bf16; the
            # inference (forward-only) path has no such constraint. Retry the
            # execution under an inference-intent plan before recording an error.
            if "backward" in str(first_exc) and library_plan.intent.value == "training":
                library_plan = _compile(
                    recipe, MixerBackend.LIBRARY, "inference", library_dtype_name
                )
                library_result = library_plan.execute(**cuda_ops)
            else:
                raise
        lib_out = library_result.output.detach().float().cpu().numpy()
        ref_out = reference.output.detach().float().cpu().numpy()
        abs_err = float(np.abs(lib_out - ref_out).max())
        rel_err = abs_err / max(float(np.abs(ref_out).max()), 1e-12)
        row.upstream_abs_err = abs_err
        row.upstream_rel_err = rel_err
        row.upstream_status = (
            "pass" if (abs_err < ABS_TOL or rel_err < REL_TOL) else "fail"
        )
    except Exception as exc:
        library_plan = None
        message = str(exc)
        # "no upstream adapter" only when the recipe genuinely has no upstream
        # anchor: K3 (sparse_delta_memory) declines LIBRARY at compile time.
        # Everything else - a missing pinned dependency (ModuleNotFoundError),
        # an FLA pin/source mismatch, a Triton kernel compile crash - is an
        # "upstream error": the adapter exists but could not run here.
        if "K3 uses the URM-native" in message:
            row.upstream_status = "no upstream adapter"
            row.upstream_note = f"{type(exc).__name__}: {message[:160]}"
        else:
            row.upstream_status = "upstream error"
            row.upstream_note = f"{type(exc).__name__}: {message[:160]}"

    # 3. Upstream performance: NATIVE vs LIBRARY paired timing on CUDA.
    try:
        native_plan = _compile(recipe, MixerBackend.NATIVE, "training", "float32")
        row.native_status = "available"
        row.native_anchor = native_plan.anchor
    except Exception as exc:
        native_plan = None
        row.native_status = "no native kernel"
        row.native_note = f"{type(exc).__name__}: {str(exc)[:160]}"

    if native_plan is not None and library_plan is not None:
        try:
            lib_dtype = getattr(torch, library_dtype_name)
            native_ops = _to_cuda_operands(operands, torch.float32)
            library_ops = _to_cuda_operands(operands, lib_dtype)

            def native_fn():
                native_plan.execute(**native_ops)

            def library_fn():
                library_plan.execute(**library_ops)

            median, n_ms, l_ms, samples = _paired_overhead(native_fn, library_fn)
            row.performance_overhead = median
            row.performance_native_ms = n_ms
            row.performance_library_ms = l_ms
            row.performance_samples = samples
        except Exception as exc:
            row.performance_note = f"timing error: {type(exc).__name__}: {str(exc)[:120]}"
    elif native_plan is None:
        row.performance_note = "no native kernel"
    else:
        row.performance_note = "no upstream"

    return row


def validate_catalog(names=COVERED_RECIPES, seed: int = 0) -> list[RecipeValidation]:
    return [validate_recipe(name, seed=seed) for name in names]


def _one_line(text: str, limit: int = 140) -> str:
    """Collapse a (possibly multi-line) note to a single table-safe line."""
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _fmt_err(value: float | None) -> str:
    return "-" if value is None else f"{value:.2e}"


def _fmt_overhead(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:+.1f}%"


def render_markdown(rows: list[RecipeValidation]) -> str:
    parity_pass = [r for r in rows if r.upstream_status == "pass"]
    parity_fail = [r for r in rows if r.upstream_status == "fail"]
    upstream_error = [r for r in rows if r.upstream_status == "upstream error"]
    no_upstream = [r for r in rows if r.upstream_status == "no upstream adapter"]
    perf = [r for r in rows if r.performance_overhead is not None]
    no_native = [r for r in rows if r.native_status == "no native kernel"]
    coverage_surprises = [r for r in rows if not r.coverage]

    # (a) native kernel exists + upstream parity pass + performance measured
    group_a = [
        r for r in rows
        if r.native_status == "available"
        and r.upstream_status == "pass"
        and r.performance_overhead is not None
    ]
    # (b) upstream parity only (parity measured, no native-vs-upstream timing)
    group_b = [
        r for r in rows
        if r.upstream_status in {"pass", "fail"} and r not in group_a
    ]
    # (c) no upstream adapter
    group_c = list(no_upstream)
    # (d) no native kernel
    group_d = list(no_native)

    lines = [
        "# Catalog upstream validation: coverage, upstream parity, upstream performance",
        "",
        "Status: evidence record, regenerated from the live compiler by",
        "`benchmarks/catalog_upstream_validation.py`. For each of the 62",
        "representation-covered recipes (recipes that lower into a canonical core and",
        "match their independent equation), this table records three measurements:",
        "",
        "- **coverage**: the recipe lowers into a canonical core",
        "  (`urm.oracles.composition.execute_canonical`).",
        "- **upstream parity**: the LIBRARY plan (pinned upstream FLA / SDM / SDPA",
        "  adapter) on CUDA vs the REFERENCE plan (independent equation) on identical",
        f"  operands; pass when max abs err < {ABS_TOL:g} or max rel err < {REL_TOL:g}.",
        "  Upstream adapters run in float32/bfloat16 with chunked kernels, so the",
        "  criterion is looser than the FP64 representation-coverage gate.",
        "- **upstream performance**: when both NATIVE and LIBRARY compile, paired",
        "  interleaved synchronized wall timing on CUDA; overhead is the paired median",
        "  of `(native - library) / library` (negative = URM-native is faster).",
        "",
        f"**{len(parity_pass)} of {len(rows)} covered recipes pass upstream parity; "
        f"{len(perf)} have native-vs-upstream performance measured; "
        f"{len(no_upstream)} have no upstream adapter; "
        f"{len(no_native)} have no native kernel.**",
        "",
    ]
    if coverage_surprises:
        lines += [
            "## Coverage surprises (expected to lower, did not)",
            "",
            "| Recipe | Family | note |",
            "|---|---|---|",
        ]
        for r in coverage_surprises:
            lines.append(f"| `{r.name}` | {r.family} | {r.coverage_note} |")
        lines.append("")

    lines += [
        "## (a) Native kernel + upstream parity pass + performance measured",
        "",
        "| Recipe | Family | upstream anchor | dtype | abs err | overhead (native vs upstream) | native ms | upstream ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(group_a, key=lambda r: r.performance_overhead or 0.0):
        lines.append(
            f"| `{r.name}` | {r.family} | `{r.upstream_anchor}` | {r.upstream_dtype} "
            f"| {_fmt_err(r.upstream_abs_err)} | {_fmt_overhead(r.performance_overhead)} "
            f"| {r.performance_native_ms:.3f} | {r.performance_library_ms:.3f} |"
        )
    lines += [
        "",
        "## (b) Upstream parity only (no native-vs-upstream timing)",
        "",
        "| Recipe | Family | upstream anchor | dtype | abs err | rel err | parity | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in group_b:
        note = r.performance_note or r.native_note or "-"
        lines.append(
            f"| `{r.name}` | {r.family} | `{r.upstream_anchor}` | {r.upstream_dtype} "
            f"| {_fmt_err(r.upstream_abs_err)} | {_fmt_err(r.upstream_rel_err)} "
            f"| {r.upstream_status} | {_one_line(note)} |"
        )
    lines += [
        "",
        "## (c) No upstream adapter",
        "",
        "| Recipe | Family | reason |",
        "|---|---|---|",
    ]
    for r in group_c:
        lines.append(f"| `{r.name}` | {r.family} | {_one_line(r.upstream_note)} |")
    lines += [
        "",
        "## (d) No native kernel",
        "",
        "| Recipe | Family | upstream parity | note |",
        "|---|---|---|---|",
    ]
    for r in group_d:
        lines.append(
            f"| `{r.name}` | {r.family} | {r.upstream_status} | {_one_line(r.native_note, 90)} |"
        )
    if parity_fail or upstream_error:
        lines += [
            "",
            "## Upstream parity failures and errors",
            "",
            "| Recipe | Family | status | abs err | rel err | note |",
            "|---|---|---|---|---|---|",
        ]
        for r in parity_fail + upstream_error:
            lines.append(
                f"| `{r.name}` | {r.family} | {r.upstream_status} "
                f"| {_fmt_err(r.upstream_abs_err)} | {_fmt_err(r.upstream_rel_err)} "
                f"| {_one_line(r.upstream_note)} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_outputs(rows: list[RecipeValidation]) -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    MD_OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "gate": "catalog_upstream_validation",
        "recipe_count": len(rows),
        "tolerances": {"abs": ABS_TOL, "rel": REL_TOL},
        "timing": {"warmup": WARMUP, "paired_samples": PAIRED_SAMPLES},
        "summary": {
            "coverage": sum(1 for r in rows if r.coverage),
            "upstream_parity_pass": sum(1 for r in rows if r.upstream_status == "pass"),
            "upstream_parity_fail": sum(1 for r in rows if r.upstream_status == "fail"),
            "upstream_error": sum(1 for r in rows if r.upstream_status == "upstream error"),
            "no_upstream_adapter": sum(
                1 for r in rows if r.upstream_status == "no upstream adapter"
            ),
            "performance_measured": sum(
                1 for r in rows if r.performance_overhead is not None
            ),
            "no_native_kernel": sum(
                1 for r in rows if r.native_status == "no native kernel"
            ),
        },
        "recipes": [asdict(r) for r in rows],
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    MD_OUT.write_text(render_markdown(rows), encoding="utf-8")


def main() -> None:
    rows = validate_catalog()
    write_outputs(rows)
    parity_pass = sum(1 for r in rows if r.upstream_status == "pass")
    parity_fail = [r.name for r in rows if r.upstream_status == "fail"]
    perf = [r for r in rows if r.performance_overhead is not None]
    no_upstream = [r.name for r in rows if r.upstream_status == "no upstream adapter"]
    print(f"[catalog] {len(rows)} recipes validated")
    print(f"[catalog] upstream parity measured & passing: {parity_pass}/{len(rows)}")
    print(f"[catalog] native-vs-upstream performance measured: {len(perf)}")
    for r in sorted(perf, key=lambda r: r.performance_overhead or 0.0):
        print(f"  {r.name}: {_fmt_overhead(r.performance_overhead)}")
    print(f"[catalog] no upstream adapter: {len(no_upstream)} -> {no_upstream}")
    print(f"[catalog] upstream parity failed: {parity_fail}")
    print(f"[catalog] wrote {JSON_OUT}")
    print(f"[catalog] wrote {MD_OUT}")


if __name__ == "__main__":
    main()
