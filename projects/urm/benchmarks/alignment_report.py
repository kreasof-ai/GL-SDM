"""Gradient alignment and inference-decoding KL divergence: native vs upstream.

Two correctness-evidence tables the product owes, beyond the release gate's
pass/fail:

- **Gradient alignment**: the per-operand gradient errors (native vs the
  upstream/oracle backward) recorded in the committed qualification artifacts.
  This shows the native backward pass produces gradients that align with the
  reference, per operand (the "weights/gradient alignment" evidence). The mixer
  kernels are operations (no learned weights of their own), so the alignment is
  over the input/state operand gradients - the quantities a model's weights
  depend on through the chain rule.

- **Inference decoding KL divergence**: run a single-token decode step with the
  native kernel and the upstream comparator on the same operands, softmax the
  outputs over the feature dimension to get a per-token distribution, and report
  the KL divergence between them. This shows the native decode path produces the
  same output distribution as upstream - the quantity that matters for sampling
  / generation quality. KL is near zero when the distributions agree.

Run ``PYTHONPATH=src:benchmarks python benchmarks/alignment_report.py`` to
regenerate ``docs/validation/alignment.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "qualification"
DOC = PROJECT_ROOT / "docs" / "validation" / "alignment.md"

# Workload -> (artifact, gradient field, gradient base name)
_GRADIENT_SOURCES = {
    "k1-mha": ("native-k1-mha.json", "input_gradient_max_abs_errors_vs_upstream"),
    "k1-gqa": ("native-k1-gqa.json", "input_gradient_max_abs_errors_vs_upstream"),
    "k1-masked-variant": ("native-k1-masked.json", "input_gradient_max_abs_errors_vs_upstream"),
    "k2-diagonal-recurrence": ("native-k2-hgrn.json", "input_gradient_max_abs_errors_vs_upstream"),
    "k2-gated-delta-recurrence": ("native-k2-gated-delta.json", "input_gradient_max_abs_errors_vs_oracle"),
}


def _gradient_rows() -> list[dict]:
    """Per-operand gradient-alignment rows from the committed artifacts."""
    rows = []
    for workload, (artifact_name, grad_field) in _GRADIENT_SOURCES.items():
        path = RESULTS / artifact_name
        if not path.exists():
            continue
        artifact = json.loads(path.read_text())
        for case_key, case_payload in artifact.get("cases", {}).items():
            parity = case_payload.get("parity", {})
            gradients = parity.get(grad_field)
            if not isinstance(gradients, dict):
                continue
            state_gradients = parity.get("state_gradient_max_abs_errors")
            case_id, _, dtype_name = case_key.partition("/")
            for operand, error in gradients.items():
                rows.append(
                    {
                        "workload": workload,
                        "case": case_id,
                        "dtype": dtype_name,
                        "operand": operand,
                        "gradient_max_abs_error": error,
                        "is_state_gradient": bool(
                            state_gradients and operand in state_gradients
                        ),
                        "parity_status": parity.get("status"),
                    }
                )
    return rows


def _kl_divergence_rows() -> list[dict]:
    """Live decode-step KL divergence between native and upstream outputs.

    Runs a single-token decode step with the native kernel and the upstream
    comparator on the same operands, softmaxes the outputs over the feature
    dimension, and reports KL(native || upstream) and KL(upstream || native).
    """
    import torch

    if not torch.cuda.is_available():
        return []
    rows = []
    rows.extend(_k1_decode_kl(torch))
    rows.extend(_k2_gated_delta_decode_kl(torch))
    rows.extend(_k2_diagonal_decode_kl(torch))
    return rows


def _kl(p: "torch.Tensor", q: "torch.Tensor") -> float:
    """KL(p || q) for distributions p, q (already softmaxed over the last dim)."""
    import torch

    p = p.float().clamp_min(1e-12)
    q = q.float().clamp_min(1e-12)
    return float((p * (p.log() - q.log())).sum(dim=-1).mean().item())


def _k1_decode_kl(torch) -> list[dict]:
    """K1 attention decode: KL between native and SDPA output distributions."""
    import torch.nn.functional as F

    from urm.backends.triton.softmax.online import execute_online_softmax_decode

    rows = []
    for dtype_name, dtype in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        B, H, K, V, S = 1, 8, 64, 64, 2048
        generator = torch.Generator(device="cuda").manual_seed(7701)
        q = torch.randn(B, 1, H, K, device="cuda", dtype=dtype, generator=generator)
        k = torch.randn(B, S, H, K, device="cuda", dtype=dtype, generator=generator)
        v = torch.randn(B, S, H, V, device="cuda", dtype=dtype, generator=generator)
        scale = K**-0.5
        native = execute_online_softmax_decode(q[:, 0], k, v, scale=scale, causal=True)
        with torch.no_grad():
            upstream = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=False, scale=scale,
            ).transpose(1, 2)[:, 0]
        native_dist = torch.softmax(native.float(), dim=-1)
        upstream_dist = torch.softmax(upstream.float(), dim=-1)
        rows.append(
            {
                "workload": "k1-mha",
                "mode": "decode",
                "dtype": dtype_name,
                "kl_native_vs_upstream": _kl(native_dist, upstream_dist),
                "kl_upstream_vs_native": _kl(upstream_dist, native_dist),
            }
        )
    return rows


def _k2_gated_delta_decode_kl(torch) -> list[dict]:
    """K2 gated-delta decode: KL between the native decode-step kernel and the
    exact upstream sequential operator, on a single-token step."""
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

    from urm.backends.triton.recurrence.matrix_state import (
        execute_matrix_state_decode_step,
    )

    rows = []
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        B, H, K, V = 2, 8, 64, 64
        generator = torch.Generator(device="cuda").manual_seed(7702)
        state = torch.randn(B, H, K, V, device="cuda", dtype=torch.float32, generator=generator) * 0.1
        q = torch.randn(B, 1, H, K, device="cuda", dtype=dtype, generator=generator)
        k = torch.nn.functional.normalize(
            torch.randn(B, 1, H, K, device="cuda", dtype=dtype, generator=generator).float(), dim=-1
        ).to(dtype)
        v = torch.randn(B, 1, H, V, device="cuda", dtype=dtype, generator=generator)
        g = (-torch.rand(B, 1, H, device="cuda", dtype=dtype, generator=generator) * 0.3)
        beta = torch.rand(B, 1, H, device="cuda", dtype=dtype, generator=generator)
        native = execute_matrix_state_decode_step(
            query=q[:, 0].float(), key=k[:, 0].float(), value=v[:, 0].float(),
            log_decay=g[:, 0].float(), beta=beta[:, 0].float(), state=state.clone(),
            scale=1.0, decay_granularity="head", is_delta=True, read_before=False,
        )
        with torch.no_grad():
            upstream, _ = fused_recurrent_gated_delta_rule(
                q, k, v, g=g, beta=beta, scale=1.0,
                initial_state=state.clone(), output_final_state=True,
            )
        native_dist = torch.softmax(native.float(), dim=-1)
        upstream_dist = torch.softmax(upstream[:, 0].float(), dim=-1)
        rows.append(
            {
                "workload": "k2-gated-delta-recurrence",
                "mode": "decode",
                "dtype": dtype_name,
                "kl_native_vs_upstream": _kl(native_dist, upstream_dist),
                "kl_upstream_vs_native": _kl(upstream_dist, native_dist),
            }
        )
    return rows


def _k2_diagonal_decode_kl(torch) -> list[dict]:
    """K2 diagonal (HGRN) decode: KL between the native decode-step kernel and
    the exact upstream sequential operator, on a single-token step."""
    from fla.ops.hgrn import fused_recurrent_hgrn

    from urm.backends.triton.recurrence.diagonal_recurrence import (
        execute_diagonal_decode_step,
    )

    rows = []
    for dtype_name, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        B, C = 2, 512  # heads8 * key_dim64
        generator = torch.Generator(device="cuda").manual_seed(7703)
        state = torch.randn(B, C, 1, device="cuda", dtype=torch.float32, generator=generator) * 0.1
        x = (torch.randn(B, 1, C, device="cuda", dtype=dtype, generator=generator) * 0.2)
        ld = (-torch.rand(B, 1, C, device="cuda", dtype=dtype, generator=generator) * 0.05)
        native = execute_diagonal_decode_step(
            x=x[:, 0].float(), log_decay=ld[:, 0].float(), input_gate=None,
            read_gate=None, state=state.clone(), read_before=False,
        )
        with torch.no_grad():
            upstream, _ = fused_recurrent_hgrn(
                x, ld, initial_state=state.clone().squeeze(-1), output_final_state=True,
            )
        native_dist = torch.softmax(native.float(), dim=-1)
        upstream_dist = torch.softmax(upstream[:, 0].float(), dim=-1)
        rows.append(
            {
                "workload": "k2-diagonal-recurrence",
                "mode": "decode",
                "dtype": dtype_name,
                "kl_native_vs_upstream": _kl(native_dist, upstream_dist),
                "kl_upstream_vs_native": _kl(upstream_dist, native_dist),
            }
        )
    return rows


def render_markdown(gradient_rows: list[dict], kl_rows: list[dict]) -> str:
    lines = [
        "# Gradient alignment and inference-decoding KL divergence",
        "",
        "Status: evidence record. Gradient-alignment rows regenerate from the",
        "committed qualification artifacts; the decoding KL divergence is a live",
        "single-token decode-step measurement. Regenerate with",
        "`PYTHONPATH=src:benchmarks python benchmarks/alignment_report.py`.",
        "",
        "## Gradient alignment (native vs upstream/oracle backward)",
        "",
        "Per-operand max-abs gradient error recorded in the committed qualification",
        "artifacts. The mixer kernels are operations (no learned weights of their",
        "own), so the alignment is over the input/state operand gradients - the",
        "quantities a model's weights depend on through the chain rule. The state",
        "gradient (the recurrent-state input) is the state-continuation path.",
        "",
        "| Workload | Case | dtype | Operand | gradient max abs err | kind |",
        "|---|---|---|---|---|---|",
    ]
    for r in gradient_rows:
        kind = "state" if r["is_state_gradient"] else "input"
        lines.append(
            f"| {r['workload']} | {r['case']} | {r['dtype']} | `{r['operand']}` | "
            f"{r['gradient_max_abs_error']:.2e} | {kind} |"
        )
    lines += [
        "",
        "## Inference decoding KL divergence (native vs upstream)",
        "",
        "A single-token decode step with the native kernel and the upstream",
        "comparator on the same operands; the outputs are softmaxed over the",
        "feature dimension to a per-token distribution, and the KL divergence is",
        "reported in both directions. Near-zero KL means the native decode path",
        "produces the same output distribution as upstream (the quantity that",
        "matters for sampling / generation quality).",
        "",
        "| Workload | dtype | KL(native \\|\\| upstream) | KL(upstream \\|\\| native) |",
        "|---|---|---|---|",
    ]
    for r in kl_rows:
        lines.append(
            f"| {r['workload']} | {r['dtype']} | "
            f"{r['kl_native_vs_upstream']:.3e} | {r['kl_upstream_vs_native']:.3e} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    gradient_rows = _gradient_rows()
    kl_rows = _kl_divergence_rows()
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(render_markdown(gradient_rows, kl_rows), encoding="utf-8")
    print(f"wrote {DOC} ({len(gradient_rows)} gradient rows, {len(kl_rows)} KL rows)")


if __name__ == "__main__":
    main()
