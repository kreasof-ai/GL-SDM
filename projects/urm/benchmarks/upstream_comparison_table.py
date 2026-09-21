"""Generate the consolidated upstream comparison table from committed artifacts.

Reads the architecture coverage register and the per-architecture result
artifacts under ``results/unified-mixer/`` and emits a Markdown table of
coverage, parity, and dispatch-overhead versus the pinned upstream comparators.

This is a reporting rollup over validated, committed measurements - it does not
re-run the comparisons. Each row's numbers come from the artifact recorded in
the register, measured on the validated A10G / torch 2.8.0 / triton 3.4.0 line
against the pinned upstream revision named in that artifact. Overhead is the
paired median of per-pair ``(compiled - direct) / direct`` fractions: negative
is faster than the upstream call, positive is slower.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER = PROJECT_ROOT / "benchmarks" / "architecture-coverage.json"


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _coarse_family(proposed_lowering: str | None) -> str:
    if not proposed_lowering:
        return "other"
    for family in ("K1", "K2", "K3"):
        if proposed_lowering.startswith(family):
            return family
    return "other"


def _select_cases(data: dict | None, comparison: dict) -> tuple[list[dict], str | None]:
    """Resolve the exact artifact cases this comparison is qualified on.

    The register names the cases explicitly: ``recipe`` for a single case or
    ``recipes`` for a multi-case artifact. There is no fallback to "all cases" -
    an unnamed or missing case is an evidence error, never a silent substitution
    of unrelated cases. Returns ``(cases, error)``; ``error`` is set when the
    evidence is missing or malformed so the caller cannot emit a pass claim.
    """
    if not data or not isinstance(data.get("cases"), dict):
        return [], "artifact_missing"
    cases = data["cases"]
    named = comparison.get("recipes")
    if named is None:
        recipe = comparison.get("recipe")
        named = [recipe] if recipe is not None else []
    if not named:
        return [], "no_cases_named"
    selected: list[dict] = []
    for name in named:
        case = cases.get(name)
        if not isinstance(case, dict):
            return [], f"case_missing:{name}"
        selected.append(case)
    return selected, None


def _aggregate_status(statuses: list[str | None]) -> str:
    """Combine per-case parity verdicts into one order-independent status.

    A single failed case fails the comparison; otherwise any case that is not a
    clean pass (missing, incomplete, or an unexpected verdict) makes the rollup
    incomplete. Only when every named case passes does the comparison pass.
    """
    if any(status == "fail" for status in statuses):
        return "fail"
    if statuses and all(status == "pass" for status in statuses):
        return "pass"
    return "incomplete"


def _aggregate_case(data: dict | None, comparison: dict) -> tuple[str, float | None, float | None]:
    """Return (parity_status, worst forward overhead, worst fwd+bwd overhead)."""
    selected, error = _select_cases(data, comparison)
    if error is not None:
        # Missing or malformed evidence must never surface as a pass.
        return "incomplete", None, None
    statuses: list[str | None] = []
    forwards: list[float] = []
    forward_backwards: list[float] = []
    for case in selected:
        statuses.append(case.get("parity", {}).get("status"))
        measurements = case.get("performance", {}).get("measurements", {})
        forward = (
            measurements.get("forward", {}).get("paired_compiled_overhead_fraction", {})
            or {}
        ).get("median")
        forward_backward = (
            measurements.get("forward_backward", {}).get("paired_compiled_overhead_fraction", {})
            or {}
        ).get("median")
        if forward is not None:
            forwards.append(forward)
        if forward_backward is not None:
            forward_backwards.append(forward_backward)
    # Report the least-favorable (highest) overhead so a multi-case artifact is
    # not flattered by its best case.
    return (
        _aggregate_status(statuses),
        max(forwards) if forwards else None,
        max(forward_backwards) if forward_backwards else None,
    )


def _pct(fraction: float | None) -> str:
    if fraction is None:
        return "n/a"
    return f"{fraction * 100:+.1f}%"


def build_rows() -> list[dict[str, object]]:
    register = _load(REGISTER)
    if register is None:
        raise RuntimeError(f"coverage register not found: {REGISTER}")
    rows: list[dict[str, object]] = []
    for architecture in register["architectures"]:
        comparison = architecture.get("kernel_upstream_comparison")
        if not comparison:
            continue
        artifact = comparison.get("artifact")
        data = _load(PROJECT_ROOT / artifact) if artifact else None
        parity, forward, forward_backward = _aggregate_case(data, comparison)
        rows.append(
            {
                "architecture": architecture["architecture"],
                "family": _coarse_family(architecture.get("proposed_lowering")),
                "source": comparison.get("source"),
                "recipe": comparison.get("recipe"),
                "parity": parity,
                "forward": forward,
                "forward_backward": forward_backward,
                "components": comparison.get("parity_components", []),
                "scope": comparison.get("scope", ""),
            }
        )
    return rows


def render_markdown(rows: list[dict[str, object]]) -> str:
    by_family: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    compared = len(rows)
    parity_pass = sum(1 for row in rows if row["parity"] == "pass")
    sources = sorted({str(row["source"]) for row in rows})
    lines = [
        "# Upstream comparison register: coverage, parity, and dispatch overhead",
        "",
        "Consolidated rollup of the validated per-architecture comparisons against",
        "pinned upstream sources. Each row is a kernel-slice comparison recorded in",
        "the named artifact under `results/unified-mixer/`; it is not full-layer or",
        "end-to-end qualification. Numbers are reproduced from the committed",
        "artifacts, measured on the validated A10G / torch 2.8.0 / triton 3.4.0 line",
        "against the pinned upstream revision recorded in each artifact.",
        "",
        f"**{parity_pass}/{compared} compared architectures pass parity** against their",
        f"pinned upstream callable. Upstream sources compared: {', '.join(sources)}.",
        "",
        "- **Parity** is the artifact's output/gradient/state correctness verdict",
        "  against the exact upstream callable. A row shows `pass` only when every",
        "  case named in the register passes within the frozen tolerances; a failed",
        "  case shows `fail`, and missing or incomplete evidence shows `incomplete`.",
        "- **Overhead** is the paired median of per-pair `(compiled - direct) / direct`",
        "  dispatch fractions: negative is faster than the upstream call, positive is",
        "  slower. Forward and forward+backward are reported separately; the",
        "  least-favorable case is shown for multi-case artifacts.",
        "",
        "**These are dispatch-overhead numbers, and dispatch coverage is not native",
        "coverage.** For most rows the compiled plan invokes the pinned upstream",
        "kernel through URM's library adapter, so the overhead measures only the",
        "compiler's dispatch cost on top of that shared upstream kernel (typically",
        "a few percent). It does **not** mean URM computes the operation with its",
        "own kernel - a generator that only dispatches would be a thin wrapper.",
        "The honest measure of the unified generator's reach is native generation:",
        "see `docs/validation/native-coverage.md` for which recipes URM computes",
        "natively, and `docs/planning/production-matrix.md` plus",
        "`results/qualification/` for native-replacement performance qualification.",
        "",
    ]
    for family in ("K1", "K2", "K3", "other"):
        family_rows = by_family.get(family)
        if not family_rows:
            continue
        lines.append(f"## {family}")
        lines.append("")
        lines.append(
            "| Architecture | Upstream | Parity | Forward overhead | Fwd+Bwd overhead | Scope |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in sorted(family_rows, key=lambda r: str(r["architecture"])):
            scope = str(row["scope"])
            scope = (scope[:117] + "…") if len(scope) > 117 else scope
            lines.append(
                "| {architecture} | {source} | {parity} | {forward} | {forward_backward} | {scope} |".format(
                    architecture=row["architecture"],
                    source=row["source"],
                    parity=row["parity"],
                    forward=_pct(row["forward"]),
                    forward_backward=_pct(row["forward_backward"]),
                    scope=scope,
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    rows = build_rows()
    print(render_markdown(rows))


if __name__ == "__main__":
    main()
