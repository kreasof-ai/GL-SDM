"""Render the source-architecture identity and construction backlog.

The register of record is ``extra/architecture-coverage.json`` (validated by
``tests/test_architecture_coverage.py``). This script renders it into
``docs/planning/coverage.md`` so the human-readable table can never drift from
the machine-readable register. Regenerate with::

    python extra/coverage_register.py > docs/planning/coverage.md

Current parity/performance measurements live under ``results/sweep/`` and
``results/upstream/`` (joined in ``results/report.md``). This page is an index,
not a current public-graph or complete-source-model coverage claim.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER = PROJECT_ROOT / "extra" / "architecture-coverage.json"

WAVE_NAMES = {
    1: "Core native closure",
    2: "Structured variants and composite memories",
    3: "Generalized state, routing and axis coverage",
    4: "Nonlinear updates and newly resolved targets",
}

_STATUS = {
    "measured_pass": "pass",
    "not_measured": "—",
    "not_applicable": "n/a",
    "measured_fail": "fail",
    "upstream_unavailable": "no upstream",
}


def _cell(value: object) -> str:
    return _STATUS.get(str(value), str(value))


def build_rows(register: dict) -> dict[int, list[dict]]:
    rows = register["architectures"]
    by_wave: dict[int, list[dict]] = {}
    for row in rows:
        by_wave.setdefault(row["wave"], []).append(row)
    return by_wave


def render_markdown(register: dict) -> str:
    sources = register["sources"]
    by_wave = build_rows(register)
    total = sum(len(rows) for rows in by_wave.values())
    mixer = [
        row
        for rows in by_wave.values()
        for row in rows
        if row["mapping_status"] != "not_applicable"
    ]
    live_prototypes = sum(
        1 for row in mixer if row.get("prototype_status") == "kernel_prototype_only"
    )
    pending = sum(
        1 for row in mixer if row.get("prototype_status") == "pending_graph_migration"
    )

    out: list[str] = []
    out.append("# Source architecture index")
    out.append("")
    out.append(
        "Generated from the [machine register](../../extra/architecture-coverage.json). "
        "This is the source identity and construction backlog. The "
        "[architecture catalog](../catalog.md) maps the registered rows; "
        "[evidence rules](../evidence.md) "
        "define what each status can claim. Inclusion is not model support."
    )
    out.append("")
    out.append(
        f"Of {total} catalog rows, {len(mixer)} are mixer-relevant and "
        f"{total - len(mixer)} are outside mixer scope. {live_prototypes} rows "
        "map to live public-graph fragments and "
        f"{pending} retain the `pending_graph_migration` label. These "
        "figures do not qualify a complete source model. No K2 graph recipe is live. "
        "Parity and performance evidence for the current rows is the "
        "training-harness campaign (`results/sweep/`, `results/upstream/`, "
        "`results/report.md`)."
    )
    out.append("")
    out.append(
        "Proposed lowerings in the register are hypotheses until closed "
        "descriptors, references and public plans qualify them."
    )
    out.append("")

    for wave in sorted(by_wave):
        rows = by_wave[wave]
        out.append(f"## Wave {wave}: {WAVE_NAMES[wave]}")
        out.append("")
        out.append("| ID | Architecture | Proposed fragment | Comparator | Prototype |")
        out.append("|---|---|---|---|---|")
        for row in rows:
            source = sources[row["source"]]
            repo = source["repository"].rstrip("/").split("/")[-1]
            revision = source.get("revision")
            comparator = f"{repo} `{revision[:10]}`" if revision else f"{repo} (unresolved)"
            lowering = str(row.get("proposed_lowering") or "—")
            out.append(
                f"| {row['id']} | {row['architecture']} | {lowering} | {comparator} "
                f"| {_cell(row.get('prototype_status', '—'))} |"
            )
        out.append("")

    out.append("## Source register")
    out.append("")
    out.append(
        "Each comparator is pinned to an exact revision; a resolved identity does "
        "not imply an executable comparator, and source blockers stay attached to "
        "the pinned row. ATMA is a local comparator at its recorded revision."
    )
    out.append("")
    out.append("| Key | Repository | Pin |")
    out.append("|---|---|---|")
    for key in sorted(sources):
        source = sources[key]
        repo = source["repository"]
        revision = source.get("revision")
        pin = f"`{revision[:10]}`" if revision else "unresolved"
        out.append(f"| {key} | {repo} | {pin} |")
    out.append("")
    out.append("## What counts as coverage")
    out.append("")
    out.append(
        "Represented, reference-executable, native-qualified, performance-qualified "
        "and source-model-qualified are separate verdicts. A fragment never covers "
        "its router, frontend, cache or full layer by implication. Training, prefill "
        "and decode qualify separately. See the [evidence protocol](../evidence.md) "
        "and [generality axes](../compiler/generality-axes.md)."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    print(render_markdown(register), end="")


if __name__ == "__main__":
    main()
