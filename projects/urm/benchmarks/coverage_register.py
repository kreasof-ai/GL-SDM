"""Render the named-architecture coverage register as a concise Markdown table.

The register of record is ``benchmarks/architecture-coverage.json`` (validated by
``tests/test_architecture_coverage.py``). This script renders it into
``docs/planning/coverage.md`` so the human-readable table can never drift from
the machine-readable register. Regenerate with::

    python benchmarks/coverage_register.py > docs/planning/coverage.md

The detailed per-architecture measurement prose lives in the committed kernel-slice
artifacts under ``results/unified-mixer/``; this table is the index, not the evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTER = PROJECT_ROOT / "benchmarks" / "architecture-coverage.json"

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
        if row["kernel_upstream_parity_status"] != "not_applicable"
    ]
    measured = sum(
        1 for row in mixer if row["kernel_upstream_parity_status"] == "measured_pass"
    )
    native = sum(1 for row in mixer if row["native_parity_status"] == "measured_pass")
    live_prototypes = sum(
        1 for row in mixer if row.get("prototype_status") == "kernel_prototype_only"
    )
    pending = sum(
        1 for row in mixer if row.get("prototype_status") == "pending_graph_migration"
    )

    out: list[str] = []
    out.append("# Named architecture coverage register")
    out.append("")
    out.append(
        "Index of every named architecture URM tracks, rendered from the "
        "[machine-readable register](../../benchmarks/architecture-coverage.json) "
        "(`tests/test_architecture_coverage.py` keeps the two in sync). This is the "
        "production construction backlog, not a claim of full architecture support: "
        "inclusion commits us to resolve the mapping and attempt a fair comparison."
    )
    out.append("")
    out.append(
        f"Of {total} catalog rows, {len(mixer)} are mixer-relevant and "
        f"{total - len(mixer)} are classified outside K1/K2/K3 mixer scope. All "
        f"{measured} mixer-relevant rows have measured kernel-upstream parity and "
        f"paired profiling evidence against a pinned source; {native} rows "
        "(MHA/MQA/GQA/BitAttention) additionally have a measured URM-native K1 "
        f"profile. {live_prototypes} rows keep a live `kernel_prototype_only` "
        "slice through the public graph path (the schema-v2 recipes); "
        f"{pending} rows are `pending_graph_migration`: their equation cores were "
        "qualified against the pinned sources, and their prototypes are being "
        "re-authored as typed graph documents. Projections, frontends, caches and "
        "full-layer integration remain open per row."
    )
    out.append("")
    out.append(
        "K1 = [softmax](../kernels/softmax-attention.md), K2 = "
        "[linear/delta](../kernels/linear-delta.md), K3 = "
        "[sparse delta](../kernels/sparse-delta.md). "
        "**Kernel** = output/state/gradient parity plus paired overhead vs the pinned "
        "upstream kernel slice. **Native** = a URM-generated kernel (not an upstream "
        "dispatch) measured against upstream. Per-recipe model-level numbers are in "
        "the [master coverage table](../validation/master-table.md) (rebuilt on the "
        "public graph path as the graph migration completes)."
    )
    out.append("")

    for wave in sorted(by_wave):
        rows = by_wave[wave]
        out.append(f"## Wave {wave}: {WAVE_NAMES[wave]}")
        out.append("")
        out.append("| ID | Architecture | Lowering | Comparator | Kernel | Native |")
        out.append("|---|---|---|---|---|---|")
        for row in rows:
            source = sources[row["source"]]
            repo = source["repository"].rstrip("/").split("/")[-1]
            revision = source.get("revision")
            comparator = f"{repo} `{revision[:10]}`" if revision else f"{repo} (unresolved)"
            lowering = str(row.get("proposed_lowering") or "—")
            out.append(
                f"| {row['id']} | {row['architecture']} | {lowering} | {comparator} "
                f"| {_cell(row['kernel_upstream_parity_status'])} "
                f"| {_cell(row['native_parity_status'])} |"
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
        "Frontend expression, external-adapter execution, native execution and "
        "performance parity are reported separately. A kernel does not implicitly "
        "cover a router, convolution, positional transform, normalization, cache or "
        "inner optimizer. Training, prefill and decode qualify separately; "
        "unsupported upstream modes are recorded as `upstream_unavailable`, never "
        "counted as passes. Catalog items that are MLP, MoE or meta-learning "
        "algorithms are `not_applicable` for mixer-kernel parity and point to the "
        "separate compiler domain. See the [parity plan](../validation/parity.md) "
        "and [generality axes](../compiler/generality-axes.md) for the per-row "
        "qualification gates."
    )
    out.append("")
    return "\n".join(out)


def main() -> None:
    register = json.loads(REGISTER.read_text(encoding="utf-8"))
    print(render_markdown(register), end="")


if __name__ == "__main__":
    main()
