"""Reproduce the historical speed counts in coverage-recovery-analysis.md.

This reads committed artifacts and pre-cut recipe metadata. It performs no GPU
work and does not qualify any current graph or source model.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
REPOSITORY = PROJECT.parents[1]
OLD_RECIPES = "0377bb3^"
NEAR_RATIO = 1 / 1.05


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _train_ratio(row: dict) -> float | None:
    native = row.get("native_train_tok_s")
    upstream = row.get("upstream_train_tok_s")
    if not isinstance(native, (int, float)) or not isinstance(upstream, (int, float)):
        return None
    return native / upstream if native > 0 and upstream > 0 else None


def _inference_ratio(row: dict, mode: str, size: str) -> float | None:
    inference = row.get("inference") or {}
    native = ((inference.get(f"native_{mode}") or {}).get(size) or {}).get("tok_s")
    upstream = ((inference.get(f"upstream_{mode}") or {}).get(size) or {}).get("tok_s")
    if not isinstance(native, (int, float)) or not isinstance(upstream, (int, float)):
        return None
    return native / upstream if native > 0 and upstream > 0 else None


def _counts(ratios: list[float]) -> dict:
    return {
        "pairs": len(ratios),
        "median": statistics.median(ratios) if ratios else None,
        "within_5_percent_proxy": sum(value >= NEAR_RATIO for value in ratios),
        "within_2x": sum(value >= 0.5 for value in ratios),
        "more_than_10x_slower": sum(value < 0.1 for value in ratios),
    }


def _old_recipe_ids() -> dict[str, set[str]]:
    base = "projects/urm/recipes/kernels/"
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", OLD_RECIPES, base],
        cwd=REPOSITORY,
        text=True,
    ).splitlines()
    by_id: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.endswith(".json"):
            continue
        raw = subprocess.check_output(
            ["git", "show", f"{OLD_RECIPES}:{path}"],
            cwd=REPOSITORY,
            text=True,
        )
        document = json.loads(raw)
        for architecture_id in document.get("architecture_ids", []):
            by_id[architecture_id].add(document["name"])
    for path in (PROJECT / "recipes/kernels").glob("*.json"):
        document = _json(path)
        for architecture_id in document.get("architecture_ids", []):
            by_id[architecture_id].add(document["name"])
    return by_id


def main() -> None:
    rows = _json(PROJECT / "results/validation/master-table.json")["recipes"]
    by_name = {row["recipe"]: row for row in rows}
    by_family = defaultdict(list)
    joint = defaultdict(list)
    for row in rows:
        family = row["family"]
        training = _train_ratio(row)
        if training is not None:
            by_family[family].append(training)
        prefill = _inference_ratio(row, "prefill", "4096")
        decode = _inference_ratio(row, "decode", "256")
        if training is not None and prefill is not None and decode is not None:
            joint[family].append(min(training, prefill, decode))

    catalog_bins = defaultdict(list)
    recipe_ids = _old_recipe_ids()
    register = _json(PROJECT / "benchmarks/architecture-coverage.json")["architectures"]
    for entry in register:
        if entry["kernel_upstream_parity_status"] == "not_applicable":
            continue
        ratios = [
            ratio
            for name in recipe_ids[entry["id"]]
            if name in by_name
            if (ratio := _train_ratio(by_name[name])) is not None
        ]
        best = max(ratios) if ratios else None
        if best is None:
            category = "no_old_master_measurement"
        elif best >= NEAR_RATIO:
            category = "within_5_percent_proxy"
        elif best >= 0.5:
            category = "within_2x"
        elif best >= 0.1:
            category = "slower_2_to_10x"
        else:
            category = "more_than_10x_slower"
        catalog_bins[category].append(entry["id"])

    report = {
        "basis": "historical generic decoder; throughput arithmetic, not qualification",
        "training_by_family": {
            family: _counts(values) for family, values in sorted(by_family.items())
        },
        "all_three_modes_by_family": {
            family: _counts(values) for family, values in sorted(joint.items())
        },
        "named_catalog_bins": {
            category: {"count": len(ids), "ids": ids}
            for category, ids in sorted(catalog_bins.items())
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
