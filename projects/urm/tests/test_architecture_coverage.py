"""Keep the named construction register and human-readable plan synchronized."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_named_register_is_complete_and_source_pinned_or_explicitly_blocked():
    manifest = json.loads(
        (ROOT / "benchmarks/architecture-coverage.json").read_text(encoding="utf-8")
    )
    rows = manifest["architectures"]
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["architecture"] for row in rows}) == len(rows)
    coverage = (ROOT / "docs/planning/coverage.md").read_text(encoding="utf-8")
    assert set(re.findall(r"arch-\d{3}", coverage)) == {row["id"] for row in rows}
    for row in rows:
        assert row["architecture"] in coverage
        assert row["wave"] in {1, 2, 3, 4}
        assert row["work_required"]
        assert set(row["mode_qualification"]) == {"training", "prefill", "decode"}
        source = manifest["sources"][row["source"]]
        if source["revision"] is None:
            assert row["audit_status"] == "identity_unresolved"
        else:
            assert re.fullmatch(r"[a-f0-9]{40}", source["revision"])
        # A catalog entry is not yet an executable, qualified benchmark case.
        if row["native_parity_status"] == "not_measured":
            assert row["mapping_status"] != "parity_qualified"


def test_archived_extension_axes_are_explicit_construction_tasks():
    text = (ROOT / "docs/compiler/generality-axes.md").read_text(encoding="utf-8")
    for axis in range(1, 14):
        assert f"A{axis}:" in text
