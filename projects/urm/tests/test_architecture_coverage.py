"""Keep the named construction register and human-readable plan synchronized."""

import json
import re
from pathlib import Path

from extra.comparators.anchors import UPSTREAM_ANCHORS
from extra.recipe_catalog import kernel_recipe_names, recipe_document
from urm.compiler.select.anchors import TRUSTED_ANCHORS

ROOT = Path(__file__).resolve().parents[1]


def test_named_register_is_complete_and_source_pinned_or_explicitly_blocked():
    manifest = json.loads(
        (ROOT / "extra/architecture-coverage.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 4
    rows = manifest["architectures"]
    test_sources: dict[str, str] = {}

    def _test_names(file_ref: str) -> str:
        if file_ref not in test_sources:
            test_sources[file_ref] = (ROOT / file_ref).read_text(encoding="utf-8")
        return test_sources[file_ref]
    # Anchor coverage spans the URM-owned core anchors plus the consumer-owned
    # upstream providers (the comparator suite registers them).
    trusted_anchor_names = {anchor.name for anchor in (*TRUSTED_ANCHORS, *UPSTREAM_ANCHORS)}
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["architecture"] for row in rows}) == len(rows)
    coverage = (ROOT / "docs/planning/coverage.md").read_text(encoding="utf-8")
    assert set(re.findall(r"arch-\d{3}", coverage)) == {row["id"] for row in rows}
    for row in rows:
        assert row["architecture"] in coverage
        assert row["wave"] in {1, 2, 3, 4}
        assert row["work_required"]
        source = manifest["sources"][row["source"]]
        if source["revision"] is None:
            assert row["audit_status"] == "identity_unresolved"
        else:
            assert re.fullmatch(r"[a-f0-9]{40}", source["revision"])
        # A catalog entry is not yet an executable, qualified benchmark case.
        if row.get("prototype_status") == "kernel_prototype_only":
            assert row["prototype_kernel"] in {
                "K1_softmax_reduction",
                "K2_state_recurrence",
                "K3_sparse_delta_state",
            }
            assert row["prototype_scope"]
            assert row["mapping_status"] != "parity_qualified"
            assert set(row["mode_qualification"].values()) == {"unqualified"}
            assert row["prototype_validated_anchors"]
            assert set(row["prototype_validated_anchors"]) <= trusted_anchor_names
            ref_file, ref_test = row["prototype_reference_test"].split("::")
            assert f"def {ref_test}" in _test_names(ref_file)
            if "prototype_accelerated_test" in row:
                accel_file, accel_test = row["prototype_accelerated_test"].split("::")
                assert f"def {accel_test}" in _test_names(accel_file)
    # 19 architectures keep a live prototype through the graph path (the 17
    # schema-v2 recipes); the remaining 57 are pending graph migration.
    assert (
        sum(row.get("prototype_status") == "kernel_prototype_only" for row in rows)
        == 19
    )
    assert (
        sum(row.get("prototype_status") == "pending_graph_migration" for row in rows)
        == 57
    )
    recipe_ids = {
        architecture_id
        for recipe_name in kernel_recipe_names()
        for architecture_id in recipe_document(recipe_name)["architecture_ids"]
    }
    prototype_ids = {
        row["id"]
        for row in rows
        if row.get("prototype_status") == "kernel_prototype_only"
    }
    assert recipe_ids == prototype_ids


def test_archived_extension_axes_are_explicit_construction_tasks():
    text = (ROOT / "docs/compiler/generality-axes.md").read_text(encoding="utf-8")
    for axis in range(1, 14):
        assert f"A{axis}:" in text
