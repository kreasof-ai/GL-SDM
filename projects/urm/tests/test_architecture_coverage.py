"""Keep the named construction register and human-readable plan synchronized."""

import json
import re
from pathlib import Path

from benchmarks.comparators.anchors import UPSTREAM_ANCHORS
from urm.compiler.select.anchors import TRUSTED_ANCHORS
from benchmarks.recipe_catalog import kernel_recipe_names, recipe_document

ROOT = Path(__file__).resolve().parents[1]


def test_named_register_is_complete_and_source_pinned_or_explicitly_blocked():
    manifest = json.loads(
        (ROOT / "benchmarks/architecture-coverage.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3
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
    profile_cache = {}
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
        assert row["kernel_upstream_parity_status"] in {
            "not_measured",
            "measured_pass",
            "measured_fail",
            "upstream_unavailable",
            "not_applicable",
        }
        assert row["kernel_upstream_profile_status"] in {
            "not_measured",
            "measured_pass",
            "measured_fail",
            "upstream_unavailable",
            "not_applicable",
        }
        comparison = row.get("kernel_upstream_comparison")
        if row["kernel_upstream_parity_status"] == "measured_pass":
            assert row["kernel_upstream_profile_status"] == "measured_pass"
            assert comparison is not None
            assert set(row["mode_qualification"].values()) == {"unqualified"}
            artifact_path = ROOT / comparison["artifact"]
            assert artifact_path.is_file()
            artifact = profile_cache.setdefault(
                str(artifact_path),
                json.loads(artifact_path.read_text(encoding="utf-8")),
            )
            assert artifact["upstream"]["revision"] == comparison["revision"]
            recipes = comparison.get("recipes", [comparison.get("recipe")])
            assert recipes and all(recipe is not None for recipe in recipes)
            assert (
                comparison["performance_gate_fraction"]
                == artifact["methodology"]["overhead_gate_fraction"]
            )
            for recipe in recipes:
                case = artifact["cases"][recipe]
                assert row["id"] in case["architecture_ids"]
                assert case["parity"]["status"] == "pass"
                # The register qualifies the comparison on its declared
                # ``intent_modes`` only; other measured modes (e.g. decode) are
                # reported in the artifact for transparency but are outside this
                # register claim's scope. Map the declared intent modes to the
                # artifact's measurement keys and check exactly those.
                intent_to_measurement = {
                    "training_forward": "forward",
                    "training_forward_backward": "forward_backward",
                    "prefill": "prefill",
                    "decode": "decode",
                }
                claimed = comparison.get("intent_modes")
                measurement_keys = (
                    [intent_to_measurement[m] for m in claimed]
                    if claimed
                    else list(case["performance"]["measurements"].keys())
                )
                for key in measurement_keys:
                    measurement = case["performance"]["measurements"][key]
                    assert measurement["paired_compiled_overhead_fraction"]["gate"][
                        "pass"
                    ]
        else:
            assert comparison is None
        native_profile = row.get("native_urm_profile")
        if native_profile is not None:
            artifact_path = ROOT / native_profile["artifact"]
            assert artifact_path.is_file()
            artifact = profile_cache.setdefault(
                str(artifact_path),
                json.loads(artifact_path.read_text(encoding="utf-8")),
            )
            assert row["native_parity_status"] == "measured_pass"
            assert native_profile["upstream_parity_status"] == "measured_pass"
            assert artifact["upstream"]["revision"] == native_profile["revision"]
            case = artifact["cases"][native_profile["case"]]
            assert case["parity"]["status"] == "pass"
            assert row["id"] in case["architecture_ids"]
            assert native_profile["performance_mode"] == "cuda_graph_replay"
            for measurement in case["performance"]["measurements"].values():
                graph = measurement["cuda_graph_replay"]
                assert graph["paired_native_overhead_fraction"]["gate"]["pass"]
                assert not measurement["paired_native_overhead_fraction"]["gate"][
                    "pass"
                ]
        for extra in row.get("additional_kernel_upstream_comparisons", []):
            artifact_path = ROOT / extra["artifact"]
            assert artifact_path.is_file()
            artifact = profile_cache.setdefault(
                str(artifact_path),
                json.loads(artifact_path.read_text(encoding="utf-8")),
            )
            assert artifact["upstream"]["revision"] == extra["revision"]
            case = artifact["case"]
            assert row["id"] in case["architecture_ids"]
            assert case["parity"]["status"] == "pass"
            assert (
                extra["performance_gate_fraction"]
                == artifact["methodology"]["overhead_gate_fraction"]
            )
            profile = case["performance"]["cuda_graph_replay"]
            assert profile["execution_mode"] == extra["performance_mode"]
            assert profile["paired_compiled_overhead_fraction"]["gate"]["pass"]
        if row["kernel_upstream_parity_status"] == "upstream_unavailable":
            assert row["kernel_upstream_profile_status"] == "upstream_unavailable"
            assert row["kernel_upstream_blocker"]["reason"]
            assert row["kernel_upstream_blocker"]["required_action"]
        if row["kernel_upstream_parity_status"] == "not_applicable":
            assert row["kernel_upstream_profile_status"] == "not_applicable"
            assert comparison is None
            assert row["kernel_upstream_blocker"]["reason"]
            assert row["kernel_upstream_blocker"]["required_action"]
    # 17 architectures keep a live prototype through the graph path (the 15
    # schema-v2 recipes); the remaining 59 are pending graph migration.
    assert (
        sum(row.get("prototype_status") == "kernel_prototype_only" for row in rows)
        == 17
    )
    assert (
        sum(row.get("prototype_status") == "pending_graph_migration" for row in rows)
        == 59
    )
    assert sum(row.get("native_urm_profile") is not None for row in rows) == 4
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
