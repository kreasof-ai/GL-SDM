"""Keep the named construction register and human-readable plan synchronized."""

import json
import re
from pathlib import Path

from urm.compiler.execution import TRUSTED_ANCHORS
from urm.frontend.mixer_recipes import MIXER_RECIPE_NAMES, named_mixer_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_named_register_is_complete_and_source_pinned_or_explicitly_blocked():
    manifest = json.loads(
        (ROOT / "benchmarks/architecture-coverage.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 3
    rows = manifest["architectures"]
    unified_tests = (ROOT / "tests/test_unified_mixer.py").read_text(encoding="utf-8")
    trusted_anchor_names = {anchor.name for anchor in TRUSTED_ANCHORS}
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
            assert row["prototype_reference_test"].split("::")[-1] in unified_tests
            if "prototype_accelerated_test" in row:
                assert (
                    row["prototype_accelerated_test"].split("::")[-1] in unified_tests
                )
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
                for measurement in case["performance"]["measurements"].values():
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
    assert (
        sum(row.get("prototype_status") == "kernel_prototype_only" for row in rows)
        == 76
    )
    assert sum(row.get("native_urm_profile") is not None for row in rows) == 4
    recipe_ids = {
        architecture_id
        for recipe_name in MIXER_RECIPE_NAMES
        for architecture_id in named_mixer_recipe(recipe_name).architecture_ids
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
