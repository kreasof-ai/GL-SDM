"""Compatibility and archival integrity checks for the compiler migration."""

import hashlib
import json
import re
from pathlib import Path


def test_active_documentation_links_resolve():
    root = Path(__file__).resolve().parents[1]
    for document in [root / "README.md", *(root / "docs").rglob("*.md")]:
        for target in re.findall(
            r"\]\(([^)\s]+)\)", document.read_text(encoding="utf-8")
        ):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            assert (document.parent / target.split("#")[0]).exists(), (document, target)


def test_public_compatibility_types_keep_identity():
    import urm
    from urm.backend import BackendRegistry as LegacyRegistry
    from urm.frontend import MixerSpec
    from urm.ir import MixerSpec as LegacySpec
    from urm.oracles import execute
    from urm.reference import execute as legacy_execute
    from urm.runtime import BackendRegistry

    assert urm.MixerSpec is LegacySpec is MixerSpec
    assert urm.BackendRegistry is LegacyRegistry is BackendRegistry
    assert urm.execute is legacy_execute is execute


def test_archived_evidence_is_preserved_byte_for_byte():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "archive/manifest.json").read_text())
    assert manifest["entries"]
    for entry in manifest["entries"]:
        path = (root / entry["archived"]).resolve()
        assert path.is_relative_to(root / "archive")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]


def test_numpy_backend_catalog_does_not_load_experimental_kernels():
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import urm.backends; import urm.compiler.planner; "
                "assert 'torch' not in sys.modules; "
                "assert 'urm.experimental' not in sys.modules; "
                "import importlib.util; "
                "assert importlib.util.find_spec('urm.experimental') is None; "
                "assert importlib.util.find_spec('urm.backends.dual_form_sdm') is None; "
                "assert importlib.util.find_spec('urm.triton_kernels.dual_form_sdm') is None"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_dual_form_names_are_absent_from_production_backend_catalog():
    import urm.backends as backends
    import urm.triton_kernels as triton_kernels

    for module, names in (
        (backends, ("DualFormSDMFunction", "dual_form_sdm")),
        (
            triton_kernels,
            (
                "TritonDualFormSDMFunction",
                "triton_dual_form_sdm",
                "_triton_dual_form_fwd_kernel",
                "_triton_dual_form_bwd_kernel",
            ),
        ),
    ):
        for name in names:
            assert not hasattr(module, name)
