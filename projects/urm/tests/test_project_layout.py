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


def test_public_canonical_types_keep_identity():
    import urm
    from urm.frontend import MixerSpec
    from urm.ir import MixerSpec as IrSpec
    from urm.oracles import execute
    from urm.runtime import BackendRegistry

    assert urm.MixerSpec is IrSpec is MixerSpec
    assert urm.BackendRegistry is BackendRegistry
    assert urm.execute is execute


def test_removed_compatibility_shims_are_not_importable():
    """The wildcard compatibility shims were removed; canonical paths remain."""
    import importlib.util

    for removed in ("urm.reference", "urm.backend"):
        assert importlib.util.find_spec(removed) is None, removed


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
                "assert 'triton' not in sys.modules; "
                "assert 'tilelang' not in sys.modules; "
                "assert 'urm.experimental' not in sys.modules; "
                "import importlib.util; "
                "assert importlib.util.find_spec('urm.experimental') is None; "
                "assert importlib.util.find_spec('urm.backends.dual_form_sdm') is None; "
                "assert importlib.util.find_spec('urm.triton_kernels') is None"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_importing_urm_and_numpy_backend_stays_dependency_light():
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import sys; import urm; from urm.backends.numpy.softmax import NumpyBackend; "
                "assert NumpyBackend; "
                "assert 'torch' not in sys.modules; "
                "assert 'triton' not in sys.modules; "
                "assert 'tilelang' not in sys.modules"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_dual_form_names_are_absent_from_production_backend_catalog():
    import urm.backends as backends

    for name in ("DualFormSDMFunction", "dual_form_sdm"):
        assert not hasattr(backends, name)
