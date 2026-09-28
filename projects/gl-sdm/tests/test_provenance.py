from gl_sdm.experiments import provenance


def test_source_fingerprint_tracks_nested_implementation(tmp_path, monkeypatch):
    package = tmp_path / "gl_sdm"
    experiment = package / "experiments" / "provenance.py"
    backend = package / "memory" / "backends" / "commit.py"
    for path in (experiment, backend):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original implementation\n")
    monkeypatch.setattr(provenance, "__file__", str(experiment))
    original = provenance.metadata("gl_sdm")["source_sha256"]
    backend.write_text("# changed nested implementation\n")
    changed = provenance.metadata("gl_sdm")["source_sha256"]
    assert original != changed
    (package / "README.md").write_text("Documentation does not change the implementation.\n")
    assert provenance.metadata("gl_sdm")["source_sha256"] == changed
