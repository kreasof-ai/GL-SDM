"""Provision the pinned upstream comparator checkouts for qualification.

The acceptance contract requires comparing URM-native kernels against the pinned
upstream implementations, which live in scattered public repositories. This script
clones each source at the exact revision frozen in
``benchmarks/architecture-coverage.json`` into a local comparator-pins directory
(default ``/tmp/urm-comparator-pins/<name>``), so the qualification runners and the
GPU parity tests can import them.

ATMA's register entry records ``repository: null`` (it was originally a local
checkout); its public repository is https://github.com/kreasof-ai/atma, confirmed
against the pinned revision's tests. We record the public URL here so the
provisioning is reproducible rather than depending on a local machine path.

Usage: PYTHONPATH=src python benchmarks/provision_comparators.py [name ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COVERAGE = PROJECT_ROOT / "benchmarks" / "architecture-coverage.json"
DEFAULT_PINS = Path("/tmp/urm-comparator-pins")

# Public repository URLs for sources whose register entry lacks one.
REPOSITORY_OVERRIDES = {
    "atma": "https://github.com/kreasof-ai/atma",
}


def _sources() -> dict[str, tuple[str, str]]:
    data = json.loads(COVERAGE.read_text())
    sources = data.get("upstream_sources", data.get("sources", {}))
    out = {}
    for name, info in sources.items():
        if not isinstance(info, dict):
            continue
        repo = info.get("repository") or REPOSITORY_OVERRIDES.get(name)
        revision = info.get("revision")
        if repo and revision:
            out[name] = (repo, revision)
    return out


def provision(name: str, repo: str, revision: str, pins_dir: Path) -> str:
    dest = pins_dir / name
    if dest.exists() and (dest / ".git").exists():
        current = subprocess.check_output(
            ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
        ).strip()
        if current == revision:
            return f"{name}: already at {revision[:12]}"
    # Fresh clone at the pinned revision.
    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(dest)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "fetch", "origin", revision],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", revision],
        check=True, capture_output=True,
    )
    current = subprocess.check_output(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], text=True
    ).strip()
    if current != revision:
        raise RuntimeError(f"{name}: checked out {current}, expected {revision}")
    return f"{name}: cloned {repo} @ {revision[:12]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="sources to provision (default: all)")
    parser.add_argument("--pins-dir", type=Path, default=DEFAULT_PINS)
    args = parser.parse_args()
    sources = _sources()
    names = args.names or sorted(sources)
    args.pins_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in names:
        if name not in sources:
            print(f"{name}: no repository/revision in the register; skipped", file=sys.stderr)
            failures.append(name)
            continue
        repo, revision = sources[name]
        try:
            print(provision(name, repo, revision, args.pins_dir))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"{name}: FAILED ({exc})", file=sys.stderr)
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} source(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(names)} comparator sources provisioned at {args.pins_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
