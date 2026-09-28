"""Clone exact upstream pins into a user cache; never modify sibling URM."""
import argparse
from pathlib import Path
import subprocess

PINS = {
    "sdm": ("https://github.com/facebookresearch/sparse-delta-memory.git", "183e7df809131b80ad4393741029d0f20fc3640b"),
    "fla": ("https://github.com/fla-org/flash-linear-attention.git", "864a87f6ce5be8828bef81eb22baafd41937cdf2"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--directory", type=Path, default=Path.home() / ".cache/gl-sdm/sources")
args = parser.parse_args()
for name, (url, revision) in PINS.items():
    root = args.directory / name
    if not root.exists():
        root.mkdir(parents=True)
        subprocess.run(["git", "init", str(root)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(root), "remote", "add", "origin", url], check=True)
        subprocess.run(["git", "-C", str(root), "fetch", "--depth=1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise RuntimeError(f"refusing to replace existing {root} revision {actual}")
    print(f"export GL_SDM_{name.upper()}_ROOT={root}")
