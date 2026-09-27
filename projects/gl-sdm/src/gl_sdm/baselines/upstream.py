"""Load verified external sources; production kernels never become references."""
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache

PINS = {
    "sdm": ("https://github.com/facebookresearch/sparse-delta-memory", "183e7df809131b80ad4393741029d0f20fc3640b"),
    "fla": ("https://github.com/fla-org/flash-linear-attention", "864a87f6ce5be8828bef81eb22baafd41937cdf2"),
}


@lru_cache(None)
def source(name):
    root = Path(os.environ.get(f"GL_SDM_{name.upper()}_ROOT", Path.home() / ".cache/gl-sdm/sources" / name)).resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if revision != PINS[name][1] or dirty:
        raise RuntimeError(f"{name} requires clean pinned source {PINS[name][1]} at {root}")
    sys.path.insert(0, str(root))
    return root


def verified_import(name, module):
    root = source(name)
    loaded = importlib.import_module(module)
    if not Path(inspect.getfile(loaded)).resolve().is_relative_to(root):
        raise RuntimeError(f"{module} was imported from outside the pinned {name} checkout")
    return loaded


@lru_cache(None)
def sdm_layer():
    import torch.utils.cpp_extension as extension
    cuda_home = os.environ.get("GL_SDM_CUDA_HOME")
    cached_cuda = Path.home() / ".cache/gl-sdm/cuda"
    if cuda_home is None and (cached_cuda / "bin/nvcc").is_file():
        cuda_home = str(cached_cuda.resolve())
    if cuda_home:
        extension.CUDA_HOME = cuda_home
        os.environ["CUDA_HOME"] = cuda_home
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str((Path.home() / ".cache/gl-sdm/extensions").resolve()))
    os.environ.setdefault("MAX_JOBS", "2")
    # Load both CUDA extensions now: an upstream optional gather must not silently
    # take its Triton fallback after a failed extension build.
    identities = {}
    for mod in ("sparse_ip_cuda", "warp_cooperative_gather_cuda"):
        loader = verified_import("sdm", f"lingua.sparse_delta_memory.cuda.{mod}")
        binary = Path(loader._get_cuda_module().__file__)
        identities[mod] = {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    layer = verified_import("sdm", "lingua.sparse_delta_memory.layer")
    return layer, identities
