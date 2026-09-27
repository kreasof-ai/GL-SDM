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


def metadata(arch, cfg=None):
    import torch
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "reference_fallback": False}
    if arch == "gl_sdm":
        backend = (cfg or {}).get("gl_memory_backend", "torch")
        info.update(implementation="GL-SDM transaction kernels + frozen URM" if backend == "urm" else "project-owned PyTorch GL-SDM",
                    memory_backend=backend, transaction="one token", state_dtype="float32", optimized_kernel=backend == "urm")
        chunk = (cfg or {}).get("gl_chunk_size", 1)
        info["dense_compilation"] = (cfg or {}).get("gl_compile", False)
        info["training_cuda_graph"] = (cfg or {}).get("gl_cuda_graph", False)
        if chunk > 1:
            info.update(transaction=f"one commit per {chunk} tokens", local_context="causal SDPA within chunk",
                        write_mass="per-token reasoning mass divided by chunk size", routing="highest-address ties")
        digest = hashlib.sha256()
        for path in sorted(Path(__file__).parent.glob("*.py")):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        info["source_sha256"] = digest.hexdigest()
        if backend == "urm":
            from .urm_adapter import verify_dependency, read_plan, routed_read_plan
            info["urm"] = verify_dependency()
            if cfg and "hidden_size" in cfg:
                heads = cfg["hidden_size"] // cfg["head_dim"]
                queries = cfg["mbs"] * heads
                if chunk > 1:
                    physical_dim = 128 if cfg["head_dim"] == 64 else cfg["head_dim"]
                    info["logical_value_dim"] = cfg["head_dim"]
                    info["physical_read_value_dim"] = physical_dim
                    info["initial_batch_read_plan"] = routed_read_plan(queries, min(chunk, cfg["seq_len"]),
                        cfg.get("gl_slots", 1024), physical_dim, cfg.get("gl_reads", 8)).serialized_plan()
                else:
                    info["initial_batch_read_plan"] = read_plan(queries * cfg.get("gl_slots", 1024), queries,
                        cfg["head_dim"], cfg.get("gl_reads", 8)).serialized_plan()
            info["routing"] = "URM product-key top-k; highest address wins ties" if chunk > 1 else "GL-SDM stable product-key top-k; smaller address wins ties"
            info["commit"] = "stable ordered sum per address; no forward atomics"
            info["backward_storage"] = "URM saves one aliased snapshot per chunk; selected write rows" if chunk > 1 else "selected rows; full bank versions are not saved"
    if arch in {"sdm", "gdn2"}:
        name = "sdm" if arch == "sdm" else "fla"
        info.update(repository=PINS[name][0], revision=PINS[name][1], source=str(source(name)))
    if arch == "sdm":
        info["cuda_extensions"] = sdm_layer()[1]
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name()
    return info
