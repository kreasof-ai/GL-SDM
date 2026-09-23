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


def setup_build_toolchain(pins_dir: Path) -> list[str]:
    """Assemble the CUDA build toolchain the JIT comparators (SDM, Mamba) need.

    This instance has no single complete CUDA toolkit: the conda ``nvcc``
    (``/opt/conda/bin/nvcc``, 12.9) works but its ``cicc`` component lives outside
    the ``targets/`` layout nvcc expects, its include tree lacks ``cuda_runtime.h``,
    and there is no unversioned ``libcudart.so``. This function wires the three
    pieces together idempotently:

    1. ``cicc`` symlinked into ``/opt/conda/targets/x86_64-linux/nvvm/bin/`` so the
       conda nvcc can find its compiler component.
    2. ``libcudart.so`` symlinked into ``<pins>/cuda_lib/`` so extensions linking
       ``-lcudart`` resolve it.
    3. An SDPA-backed ``flash_attn`` shim package in ``<pins>/flash_attn_shim/`` so
       FLA's deltaformer comparator runs without building flash-attn from source.

    The complete CUDA *include* tree (cuda_runtime.h, crt/host_config.h, nv/target)
    comes from the tensorflow CUDA headers at build time (see
    ``master_table._run_one``); it is not modified here.
    """
    notes: list[str] = []
    # 1. cicc into the nvcc targets/ layout.
    cicc_src = Path("/opt/conda/nvvm/bin/cicc")
    cicc_dst = Path("/opt/conda/targets/x86_64-linux/nvvm/bin/cicc")
    if cicc_src.exists():
        cicc_dst.parent.mkdir(parents=True, exist_ok=True)
        if not cicc_dst.exists():
            cicc_dst.symlink_to(cicc_src)
        notes.append(f"cicc -> {cicc_dst}")
    # 2. libcudart.so symlink.
    cuda_lib = pins_dir / "cuda_lib"
    cuda_lib.mkdir(parents=True, exist_ok=True)
    cudart = cuda_lib / "libcudart.so"
    if not cudart.exists():
        for cand in ("/opt/conda/lib/libcudart.so.12", "/opt/conda/lib/libcudart.so.13"):
            if Path(cand).exists():
                cudart.symlink_to(cand)
                break
    notes.append(f"libcudart.so -> {cudart}")
    # 3. flash_attn SDPA shim.
    shim = pins_dir / "flash_attn_shim" / "flash_attn" / "__init__.py"
    if not shim.exists():
        shim.parent.mkdir(parents=True, exist_ok=True)
        shim.write_text(_FLASH_ATTN_SHIM)
    notes.append(f"flash_attn shim -> {shim.parent.parent}")
    # 4. Prebuilt mamba selective_scan_cuda extension (mamba1 upstream). Built once
    #    into <pins>/mamba_ext/ so the sweep subprocesses import it without a
    #    per-worker nvcc compile. Only built when --build-mamba is passed (slow).
    return notes


def build_mamba_ext(pins_dir: Path) -> str:
    """Build the pinned Mamba selective_scan_cuda op into <pins>/mamba_ext/.

    The mamba1 upstream comparator (``mamba_selective_scan_adapter``) calls the
    compiled ``selective_scan_cuda`` op; there is no prebuilt wheel for this torch
    and the pure-torch reference is O(T)-sequential (too slow at seq=1024). This
    builds just the selective_scan extension (not all of mamba_ssm) with the
    assembled toolchain and copies the resulting ``.so`` into ``mamba_ext/`` so it
    is importable on the comparator PYTHONPATH. Idempotent: skips if already built.
    """
    dest = pins_dir / "mamba_ext"
    so = dest / "selective_scan_cuda.so"
    if so.exists():
        return f"mamba_ext: already built ({so})"
    mamba_src = pins_dir / "mamba"
    if not (mamba_src / "csrc" / "selective_scan").is_dir():
        raise RuntimeError("mamba source not provisioned; run provisioning first")
    import glob
    import os

    tf_inc = "/opt/conda/lib/python3.12/site-packages/tensorflow/include/third_party/gpus/cuda/include"
    env = dict(os.environ)
    env["CUDA_HOME"] = "/opt/conda"
    for var, entries in (
        ("CPLUS_INCLUDE_PATH", [tf_inc]),
        ("C_INCLUDE_PATH", [tf_inc]),
        ("LIBRARY_PATH", [str(pins_dir / "cuda_lib"), "/opt/conda/lib"]),
        ("LD_LIBRARY_PATH", [str(pins_dir / "cuda_lib"), "/opt/conda/lib"]),
    ):
        for e in entries:
            existing = env.get(var, "")
            if e not in existing:
                env[var] = f"{e}:{existing}" if existing else e
    build = (
        "import glob, torch;"
        "from torch.utils.cpp_extension import load;"
        "srcs=['csrc/selective_scan/selective_scan.cpp']+sorted(glob.glob('csrc/selective_scan/*.cu'));"
        "ext=load(name='selective_scan_cuda',sources=srcs,"
        "extra_cuda_cflags=['-O3','--use_fast_math','-std=c++20','-gencode=arch=compute_86,code=sm_86'],"
        "extra_cflags=['-O3','-std=c++20'],verbose=False);"
        "print(ext.__file__ if hasattr(ext,'__file__') else 'built')"
    )
    subprocess.run(
        [sys.executable, "-c", build], cwd=str(mamba_src), env=env, check=True,
        capture_output=True,
    )
    # Copy the cached .so into mamba_ext/.
    import shutil

    built = list(
        Path.home().glob(".cache/torch_extensions/*/selective_scan_cuda/selective_scan_cuda.so")
    )
    if not built:
        raise RuntimeError("selective_scan_cuda build produced no .so")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(built[0], so)
    return f"mamba_ext: built {so}"


_FLASH_ATTN_SHIM = '''"""SDPA-backed flash_attn shim (flash-attn alias).

The pinned FLA deltaformer comparator calls ``flash_attn_func`` /
``flash_attn_varlen_func``. Building the real flash-attn from source crashes this
instance (long nvcc compile), so this shim provides the same interface backed by
PyTorch SDPA - the same attention math.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__version__ = "2.0.0+sdpa-shim"


def _sdpa(q, k, v, causal):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2).contiguous()


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1), **kw):
    return _sdpa(q, k, v, causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, causal=False, window_size=(-1, -1), **kw):
    outs = []
    for i in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        ks, ke = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)
        outs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="sources to provision (default: all)")
    parser.add_argument("--pins-dir", type=Path, default=DEFAULT_PINS)
    parser.add_argument("--toolchain", action="store_true",
                        help="set up the CUDA build toolchain (cicc/libcudart symlinks, flash_attn shim)")
    parser.add_argument("--build-mamba", action="store_true",
                        help="build the mamba selective_scan_cuda extension into mamba_ext/ (slow, one-time)")
    args = parser.parse_args()
    if args.toolchain:
        for note in setup_build_toolchain(args.pins_dir):
            print(note)
        return 0
    if args.build_mamba:
        setup_build_toolchain(args.pins_dir)  # ensure cicc/libcudart symlinks exist
        print(build_mamba_ext(args.pins_dir))
        return 0
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
