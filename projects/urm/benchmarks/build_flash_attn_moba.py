"""Build a narrow, reproducible FlashAttention extension for pinned MoBA checks.

The reference A10G profile needs only BF16, head dimension 32, and both causal
modes. The upstream C++ dispatcher is copied into a temporary directory and
narrowed to that contract; the four compiled CUDA translation units come from
the pinned FlashAttention checkout unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_REVISION = "1bda8f9290cd48d030f1516f0e680cd464ef3554"
KERNEL_SOURCES = (
    "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
    "csrc/flash_attn/src/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
    "csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _narrow_dispatch(source: str) -> str:
    forward = """void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                if (params.num_splits <= 1 && !force_split_kernel) {  // If we don't set it num_splits == 0
                    run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
                } else {
                    run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim, Is_causal>(params, stream);
                }
            });
        });
    });
}"""
    forward_narrow = """void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) {
    TORCH_CHECK(params.is_bf16 && params.d == 32, "narrow MoBA comparator supports BF16 D=32 only");
    TORCH_CHECK(params.num_splits <= 1 && !force_split_kernel, "narrow MoBA comparator excludes split-KV calls");
    if (params.is_causal) {
        run_mha_fwd_<cutlass::bfloat16_t, 32, true>(params, stream);
    } else {
        run_mha_fwd_<cutlass::bfloat16_t, 32, false>(params, stream);
    }
}"""
    backward = """void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_bwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}"""
    backward_narrow = """void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(params.is_bf16 && params.d == 32, "narrow MoBA comparator supports BF16 D=32 only");
    if (params.is_causal) {
        run_mha_bwd_<cutlass::bfloat16_t, 32, true>(params, stream);
    } else {
        run_mha_bwd_<cutlass::bfloat16_t, 32, false>(params, stream);
    }
}"""
    if source.count(forward) != 1 or source.count(backward) != 1:
        raise RuntimeError("pinned FlashAttention dispatcher blocks changed")
    return source.replace(forward, forward_narrow).replace(backward, backward_narrow)


def build(source_root: Path, output_root: Path, cuda_home: Path) -> dict[str, object]:
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    )
    if revision != EXPECTED_REVISION or dirty:
        raise RuntimeError(
            f"expected clean FlashAttention {EXPECTED_REVISION}, got {revision}; dirty={bool(dirty)}"
        )

    source_files = [source_root / path for path in KERNEL_SOURCES]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    build_root = output_root / "build"
    build_root.mkdir(parents=True, exist_ok=True)
    narrowed_cpp = build_root / "flash_api_moba_narrow.cpp"
    original_cpp = source_root / "csrc/flash_attn/flash_api.cpp"
    narrowed_cpp.write_text(
        _narrow_dispatch(original_cpp.read_text(encoding="utf-8")), encoding="utf-8"
    )

    setup_file = build_root / "setup.py"
    include_dirs = [
        source_root / "csrc/flash_attn",
        source_root / "csrc/flash_attn/src",
        source_root / "csrc/cutlass/include",
    ]
    setup_file.write_text(
        "from setuptools import setup\n"
        "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n"
        f"sources = {[str(narrowed_cpp), *(str(path) for path in source_files)]!r}\n"
        f"includes = {[str(path) for path in include_dirs]!r}\n"
        "setup(name='flash-attn-moba-narrow', ext_modules=[CUDAExtension(\n"
        "    name='flash_attn_2_cuda', sources=sources, include_dirs=includes,\n"
        "    extra_compile_args={'cxx': ['-O3', '-std=c++20'], 'nvcc': [\n"
        "        '-O3', '-std=c++20', '--use_fast_math',\n"
        "        '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__',\n"
        "        '-U__CUDA_NO_HALF2_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',\n"
        "    ]},\n"
        ")], cmdclass={'build_ext': BuildExtension.with_options(use_ninja=True)})\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["TORCH_CUDA_ARCH_LIST"] = "8.6"
    env["MAX_JOBS"] = "2"
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    cuda_lib = Path(sys.prefix) / f"lib/{python_version}/site-packages/nvidia/cu13/lib"
    if cuda_lib.is_dir():
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (str(cuda_lib), env.get("LD_LIBRARY_PATH", "")))
        )
        versioned_cudart = cuda_lib / "libcudart.so.13"
        if versioned_cudart.is_file():
            link_dir = output_root / "cuda-link"
            link_dir.mkdir(parents=True, exist_ok=True)
            cudart_link = link_dir / "libcudart.so"
            if not cudart_link.exists():
                cudart_link.symlink_to(versioned_cudart)
            env["LIBRARY_PATH"] = os.pathsep.join(
                filter(None, (str(link_dir), env.get("LIBRARY_PATH", "")))
            )
    subprocess.run(
        [
            sys.executable,
            str(setup_file),
            "build_ext",
            "--build-lib",
            str(output_root / "lib"),
            "--build-temp",
            str(build_root / "temp"),
            "-j",
            "2",
        ],
        cwd=build_root,
        env=env,
        check=True,
    )
    extension = next((output_root / "lib").glob("flash_attn_2_cuda*.so"))
    identity = {
        "revision": revision,
        "source_api_sha256": sha256(original_cpp),
        "narrowed_dispatch_sha256": sha256(narrowed_cpp),
        "kernel_source_sha256": {str(path.relative_to(source_root)): sha256(path) for path in source_files},
        "extension_sha256": sha256(extension),
        "extension": str(extension),
        "supported_contract": "BF16, head_dim=32, causal and noncausal, no split-KV",
    }
    (output_root / "build_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="clean pinned FlashAttention checkout")
    parser.add_argument("--output", type=Path, required=True, help="temporary output directory")
    parser.add_argument("--cuda-home", type=Path, required=True, help="CUDA 13 toolkit root")
    args = parser.parse_args()
    result = build(args.source.resolve(), args.output.resolve(), args.cuda_home.resolve())
    print(result)


if __name__ == "__main__":
    main()
