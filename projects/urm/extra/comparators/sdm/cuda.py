"""Actual pinned Meta SDM CUDA/Triton baseline; no reference substitution."""
from __future__ import annotations

import inspect
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from .upstream import EXPECTED_SDM_COMMIT, SDM_REPOSITORY


def configure_toolchain():
    root = Path(os.environ.get('URM_SDM_CUDA_HOME',
                Path.home() / '.cache/urm/sdm-cuda13/nvidia/cu13')).resolve()
    if not (root / 'bin/nvcc').is_file() or not (root / 'include/nv/target').exists():
        raise RuntimeError('SDM CUDA toolchain is missing; run python extra/provision_sdm_cuda.py')
    import torch.utils.cpp_extension as extension
    extension.CUDA_HOME = str(root)
    os.environ['CUDA_HOME'] = str(root)
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(Path.home() / '.cache/urm/sdm-extensions'))
    os.environ.setdefault('MAX_JOBS', '2')
    return root


def load_pinned_sdm():
    checkout = Path(os.environ.get('URM_SDM_UPSTREAM_ROOT', '/tmp/urm-comparator-pins/sdm')).resolve()
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True).strip()
    if revision != EXPECTED_SDM_COMMIT or dirty:
        raise RuntimeError('SDM production baseline requires the clean pinned upstream checkout')
    configure_toolchain()
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from lingua.sparse_delta_memory.memory_ops import GatedSparseMemoryWriteRead
    source = Path(inspect.getfile(GatedSparseMemoryWriteRead)).resolve()
    if not source.is_relative_to(checkout):
        raise RuntimeError('Loaded SDM implementation is outside the verified checkout')
    return GatedSparseMemoryWriteRead


def sdm_cuda_identity():
    kernel = load_pinned_sdm()
    root = configure_toolchain()
    from lingua.sparse_delta_memory.cuda import sparse_ip_cuda, warp_cooperative_gather_cuda
    libraries = {}
    for name, loader in (("sparse_ip", sparse_ip_cuda),
                         ("warp_cooperative_gather", warp_cooperative_gather_cuda)):
        path = Path(loader._get_cuda_module().__file__).resolve()
        libraries[name] = {"path": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return dict(repository=SDM_REPOSITORY, revision=EXPECTED_SDM_COMMIT,
                source_file=inspect.getfile(kernel), cuda_home=str(root),
                cuda_extensions=libraries, reference_fallback=False,
                nvcc=subprocess.check_output([str(root / 'bin/nvcc'), '--version'], text=True).strip())


def pinned_cuda_write_read(memory, read_indices, read_weights, *, write_indices,
                           write_weights, values, beta, log_decay, chunk_size=256,
                           grad_final_memory=None, kernel=None):
    """Partition-safe launch of the original CUDA-backed WY implementation.

    Pad each partition independently: padding only the flattened sequence would
    mix heads at chunk boundaries. Padding has zero write/read weights and zero
    decay/gates, so it is an identity state transition. A snapshot is cloned
    because upstream mutates its memory argument. Terminal-state cotangents use
    the pinned kernel's explicit grad_final_memory API.
    """
    kernel = kernel or load_pinned_sdm()
    p, t, d = values.shape
    slots = memory.shape[1]
    pad = (-t) % chunk_size
    padded_t = t + pad
    offsets = torch.arange(p, device=memory.device).view(p, 1, 1) * slots

    def indices(x):
        return (F.pad(x.long(), (0, 0, 0, pad)) + offsets).reshape(p * padded_t, -1).contiguous()

    def operand(x):
        return F.pad(x, (0, 0, 0, pad)).reshape(p * padded_t, -1).contiguous()

    flat_memory = memory.clone().reshape(p * slots, d)
    # The pinned Function has no custom_fwd autocast guard. Its explicitly FP32
    # WY matmuls must remain FP32 before their fused Triton consumers.
    with torch.autocast(memory.device.type, enabled=False):
        output, _ = kernel.apply(flat_memory,
            indices(write_indices), operand(write_weights), operand(values),
            operand(beta), operand(log_decay), indices(read_indices), operand(read_weights),
            chunk_size, True, slots, p, False, 'none',
            None if grad_final_memory is None else grad_final_memory.reshape(p * slots, d).contiguous())
    # The pin returns an empty placeholder in its second output; the state is
    # its mutated memory argument. Terminal gradients use the explicit API.
    # Backward reconstructs earlier memory in this same workspace. Preserve the
    # terminal snapshot now, before that reconstruction overwrites it.
    return output.reshape(p, padded_t, d)[:, :t], flat_memory.detach().clone().reshape(p, slots, d)


__all__ = ['pinned_cuda_write_read', 'load_pinned_sdm', 'sdm_cuda_identity']
