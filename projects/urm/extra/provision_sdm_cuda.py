"""Install an isolated matching CUDA 13 toolkit and build unmodified SDM kernels."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

PACKAGES = ('nvidia-cuda-nvcc==13.0.88', 'nvidia-cuda-runtime==13.0.96',
            'nvidia-cuda-crt==13.0.88', 'nvidia-nvvm==13.0.88', 'nvidia-cuda-cccl==13.0.85')


def main():
    project = Path(__file__).resolve().parents[1]
    for path in (project, project / 'src'):
        sys.path.insert(0, str(path))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path.home() / '.cache/urm/sdm-cuda13')
    args = parser.parse_args()
    destination = args.destination.expanduser().resolve()
    root = destination / 'nvidia/cu13'
    if not (root / 'bin/nvcc').is_file() or not (root / 'include/nv/target').exists():
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--target', str(destination),
                        *PACKAGES], check=True)
    for name, target in [('lib/libcudart.so', 'libcudart.so.13'), ('lib64', 'lib')]:
        path = root / name
        if not path.exists():
            path.symlink_to(target)
    os.environ['URM_SDM_CUDA_HOME'] = str(root)
    from extra.comparators.sdm.cuda import load_pinned_sdm, sdm_cuda_identity
    load_pinned_sdm()
    from lingua.sparse_delta_memory.cuda import sparse_ip_cuda, warp_cooperative_gather_cuda
    print('Building the clean pinned SDM sparse-IP CUDA extension', flush=True)
    sparse_ip_cuda._get_cuda_module()
    print('Building the clean pinned SDM gather CUDA extension', flush=True)
    warp_cooperative_gather_cuda._get_cuda_module()
    print(sdm_cuda_identity(), flush=True)


if __name__ == '__main__':
    main()
