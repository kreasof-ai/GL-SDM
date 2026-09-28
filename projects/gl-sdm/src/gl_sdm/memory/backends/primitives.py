"""Shared CUDA operand checks and row gather."""
import torch
import triton
import triton.language as tl


def require_cuda(values):
    if not values.is_cuda or not values.is_contiguous():
        raise ValueError("GL-SDM native kernels require contiguous CUDA tensors")
    if torch.cuda.get_device_capability(values.device) < (8, 0):
        raise ValueError("GL-SDM native kernels require SM80 or newer")


@triton.jit
def _gather(values, indices, output, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BD)
    idx = tl.load(indices + row)
    value = tl.load(values + idx * D + d, d < D, other=0.0)
    tl.store(output + row * D + d, value, d < D)
