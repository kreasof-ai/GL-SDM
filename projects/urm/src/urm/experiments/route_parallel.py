"""Explicit CUDA route-parallel experiments, NOT a production backend.

CUDA custom ops make actual fullgraph forward/backward paths testable. Compilation,
pointer validation and optional audits are separated from measurement. Certified
routes must be unique within each token, contiguous and width 64; D stays 4.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import torch

BUILD_FLAGS = [
    "-O3",
    "-std=c++17",
    "--shared",
    "-Xcompiler=-fPIC",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_86,code=compute_86",
    "--fmad=false",
    "-lineinfo",
    "--ptxas-options=-v",
]
AUDIT: list[dict] | None = None


@functools.lru_cache(None)
def library():
    source = Path(__file__).with_suffix(".cu")
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("isolated CUDA experiment requires nvcc")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    digest = hashlib.sha256(
        source.read_bytes() + repr(BUILD_FLAGS).encode() + version.encode()
    ).hexdigest()
    root = (
        Path(os.environ.get("URM_EXPERIMENT_CACHE", "/tmp/urm-route-parallel-build"))
        / digest
    )
    root.mkdir(parents=True, exist_ok=True)
    binary = root / "route_parallel.so"
    if not binary.exists():
        command = [nvcc, *BUILD_FLAGS, str(source), "-o", str(binary)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        (root / "build.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
    lib = ctypes.CDLL(str(binary))
    lib.launch.argtypes = (
        [ctypes.c_int] * 5
        + [ctypes.POINTER(ctypes.c_void_p)]
        + [ctypes.c_int] * 4
        + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    )
    lib.launch.restype = ctypes.c_int
    lib.set_audit.argtypes = [ctypes.c_int]
    lib.get_launch.argtypes = [ctypes.POINTER(ctypes.c_int)]
    return lib, {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "binary": str(binary),
        "build_log": str(root / "build.log"),
        "nvcc": version,
        "flags": BUILD_FLAGS,
    }


def schedule(memory, indices, resident, before, backward=False):
    return {
        "backward": int(backward),
        "grid": [memory.shape[0], memory.shape[2] // 4],
        "threads": 256,
        "dynamic_shared_bytes": memory.shape[1] * 4 * memory.element_size()
        if resident
        else 0,
        "sequence": indices.shape[1],
        "slots": memory.shape[1],
        "value_dim": memory.shape[2],
        "block_d": 4,
        "write_width": 64,
        "read_width": 64,
        "resident": resident,
        "before": before,
    }


def _call(tensors, memory, indices, resident, before, backward=False, resources=None):
    lib, _ = library()
    ptrs = (ctypes.c_void_p * len(tensors))(*(x.data_ptr() for x in tensors))
    status = lib.launch(
        int(backward),
        int(resident),
        int(before),
        int(memory.dtype == torch.bfloat16),
        int(indices.dtype == torch.int64),
        ptrs,
        memory.shape[0],
        indices.shape[1],
        memory.shape[1],
        memory.shape[2],
        torch.cuda.current_stream(memory.device).cuda_stream,
        resources,
    )
    if status:
        raise RuntimeError(f"CUDA route-parallel launch failed: {status}")
    if AUDIT is not None and resources is None:
        observed = (ctypes.c_int * 12)()
        lib.get_launch(observed)
        plan = schedule(memory, indices, resident, before, backward)
        expected = [
            int(backward),
            *plan["grid"],
            256,
            plan["dynamic_shared_bytes"],
            plan["sequence"],
            plan["slots"],
            plan["value_dim"],
            int(resident),
            int(before),
            memory.element_size(),
            indices.element_size(),
        ]
        if list(observed) != expected:
            raise RuntimeError(
                f"actual CUDA launch differs from plan: {list(observed)} != {expected}"
            )
        AUDIT.append({"plan": plan, "observed": list(observed), "passed": True})


def resource_report(memory, indices, resident, before, backward):
    output = (ctypes.c_int * 6)()
    _call([], memory, indices, resident, before, backward, resources=output)
    names = (
        "registers_per_thread",
        "local_bytes_per_thread",
        "static_shared_bytes",
        "dynamic_shared_bytes",
        "max_active_blocks_per_sm",
        "binary_sm",
    )
    report = dict(zip(names, output, strict=True))
    report["theoretical_active_warps_per_sm"] = output[4] * 8
    # GA102 (A10G): 48 resident warps per SM. This is a resource ceiling, not achieved occupancy.
    report["theoretical_occupancy_fraction_sm86"] = output[4] * 8 / 48
    return report


def enable_audit(enabled):
    global AUDIT
    AUDIT = [] if enabled else None
    library()[0].set_audit(int(enabled))


def _check(memory, wi, ww, v, b, g, ri, rw):
    p, s, d = memory.shape
    t = wi.shape[1]
    if d % 4 or s > 4096 or wi.shape != (p, t, 64) or ri.shape != wi.shape or t < 1:
        raise ValueError(
            "experiment requires D divisible by 4, S<=4096, positive T, 64/64 routes"
        )
    if (
        memory.dtype not in (torch.float32, torch.bfloat16)
        or wi.dtype not in (torch.int32, torch.int64)
        or ri.dtype != wi.dtype
    ):
        raise ValueError("unsupported experimental dtype")
    for x, shape in zip(
        (ww, v, b, g, rw),
        ((p, t, 64), (p, t, d), (p, t, 1), (p, t, 1), (p, t, 64)),
        strict=True,
    ):
        if x.shape != shape or x.dtype != memory.dtype:
            raise ValueError("operand shape/dtype mismatch")
    if not all(
        x.is_cuda and x.is_contiguous() and x.device == memory.device
        for x in (memory, wi, ww, v, b, g, ri, rw)
    ):
        raise ValueError(
            "all experimental operands must be contiguous on one CUDA device"
        )


@torch.library.custom_op("urm_experiment::route_forward", mutates_args=())
def forward(
    memory: torch.Tensor,
    wi: torch.Tensor,
    ww: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    ri: torch.Tensor,
    rw: torch.Tensor,
    resident: bool,
    before: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _check(memory, wi, ww, v, b, g, ri, rw)
    final = torch.empty_like(memory) if resident else memory.clone()
    y = torch.empty_like(v)
    wh = torch.empty(
        (*wi.shape, memory.shape[-1]), device=memory.device, dtype=memory.dtype
    )
    rh = torch.empty_like(wh)
    _call(
        [memory, final, wi, ww, v, b, g, ri, rw, y, wh, rh],
        memory,
        wi,
        resident,
        before,
    )
    return y, final, wh, rh


@forward.register_fake
def _fake_forward(memory, wi, ww, v, b, g, ri, rw, resident, before):
    wh = memory.new_empty((*wi.shape, memory.shape[-1]))
    return torch.empty_like(v), torch.empty_like(memory), wh, torch.empty_like(wh)


@torch.library.custom_op("urm_experiment::route_backward", mutates_args=())
def backward(
    dy: torch.Tensor,
    df: torch.Tensor,
    wh: torch.Tensor,
    rh: torch.Tensor,
    wi: torch.Tensor,
    ww: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    ri: torch.Tensor,
    rw: torch.Tensor,
    resident: bool,
    before: bool,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    dy, df = dy.contiguous(), df.contiguous()
    dm = torch.empty_like(df) if resident else df.clone()
    dw, db, dg, dq = [torch.zeros_like(x, dtype=torch.float32) for x in (ww, b, g, rw)]
    dv = torch.empty_like(v, dtype=torch.float32)
    _call(
        [df, dm, wh, rh, wi, ww, v, b, g, ri, rw, dy, dw, dv, db, dg, dq],
        df,
        wi,
        resident,
        before,
        True,
    )
    return (
        dm,
        dw.to(ww.dtype),
        dv.to(v.dtype),
        db.to(b.dtype),
        dg.to(g.dtype),
        dq.to(rw.dtype),
    )


@backward.register_fake
def _fake_backward(dy, df, wh, rh, wi, ww, v, b, g, ri, rw, resident, before):
    return tuple(torch.empty_like(x) for x in (df, ww, v, b, g, rw))


def _setup(ctx, inputs, output):
    memory, wi, ww, v, b, g, ri, rw, resident, before = inputs
    _, _, wh, rh = output
    ctx.save_for_backward(wh, rh, wi, ww, v, b, g, ri, rw)
    ctx.resident, ctx.before = resident, before
    ctx.shape = memory.shape
    ctx.mark_non_differentiable(wh, rh)


def _vjp(ctx, dy, df, _dwh, _drh):
    wh, rh, wi, ww, v, b, g, ri, rw = ctx.saved_tensors
    if dy is None:
        dy = torch.zeros_like(v)
    if df is None:
        df = v.new_zeros(ctx.shape)
    dm, dw, dv, db, dg, dq = backward(
        dy, df, wh, rh, wi, ww, v, b, g, ri, rw, ctx.resident, ctx.before
    )
    return dm, None, dw, dv, db, dg, None, dq, None, None


forward.register_autograd(_vjp, setup_context=_setup)


def update(memory, wi, ww, v, b, g, ri, rw, *, resident, before=False):
    """Functional training experiment. Never redirects production in-place decode."""
    y, final, _, _ = forward(memory, wi, ww, v, b, g, ri, rw, resident, before)
    return y, final
