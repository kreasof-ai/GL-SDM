"""Separate-pass CUDA spans including state allocation and host orchestration."""

from contextlib import contextmanager

EVENTS = []


@contextmanager
def state_stage(phase):
    import torch

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    with torch.autograd.profiler.record_function(
        f"pretraining::sparse_memory::native_state_{phase}"
    ):
        yield
    end.record()
    EVENTS.append((phase, start, end))
