"""Provider decline contract for the K1 score/reducer/indexed laws (A13/A2).

A provider must decline a descriptor it cannot execute — silently running the
wrong law is a correctness hole, not a graceful fallback. The Triton native
online-softmax kernel and the SDPA library path execute only the plain
dot-product score with a softmax reducer; they must decline the channel-decay
score law, the threshold/squared-sum reducers, and the indexed path. The NumPy
oracle and Torch reference executors dispatch on every current law, so they
accept all of these.

Regression coverage for the hole where ``K1NativeTritonProvider.decline`` and
``K1SdpaLibraryProvider.decline`` accepted any descriptor, then ran online
softmax regardless.
"""

from __future__ import annotations

import pytest

from urm.backends.contract import ProviderRequest
from urm.backends.numpy.k1 import K1NumpyProvider
from urm.backends.torch.k1 import K1SdpaLibraryProvider, K1TorchReferenceProvider
from urm.backends.triton.k1 import K1NativeTritonProvider
from urm.ir.program import (
    K1Descriptor,
    K1HeadMap,
    K1ReducerLaw,
    K1ScoreLaw,
)


def _request(descriptor: K1Descriptor) -> ProviderRequest:
    return ProviderRequest(
        family="k1", descriptor=descriptor, mode="inference", accumulation_dtype="float32"
    )


DOT_SOFTMAX = K1Descriptor()  # defaults: DOT score, SOFTMAX reducer, non-indexed
CHANNEL_DECAY = K1Descriptor(score_law=K1ScoreLaw.CHANNEL_DECAY)
THRESHOLD = K1Descriptor(
    reducer_law=K1ReducerLaw.THRESHOLD_RELU_POWER, threshold_beta=0.5, relu_power=2.0
)
SQUARED_SUM = K1Descriptor(reducer_law=K1ReducerLaw.SQUARED_SUM, squared_sum_groups=2)
INDEXED = K1Descriptor(indexed=True)
GQA = K1Descriptor(head_map=K1HeadMap.GROUPED, group_size=2)


# --- native Triton: declines every non-DOT-softmax / indexed descriptor ---

@pytest.mark.parametrize(
    "descriptor,reason",
    [
        (CHANNEL_DECAY, "DOT score law"),
        (THRESHOLD, "SOFTMAX reducer law"),
        (SQUARED_SUM, "SOFTMAX reducer law"),
        (INDEXED, "indexed"),
    ],
)
def test_triton_declines_unsupported_laws(descriptor, reason):
    decline = K1NativeTritonProvider().decline(_request(descriptor))
    assert decline is not None, "native K1 must decline a descriptor it cannot execute"
    assert reason in decline


def test_triton_accepts_plain_softmax_shape():
    # DOT + SOFTMAX + non-indexed passes the law checks; on a CUDA-less host it
    # may still decline for CUDA, but it must NOT decline for a law reason.
    decline = K1NativeTritonProvider().decline(_request(DOT_SOFTMAX))
    assert decline is None or "CUDA" in decline


def test_triton_accepts_gqa_shape():
    # Grouped head map is supported by the online-softmax kernel (GQA expand).
    decline = K1NativeTritonProvider().decline(_request(GQA))
    assert decline is None or "CUDA" in decline


# --- SDPA library: declines every non-DOT-softmax / indexed descriptor ---

@pytest.mark.parametrize(
    "descriptor,reason",
    [
        (CHANNEL_DECAY, "DOT score law"),
        (THRESHOLD, "SOFTMAX reducer law"),
        (SQUARED_SUM, "SOFTMAX reducer law"),
        (INDEXED, "indexed"),
    ],
)
def test_sdpa_declines_unsupported_laws(descriptor, reason):
    decline = K1SdpaLibraryProvider().decline(_request(descriptor))
    assert decline is not None, "SDPA must decline a descriptor it cannot execute"
    assert reason in decline


def test_sdpa_accepts_plain_softmax():
    assert K1SdpaLibraryProvider().decline(_request(DOT_SOFTMAX)) is None


def test_sdpa_accepts_gqa():
    assert K1SdpaLibraryProvider().decline(_request(GQA)) is None


# --- reference tiers ---

# The Torch reference executor dispatches on every current law, so it accepts all.
@pytest.mark.parametrize(
    "descriptor", [DOT_SOFTMAX, CHANNEL_DECAY, THRESHOLD, SQUARED_SUM, INDEXED, GQA]
)
def test_torch_reference_accepts_all_current_laws(descriptor):
    assert K1TorchReferenceProvider().decline(_request(descriptor)) is None


# The NumPy oracle implements only DOT + SOFTMAX (non-indexed); it declines the rest
# rather than accept and compute the wrong equation.
@pytest.mark.parametrize(
    "descriptor,reason",
    [
        (CHANNEL_DECAY, "DOT score law"),
        (THRESHOLD, "SOFTMAX reducer law"),
        (SQUARED_SUM, "SOFTMAX reducer law"),
        (INDEXED, "indexed"),
    ],
)
def test_numpy_oracle_declines_unsupported_laws(descriptor, reason):
    decline = K1NumpyProvider().decline(_request(descriptor))
    assert decline is not None
    assert reason in decline


@pytest.mark.parametrize("descriptor", [DOT_SOFTMAX, GQA])
def test_numpy_oracle_accepts_dot_softmax(descriptor):
    assert K1NumpyProvider().decline(_request(descriptor)) is None
