"""K1 family — Torch reference ops over the full source history (score/select/reduce).

One file per op. The family re-exports the combined provider tuple (the differentiable
reference and the trusted SDPA library provider) so the registry finds them.
"""

from .softmax import K1SdpaLibraryProvider, K1TorchReferenceProvider

PROVIDERS = (K1TorchReferenceProvider(), K1SdpaLibraryProvider())

__all__ = ["PROVIDERS", "K1TorchReferenceProvider", "K1SdpaLibraryProvider"]
