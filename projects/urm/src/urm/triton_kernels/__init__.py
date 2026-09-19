"""Triton kernels are loaded only when explicitly requested."""


def __getattr__(name):
    legacy = {
        "TritonDualFormSDMFunction": "TritonSparseDeltaFunction",
        "_triton_dual_form_bwd_kernel": "_sparse_delta_bwd_kernel",
        "_triton_dual_form_fwd_kernel": "_sparse_delta_fwd_kernel",
        "triton_dual_form_sdm": "triton_sparse_delta",
    }
    if name in legacy:
        from importlib import import_module

        return getattr(
            import_module("urm.experimental.triton_sparse_delta"), legacy[name]
        )
    raise AttributeError(name)
