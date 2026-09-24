"""Independent reference backend families (NumPy and PyTorch).

These implement the K1/K2/K3 equations independently of the native Triton
kernels and reject semantics they do not represent. The float64 NumPy family is
the correctness oracle; the PyTorch family is the differentiable reference.
"""
