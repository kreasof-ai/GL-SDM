"""Native K1 Triton kernels: normalized and routed reductions.

Contains the GPU kernel bodies, autograd/backward implementations, and narrow
launchers. Candidate choice, schedule, and capability checks belong to the
compiler.
"""
