"""Native K2 Triton kernels: canonical matrix/diagonal state transitions.

Contains the recurrence/scan kernel bodies with narrow launchers. Schedule
choice belongs to the compiler. Source-named single-architecture kernels are
not core backend content; they live with their comparators until they satisfy
the backend branch admission rule.
"""
