"""Graph partitioning into legal execution regions.

Owns the partitioning of a typed semantic graph into per-family regions before
backend selection. The K3 route→state composition is an ordinary typed graph
(``sparse_route_generation`` → ``sparse_state_mixer``), not a special plan; see
:mod:`urm.compiler.normalize.graph` and the graph compile path.
"""
