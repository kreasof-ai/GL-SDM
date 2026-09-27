"""Backend packages: the single Provider contract plus one directory per backend.

Every backend — reference NumPy oracle, reference Torch, native Triton — lives
in its own directory (``backends/<name>/``) and implements each family it
supports in ``backends/<name>/<family>.py`` behind the uniform
:class:`~urm.backends.contract.Provider` surface. The dispatch table is built by
filesystem auto-discovery (:mod:`urm.backends.registry`); adding a backend is
creating its directory. The compiler owns candidate choice, cost and schedule
decisions; route/operand certification lives in :mod:`urm.runtime.certification`.
"""

from __future__ import annotations

__all__: list[str] = []
