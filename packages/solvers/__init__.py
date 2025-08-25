from __future__ import annotations
from typing import List
from .base import BaseSolver, REGISTRY, register

from . import random_consistent  # noqa: F401
from . import letter_freq  # noqa: F401
from . import positional_freq  # noqa: F401
from . import expected_left  # noqa: F401
from . import max_patterns  # noqa: F401
from . import entropy  # noqa: F401
from . import entropy_weighted  # noqa: F401
from . import two_stage_probe  # noqa: F401
from . import two_ply_mc  # noqa: F401


def create_solver(solver_id: str) -> BaseSolver:
    """
    Factory: instantiate a registered solver by id.
    """
    try:
        cls = REGISTRY[solver_id]
    except KeyError as e:
        raise ValueError(
            f"Unknown solver id: {solver_id}. Available: {sorted(REGISTRY.keys())}") from e
    return cls()


def get_solver_ids() -> List[str]:
    """
    Return all registered solver ids (sorted for stable CLI help).
    """
    return sorted(REGISTRY.keys())
