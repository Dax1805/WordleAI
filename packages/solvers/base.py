from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Iterable
import random

# Simple registry so CLI can instantiate by id
_SOLVERS: Dict[str, type] = {}


def register(cls):
    if not getattr(cls, "id", None):
        raise ValueError("Solver class must define an 'id' attribute.")
    _SOLVERS[cls.id] = cls
    return cls


def create_solver(solver_id: str):
    if solver_id not in _SOLVERS:
        raise KeyError(f"Unknown solver id: {solver_id}. Known: {sorted(_SOLVERS.keys())}")
    return _SOLVERS[solver_id]()


def get_solver_ids() -> List[str]:
    return sorted(_SOLVERS.keys())


class BaseSolver(ABC):
    """
    All solvers implement next_guess(state) -> str.
    State keys: turn, history [(guess, pattern)], candidates, allowed, N, rng
    """
    id: str = "base"
    name: str = "Base Solver"
    version: str = "0.0.0"

    def reset(self, *, allowed: Iterable[str], answers: Iterable[str], N: int,
              seed: int | None = None):
        self.allowed = list(allowed)
        self.answers = list(answers)
        self.N = N
        self.rng = random.Random(seed)

    @abstractmethod
    def next_guess(self, state: dict) -> str:
        ...
