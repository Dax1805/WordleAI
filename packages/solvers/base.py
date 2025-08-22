"""
Solver base class + registry.

A "solver" is a strategy module that decides the next guess given the
current game state. All solvers subclass `BaseSolver` and implement
`next_guess(state) -> str`.

We keep a simple in-process registry so the CLI (and tests) can
instantiate solvers by id without hardcoding imports everywhere.

Typical flow:
    from packages.solvers import create_solver
    solver = create_solver("entropy")  # returns an instance
    solver.reset(allowed=..., answers=..., N=5, seed=123)
    guess = solver.next_guess(state)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Type
import random

# Internal registry mapping solver id -> class
_SOLVERS: Dict[str, Type["BaseSolver"]] = {}


def register(cls: Type["BaseSolver"]) -> Type["BaseSolver"]:
    """
    Class decorator to register a solver by its `id` attribute.

    Usage:
        @register
        class MySolver(BaseSolver):
            id = "my_solver"
            ...
    """
    sid = getattr(cls, "id", None)
    if not sid:
        raise ValueError("Solver class must define a non-empty 'id' attribute.")
    _SOLVERS[sid] = cls
    return cls


def create_solver(solver_id: str) -> "BaseSolver":
    """
    Instantiate a solver by id. Raises KeyError if unknown.
    """
    if solver_id not in _SOLVERS:
        known = ", ".join(sorted(_SOLVERS.keys()))
        raise KeyError(f"Unknown solver id: {solver_id}. Known: [{known}]")
    return _SOLVERS[solver_id]()


def get_solver_ids() -> List[str]:
    """
    Return the list of registered solver ids (sorted for stable help text).
    """
    return sorted(_SOLVERS.keys())


class BaseSolver(ABC):
    """
    Abstract base class for all solvers.

    Required override:
        - next_guess(self, state) -> str

    Lifecycle:
        - reset(...) is called once per game to provide:
            * allowed: iterable of permitted guesses (strings)
            * answers: iterable of official answer words (strings)
            * N      : word length (int)
            * seed   : optional RNG seed (for deterministic tie-breaks)
        - next_guess(state) is called every turn with:
            state = {
                "turn": int,
                "history": list[(guess, pattern)],
                "candidates": list[str],   # remaining valid answers
                "allowed": list[str],      # valid guess universe (length N)
                "N": int,
                "rng": random.Random,      # same seeded RNG as in reset()
            }
    """

    # Metadata (override in subclasses)
    id: str = "base"
    name: str = "Base Solver"
    version: str = "0.0.0"

    # Provided at reset()
    allowed: List[str]
    answers: List[str]
    N: int
    rng: random.Random

    def reset(
            self,
            *,
            allowed: Iterable[str],
            answers: Iterable[str],
            N: int,
            seed: int | None = None,
    ) -> None:
        """
        Initialize per-game state shared by all solvers.

        Subclasses may override to precompute statistics, but should call
        super().reset(...) to keep behavior consistent.
        """
        self.allowed = list(allowed)
        self.answers = list(answers)
        self.N = int(N)
        self.rng = random.Random(seed)

    @abstractmethod
    def next_guess(self, state: dict) -> str:
        """
        Decide the next guess given the current game state.

        Must return a lowercased string of length N. The harness does not
        enforce the guess being in `allowed` here—solvers are expected
        to respect that contract themselves (our harness *provides* the
        already-length-filtered `state["allowed"]` for convenience).
        """
        ...
