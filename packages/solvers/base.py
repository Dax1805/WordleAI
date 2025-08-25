from __future__ import annotations
import random
from typing import Dict, List, Type

# ---- Global solver registry ----
REGISTRY: Dict[str, Type["BaseSolver"]] = {}


def register(cls: Type["BaseSolver"]) -> Type["BaseSolver"]:
    """
    Decorator: @register on a solver class adds it to REGISTRY by its `id`.
    """
    sid = getattr(cls, "id", None)
    if not sid:
        raise ValueError(f"{cls.__name__} must define a non-empty `id`")
    if sid in REGISTRY:
        raise ValueError(f"Duplicate solver id: {sid}")
    REGISTRY[sid] = cls
    return cls


# ---- Base class that solvers inherit ----
class BaseSolver:
    id = "base"
    name = "Base"
    version = "0.0.0"

    def __init__(self):
        self.N: int = 5
        self.allowed: List[str] = []
        self.answers: List[str] = []
        self.rng = random.Random()

    def reset(self, *, allowed: List[str], answers: List[str], N: int,
              seed: int | None = None) -> None:
        self.allowed = list(allowed)
        self.answers = list(answers)
        self.N = int(N)
        if seed is not None:
            self.rng.seed(seed)

    def next_guess(self, state: dict) -> str:
        raise NotImplementedError("Override in subclass")
