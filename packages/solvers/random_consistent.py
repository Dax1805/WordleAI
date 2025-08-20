from __future__ import annotations
from typing import List
from .base import BaseSolver, register

@register
class RandomConsistentSolver(BaseSolver):
    id = "random_consistent"
    name = "Random Consistent"
    version = "1.0.0"

    def next_guess(self, state: dict) -> str:
        """
        Pick any candidate uniformly at random (deterministic w.r.t seed).
        """
        candidates: List[str] = state["candidates"]
        # Fallback: if no candidates (shouldn’t happen if engine is correct), guess from allowed.
        pool = candidates if candidates else state["allowed"]
        i = self.rng.randrange(len(pool))
        return pool[i]
