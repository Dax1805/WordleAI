"""
Random Consistent solver.

Strategy:
  - Choose uniformly at random from the CURRENT candidate set (words still
    consistent with all feedback so far).
  - If (unexpectedly) the candidate set is empty, fall back to the allowed list.

Notes:
  - Deterministic across runs with the same seed (via BaseSolver.rng).
  - This is a baseline to verify the pipeline; it does not try to maximize
    information gain or positional coverage.
"""

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
        Pick any candidate uniformly at random (seeded RNG).

        Args:
            state: dict with keys:
                - "candidates": current consistent answer set (List[str])
                - "allowed":    valid guess universe (List[str], already length N)
                - "rng":        random.Random initialized in reset()

        Returns:
            A single lowercase guess string of length N.
        """
        candidates: List[str] = state["candidates"]
        allowed: List[str] = state["allowed"]

        # Primary pool = candidates; fallback to allowed if somehow empty.
        pool: List[str] = candidates if candidates else allowed

        # Final safeguard: if both are empty (shouldn't happen), return a stub
        # to avoid IndexError; harness will score and quickly fail.
        if not pool:
            return "a" * self.N  # degenerate but safe

        i = self.rng.randrange(len(pool))
        return pool[i]
