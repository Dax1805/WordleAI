"""
Positional Letter Frequency (PLF).

Idea:
  Build per-position histograms from the CURRENT candidate set.
  Score each word by sum(counts[pos][word[pos]]) across positions.
  (Optional) penalize repeated letters slightly to keep coverage early.

Fast: O(|candidates|*N) to build + O(|pool|*N) to score.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import List
from .base import BaseSolver, register


@register
class PositionalFreqSolver(BaseSolver):
    id = "positional_freq"
    name = "Positional Letter Frequency"
    version = "1.0.0"

    # When candidates are large, evaluate guesses from allowed; else use candidates.
    CAND_POOL_LIMIT = 200
    DUPLICATE_PENALTY = 0.25  # subtract this per repeated letter instance beyond the first

    def _build_pos_counts(self, candidates: List[str]) -> List[Counter]:
        counts = [Counter() for _ in range(self.N)]
        for w in candidates:
            for i, ch in enumerate(w):
                counts[i][ch] += 1
        return counts

    def _score_word(self, w: str, pos_counts: List[Counter]) -> float:
        s = 0.0
        seen = set()
        for i, ch in enumerate(w):
            s += pos_counts[i][ch]
            if ch in seen:
                s -= self.DUPLICATE_PENALTY
            else:
                seen.add(ch)
        return s

    def next_guess(self, state: dict) -> str:
        candidates: List[str] = state["candidates"]
        allowed: List[str]    = state["allowed"]

        pool = candidates if len(candidates) <= self.CAND_POOL_LIMIT else allowed
        if not pool:
            return "a" * self.N

        pos_counts = self._build_pos_counts(candidates if candidates else allowed)

        best_score = None
        best: List[str] = []
        for w in pool:
            s = self._score_word(w, pos_counts)
            if best_score is None or s > best_score:
                best_score = s; best = [w]
            elif s == best_score:
                best.append(w)

        return best[self.rng.randrange(len(best))]