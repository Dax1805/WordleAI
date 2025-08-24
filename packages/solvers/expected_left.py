"""
Expected Remaining Candidates (ERC).

Idea:
  For guess g, if CURRENT candidates partition into buckets of sizes {c_i},
  the expected leftover after seeing the pattern is:
      E[left | g] = sum_i ( (c_i / n) * c_i ) = (1/n) * sum_i c_i^2
  Minimize sum_i c_i^2 (equivalently E[left]). Tie-break: smaller worst bucket.

Tracks entropy closely but simpler to compute/compare.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
from .base import BaseSolver, register
from packages.engine import score as score_fn


def _sum_c2_and_worst(guess: str, candidates: List[str]) -> Tuple[int, int]:
    buckets: Dict[str, int] = defaultdict(int)
    _score = score_fn
    for ans in candidates:
        buckets[_score(guess, ans)] += 1
    worst = max(buckets.values()) if buckets else 0
    sum_c2 = sum(c*c for c in buckets.values())
    return sum_c2, worst


@register
class ExpectedLeftSolver(BaseSolver):
    id = "expected_left"
    name = "Expected Remaining Candidates"
    version = "1.0.0"

    CANDIDATE_ONLY_LIMIT = 200
    POOL_CAP = 400

    def _distinct_letter_score(self, w: str, alpha: Dict[str, int]) -> int:
        seen = set(); s = 0
        for ch in w:
            if ch not in seen:
                seen.add(ch); s += alpha.get(ch, 0)
        return s

    def _select_pool(self, candidates: List[str], allowed: List[str]) -> List[str]:
        if len(candidates) <= self.CANDIDATE_ONLY_LIMIT:
            return candidates
        alpha: Dict[str, int] = {}
        for w in candidates:
            for ch in set(w):
                alpha[ch] = alpha.get(ch, 0) + 1
        ranked = sorted(allowed, key=lambda w: self._distinct_letter_score(w, alpha), reverse=True)
        return ranked[: self.POOL_CAP]

    def next_guess(self, state: dict) -> str:
        candidates: List[str] = state["candidates"]
        allowed: List[str]    = state["allowed"]

        pool = self._select_pool(candidates, allowed)
        if not pool:
            return "a" * self.N

        best_sum = None
        best_worst = None
        best: List[str] = []

        for g in pool:
            sum_c2, worst = _sum_c2_and_worst(g, candidates)
            if (best_sum is None) or (sum_c2 < best_sum) or (sum_c2 == best_sum and (best_worst is None or worst < best_worst)):
                best_sum, best_worst, best = sum_c2, worst, [g]
            elif sum_c2 == best_sum and worst == best_worst:
                best.append(g)

        return best[self.rng.randrange(len(best))]
