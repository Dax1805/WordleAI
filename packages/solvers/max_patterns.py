"""
Max Pattern Diversity (MPD).

Idea:
  For each guess g, count how many DISTINCT feedback patterns it produces
  against the CURRENT candidates. Pick the guess with the MOST unique patterns.
  Tie-break: smaller worst bucket, then RNG.

Cheaper than entropy (no logs or probabilities), but still buckets candidates.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
from .base import BaseSolver, register
from packages.engine import score as score_fn


def _pattern_stats(guess: str, candidates: List[str]) -> Tuple[int, int]:
    """
    Return (num_distinct_patterns, worst_bucket_size) for guess.
    """
    buckets: Dict[str, int] = defaultdict(int)
    _score = score_fn
    for ans in candidates:
        buckets[_score(guess, ans)] += 1
    if not buckets:
        return 0, 0
    worst = max(buckets.values())
    return len(buckets), worst


@register
class MaxPatternsSolver(BaseSolver):
    id = "max_patterns"
    name = "Max Pattern Diversity"
    version = "1.0.0"

    CANDIDATE_ONLY_LIMIT = 200
    POOL_CAP = 400  # cap pool when using allowed (pre-filtered by a quick heuristic)

    def _distinct_letter_score(self, w: str, alphabet_counts: Dict[str, int]) -> int:
        seen = set(); s = 0
        for ch in w:
            if ch not in seen:
                seen.add(ch); s += alphabet_counts.get(ch, 0)
        return s

    def _select_pool(self, candidates: List[str], allowed: List[str]) -> List[str]:
        if len(candidates) <= self.CANDIDATE_ONLY_LIMIT:
            return candidates
        # quick prefilter by distinct-letter coverage vs candidates (like letter_freq)
        alphabet_counts: Dict[str, int] = {}
        for w in candidates:
            for ch in set(w):
                alphabet_counts[ch] = alphabet_counts.get(ch, 0) + 1
        ranked = sorted(allowed, key=lambda w: self._distinct_letter_score(w, alphabet_counts), reverse=True)
        return ranked[: self.POOL_CAP]

    def next_guess(self, state: dict) -> str:
        candidates: List[str] = state["candidates"]
        allowed: List[str]    = state["allowed"]

        pool = self._select_pool(candidates, allowed)
        if not pool:
            return "a" * self.N

        best_m = None
        best_worst = None
        best: List[str] = []

        for g in pool:
            m, worst = _pattern_stats(g, candidates)
            if (best_m is None) or (m > best_m) or (m == best_m and (best_worst is None or worst < best_worst)):
                best_m, best_worst, best = m, worst, [g]
            elif m == best_m and worst == best_worst:
                best.append(g)

        return best[self.rng.randrange(len(best))]
