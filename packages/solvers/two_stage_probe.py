"""
Two-Stage Coverage (dynamic).

Idea:
  Turn 1: play a high-coverage "probe" word picked dynamically (no hardcoding).
  Turn 2: IF the space is still large, play a second probe biased toward NEW letters.
  Else / afterwards: switch to positional frequency (PLF) for precision.

Cheap, improves worst-case tails; no fixed word lists required.
"""

from __future__ import annotations
from collections import Counter
from typing import List, Dict, Set
from .base import BaseSolver, register


@register
class TwoStageProbeSolver(BaseSolver):
    id = "two_stage_probe"
    name = "Two-Stage Coverage"
    version = "1.0.0"

    LARGE_THRESHOLD = 1200  # if |candidates| > this after turn 1, fire second probe
    CAND_POOL_LIMIT = 200

    def _alphabet_counts(self, words: List[str]) -> Dict[str, int]:
        c = Counter()
        for w in words:
            for ch in set(w):
                c[ch] += 1
        return c

    def _coverage_score(self, w: str, counts: Dict[str, int], banned: Set[str] = None) -> int:
        banned = banned or set()
        seen = set(); s = 0
        for ch in w:
            if ch in banned:
                continue
            if ch not in seen:
                seen.add(ch); s += counts.get(ch, 0)
        return s

    def _pick_probe(self, pool: List[str], counts: Dict[str, int], banned: Set[str] = None) -> str:
        best_s = None; best: List[str] = []
        for w in pool:
            s = self._coverage_score(w, counts, banned)
            if best_s is None or s > best_s:
                best_s = s; best = [w]
            elif s == best_s:
                best.append(w)
        return best[self.rng.randrange(len(best))] if best else ("a" * self.N)

    def _plf_pick(self, candidates: List[str], allowed: List[str]) -> str:
        # minimal PLF for the decision step (reuse logic locally to avoid import)
        pos_counts = [Counter() for _ in range(self.N)]
        for w in candidates:
            for i, ch in enumerate(w):
                pos_counts[i][ch] += 1
        pool = candidates if len(candidates) <= self.CAND_POOL_LIMIT else allowed
        best_s = None; best: List[str] = []
        for w in pool:
            s = 0
            seen = set()
            for i, ch in enumerate(w):
                s += pos_counts[i][ch]
                if ch in seen:
                    s -= 0.25
                else:
                    seen.add(ch)
            if best_s is None or s > best_s:
                best_s = s; best = [w]
            elif s == best_s:
                best.append(w)
        return best[self.rng.randrange(len(best))] if best else ("a" * self.N)

    def next_guess(self, state: dict) -> str:
        turn = state["turn"]
        candidates: List[str] = state["candidates"]
        allowed: List[str]    = state["allowed"]

        if turn == 1:
            counts = self._alphabet_counts(candidates if candidates else allowed)
            return self._pick_probe(allowed, counts)

        if turn == 2 and len(candidates) > self.LARGE_THRESHOLD:
            # pick second probe emphasizing NEW letters vs guess_1
            first_guess = state["history"][0][0].lower()
            banned = set(first_guess)  # prefer disjoint letter set
            counts = self._alphabet_counts(candidates if candidates else allowed)
            return self._pick_probe(allowed, counts, banned=banned)

        # afterwards: use positional frequency
        return self._plf_pick(candidates, allowed)
