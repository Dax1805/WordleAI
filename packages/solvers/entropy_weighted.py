"""
Weighted Entropy.

Idea:
  Like entropy, but bucket probabilities use PRIOR WEIGHTS w(a) for answers:
      p(bucket) = sum_{a in bucket} w(a) / sum_{a in candidates} w(a)
  This makes the solver robust when the answers list is imperfect or when
  running "open-world" (hypothesis = allowed).

We build a simple prior from global letter frequencies over ALLOWED:
  w(word) = sum of distinct-letter frequencies (positive, fast to compute).
You can swap this for a corpus/LM prior later.
"""

from __future__ import annotations
from collections import defaultdict, Counter
from math import log2
from typing import Dict, List, Tuple
from .base import BaseSolver, register
from packages.engine import score as score_fn


def _weighted_entropy(guess: str, candidates: List[str], weight: Dict[str, float]) -> Tuple[float, float, int]:
    """
    Return (entropy_bits, total_weight, worst_bucket_size) for guess.
    """
    buckets_w: Dict[str, float] = defaultdict(float)
    buckets_c: Dict[str, int]   = defaultdict(int)

    total_w = 0.0
    _score = score_fn
    for ans in candidates:
        patt = _score(guess, ans)
        w = weight.get(ans, 1.0)
        buckets_w[patt] += w
        buckets_c[patt] += 1
        total_w += w

    if total_w <= 0:
        return 0.0, 0.0, 0

    H = 0.0
    worst = 0
    for patt, w in buckets_w.items():
        p = w / total_w
        H += p * log2(1 / p)
        if buckets_c[patt] > worst:
            worst = buckets_c[patt]
    return H, total_w, worst

@register
class WeightedEntropySolver(BaseSolver):
    id = "entropy_weighted"
    name = "Entropy (Weighted)"
    version = "1.0.0"

    CANDIDATE_ONLY_LIMIT = 200
    POOL_CAP = 400

    def reset(self, *, allowed, answers, N, seed=None) -> None:
        super().reset(allowed=allowed, answers=answers, N=N, seed=seed)
        # Global counts over allowed to define a prior
        self._global_counts = Counter("".join(self.allowed)) if self.allowed else Counter()

        # Precompute prior weights for all allowed words (dict lookup is fast)
        self._prior_weight: Dict[str, float] = {}
        for w in self.allowed:
            s = 0.0; seen = set()
            for ch in w:
                if ch not in seen:
                    s += self._global_counts[ch]; seen.add(ch)
            # keep weights positive; no need to normalize for probabilities
            self._prior_weight[w] = max(1.0, s)

    def _select_pool(self, candidates: List[str], allowed: List[str]) -> List[str]:
        if len(candidates) <= self.CANDIDATE_ONLY_LIMIT:
            return candidates
        # rank by prior weight as a cheap proxy
        ranked = sorted(allowed, key=lambda w: self._prior_weight.get(w, 1.0), reverse=True)
        return ranked[: self.POOL_CAP]

    def next_guess(self, state: dict) -> str:
        candidates: List[str] = state["candidates"]
        allowed: List[str]    = state["allowed"]

        pool = self._select_pool(candidates, allowed)
        if not pool:
            return "a" * self.N

        best_H = None; best_worst = None; best: List[str] = []
        weight = self._prior_weight

        for g in pool:
            H, _, worst = _weighted_entropy(g, candidates, weight)
            if (best_H is None) or (H > best_H) or (H == best_H and (best_worst is None or worst < best_worst)):
                best_H, best_worst, best = H, worst, [g]
            elif H == best_H and worst == best_worst:
                best.append(g)

        return best[self.rng.randrange(len(best))]
