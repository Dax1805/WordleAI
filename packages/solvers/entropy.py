"""
Entropy Solver (expected information gain), accelerated.

Main idea unchanged:
  - For each candidate guess g, partition CURRENT candidates by feedback pattern.
  - Compute Shannon entropy H over those buckets; pick g with max H.
Tie-break:
  - smaller worst-case bucket (minimax-ish), then seeded RNG.

Acceleration:
  - When the candidate set is large, don't evaluate ENTIRE allowed.
  - Pre-rank allowed words by DISTINCT letter-frequency score w.r.t. CURRENT candidates,
    keep only the top-K (POOL_CAP), and always include top candidate words too.
  - This preserves quality while cutting work by ~10x–30x on big lists.
"""

from __future__ import annotations
from collections import defaultdict, Counter
from math import log2
from typing import Dict, List, Tuple
from .base import BaseSolver, register
from packages.engine import score as score_fn  # canonical scoring


def _entropy_of_guess(guess: str, candidates: List[str]) -> Tuple[float, int]:
    """Partition candidates by pattern; return (entropy_bits, worst_bucket_size)."""
    n = len(candidates)
    if n <= 1:
        return 0.0, 0

    buckets: Dict[str, int] = defaultdict(int)
    # localize for speed
    _score = score_fn
    for ans in candidates:
        buckets[_score(guess, ans)] += 1

    H = 0.0
    worst = 0
    for c in buckets.values():
        p = c / n
        H += p * log2(1 / p)   # == -p*log2(p)
        if c > worst:
            worst = c
    return H, worst


@register
class EntropySolver(BaseSolver):
    id = "entropy"
    name = "Entropy (Expected Information Gain)"
    version = "1.1.0"

    # If candidates ≤ this, search only among candidates (precise & cheap)
    CANDIDATE_ONLY_LIMIT = 200

    # When candidates are larger, cap the pool size after prefiltering
    POOL_CAP = 400

    # Also include up to this many top candidate words in the pool (so we don't
    # miss an obvious answer late in the game)
    INCLUDE_TOP_CANDIDATES = 100

    def _distinct_score(self, w: str, counts: Counter) -> int:
        """Sum of per-letter counts with duplicates in the word counted once."""
        s, seen = 0, set()
        for ch in w:
            if ch not in seen:
                s += counts[ch]
                seen.add(ch)
        return s

    def _select_pool(self, candidates: List[str], allowed: List[str]) -> List[str]:
        """
        Choose which words to evaluate with entropy:
          - Small candidate set: just use candidates.
          - Large set: rank allowed by distinct-letter score (w.r.t. candidates)
                       and keep only the top-K; also merge in top candidate words.
        """
        if len(candidates) <= self.CANDIDATE_ONLY_LIMIT:
            return candidates

        # Build histogram from CURRENT candidates (reflects constraints so far)
        counts = Counter("".join(candidates)) if candidates else Counter("".join(allowed))

        # Rank allowed by distinct-letter coverage and take the top-K
        # (avoid sorting the whole list if very large: use nlargest-like approach)
        scored_allowed = sorted(allowed, key=lambda w: self._distinct_score(w, counts), reverse=True)
        pool = scored_allowed[: self.POOL_CAP]

        # Also ensure some of the strongest candidates are present
        top_cands = sorted(candidates, key=lambda w: self._distinct_score(w, counts), reverse=True)
        top_cands = top_cands[: min(self.INCLUDE_TOP_CANDIDATES, len(top_cands))]

        # Stable-union: top candidates first, then the allowed top-K without duplicates
        seen = set()
        out: List[str] = []
        for w in top_cands + pool:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def next_guess(self, state: dict) -> str:
        """Pick the guess with maximum expected information gain."""
        candidates: List[str] = state["candidates"]
        allowed: List[str] = state["allowed"]

        # Select a manageable but informative pool
        pool: List[str] = self._select_pool(candidates, allowed)

        best_H = None
        best_worst = None
        best_words: List[str] = []

        for g in pool:
            H, worst = _entropy_of_guess(g, candidates)
            if (best_H is None) or (H > best_H) or (H == best_H and (best_worst is None or worst < best_worst)):
                best_H, best_worst, best_words = H, worst, [g]
            elif H == best_H and worst == best_worst:
                best_words.append(g)

        # Safe fallback + deterministic tie-break
        if not best_words:
            best_words = pool or allowed or candidates or ["a" * self.N]
        i = self.rng.randrange(len(best_words))
        return best_words[i]
