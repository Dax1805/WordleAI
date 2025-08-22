"""
Letter-Frequency Solver (distinct-letter coverage).

Idea:
  - Build a letter histogram over the CURRENT candidate set (already filtered
    by past feedback). Score each word as the sum of its DISTINCT letters'
    frequencies. Pick the max; break ties with seeded RNG.

Why it works:
  - Early turns: favors words that cover common letters (shrinks space fast).
  - Later turns: histogram reflects constraints; top-scoring word tends to fit.

Notes:
  - Ignores positions (slot-specific frequencies are for a different solver).
  - Uses ALLOWED as the scoring pool only when candidates are very large
    (cheap "probing" words can split the space better early on).
"""

from __future__ import annotations
from collections import Counter
from typing import List
from .base import BaseSolver, register


@register
class LetterFreqSolver(BaseSolver):
    id = "letter_freq"
    name = "Letter Frequency (distinct)"
    version = "1.0.0"

    # If candidate set is huge, score the bigger allowed pool for better probes.
    # Tune this if your lists are much larger/smaller.
    CAND_POOL_LIMIT = 200

    def _score_word(self, w: str, counts: Counter[str]) -> int:
        """
        Sum letter frequencies but count each letter at most once per word
        (prefer 'slate' over 'sleet' when counts are similar).
        """
        seen = set()
        s = 0
        for ch in w:
            if ch not in seen:
                s += counts[ch]
                seen.add(ch)
        return s

    def next_guess(self, state: dict) -> str:
        """
        Decide the next guess based on distinct-letter coverage.
        """
        candidates: List[str] = state["candidates"]
        allowed: List[str] = state["allowed"]

        # Decide which pool of words to evaluate as guesses.
        pool: List[str] = candidates if len(candidates) <= self.CAND_POOL_LIMIT else allowed

        # Build frequency table from the CURRENT candidate set.
        counts = Counter("".join(candidates)) if candidates else Counter("".join(allowed))

        best_score = None
        best_words: List[str] = []

        for w in pool:
            s = self._score_word(w, counts)
            if best_score is None or s > best_score:
                best_score = s
                best_words = [w]
            elif s == best_score:
                best_words.append(w)

        # Deterministic tie-break using the solver RNG.
        if not best_words:
            best_words = pool or allowed or candidates or ["a" * self.N]
        i = self.rng.randrange(len(best_words))
        return best_words[i]
