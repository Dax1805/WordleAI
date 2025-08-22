"""
Candidate filtering given game history.

Given:
  - a pool of words (e.g., the answers list)
  - a history of (guess, pattern) pairs
  - target word length N

Return:
  - words that are consistent with ALL feedback seen so far.

This is the core step that turns feedback into a shrinking candidate set.
Solvers use this to ensure future guesses remain consistent with the past.
"""

from typing import Iterable, List, Tuple
from .scoring import score

# History is a sequence of (guess, pattern) tuples produced by the engine.
History = Iterable[Tuple[str, str]]  # (guess, pattern)


def filter_candidates(words: Iterable[str], history: History, N: int) -> List[str]:
    """
    Keep only words (length == N) that would produce exactly the recorded
    patterns for every (guess, pattern) in `history`.

    Args:
      words   : iterable of candidate words (often the answers pool)
      history : iterable of (guess, pattern) seen so far
      N       : expected word length

    Returns:
      List[str] of consistent candidates (order preserved as in `words`).
    """
    out: List[str] = []

    for w in words:
        w = w.strip().lower()

        # Basic hygiene: skip anything that isn't a clean N-letter alpha token
        if len(w) != N or not w.isalpha():
            continue

        # Check against all past (guess, pattern) pairs
        consistent = True
        for g, patt in history:
            # If scoring this candidate against the old guess doesn't reproduce
            # the same pattern, the candidate is invalid.
            if score(g, w) != patt:
                consistent = False
                break

        if consistent:
            out.append(w)

    return out
