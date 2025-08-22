"""
Lightweight guess validation.

This module answers the question: "Is this guess acceptable right now?"
For our offline experiments, a guess is valid iff:
  - it is a string
  - it is alphabetic a–z only
  - it has exact length N
  - it exists in the provided `allowed` list/set

In a UI, you'd also check "already guessed" and possibly throttling, but the
harness handles repetition implicitly (a repeated guess can never be consistent
after it's not the answer).
"""

from typing import Iterable, Set


def validate_guess(word: str, allowed: Iterable[str], N: int) -> bool:
    """
    Return True if `word` is a valid guess per the rules above.

    Args:
      word    : proposed guess
      allowed : iterable of allowed words (e.g., allowed_N.txt)
      N       : required word length

    Notes:
      - The `allowed` parameter can be a large list; we build a local set
        here for O(1) membership checks. If you’re calling this in a tight
        loop, consider precomputing the set once at a higher level.
    """
    if not isinstance(word, str):
        return False

    w = word.strip().lower()

    # Shape/characters check
    if len(w) != N or not w.isalpha():
        return False

    # Membership check (case-normalized)
    allowed_set: Set[str] = {a.strip().lower() for a in allowed}
    return w in allowed_set
