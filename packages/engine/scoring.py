"""
Wordle-style scoring (feedback) for a single (guess, answer) pair.

Conventions:
  - 'G'  : green  = correct letter in the correct position
  - 'Y'  : yellow = correct letter in the wrong position
  - '-'  : gray   = letter not present (or present fewer times than guessed)

This implementation is:
  - N-aware (any word length)
  - duplicate-safe (respects true letter multiplicities in the answer)
  - deterministic (same inputs -> same outputs)

Algorithm (two-pass, canonical for Wordle):
  1) First pass marks all greens and counts the remaining (unmatched) letters
     from the answer.
  2) Second pass marks yellows only if the letter still has remaining count.
"""

from collections import Counter
from typing import Literal

# Type alias for clarity; each pattern character is one of 'G', 'Y', '-'
PatternChar = Literal["G", "Y", "-"]


def score(guess: str, answer: str) -> str:
    """
    Compute Wordle feedback pattern for `guess` against `answer`.

    Preconditions:
      - len(guess) == len(answer)

    Returns:
      - string of length N composed only of 'G', 'Y', '-'

    Examples:
      score("belle", "level") -> "-GYYY"
      score("lemon", "level") -> "GG---"
    """
    # Normalize; Wordle is case-insensitive but canonicalizes to lowercase
    guess = guess.strip().lower()
    answer = answer.strip().lower()
    assert len(guess) == len(answer), "Guess and answer must be the same length"

    n = len(guess)
    pattern = ["-"] * n

    # Pass 1: mark greens and collect leftover counts from the answer.
    # For positions that are not green, we count the answer's letter
    # so yellows can be assigned correctly in pass 2.
    remaining = Counter()
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            pattern[i] = "G"
        else:
            remaining[a] += 1

    # Pass 2: mark yellows only if the letter still has remaining availability.
    # This caps 'Y' assignments by the true multiplicity in the answer.
    for i, g in enumerate(guess):
        if pattern[i] == "G":
            continue  # already green; skip
        if remaining[g] > 0:
            pattern[i] = "Y"
            remaining[g] -= 1  # consume one instance
        # else: stay '-', no remaining instances of this letter

    return "".join(pattern)
