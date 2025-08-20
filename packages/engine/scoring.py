from collections import Counter
from typing import Literal

PatternChar = Literal["G", "Y", "-"]


def score(guess: str, answer: str) -> str:
    """
    Wordle-style feedback for (guess, answer) for any length N.

    'G' = correct letter, correct position
    'Y' = letter in word but wrong position (capped by true letter counts)
    '-' = letter not present (or exhausted)

    Preconditions:
      - len(guess) == len(answer)
    Postconditions:
      - returns a string of length N composed of only 'G','Y','-'
      - deterministic: same inputs -> same outputs
    """
    guess = guess.strip().lower()
    answer = answer.strip().lower()
    assert len(guess) == len(answer), "Guess and answer must be the same length"

    n = len(guess)
    res = ["-"] * n

    # Pass 1: mark greens; collect remaining answer letters
    remaining = Counter()
    for i, (g, a) in enumerate(zip(guess, answer)):
        if g == a:
            res[i] = "G"
        else:
            remaining[a] += 1

    # Pass 2: mark yellows from remaining counts
    for i, g in enumerate(guess):
        if res[i] == "G":
            continue
        if remaining[g] > 0:
            res[i] = "Y"
            remaining[g] -= 1

    return "".join(res)
