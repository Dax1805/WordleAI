from typing import Iterable, Set


def validate_guess(word: str, allowed: Iterable[str], N: int) -> bool:
    """
    A guess is valid if:
      - it's a string of alphabetic chars
      - length == N
      - present in the allowed word set
    """
    if not isinstance(word, str):
        return False
    w = word.strip().lower()
    if len(w) != N or not w.isalpha():
        return False
    allowed_set: Set[str] = {a.strip().lower() for a in allowed}
    return w in allowed_set
