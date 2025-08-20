from typing import Iterable, List, Tuple
from .scoring import score

History = Iterable[Tuple[str, str]]  # (guess, pattern)


def filter_candidates(words: Iterable[str], history: History, N: int) -> List[str]:
    """
    Return words (length == N) that are consistent with all (guess, pattern) pairs.
    """
    out: List[str] = []
    for w in words:
        w = w.strip().lower()
        if len(w) != N or not w.isalpha():
            continue
        consistent = True
        for guess, patt in history:
            if score(guess, w) != patt:
                consistent = False
                break
        if consistent:
            out.append(w)
    return out
