from __future__ import annotations
import time
from typing import Dict, List, Iterable, Tuple
from packages.engine import score, filter_candidates


def run_case(solver, answer: str, *, allowed: Iterable[str], answers: Iterable[str],
             N: int, max_turns: int = 6, seed: int | None = None) -> Dict:
    """
    Run a single puzzle to completion or failure.
    Returns: dict with success, guesses, time_ms, history[(guess, patt)].
    """
    solver.reset(allowed=allowed, answers=answers, N=N, seed=seed)
    history: List[Tuple[str, str]] = []
    candidates = [w for w in answers if len(w) == N]
    t0 = time.time()
    for turn in range(1, max_turns + 1):
        state = {
            "turn": turn,
            "history": list(history),
            "candidates": candidates,
            "allowed": [w for w in allowed if len(w) == N],
            "N": N,
            "rng": solver.rng,
        }
        guess = solver.next_guess(state)
        patt = score(guess, answer)
        history.append((guess, patt))

        if guess == answer:
            dt = (time.time() - t0) * 1000.0
            return {
                "success": True,
                "guesses": turn,
                "time_ms": dt,
                "history": history,
                "answer": answer,
            }

        candidates = filter_candidates(candidates, [(guess, patt)], N)
        if not candidates:
            # dead end, but keep looping so we count turns up to max_turns
            candidates = []

    dt = (time.time() - t0) * 1000.0
    return {
        "success": False,
        "guesses": max_turns,
        "time_ms": dt,
        "history": history,
        "answer": answer,
    }


def run_batch(solver, answers: List[str], *, allowed: List[str],
              N: int, max_turns: int = 6, seed: int | None = None, sample: int | None = None) -> \
List[Dict]:
    """
    Run over a list of answers. If sample is provided, use the first K answers.
    """
    pool = [w for w in answers if len(w) == N]
    if sample is not None:
        pool = pool[:sample]
    out: List[Dict] = []
    for idx, ans in enumerate(pool, start=1):
        # vary seed per case for diversified RNG but deterministic overall
        case_seed = None if seed is None else (seed + idx)
        out.append(run_case(solver, ans, allowed=allowed, answers=answers, N=N, max_turns=max_turns,
                            seed=case_seed))
    return out
