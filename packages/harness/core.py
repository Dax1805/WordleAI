"""
Experiment harness core primitives.

- run_case:  run a single puzzle (one hidden answer) with a given solver.
- run_batch: run many puzzles in sequence (optionally a sample prefix).
- Enforces Wordle's 6-turn limit at the harness layer.

These functions are intentionally UI-agnostic so they can be reused by
a CLI app, a notebook, or future services without changes.
"""

from __future__ import annotations
import time
from typing import Dict, List, Iterable, Tuple
from packages.engine import score, filter_candidates

# Single source of truth for Wordle turn budget.
WORDLE_MAX_TURNS = 6

def _assert_wordle_turns(max_turns: int) -> None:
    """Guardrail: prevent accidental runs with >6 turns."""
    if max_turns != WORDLE_MAX_TURNS:
        raise ValueError(f"max_turns must be {WORDLE_MAX_TURNS} for Wordle-like rules; got {max_turns}")

def run_case(
        solver,
        answer: str,
        *,
        allowed: Iterable[str],
        answers: Iterable[str],
        N: int,
        max_turns: int = WORDLE_MAX_TURNS,
        seed: int | None = None,
) -> Dict:
    """
    Execute one game until the solver wins or the turn budget is exhausted.

    Args:
        solver:        an object implementing BaseSolver with next_guess(state)
        answer:        the hidden word for this case
        allowed:       all words permitted as guesses (ideally a superset of answers)
        answers:       the official answer pool (candidate universe)
        N:             word length (e.g., 5 or 6)
        max_turns:     must be 6 (Wordle rule; enforced)
        seed:          RNG seed to make solver tie-breaks reproducible

    Returns:
        dict with keys:
            success (bool), guesses (int), time_ms (float),
            history (list[(guess, pattern)]), answer (str)
    """
    _assert_wordle_turns(max_turns)

    # Initialize solver (gives it access to lists, N, RNG)
    solver.reset(allowed=allowed, answers=answers, N=N, seed=seed)

    # History accumulates (guess, pattern) tuples for logging and filtering
    history: List[Tuple[str, str]] = []

    # Start with all valid answers of length N as candidates
    candidates = [w for w in answers if len(w) == N]

    t0 = time.time()
    for turn in range(1, WORDLE_MAX_TURNS + 1):
        # Provide current state so the solver can decide intelligently
        state = {
            "turn": turn,
            "history": list(history),
            "candidates": candidates,
            "allowed": [w for w in allowed if len(w) == N],
            "N": N,
            "rng": solver.rng,
        }

        # Solver proposes a guess from the provided state
        guess = solver.next_guess(state)

        # Engine scores the guess vs the hidden answer (e.g., 'G', 'Y', '-')
        patt = score(guess, answer)
        history.append((guess, patt))

        # Win condition: all green
        if guess == answer:
            dt = (time.time() - t0) * 1000.0
            return {
                "success": True, "guesses": turn, "time_ms": dt,
                "history": history, "answer": answer
            }

        # Narrow candidate set using the new feedback before next turn
        candidates = filter_candidates(candidates, [(guess, patt)], N)

    # Out of turns: lose
    dt = (time.time() - t0) * 1000.0
    return {
        "success": False, "guesses": WORDLE_MAX_TURNS, "time_ms": dt,
        "history": history, "answer": answer
    }

def run_batch(
        solver,
        answers: List[str],
        *,
        allowed: List[str],
        N: int,
        max_turns: int = WORDLE_MAX_TURNS,
        seed: int | None = None,
        sample: int | None = None,
) -> List[Dict]:
    """
    Run many cases back-to-back. If 'sample' is provided, only the first K answers
    (after filtering to length N) are used to speed up quick experiments.

    Each case's seed is derived from the base seed to make runs reproducible
    but not identical across cases (seed + index).
    """
    _assert_wordle_turns(max_turns)

    # Pre-filter answer pool to the requested length
    pool = [w for w in answers if len(w) == N]
    if sample is not None:
        pool = pool[:sample]

    out: List[Dict] = []
    for idx, ans in enumerate(pool, start=1):
        case_seed = None if seed is None else (seed + idx)
        r = run_case(
            solver, ans, allowed=allowed, answers=answers, N=N,
            max_turns=WORDLE_MAX_TURNS, seed=case_seed
        )
        out.append(r)
    return out
