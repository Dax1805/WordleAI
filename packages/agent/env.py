# packages/agent/env.py
from __future__ import annotations
import time
import random
from typing import Dict, List, Tuple
from pathlib import Path

from packages.engine import score as score_fn
from packages.engine.constraints import filter_candidates
from packages.datasets.io import read_lines  # if you put io helpers elsewhere, adjust import
from packages.agent.features import make_features

# pull in solvers so they're registered
from packages.solvers import random_consistent, letter_freq, positional_freq, expected_left, \
    max_patterns  # noqa: F401
from packages.solvers import entropy, entropy_weighted, two_stage_probe, two_ply_mc  # noqa: F401

# Registry helper (works with your @register decorator)
try:
    # common name used in your base
    from packages.solvers.base import REGISTRY as SOLVER_REGISTRY
except Exception:
    from packages.solvers import REGISTRY as SOLVER_REGISTRY  # fallback


class WordleBanditEnv:
    """
    One episode = one Wordle game. At each turn, the agent picks a solver-id.
    The solver proposes a guess; we score and update the candidate set.
    Reward per turn: -1 - alpha_time * (time_ms/100).
    """

    def __init__(self, *, answers_path: str, allowed_path: str, N: int, seed: int | None = None,
                 actions: List[str] | None = None, alpha_time: float = 0.2):
        self.N = N
        self.rng = random.Random(seed)
        self.alpha_time = float(alpha_time)

        self.answers_all = [w.strip().lower() for w in read_lines(Path(answers_path)) if
                            len(w.strip()) == N]
        self.allowed = [w.strip().lower() for w in read_lines(Path(allowed_path)) if
                        len(w.strip()) == N]

        # default action set (fast-ish)
        self.actions = actions or ["positional_freq", "expected_left", "max_patterns",
                                   "letter_freq"]

        # Prepare solver instances (reset per episode)
        self._solvers = {aid: SOLVER_REGISTRY[aid]() for aid in self.actions}

        # episode state
        self._answer = None
        self._turn = 0
        self._history: List[Tuple[str, str]] = []
        self._candidates: List[str] = []
        self._prev_c_len: int | None = None

    def reset(self, *, answer: str | None = None, seed: int | None = None) -> Dict:
        if seed is not None:
            self.rng.seed(seed)
        self._answer = (answer or self.rng.choice(self.answers_all)).lower()
        self._turn = 0
        self._history = []
        self._candidates = list(self.answers_all)  # closed-world hypothesis
        self._prev_c_len = None

        # reset solvers for this episode
        for s in self._solvers.values():
            s.reset(allowed=self.allowed, answers=self.answers_all, N=self.N,
                    seed=self.rng.randint(0, 2 ** 31 - 1))

        return self._obs(last_pattern=None)

    def _obs(self, last_pattern: str | None) -> Dict:
        feats, names = make_features(
            turn=self._turn + 1, N=self.N,
            candidates=self._candidates,
            prev_candidates_len=self._prev_c_len,
            last_pattern=last_pattern
        )
        return {
            "turn": self._turn + 1,
            "N": self.N,
            "candidates": list(self._candidates),
            "allowed": self.allowed,
            "history": list(self._history),
            "features": feats,
            "feature_names": names,
        }

    def step(self, action_solver_id: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Agent chooses an action (solver id). We ask that solver for a guess,
        apply feedback, compute reward, return next observation.
        """
        assert action_solver_id in self._solvers, f"Unknown action {action_solver_id}"
        solver = self._solvers[action_solver_id]

        state_for_solver = {
            "turn": self._turn + 1,
            "N": self.N,
            "candidates": self._candidates,
            "allowed": self.allowed,
            "history": self._history,
        }

        t0 = time.perf_counter_ns()
        guess = solver.next_guess(state_for_solver).lower()
        t1 = time.perf_counter_ns()
        step_time_ms = (t1 - t0) / 1_000_000.0

        # guard: invalid guess → penalize (shouldn't happen if solver uses allowed)
        if (len(guess) != self.N) or (guess not in self.allowed):
            # heavy penalty and random fallback pattern
            patt = "-" * self.N
            reward = -2.0 - self.alpha_time * (step_time_ms / 100.0)
        else:
            patt = score_fn(guess, self._answer)
            reward = -1.0 - self.alpha_time * (step_time_ms / 100.0)

        self._history.append((guess, patt))
        self._prev_c_len = len(self._candidates)
        self._candidates = filter_candidates(self._candidates, [(guess, patt)], self.N)
        self._turn += 1

        done = (patt == "G" * self.N) or (self._turn >= 6)
        info = {
            "guess": guess,
            "pattern": patt,
            "time_ms": step_time_ms,
            "chosen_solver": action_solver_id,
            "answer": self._answer if done else None,
        }

        return self._obs(last_pattern=patt), reward, done, info
