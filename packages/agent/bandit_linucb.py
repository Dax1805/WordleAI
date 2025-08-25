# packages/agent/bandit_linucb.py
from __future__ import annotations
import json
import numpy as np
from typing import Dict, List

class LinUCB:
    """
    Per-action linear UCB:
      For each action a, maintain A_a (dxd) and b_a (dx1).
      Select argmax  x^T theta_a + alpha * sqrt( x^T A_a^{-1} x ).
    """
    def __init__(self, actions: List[str], d: int, alpha: float = 0.5, l2: float = 1.0, seed: int | None = None):
        self.actions = list(actions)
        self.d = int(d)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.A: Dict[str, np.ndarray] = {a: (l2 * np.eye(self.d)) for a in self.actions}
        self.b: Dict[str, np.ndarray] = {a: np.zeros((self.d, 1)) for a in self.actions}

    def select(self, x: np.ndarray) -> str:
        """
        Choose action by UCB. x is a (d,) feature vector.
        """
        x = np.asarray(x, dtype=float).reshape(-1)     # ensure shape (d,)
        best = None
        best_score = -np.inf

        for a in self.actions:
            Ainv = np.linalg.inv(self.A[a])            # (d,d)
            theta = (Ainv @ self.b[a]).reshape(-1)     # (d,)
            mean = float(x @ theta)                    # scalar
            quad = float(x @ Ainv @ x)                 # scalar, >=0
            bonus = self.alpha * np.sqrt(max(quad, 0.0))
            ucb = mean + bonus
            if (best is None) or (ucb > best_score):
                best = a
                best_score = ucb

        return best  # type: ignore

    def update(self, a: str, x: np.ndarray, reward: float) -> None:
        """
        One-step update for the chosen action a.
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)  # (d,1)
        self.A[a] += (x @ x.T)                         # rank-1 update
        self.b[a] += reward * x

    # ---- persistence ----
    def to_json(self) -> str:
        pack = {
            "actions": self.actions,
            "d": self.d,
            "alpha": self.alpha,
            "A": {a: self.A[a].tolist() for a in self.actions},
            "b": {a: self.b[a].reshape(-1).tolist() for a in self.actions},
        }
        return json.dumps(pack)

    @staticmethod
    def from_json(s: str) -> "LinUCB":
        obj = json.loads(s)
        m = LinUCB(actions=obj["actions"], d=int(obj["d"]), alpha=float(obj["alpha"]))
        for a in m.actions:
            m.A[a] = np.array(obj["A"][a], dtype=float)
            m.b[a] = np.array(obj["b"][a], dtype=float).reshape(-1, 1)
        return m
