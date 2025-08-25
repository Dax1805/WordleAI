from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List, Tuple

PATTERN_TYPES = ("all_gray", "some_green", "some_yellow", "mix_GY", "other")


def pattern_type(p: str) -> str:
    if not p:
        return "other"
    g = p.count("G")
    y = p.count("Y")
    if g == 0 and y == 0:
        return "all_gray"
    if g > 0 and y == 0:
        return "some_green"
    if y > 0 and g == 0:
        return "some_yellow"
    if g > 0 and y > 0:
        return "mix_GY"
    return "other"


def per_slot_entropy(candidates: List[str], N: int) -> List[float]:
    if not candidates:
        return [0.0] * N
    counts = [Counter() for _ in range(N)]
    for w in candidates:
        for i, ch in enumerate(w):
            counts[i][ch] += 1
    H = []
    n = float(len(candidates))
    for i in range(N):
        h = 0.0
        for c in counts[i].values():
            p = c / n
            h -= p * math.log(p + 1e-12)
        H.append(h)  # nats are fine; scaling not important for the bandit
    return H


def dup_ratio(candidates: List[str]) -> float:
    if not candidates:
        return 0.0
    dup = 0
    for w in candidates:
        if len(set(w)) < len(w):
            dup += 1
    return dup / len(candidates)


def make_features(*, turn: int, N: int,
                  candidates: List[str],
                  prev_candidates_len: int | None,
                  last_pattern: str | None) -> Tuple[List[float], List[str]]:
    """Return (feature_vector, feature_names)."""
    c_len = len(candidates)
    log_c = math.log2(max(c_len, 1))
    shrink = 0.0
    if prev_candidates_len and prev_candidates_len > 0:
        shrink = (prev_candidates_len - c_len) / prev_candidates_len
    pt = pattern_type(last_pattern or "")
    onehot_pt = [1.0 if pt == k else 0.0 for k in PATTERN_TYPES]
    Hslots = per_slot_entropy(candidates, N)
    dup = dup_ratio(candidates)

    feats = [
                float(turn),
                float(N),
                log_c,
                shrink,
                dup,
            ] + Hslots + onehot_pt

    names = (
            ["turn", "N", "log2_c", "shrink", "dup_ratio"]
            + [f"H{i}" for i in range(N)]
            + [f"pt_{k}" for k in PATTERN_TYPES]
    )
    return feats, names
