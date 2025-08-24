from __future__ import annotations
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
from .base import BaseSolver, register
from packages.engine import score as score_fn
from packages.engine.constraints import filter_candidates


@register
class TwoPlyMCSolver(BaseSolver):
    id = "two_ply_mc"
    name = "Two-Ply Monte-Carlo"
    version = "1.1.0"

    # Tighter caps (tune as you like)
    FIRST_POOL_CAP = 40  # was 80
    SAMPLE_SIZE = 32  # was 64
    CAND_CAP_PLY2 = 400  # cap the 2nd-ply scoring pool to candidates-only, top-K

    def _alphabet_counts(self, words: List[str]) -> Dict[str, int]:
        c = Counter()
        for w in words:
            for ch in set(w):
                c[ch] += 1
        return c

    def _distinct_score(self, w: str, counts: Dict[str, int]) -> int:
        s, seen = 0, set()
        for ch in w:
            if ch not in seen:
                seen.add(ch);
                s += counts.get(ch, 0)
        return s

    # ---- 2nd-ply picker: candidates-only, capped, no 'allowed' scans ----
    def _plf_pick_on_candidates(self, cands: List[str]) -> str:
        if not cands:
            return "a" * self.N
        # cap pool to top-K distinct-coverage wrt candidates (cheap prefilter)
        counts = self._alphabet_counts(cands)
        pool = sorted(cands, key=lambda w: self._distinct_score(w, counts), reverse=True)[
               : self.CAND_CAP_PLY2]
        # build positional counts once
        pos_counts = [Counter() for _ in range(self.N)]
        for w in cands:
            for i, ch in enumerate(w):
                pos_counts[i][ch] += 1
        # score pool
        best_s = None;
        best: List[str] = []
        for w in pool:
            s, seen = 0.0, set()
            for i, ch in enumerate(w):
                s += pos_counts[i][ch]
                if ch in seen:
                    s -= 0.25
                else:
                    seen.add(ch)
            if best_s is None or s > best_s:
                best_s, best = s, [w]
            elif s == best_s:
                best.append(w)
        return best[self.rng.randrange(len(best))]

    # ---- smaller, smarter first-guess pool ----
    def _select_first_pool(self, candidates: List[str], allowed: List[str]) -> List[str]:
        counts = self._alphabet_counts(candidates if candidates else allowed)
        # top from allowed for probing
        base = sorted(allowed, key=lambda w: self._distinct_score(w, counts), reverse=True)[
               : self.FIRST_POOL_CAP]
        # also ensure some strong candidates are present
        top_cands = sorted(candidates, key=lambda w: self._distinct_score(w, counts), reverse=True)[
                    : self.FIRST_POOL_CAP // 2]
        seen, pool = set(), []
        for w in top_cands + base:
            if w not in seen:
                seen.add(w);
                pool.append(w)
        return pool

    def next_guess(self, state: dict) -> str:
        turn = state["turn"]
        candidates: List[str] = state["candidates"]
        allowed: List[str] = state["allowed"]

        # After turn 1: just use a fast, strong policy
        if turn > 1:
            return self._plf_pick_on_candidates(candidates)

        # Turn 1: two-ply MC on a small pool
        pool = self._select_first_pool(candidates, allowed)
        if not pool:
            return "a" * self.N

        # sample answers
        sample = list(candidates) if len(candidates) <= self.SAMPLE_SIZE else self.rng.sample(
            candidates, self.SAMPLE_SIZE)

        # Precompute pattern1 groups per guess to avoid recomputing c1 for each 'a'
        best_avg = None;
        best_worst = None;
        best: List[str] = []

        for g in pool:
            # group answers by first pattern (pattern1)
            groups: Dict[str, List[str]] = defaultdict(list)
            for a in sample:
                groups[score_fn(g, a)].append(a)

            totals = 0
            worst = 0
            processed = 0

            for patt1, group_answers in groups.items():
                # compute c1 once per pattern1
                c1 = filter_candidates(candidates, [(g, patt1)], self.N)

                if not c1:
                    # this bucket contributes 0 for all answers in it
                    # (skip)
                    continue

                # pick second guess on c1 (candidates-only; no 13k scans)
                g2 = self._plf_pick_on_candidates(c1)

                # build pattern2 buckets once for c1
                buckets2: Dict[str, int] = defaultdict(int)
                for a2 in c1:
                    buckets2[score_fn(g2, a2)] += 1

                # for each 'a' in this pattern1 group, its c2 size is
                # the size of its pattern2 bucket
                for a in group_answers:
                    patt2 = score_fn(g2, a)
                    c2_size = buckets2.get(patt2, 0)
                    totals += c2_size
                    if c2_size > worst:
                        worst = c2_size
                    processed += 1

                # --- early pruning: if partial average already worse, bail ---
                if best_avg is not None:
                    avg_so_far = totals / processed
                    # optimistic lower bound on remaining is 0, so if avg_so_far >= best_avg, stop
                    if avg_so_far >= best_avg:
                        break

            avg_left = totals / max(1, processed)

            if (best_avg is None) or (avg_left < best_avg) or (
                    avg_left == best_avg and (best_worst is None or worst < best_worst)):
                best_avg, best_worst, best = avg_left, worst, [g]
            elif avg_left == best_avg and worst == best_worst:
                best.append(g)

        return best[self.rng.randrange(len(best))]
