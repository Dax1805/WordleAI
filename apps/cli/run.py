# apps/cli/run.py
"""
CLI entry point for running wordleAI experiments.

This script:
  1) Validates the wordlists (prints counts + SHA, ensures answers ⊆ allowed).
  2) Loads the lists and instantiates the requested solver.
  3) Runs a batch of games with a live progress indicator and writes:
       - CSV:  per-case results + guess/pattern history columns
       - JSON: manifest with config, wordlist hashes, git commit, etc.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Optional rich progress bar
try:
    from tqdm import tqdm  # pip install tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from packages.datasets import validate_wordlists, pretty_summary
from packages.harness.io import write_csv, write_manifest, timestamp_id, git_commit_or_unknown
from packages.solvers import create_solver, get_solver_ids

# Importing solver modules registers them by side-effect in the solver registry.
from packages.solvers import random_consistent   # noqa: F401
from packages.solvers import letter_freq         # noqa: F401
from packages.solvers import entropy             # noqa: F401
from packages.solvers import positional_freq     # noqa: F401
from packages.solvers import max_patterns        # noqa: F401
from packages.solvers import expected_left       # noqa: F401
from packages.solvers import entropy_weighted    # noqa: F401
from packages.solvers import two_stage_probe     # noqa: F401
from packages.solvers import two_ply_mc          # noqa: F401

from packages.engine import score as score_fn
from packages.engine.constraints import filter_candidates

MAX_TURNS = 6  # Wordle hard limit (kept here to mirror harness constant)


def _load_words(path: str) -> list[str]:
    """
    Read a newline-separated word list, normalize to lowercase, drop blanks.
    """
    p = Path(path)
    return [w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()]


def _play_one_game(
        *,
        solver,
        answer: str,
        answers: List[str],
        allowed: List[str],
        N: int,
        seed: int,
) -> Dict:
    """
    Simulate a single Wordle game to completion or 6 turns.
    Returns a result dict matching the harness schema.
    """
    # Reset solver RNG per-game for reproducibility
    solver.reset(allowed=allowed, answers=answers, N=N, seed=seed)

    candidates = list(answers)  # closed-world candidates
    history: List[Tuple[str, str]] = []
    total_ms = 0.0
    success = False

    for turn in range(1, MAX_TURNS + 1):
        state = {
            "turn": turn,
            "N": N,
            "candidates": candidates,
            "allowed": allowed,
            "history": history,
        }
        t0 = time.perf_counter_ns()
        guess = solver.next_guess(state).lower()
        t1 = time.perf_counter_ns()
        total_ms += (t1 - t0) / 1_000_000.0

        # Score guess against the hidden answer
        patt = score_fn(guess, answer)
        history.append((guess, patt))

        if patt == "G" * N:
            success = True
            break

        # Narrow candidate set using feedback
        candidates = filter_candidates(candidates, [(guess, patt)], N)

    result = {
        "answer": answer,
        "success": success,
        "guesses": len(history),
        "time_ms": total_ms,
        "history": history,
    }
    return result


def main():
    """
    Parse CLI args, validate datasets, run the batch with progress, and write outputs.
    """
    # Build help text showing currently registered solver IDs
    solver_choices = ", ".join(get_solver_ids())

    ap = argparse.ArgumentParser(description="wordleAI — run solver experiments")
    ap.add_argument("--solver", default="random_consistent",
                    help=f"solver id (one of: {solver_choices})")
    ap.add_argument("--N", type=int, default=5, help="word length (e.g., 5 or 6)")
    ap.add_argument("--answers", default="packages/datasets/data/answers_5.txt",
                    help="path to answers list (ground-truth pool)")
    ap.add_argument("--allowed", default="packages/datasets/data/allowed_5.txt",
                    help="path to allowed guesses (should be a superset of answers)")
    ap.add_argument("--sample", type=int,
                    help="run only a subset of answers (deterministic by seed)")
    ap.add_argument("--seed", type=int, default=123, help="base RNG seed (for reproducibility)")
    ap.add_argument("--outdir", default="reports", help="directory for output files")
    ap.add_argument(
        "--progress",
        choices=["auto", "bar", "plain", "off"],
        default="auto",
        help="Show run progress (auto=bar if tqdm available, else plain text)."
    )
    args = ap.parse_args()

    # 1) Validate wordlists and print a one-liner summary (counts, SHAs, subset check)
    rep = validate_wordlists(args.N, args.answers, args.allowed)
    print(pretty_summary(rep))
    # If you want to hard-fail on validation issues, uncomment:
    # if not rep.get("passed", True):
    #     print("Validation failed — fix wordlists before running."); return

    # 2) Load lists into memory (lowercased, no blanks)
    answers = _load_words(args.answers)
    allowed = _load_words(args.allowed)

    # 3) Instantiate solver by id (registry populated via imports above)
    solver = create_solver(args.solver)

    # 4) Choose cases (deterministic sample by seed)
    import random
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(answers):
        # Deterministic sample without replacement
        # (convert to list first to keep original order stable when sample==len(answers))
        pool = list(answers)
        rng.shuffle(pool)
        cases = pool[: args.sample]
    else:
        cases = list(answers)

    total = len(cases)

    # 5) Progress mode
    mode = args.progress
    if mode == "auto":
        mode = "bar" if (_HAS_TQDM and sys.stderr.isatty()) else "plain"

    results = []
    start = time.time()
    last_print = 0.0

    iterator = tqdm(cases, ncols=80, desc="Running", unit="game") if mode == "bar" else cases

    # 6) Run batch with live progress
    for idx, ans in enumerate(iterator, 1):
        # Derive a per-game seed so runs are reproducible and independent
        per_seed = args.seed + idx * 1013904223  # LCG-ish stride to avoid collisions
        r = _play_one_game(
            solver=solver,
            answer=ans,
            answers=answers,
            allowed=allowed,
            N=args.N,
            seed=per_seed,
        )
        r["solver_id"] = solver.id  # stamp id for downstream tools
        results.append(r)

        if mode == "plain":
            now = time.time()
            if (now - last_print >= 1.0) or (idx == total):
                elapsed = now - start
                rate = (idx / elapsed) if elapsed > 0 else 0.0
                remaining = (total - idx) / rate if rate > 0 else 0.0
                pct = 100.0 * idx / max(1, total)
                sys.stderr.write(
                    f"\r[{idx}/{total}] {pct:5.1f}% | elapsed {elapsed:6.1f}s | ETA {remaining:5.1f}s"
                )
                sys.stderr.flush()
                last_print = now

    if mode == "plain":
        sys.stderr.write("\n"); sys.stderr.flush()

    # 7) Write outputs (CSV + manifest)
    run_id = timestamp_id()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"run_{run_id}.csv"
    manifest_path = outdir / f"run_{run_id}_manifest.json"

    write_csv(results, str(csv_path), max_turns=MAX_TURNS, N=args.N)
    manifest = {
        "run_id": run_id,
        "git_commit": git_commit_or_unknown(),
        "config": vars(args),
        "wordlists": rep,
        "num_cases": len(results),
        "solver_id": solver.id,
    }
    write_manifest(manifest, str(manifest_path))

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
