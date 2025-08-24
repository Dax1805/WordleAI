"""
CLI entry point for running wordleAI experiments.

This script:
  1) Validates the wordlists (prints counts + SHA, ensures answers ⊆ allowed).
  2) Loads the lists and instantiates the requested solver.
  3) Runs a batch of games and writes:
       - CSV:  per-case results + guess/pattern history columns
       - JSON: manifest with config, wordlist hashes, git commit, etc.
"""

import argparse
from pathlib import Path

from packages.datasets import validate_wordlists, pretty_summary
from packages.harness import run_batch, write_csv, write_manifest
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

MAX_TURNS = 6  # Wordle hard limit (kept here to mirror harness constant)


def _load_words(path: str) -> list[str]:
    """
    Read a newline-separated word list, normalize to lowercase, drop blanks.

    Args:
        path: file path to read.

    Returns:
        List of lowercase words (order preserved).
    """
    p = Path(path)
    return [w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()]


def main():
    """
    Parse CLI args, validate datasets, run the batch, and write outputs.
    """
    # Build help text showing currently registered solver IDs
    solver_choices = ", ".join(get_solver_ids() + ["random_consistent", "letter_freq", "entropy"])

    ap = argparse.ArgumentParser(description="wordleAI — run solver experiments")
    ap.add_argument("--solver", default="random_consistent",
                    help=f"solver id (one of: {solver_choices})")
    ap.add_argument("--N", type=int, default=5, help="word length (e.g., 5 or 6)")
    ap.add_argument("--answers", default="packages/datasets/data/answers_5.txt",
                    help="path to answers list (ground-truth pool)")
    ap.add_argument("--allowed", default="packages/datasets/data/allowed_5.txt",
                    help="path to allowed guesses (should be a superset of answers)")
    # We intentionally DO NOT expose --max-turns; the harness enforces 6 turns.
    ap.add_argument("--sample", type=int,
                    help="run only the first K answers (for quick experiments)")
    ap.add_argument("--seed", type=int, default=123, help="base RNG seed (for reproducibility)")
    ap.add_argument("--outdir", default="reports", help="directory for output files")
    args = ap.parse_args()

    # 1) Validate wordlists and print a one-liner summary (counts, SHAs, subset check)
    rep = validate_wordlists(args.N, args.answers, args.allowed)
    print(pretty_summary(rep))
    # If you want to hard-fail on validation issues, uncomment the next lines:
    # if not rep["passed"]:
    #     print("Validation failed — fix wordlists before running.")
    #     return

    # 2) Load lists into memory (lowercased, no blanks)
    answers = _load_words(args.answers)
    allowed = _load_words(args.allowed)

    # 3) Instantiate solver by id (registry populated via imports above)
    solver = create_solver(args.solver)

    # 4) Run the batch (max turns enforced inside the harness)
    results = run_batch(
        solver, answers,
        allowed=allowed, N=args.N,
        max_turns=MAX_TURNS,  # mirrored constant; harness will assert anyway
        seed=args.seed, sample=args.sample
    )

    # Stamp solver id on each row so downstream tools don’t guess
    for r in results:
        r["solver_id"] = solver.id

    # 5) Write outputs (CSV + manifest)
    from packages.harness.io import timestamp_id, git_commit_or_unknown
    run_id = timestamp_id()
    outdir = Path(args.outdir);
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
    }
    write_manifest(manifest, str(manifest_path))

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
