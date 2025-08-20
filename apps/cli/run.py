import argparse
from pathlib import Path

from packages.datasets import validate_wordlists, pretty_summary
from packages.harness import run_batch, write_csv, write_manifest
from packages.solvers import create_solver, get_solver_ids
from packages.solvers import random_consistent  # noqa: F401 (registers the solver)


def _load_words(path: str):
    p = Path(path)
    return [w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()]


def main():
    ap = argparse.ArgumentParser(description="wordleAI - run experiments (baseline)")
    ap.add_argument("--solver", default="random_consistent",
                    help=f"solver id (one of: {', '.join(get_solver_ids() + ['random_consistent'])})")
    ap.add_argument("--N", type=int, default=5, help="word length")
    ap.add_argument("--answers", default="packages/datasets/data/answers_5.txt",
                    help="path to answers file")
    ap.add_argument("--allowed", default="packages/datasets/data/allowed_5.txt",
                    help="path to allowed file")
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--sample", type=int, help="use first K answers")
    ap.add_argument("--seed", type=int, default=123, help="base seed for RNG")
    ap.add_argument("--outdir", default="reports", help="directory for outputs")
    args = ap.parse_args()

    # Validate wordlists
    rep = validate_wordlists(args.N, args.answers, args.allowed)
    print(pretty_summary(rep))
    if not rep["passed"]:
        print("Validation failed. Fix wordlists or override intentionally.")
        # continue anyway for dev? comment the next line to proceed despite failures
        # return

    answers = _load_words(args.answers)
    allowed = _load_words(args.allowed)

    solver = create_solver(args.solver)
    results = run_batch(solver, answers, allowed=allowed, N=args.N, max_turns=args.max_turns,
                        seed=args.seed, sample=args.sample)

    # Attach solver id to each row (for CSV)
    for r in results:
        r["solver_id"] = solver.id

    # Outputs
    from packages.harness.io import timestamp_id, git_commit_or_unknown
    run_id = timestamp_id()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"run_{run_id}.csv"
    manifest_path = outdir / f"run_{run_id}_manifest.json"

    write_csv(results, str(csv_path), max_turns=args.max_turns, N=args.N)
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
