# apps/cli/run_multi.py
"""
Run multiple solvers in one shot with shared sampling and progress.

Writes per-solver outputs to: <outdir>/<solver_id>/run_<timestamp>.csv + _manifest.json
"""

from __future__ import annotations
import argparse, sys, time, random
from pathlib import Path
from typing import Dict, List, Tuple

# Optional progress bar
try:
    from tqdm import tqdm  # pip install tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

from packages.datasets import validate_wordlists, pretty_summary
from packages.harness.io import write_csv, write_manifest, timestamp_id, git_commit_or_unknown
from packages.solvers import create_solver, get_solver_ids

# ensure all solvers self-register on import
from packages.solvers import random_consistent  # noqa: F401
from packages.solvers import letter_freq  # noqa: F401
from packages.solvers import positional_freq  # noqa: F401
from packages.solvers import expected_left  # noqa: F401
from packages.solvers import max_patterns  # noqa: F401
from packages.solvers import entropy  # noqa: F401
from packages.solvers import entropy_weighted  # noqa: F401
from packages.solvers import two_stage_probe  # noqa: F401
from packages.solvers import two_ply_mc  # noqa: F401

from packages.engine import score as score_fn
from packages.engine.constraints import filter_candidates

MAX_TURNS = 6


def _load_words(path: str) -> list[str]:
    p = Path(path)
    return [w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines() if w.strip()]


def _play_one_game(*, solver, answer: str, answers: List[str], allowed: List[str], N: int,
                   seed: int) -> Dict:
    solver.reset(allowed=allowed, answers=answers, N=N, seed=seed)
    candidates = list(answers)
    history: List[Tuple[str, str]] = []
    total_ms = 0.0
    success = False
    for turn in range(1, MAX_TURNS + 1):
        state = {"turn": turn, "N": N, "candidates": candidates, "allowed": allowed,
                 "history": history}
        t0 = time.perf_counter_ns()
        guess = solver.next_guess(state).lower()
        t1 = time.perf_counter_ns()
        total_ms += (t1 - t0) / 1_000_000.0
        patt = score_fn(guess, answer)
        history.append((guess, patt))
        if patt == "G" * N:
            success = True
            break
        candidates = filter_candidates(candidates, [(guess, patt)], N)
    return {"answer": answer, "success": success, "guesses": len(history), "time_ms": total_ms,
            "history": history}


def _progress_mode(mode: str) -> str:
    if mode == "auto":
        return "bar" if (_HAS_TQDM and sys.stderr.isatty()) else "plain"
    return mode


def _run_one_solver(solver_id: str, cases: List[str], *, answers: List[str], allowed: List[str],
                    N: int, base_seed: int, outdir: Path, progress: str) -> Tuple[str, str]:
    solver = create_solver(solver_id)
    results = []
    total = len(cases)
    mode = _progress_mode(progress)
    iterator = tqdm(cases, ncols=80, desc=f"{solver_id}", unit="game") if mode == "bar" else cases
    start = time.time();
    last_print = 0.0

    for idx, ans in enumerate(iterator, 1):
        per_seed = base_seed ^ (hash(solver_id) & 0x7fffffff) + idx * 2654435761
        r = _play_one_game(solver=solver, answer=ans, answers=answers, allowed=allowed, N=N,
                           seed=per_seed)
        r["solver_id"] = solver.id
        results.append(r)

        if mode == "plain":
            now = time.time()
            if (now - last_print >= 1.0) or (idx == total):
                elapsed = now - start
                rate = (idx / elapsed) if elapsed > 0 else 0.0
                remaining = (total - idx) / rate if rate > 0 else 0.0
                pct = 100.0 * idx / max(1, total)
                sys.stderr.write(
                    f"\r[{solver_id}] {idx}/{total} {pct:5.1f}% | elapsed {elapsed:6.1f}s | ETA {remaining:5.1f}s")
                sys.stderr.flush()
                last_print = now
    if mode == "plain":
        sys.stderr.write("\n");
        sys.stderr.flush()

    # write outputs under <outdir>/<solver_id>/
    run_id = timestamp_id()
    sdir = outdir / solver_id
    sdir.mkdir(parents=True, exist_ok=True)
    csv_path = sdir / f"run_{run_id}.csv"
    manifest_path = sdir / f"run_{run_id}_manifest.json"

    write_csv(results, str(csv_path), max_turns=MAX_TURNS, N=N)
    manifest = {
        "run_id": run_id,
        "git_commit": git_commit_or_unknown(),
        "config": {"solver": solver_id, "N": N, "seed": base_seed, "num_cases": len(cases)},
        "wordlists": None,  # filled by caller once (global)
        "num_cases": len(results),
        "solver_id": solver.id,
    }
    write_manifest(manifest, str(manifest_path))
    return str(csv_path), str(manifest_path)


def main():
    registered = get_solver_ids()
    ap = argparse.ArgumentParser(description="wordleAI — run many solvers at once")
    ap.add_argument("--solvers", nargs="+", required=True,
                    help=f"list of solver ids or 'ALL'. Registered: {', '.join(registered)}")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="solver ids to skip (only if --solvers ALL)")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--answers", default="packages/datasets/data/answers_5.txt")
    ap.add_argument("--allowed", default="packages/datasets/data/allowed_5.txt")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--outdir", default="reports/batch")
    ap.add_argument("--progress", choices=["auto", "bar", "plain", "off"], default="auto")
    args = ap.parse_args()

    # 1) validate once
    rep = validate_wordlists(args.N, args.answers, args.allowed)
    print(pretty_summary(rep))

    # 2) load lists once
    answers = _load_words(args.answers)
    allowed = _load_words(args.allowed)

    # 3) shared cases (deterministic by seed)
    rng = random.Random(args.seed)
    if args.sample and args.sample < len(answers):
        pool = list(answers);
        rng.shuffle(pool);
        cases = pool[:args.sample]
    else:
        cases = list(answers)

    # 4) expand solvers
    if len(args.solvers) == 1 and args.solvers[0].lower() == "all":
        todo = [s for s in registered if s not in set(args.exclude)]
    else:
        todo = args.solvers
        missing = [s for s in todo if s not in registered]
        if missing:
            raise SystemExit(f"Unknown solver ids: {missing}. Registered: {registered}")

    outdir = Path(args.outdir);
    outdir.mkdir(parents=True, exist_ok=True)

    # 5) run each solver sequentially (shared cases) with progress
    for sid in todo:
        if args.progress != "off":
            print(f"\n=== Running {sid} on {len(cases)} cases (N={args.N}) ===")
        csv_path, manifest_path = _run_one_solver(
            solver_id=sid, cases=cases, answers=answers, allowed=allowed,
            N=args.N, base_seed=args.seed, outdir=outdir, progress=args.progress
        )
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
