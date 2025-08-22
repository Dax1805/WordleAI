"""
I/O utilities for experiment runs.

Responsibilities:
- write_csv:     flatten per-game results into a tidy CSV (one row per game).
- write_manifest:dump a JSON manifest with config, hashes, and metadata.
- timestamp_id:  stable UTC run ID string.
- git_commit_or_unknown: best-effort short commit hash for reproducibility.

Notes:
- Patterns are prefixed with an apostrophe to keep Excel from interpreting
  strings like "-GYY-" as formulas (which would display as #NAME?).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import csv
import json
import subprocess
import datetime as dt


def _excel_safe_pattern(patt: str) -> str:
    """
    Prefix with an apostrophe so spreadsheet apps treat it as text.
    Example: "-GYY-" -> "'-GYY-"
    """
    return "'" + patt if patt else patt


def write_csv(results: List[Dict], path: str, max_turns: int, N: int) -> str:
    """
    Serialize a batch of game results to CSV.

    Schema (columns):
      solver, N, answer, success, guesses, time_ms,
      guess_1, patt_1, guess_2, patt_2, ..., guess_max_turns, patt_max_turns

    Args:
      results  : list of dicts returned by the harness per game.
      path     : output CSV path.
      max_turns: turn budget (Wordle is 6).
      N        : word length.

    Returns:
      The path written (string).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = ["solver", "N", "answer", "success", "guesses", "time_ms"]
    for i in range(1, max_turns + 1):
        fields += [f"guess_{i}", f"patt_{i}"]

    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for r in results:
            row = {
                "solver": r.get("solver_id", "?"),
                "N": N,
                "answer": r["answer"],
                "success": r["success"],
                "guesses": r["guesses"],
                "time_ms": round(float(r["time_ms"]), 3),
            }

            # Expand history into fixed columns (Excel-safe patterns)
            hist = r.get("history", [])
            for i in range(1, max_turns + 1):
                if i <= len(hist):
                    g, patt = hist[i - 1]
                    row[f"guess_{i}"] = g
                    row[f"patt_{i}"] = _excel_safe_pattern(patt)
                else:
                    row[f"guess_{i}"] = ""
                    row[f"patt_{i}"] = ""

            w.writerow(row)

    return str(p)


def write_manifest(manifest: Dict, path: str) -> str:
    """
    Write a JSON manifest with run configuration and dataset validation summary.

    Typical keys:
      - run_id, git_commit
      - config: CLI args (solver, N, paths, seed, sample, outdir)
      - wordlists: output of datasets.validate_wordlists(...)
      - num_cases: number of games in this batch
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return str(p)


def timestamp_id() -> str:
    """
    Return a compact UTC timestamp suitable for filenames, e.g. 20250820T024121Z.
    """
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def git_commit_or_unknown() -> str:
    """
    Best-effort short git hash of the current repo state.
    Returns 'unknown' if git is not available or the call fails.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
