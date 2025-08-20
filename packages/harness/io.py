from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import csv
import json
import subprocess
import datetime as dt


def write_csv(results: List[Dict], path: str, max_turns: int, N: int) -> str:
    """
    Flatten results into a CSV with columns for guess_i/patt_i up to max_turns.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "solver", "N", "answer", "success", "guesses", "time_ms"
    ]
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
                "time_ms": round(r["time_ms"], 3),
            }
            # fill history columns
            hist = r["history"]
            for i in range(1, max_turns + 1):
                if i <= len(hist):
                    row[f"guess_{i}"] = hist[i - 1][0]
                    row[f"patt_{i}"] = hist[i - 1][1]
                else:
                    row[f"guess_{i}"] = ""
                    row[f"patt_{i}"] = ""
            w.writerow(row)
    return str(p)


def write_manifest(manifest: Dict, path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return str(p)


def timestamp_id() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def git_commit_or_unknown() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"
