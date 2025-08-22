"""
Dataset validator for wordleAI.

What this module does:
- Validate a pair of word lists: answers_N.txt (ground-truth pool) and allowed_N.txt (guess universe).
- Enforce formatting rules (lowercase, a–z only, exact length N, one per line).
- Detect duplicates and invalid lines; compute SHA-256 of the raw files.
- Check that answers ⊆ allowed.
- Return a machine-readable dict (for manifests) and provide a pretty one-line summary.

Typical use:
    from packages.datasets import validate_wordlists, pretty_summary
    rep = validate_wordlists(5, "packages/datasets/data/answers_5.txt",
                                "packages/datasets/data/allowed_5.txt")
    print(pretty_summary(rep))
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib


# -----------------------------
# Dataclasses for structured reports
# -----------------------------

@dataclass
class FileReport:
    """Per-file diagnostics and metadata."""
    path: str            # file path (as given)
    exists: bool         # did the file exist on disk?
    count: int           # number of VALID words after cleaning
    sha256: str          # SHA-256 of raw file bytes (empty string if missing)
    unique_count: int    # unique valid words (after dedupe)
    invalid_lines: int   # number of invalid lines encountered


@dataclass
class ValidationReport:
    """Top-level validation result for the (answers, allowed) pair."""
    N: int
    answers: FileReport
    allowed: FileReport
    answers_subset_allowed: bool
    passed: bool
    issues: List[str]    # human-friendly list of problems (if any)


# -----------------------------
# Helpers
# -----------------------------

def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_and_check(path: Path, N: int) -> Tuple[List[str], int]:
    """
    Load words from a text file and validate them.

    Rules:
      - one token per line
      - must be lowercase a–z
      - must have exact length N
      - empty/whitespace-only lines are INVALID

    Returns:
      (valid_words, invalid_count)
    """
    valid: List[str] = []
    invalid = 0

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            w = raw.strip()
            if not w:
                invalid += 1
                continue
            wl = w.lower()
            # require already-lowercase & alphabetic & exact length
            if wl == w and wl.isalpha() and len(wl) == N:
                valid.append(wl)
            else:
                invalid += 1

    return valid, invalid


def _as_dict(rep: ValidationReport) -> Dict:
    """Dataclass → plain dict (stable ordering)."""
    return asdict(rep)


# -----------------------------
# Public API
# -----------------------------

def validate_wordlists(N: int, answers_path: str, allowed_path: str) -> Dict:
    """
    Validate the answers/allowed word lists for length N.

    Parameters
    ----------
    N : int
        Word length (e.g., 5 or 6).
    answers_path : str
        Path to the ground-truth answers file (one word per line).
    allowed_path : str
        Path to the allowed guesses file (should be a superset of answers).

    Returns
    -------
    Dict
        A JSON-serializable dictionary (see ValidationReport schema) with:
          - counts, SHA-256, duplicate/invalid flags
          - answers ⊆ allowed check
          - `passed` boolean (strict: requires non-empty, no invalids, subset OK)
          - `issues` (list of strings) to surface any problems
    """
    issues: List[str] = []

    ans_p = Path(answers_path)
    all_p = Path(allowed_path)

    ans_exists = ans_p.exists()
    all_exists = all_p.exists()

    # Early return if either file is missing
    if not ans_exists or not all_exists:
        if not ans_exists:
            issues.append(f"answers file not found: {answers_path}")
        if not all_exists:
            issues.append(f"allowed file not found: {allowed_path}")
        rep = ValidationReport(
            N=N,
            answers=FileReport(answers_path, ans_exists, 0, "", 0, 0),
            allowed=FileReport(allowed_path, all_exists, 0, "", 0, 0),
            answers_subset_allowed=False,
            passed=False,
            issues=issues,
        )
        return _as_dict(rep)

    # Load and validate content
    answers, ans_invalid = _load_and_check(ans_p, N)
    allowed, all_invalid = _load_and_check(all_p, N)

    # Deduplicate while preserving semantic meaning (set is enough post-validation)
    answers_set = set(answers)
    allowed_set = set(allowed)

    # Build file reports
    ans_report = FileReport(
        path=str(ans_p),
        exists=True,
        count=len(answers),
        sha256=_sha256_file(ans_p),
        unique_count=len(answers_set),
        invalid_lines=ans_invalid,
    )
    all_report = FileReport(
        path=str(all_p),
        exists=True,
        count=len(allowed),
        sha256=_sha256_file(all_p),
        unique_count=len(allowed_set),
        invalid_lines=all_invalid,
    )

    # Logical checks & issue collection
    subset_ok = answers_set.issubset(allowed_set)
    if not subset_ok:
        # Surface a few examples to debug quickly (limit to 5 for brevity)
        missing = list(answers_set - allowed_set)[:5]
        issues.append(f"answers not subset of allowed (e.g., {missing})")

    # Empty-file guardrails (useful to catch bad paths or preprocessing bugs)
    if ans_report.count == 0:
        issues.append("answers file contains 0 valid words")
    if all_report.count == 0:
        issues.append("allowed file contains 0 valid words")

    # Invalid-line diagnostics
    if ans_invalid:
        issues.append(f"answers has {ans_invalid} invalid line(s)")
    if all_invalid:
        issues.append(f"allowed has {all_invalid} invalid line(s)")

    # Duplicate diagnostics (count vs unique_count mismatch)
    if ans_report.count != ans_report.unique_count:
        issues.append("answers contains duplicate lines")
    if all_report.count != all_report.unique_count:
        issues.append("allowed contains duplicate lines")

    # Strict pass criteria: non-empty + no invalids + subset ok
    passed = (
            subset_ok
            and ans_invalid == 0
            and all_invalid == 0
            and ans_report.count > 0
            and all_report.count > 0
    )

    rep = ValidationReport(
        N=N,
        answers=ans_report,
        allowed=all_report,
        answers_subset_allowed=subset_ok,
        passed=passed,
        issues=issues,
    )
    return _as_dict(rep)


def pretty_summary(report: Dict) -> str:
    """
    Produce a compact, human-friendly one-liner for console/docs.

    Example:
        N=5 | answers=2315 (uniq=2315, sha=abc123...) | allowed=10657 (uniq=10657, sha=def456...) | answers⊆allowed=True | OK
    """
    N = report["N"]
    a = report["answers"]
    b = report["allowed"]
    subset = report["answers_subset_allowed"]
    status = "OK" if report["passed"] else "FAIL"
    # abbreviate sha to 12 chars for readability
    a_sha = (a.get("sha256") or "")[:12]
    b_sha = (b.get("sha256") or "")[:12]
    return (
        f"N={N} | answers={a['count']} (uniq={a['unique_count']}, sha={a_sha}) "
        f"| allowed={b['count']} (uniq={b['unique_count']}, sha={b_sha}) "
        f"| answers⊆allowed={subset} | {status}"
    )
