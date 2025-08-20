from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib


# --------- types ---------
@dataclass
class FileReport:
    path: str
    exists: bool
    count: int
    sha256: str
    unique_count: int
    invalid_lines: int


@dataclass
class ValidationReport:
    N: int
    answers: FileReport
    allowed: FileReport
    answers_subset_allowed: bool
    passed: bool
    issues: List[str]


# --------- helpers ---------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_and_check(path: Path, N: int) -> Tuple[List[str], int]:
    """
    Returns (valid_words, invalid_count). Valid words are:
      - lowercase a-z only
      - exact length N
    Lines are stripped; empty lines are ignored but counted as invalid.
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
            if wl == w and wl.isalpha() and len(wl) == N:
                valid.append(wl)
            else:
                invalid += 1
    return valid, invalid


# --------- public API ---------
def validate_wordlists(N: int, answers_path: str, allowed_path: str) -> Dict:
    """
    Validate wordlists for a given N and return a dictionary summary suitable for JSON.
    Also include human-friendly fields; use pretty_summary() to print a one-liner.
    """
    issues: List[str] = []

    ans_p = Path(answers_path)
    all_p = Path(allowed_path)

    ans_exists = ans_p.exists()
    all_exists = all_p.exists()
    if not ans_exists:
        issues.append(f"answers file not found: {answers_path}")
    if not all_exists:
        issues.append(f"allowed file not found: {allowed_path}")

    # Build default empty reports if files missing
    if not (ans_exists and all_exists):
        rep = ValidationReport(
            N=N,
            answers=FileReport(answers_path, ans_exists, 0, "", 0, 0),
            allowed=FileReport(allowed_path, all_exists, 0, "", 0, 0),
            answers_subset_allowed=False,
            passed=False,
            issues=issues,
        )
        return _to_dict(rep)

    # Load/validate content
    answers, ans_invalid = _load_and_check(ans_p, N)
    allowed, all_invalid = _load_and_check(all_p, N)

    # Dedup
    answers_set = set(answers)
    allowed_set = set(allowed)

    # Reports
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

    # Subset check
    subset_ok = answers_set.issubset(allowed_set)
    if not subset_ok:
        # Identify a few missing examples (at most 5 to keep output tidy)
        missing = list(answers_set - allowed_set)[:5]
        issues.append(f"answers not subset of allowed (e.g., {missing})")

    # Additional quality checks
    if ans_report.count == 0:
        issues.append("answers file contains 0 valid words")
    if all_report.count == 0:
        issues.append("allowed file contains 0 valid words")
    if ans_invalid:
        issues.append(f"answers has {ans_invalid} invalid line(s)")
    if all_invalid:
        issues.append(f"allowed has {all_invalid} invalid line(s)")
    if len(answers) != len(answers_set):
        issues.append("answers contains duplicate lines")
    if len(allowed) != len(allowed_set):
        issues.append("allowed contains duplicate lines")

    passed = subset_ok and ans_invalid == 0 and all_invalid == 0

    rep = ValidationReport(
        N=N,
        answers=ans_report,
        allowed=all_report,
        answers_subset_allowed=subset_ok,
        passed=passed,
        issues=issues,
    )
    return _to_dict(rep)


def pretty_summary(report: Dict) -> str:
    """
    Human-friendly one-liner you can paste into docs/wordlists.md.
    """
    N = report["N"]
    a = report["answers"]
    b = report["allowed"]
    subset = report["answers_subset_allowed"]
    status = "OK" if report["passed"] else "FAIL"
    return (
        f"N={N} | answers={a['count']} (uniq={a['unique_count']}, sha={a['sha256'][:12]}) "
        f"| allowed={b['count']} (uniq={b['unique_count']}, sha={b['sha256'][:12]}) "
        f"| answers⊆allowed={subset} | {status}"
    )


# --------- utils ---------
def _to_dict(rep: ValidationReport) -> Dict:
    d = asdict(rep)
    # Keep keys order predictable for manifests
    return d
