import io
from pathlib import Path
from packages.datasets import validate_wordlists, pretty_summary


def _write(p: Path, lines):
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_wordlists_happy_path(tmp_path: Path):
    # N=5; all lowercase alpha; answers subset of allowed
    ans = tmp_path / "answers_5.txt"
    allw = tmp_path / "allowed_5.txt"
    _write(ans, ["crane", "raise", "stare"])
    _write(allw, ["crane", "raise", "stare", "trace", "cared"])

    rep = validate_wordlists(5, str(ans), str(allw))
    assert rep["passed"] is True
    assert rep["answers_subset_allowed"] is True
    s = pretty_summary(rep)
    assert "N=5" in s and "answers⊆allowed=True" in s


def test_validate_wordlists_flags_errors(tmp_path: Path):
    # N mismatch and invalid chars should be flagged
    ans = tmp_path / "answers_6.txt"
    allw = tmp_path / "allowed_6.txt"
    # 'crane' (len 5) invalid for N=6, '???' invalid chars, 'raiser' is fine
    ans.write_text("raiser\ncrane\n???\n", encoding="utf-8")
    allw.write_text("raiser\nplanet\npalate\n", encoding="utf-8")

    rep = validate_wordlists(6, str(ans), str(allw))
    assert rep["passed"] is False
    assert any("invalid" in msg for msg in rep["issues"])


def test_validate_wordlists_subset_violation(tmp_path: Path):
    ans = tmp_path / "answers_5.txt"
    allw = tmp_path / "allowed_5.txt"
    _write(ans, ["crane", "raise", "stare"])
    _write(allw, ["crane", "stare"])  # missing 'raise'

    rep = validate_wordlists(5, str(ans), str(allw))
    assert rep["passed"] is False
    assert rep["answers_subset_allowed"] is False
    assert any("subset" in msg for msg in rep["issues"])
