import pytest
from packages.engine import score, filter_candidates, validate_guess

# --- N=5 golden tests (duplicates + placements) ---
@pytest.mark.parametrize("guess,answer,expected", [
    ("belle","level","-GYYY"),
    ("level","level","GGGGG"),
    ("lemon","level","GG---"),
    ("cools","scoop","YYG-Y"),
    ("scoop","scoop","GGGGG"),
    ("crane","crane","GGGGG"),
    ("raise","crane","YY--G"),
    ("stare","crane","--GYG"),
])
def test_score_n5_golden(guess, answer, expected):
    assert score(guess, answer) == expected

def test_filter_candidates_n5_history():
    words = ["crane","raise","stare","trace","cared","racer","scoop"]
    history = [("raise","YY--G")]
    cand = filter_candidates(words, history, N=5)
    assert "crane" in cand and "stare" not in cand and "scoop" not in cand

def test_validate_guess_n5():
    allowed = ["crane","raise","stare"]
    assert validate_guess("CRANE", allowed, N=5) is True
    assert validate_guess("cranes", allowed, N=5) is False
    assert validate_guess("???", allowed, N=5) is False

# --- N=6 sample tests ---
@pytest.mark.parametrize("guess,answer,expected", [
    ("settle","letter","-GGGYY"),
    ("little","letter","G-GG-Y"),
    ("planet","palate","GYY-YY"),
    ("kitten","tinket","YGYYGY"),
])
def test_score_n6_samples(guess, answer, expected):
    assert score(guess, answer) == expected

def test_filter_candidates_n6_basic():
    words = ["letter","settle","little","tattle","better"]
    history = [("settle","-GGGYY")]
    cand = filter_candidates(words, history, N=6)
    assert "letter" in cand and "better" not in cand
