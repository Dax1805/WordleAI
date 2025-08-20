from packages.solvers import create_solver
from packages.solvers import random_consistent  # registers
from packages.harness import run_case

def test_run_case_smoke():
    answers = ["crane", "raise", "stare"]
    allowed = ["crane", "raise", "stare", "trace", "cared"]
    solver = create_solver("random_consistent")
    r = run_case(solver, "crane", allowed=allowed, answers=answers, N=5, max_turns=6, seed=42)
    assert "success" in r and "history" in r
    # Should solve within 6 in this tiny set
    assert r["success"] is True
