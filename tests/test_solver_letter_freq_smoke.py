from packages.solvers import create_solver
from packages.solvers import letter_freq  # registers
from packages.harness import run_case


def test_letter_freq_smoke():
    answers = ["crane","raise","stare","trace","cared"]
    allowed = answers + ["adieu","arise"]
    solver = create_solver("letter_freq")
    r = run_case(solver, "crane", allowed=allowed, answers=answers, N=5, max_turns=6, seed=42)
    assert r["success"] is True
