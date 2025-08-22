from packages.solvers import create_solver
from packages.solvers import entropy  # registers
from packages.harness import run_case


def test_entropy_smoke():
    answers = ["crane","raise","stare","trace","cared","adieu","alone"]
    allowed = answers + ["slate","salet","roate"]
    solver = create_solver("entropy")
    r = run_case(solver, "crane", allowed=allowed, answers=answers, N=5, max_turns=6, seed=7)
    assert r["success"] is True
