# apps/cli/eval_bandit.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from packages.agent.env import WordleBanditEnv
from packages.agent.bandit_linucb import LinUCB


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained LinUCB policy.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--allowed", required=True)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--alpha-time", type=float, default=0.2)
    ap.add_argument("--outdir", default="reports/bandit_eval")
    args = ap.parse_args()

    outdir = Path(args.outdir);
    outdir.mkdir(parents=True, exist_ok=True)
    model = LinUCB.from_json(Path(args.model).read_text(encoding="utf-8"))

    env = WordleBanditEnv(answers_path=args.answers, allowed_path=args.allowed, N=args.N,
                          seed=args.seed, actions=model.actions, alpha_time=args.alpha_time)

    # run subset deterministically by seed
    rng = np.random.default_rng(args.seed)
    answers_all = env.answers_all
    if args.sample and args.sample < len(answers_all):
        eval_answers = list(rng.choice(answers_all, size=args.sample, replace=False))
    else:
        eval_answers = answers_all

    out_csv = outdir / "bandit_eval.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solver", "N", "answer", "success", "guesses", "time_ms",
                    "guess_1", "guess_2", "guess_3", "guess_4", "guess_5", "guess_6"])
        for ans in eval_answers:
            obs = env.reset(answer=ans)
            traj = []
            total_ms = 0.0
            while True:
                x = np.array(obs["features"], dtype=float)
                action = model.select(x)
                obs, r, done, info = env.step(action)
                traj.append(
                    (info["guess"], info["pattern"], info["chosen_solver"], info["time_ms"]))
                total_ms += info["time_ms"]
                if done:
                    solved = (traj[-1][1] == "G" * args.N)
                    guesses = len(traj)
                    row = ["meta_linucb", args.N, ans, solved, guesses, round(total_ms, 2)]
                    row += [g for g, _, _, _ in traj] + [""] * (6 - len(traj))
                    w.writerow(row)
                    break
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
